from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Protocol

import numpy as np

from common.disk_audio import (
    DiskAudioStore,
    DiskFloat32Spool,
    DiskFrameSeries,
    DiskFrameSpool,
    DiskUInt8Series,
    DiskUInt8Timeline,
)

from layer4_speech_separation import (
    BandMagnitudeMatcher,
    Layer4CandidatePair,
    Layer4LongAudioInput,
    Layer4OfflineResult,
    Layer4ProcessedAudio,
    Layer4Resampler,
    SpeakerCountDecision,
)
from layer4_speech_separation.offline import OfflineLayer4Pipeline
from layer4_speech_separation.streaming import (
    Layer4StreamInputChunk,
    Layer4StreamOutputChunk,
    Layer4StreamSession,
)
from layer6_speaker_consolidation import OfflineLayer6Pipeline

from .realtime_postprocessing import RealtimePostprocessingSnapshot


class _QualityScorer(Protocol):
    def score(self, waveform_16k: np.ndarray) -> tuple[float, float, float]: ...


class _EmbeddingBackend(Protocol):
    def embed(self, waveform_16k: np.ndarray) -> np.ndarray: ...

    def embed_batch(self, waveforms_16k: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]: ...


def _digest(waveform: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(waveform)).cast("B")).hexdigest()


def _mos_score(values: tuple[float, float, float]) -> float:
    sig, bak, ovrl = (float(value) for value in values)
    if any(not np.isfinite(value) or not 1.0 <= value <= 5.0 for value in (sig, bak, ovrl)):
        raise ValueError("realtime L4 DNSMOS scores must be in [1,5]")
    return float(np.clip(((0.25 * sig + 0.25 * bak + 0.50 * ovrl) - 1.0) / 4.0, 0.0, 1.0))


class CachingEmbeddingBackend:
    """Reuse unchanged 2 s CAMPPlus evidence across provisional L6 revisions."""

    def __init__(
        self,
        backend: _EmbeddingBackend,
        *,
        max_segments: int | None = None,
    ) -> None:
        if max_segments is not None and (
            type(max_segments) is not int or max_segments <= 0
        ):
            raise ValueError("CAMPPlus cache limit must be a positive integer or None")
        self.backend = backend
        self.max_segments = max_segments
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @property
    def cached_segments(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()

    def embed(self, waveform_16k: np.ndarray) -> np.ndarray:
        return self.embed_batch((waveform_16k,))[0]

    def embed_batch(self, waveforms_16k: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        waveforms = tuple(np.ascontiguousarray(value, dtype=np.float32) for value in waveforms_16k)
        keys = tuple(_digest(value) for value in waveforms)
        resolved: dict[str, np.ndarray] = {}
        missing_keys: list[str] = []
        missing_key_set: set[str] = set()
        missing_audio: list[np.ndarray] = []
        for key, waveform in zip(keys, waveforms, strict=True):
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                resolved[key] = cached
            elif key not in missing_key_set:
                missing_keys.append(key)
                missing_key_set.add(key)
                missing_audio.append(waveform)
        if missing_audio:
            batch = getattr(self.backend, "embed_batch", None)
            values = (
                tuple(batch(tuple(missing_audio)))
                if callable(batch)
                else tuple(self.backend.embed(value) for value in missing_audio)
            )
            if len(values) != len(missing_keys):
                raise RuntimeError("CAMPPlus did not return every requested embedding")
            for key, value in zip(missing_keys, values, strict=True):
                embedding = np.ascontiguousarray(value, dtype=np.float32)
                embedding.flags.writeable = False
                self._cache[key] = embedding
                resolved[key] = embedding
        if self.max_segments is not None:
            while len(self._cache) > self.max_segments:
                self._cache.popitem(last=False)
        return tuple(resolved[key] for key in keys)


_L5_CONTEXT_SAMPLES_16K = 80 * 320  # 1.6 s on each side of a stable decision.
_DNSMOS_WINDOW_SAMPLES_16K = 451 * 320  # 9.02 s, complete 20 ms hops.


@dataclass(slots=True)
class _BranchState:
    stable_branch_id: int
    start_sample_48k: int
    audio_spool: DiskFloat32Spool
    probability_spool: DiskFrameSpool
    decision_spool: DiskFrameSpool
    audio_samples_16k: int = 0
    l5_samples_16k: int = 0
    l5_probability_sum: float = 0.0
    l5_probability_count: int = 0
    l5_probability_tail: deque[float] = field(
        default_factory=lambda: deque(maxlen=3)
    )
    l5_summary_probability: float = 0.0
    l5_model_id: str | None = None
    l5_metadata: dict[str, object] = field(default_factory=dict)
    match_weighted_sum: float = 0.0
    match_weight_samples: int = 0
    commit_count: int = 0
    minimum_peak_gain: float = 1.0
    dnsmos_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    dnsmos_sample_count: int = 0
    next_dnsmos_sample_16k: int = _DNSMOS_WINDOW_SAMPLES_16K
    last_dnsmos_end_sample_16k: int = 0
    final_dnsmos: tuple[float, float, float] | None = None
    published_audio_hasher: object = field(default_factory=hashlib.sha256, repr=False)
    published_audio_samples_16k: int = 0

    def append_audio(self, waveform: np.ndarray) -> None:
        value = np.ascontiguousarray(waveform, dtype=np.float32)
        start, end = self.audio_spool.append(value)
        if start != self.audio_samples_16k:
            raise RuntimeError("realtime branch spool lost its append position")
        self.audio_samples_16k += len(value)
        if end != self.audio_samples_16k:
            raise RuntimeError("realtime branch spool length is inconsistent")

    def audio_range(self, start: int, end: int) -> np.ndarray:
        if start < 0 or end <= start or end > self.audio_samples_16k:
            raise ValueError("realtime branch audio range is invalid")
        return self.audio_spool.view(start, end)

    def audio_digest(self, start: int, end: int) -> str:
        return self.audio_spool.digest(start, end)

    def published_audio_digest(self, end: int) -> str:
        """Hash a monotonically growing published prefix exactly once."""

        if end < self.published_audio_samples_16k:
            # This is not expected in the progressive path, but retaining a
            # bounded fallback keeps diagnostic callers total.
            return self.audio_digest(0, end)
        if end > self.published_audio_samples_16k:
            self.audio_spool.update_digest(
                self.published_audio_hasher,  # type: ignore[arg-type]
                self.published_audio_samples_16k,
                end,
            )
            self.published_audio_samples_16k = end
        return self.published_audio_hasher.copy().hexdigest()  # type: ignore[attr-defined]

    def probabilities(self, samples_16k: int) -> DiskFrameSeries:
        frame_count = samples_16k // 320
        if self.probability_spool.count < frame_count:
            raise ValueError("realtime L5 probability archive is shorter than its watermark")
        return self.probability_spool.series(frame_count)

    def decisions(self, samples_16k: int) -> DiskFrameSeries:
        frame_count = samples_16k // 320
        if self.decision_spool.count < frame_count:
            raise ValueError("realtime L5 decision archive is shorter than its watermark")
        return self.decision_spool.series(frame_count)

    def observe_probabilities(self, values: np.ndarray) -> None:
        for raw in values:
            probability = float(raw)
            self.l5_probability_sum += probability
            self.l5_probability_count += 1
            self.l5_probability_tail.append(probability)
            if self.l5_probability_count < 3:
                self.l5_summary_probability = (
                    self.l5_probability_sum / self.l5_probability_count
                )
            else:
                self.l5_summary_probability = max(
                    self.l5_summary_probability,
                    sum(self.l5_probability_tail) / 3.0,
                )

    @property
    def match_score(self) -> float:
        if not self.match_weight_samples:
            return 1.0
        return float(self.match_weighted_sum / self.match_weight_samples)

    @property
    def dnsmos(self) -> tuple[float, float, float]:
        if self.final_dnsmos is not None:
            return self.final_dnsmos
        if self.dnsmos_sample_count:
            return tuple(float(value) for value in self.dnsmos_sum / self.dnsmos_sample_count)  # type: ignore[return-value]
        return 3.0, 3.0, 3.0


@dataclass(slots=True)
class _SourcePart:
    asset_id: str
    sha256: str
    session_id: str
    stream_epoch: int
    track_id: int
    theta_deg: float
    start_sample: int
    end_sample: int
    l2_direction_counts: tuple[tuple[int, int], ...]

    @classmethod
    def from_input(cls, source: Layer4LongAudioInput) -> "_SourcePart":
        return cls(
            source.asset_id,
            source.sha256,
            source.session_id,
            source.stream_epoch,
            source.track_id,
            source.theta_deg,
            source.start_sample,
            source.end_sample,
            source.l2_direction_counts,
        )


@dataclass(slots=True)
class _TrackState:
    speaker_count: int
    session: Layer4StreamSession
    source_spool: DiskFloat32Spool
    direction_spool: DiskUInt8Timeline
    source_base_sample_48k: int
    original_start_sample_48k: int
    track_identity: tuple[str, int, int]
    inputs: deque[_SourcePart] = field(default_factory=deque)
    branches: dict[int, _BranchState] = field(default_factory=dict)
    committed_end_sample_48k: int | None = None
    preview_start_sample_48k: int | None = None
    preview_degraded_reason: str | None = None
    replay_dropped_samples_48k: int = 0
    finalized: bool = False
    published_source_hasher: object = field(default_factory=hashlib.sha256, repr=False)
    published_source_start_sample_48k: int | None = None
    published_source_end_sample_48k: int | None = None

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.track_identity

    @property
    def start_sample_48k(self) -> int:
        return self.original_start_sample_48k


class _DirectionCountSeries(Sequence[tuple[int, int]]):
    _disk_direction_trusted = True

    def __init__(
        self,
        values: DiskUInt8Series,
        *,
        start_sample: int,
        maximum_count: int,
    ) -> None:
        self.values = values
        self.start_sample = int(start_sample)
        self.maximum_count = int(maximum_count)

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for index, count in enumerate(self.values):
            yield self.start_sample + (index + 1) * 960, int(count)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return tuple(self)[index]
        resolved = index + len(self) if index < 0 else index
        if resolved < 0 or resolved >= len(self):
            raise IndexError("direction-count history index out of range")
        return self.start_sample + (resolved + 1) * 960, int(self.values[resolved])


class IncrementalLayer456Processor:
    """Progressive MF2 -> context-stable L5 -> throttled provisional L6 processor.

    The class runs only on the sidecar worker.  It never touches the L1-L3
    queues and deliberately publishes complete replaceable snapshots so the
    final sealed offline pass can supersede them without rewriting window
    audit records. Snapshot ``is_final`` means only that this preview's tails
    were flushed; realtime output remains non-canonical.
    """

    def __init__(
        self,
        *,
        backend: object,
        layer5: object,
        quality_scorer: _QualityScorer,
        embedder: _EmbeddingBackend,
        layer6_config: object,
        chunk_samples_48k: int = 480_000,
        overlap_samples_48k: int = 48_000,
        l5_context_samples_16k: int = _L5_CONTEXT_SAMPLES_16K,
        dnsmos_interval_samples_16k: int = 30 * 16_000,
        l6_interval_samples_48k: int | None = None,
        max_replay_samples_48k: int = 60 * 48_000,
        embedding_cache_segments: int | None = None,
        spool_min_free_bytes: int = 0,
    ) -> None:
        if l6_interval_samples_48k is None:
            l6_interval_samples_48k = chunk_samples_48k
        if (
            type(chunk_samples_48k) is not int
            or not 3 * 48_000 <= chunk_samples_48k <= 15 * 48_000
            or chunk_samples_48k <= overlap_samples_48k
            or chunk_samples_48k % 960
        ):
            raise ValueError("realtime L4 chunk must be 3..15 s, exceed overlap, and align to 20 ms")
        if (
            type(overlap_samples_48k) is not int
            or overlap_samples_48k <= 0
            or overlap_samples_48k % 960
        ):
            raise ValueError("realtime L4 overlap must align to 20 ms")
        if (
            type(l5_context_samples_16k) is not int
            or l5_context_samples_16k <= 0
            or l5_context_samples_16k % 320
        ):
            raise ValueError("realtime L5 context must align to 20 ms")
        if type(dnsmos_interval_samples_16k) is not int or dnsmos_interval_samples_16k <= 0:
            raise ValueError("realtime DNSMOS interval must be positive")
        if (
            type(l6_interval_samples_48k) is not int
            or l6_interval_samples_48k <= 0
            or l6_interval_samples_48k % 960
        ):
            raise ValueError("realtime L6 interval must align to 20 ms")
        if (
            type(max_replay_samples_48k) is not int
            or max_replay_samples_48k <= 0
            or max_replay_samples_48k % 960
        ):
            raise ValueError("realtime replay limit must align to 20 ms")
        if type(spool_min_free_bytes) is not int or spool_min_free_bytes < 0:
            raise ValueError("realtime L4-L6 disk reserve must be a non-negative integer")
        self.backend = backend
        self.layer5 = layer5
        self.quality_scorer = quality_scorer
        self.resampler = Layer4Resampler()
        self.matcher = BandMagnitudeMatcher()
        self.chunk_samples_48k = int(chunk_samples_48k)
        self.overlap_samples_48k = int(overlap_samples_48k)
        self.l5_context_samples_16k = int(l5_context_samples_16k)
        self.dnsmos_interval_samples_16k = int(dnsmos_interval_samples_16k)
        self.l6_interval_samples_48k = int(l6_interval_samples_48k)
        self.max_replay_samples_48k = int(max_replay_samples_48k)
        self.cached_embedder = (
            embedder
            if isinstance(embedder, CachingEmbeddingBackend)
            else CachingEmbeddingBackend(
                embedder, max_segments=embedding_cache_segments,
            )
        )
        self.layer6 = OfflineLayer6Pipeline(
            self.cached_embedder,
            layer6_config,
            spool_min_free_bytes=spool_min_free_bytes,
        )
        # Reuse the proven L5 conversion/validation path. The offline object is
        # model-light: it references the already resident backend and engines.
        backend_id = str(getattr(backend, "backend", "mossformer2_ss_16k"))
        self._l5_adapter = OfflineLayer4Pipeline(
            speaker_counter=object(),  # process_l5 never consults this dependency
            backends={backend_id: backend},
            layer5=layer5,
            quality_scorer=quality_scorer,
            default_backend=backend_id,
            resampler=self.resampler,
            matcher=self.matcher,
        )
        self._storage = DiskAudioStore(
            prefix="micarray_realtime_l456_",
            minimum_free_bytes=spool_min_free_bytes,
        )
        self._tracks: dict[tuple[str, int, int], _TrackState] = {}
        self._session_id: str | None = None
        self._revision = 0
        self._processed_blocks = 0
        self._finalized = False
        self._stage_seconds = {
            "l4": 0.0,
            "dnsmos": 0.0,
            "l5": 0.0,
            "l6": 0.0,
            "snapshot": 0.0,
        }
        self._topology_revision = 0
        self._last_l6_result: object | None = None
        self._last_l6_watermark_48k: int | None = None
        self._last_l6_topology_revision = -1
        self._last_l6_ready_keys: tuple[tuple[str, int, int], ...] = ()
        self._last_published_signature: tuple[object, ...] | None = None

    def _new_session(self, speaker_count: int) -> Layer4StreamSession:
        return Layer4StreamSession(
            speaker_count=speaker_count,  # type: ignore[arg-type]
            backend=None if speaker_count == 1 else self.backend,  # type: ignore[arg-type]
            resampler=self.resampler,
            batch_samples_48k=self.chunk_samples_48k,
            overlap_samples_48k=self.overlap_samples_48k,
        )

    @staticmethod
    def _speaker_count(source: Layer4LongAudioInput) -> int:
        return min(2, max(count for _, count in source.l2_direction_counts))

    @staticmethod
    def _validate_next_source(state: _TrackState, source: _SourcePart) -> None:
        previous = state.inputs[-1]
        if (
            (source.session_id, source.stream_epoch, source.track_id) != state.identity
            or source.start_sample != previous.end_sample
        ):
            raise ValueError("realtime L4 source chunks must be contiguous per authoritative ID")

    @staticmethod
    def _source_range(
        state: _TrackState,
        start_sample: int,
        end_sample: int,
        *,
        sha256: str | None = None,
    ) -> Layer4LongAudioInput:
        if (
            start_sample < state.source_base_sample_48k
            or end_sample <= start_sample
            or end_sample - state.source_base_sample_48k > state.source_spool.length_samples
        ):
            raise ValueError("realtime output exceeds retained L3 source chunks")
        waveform = state.source_spool.view(
            start_sample - state.source_base_sample_48k,
            end_sample - state.source_base_sample_48k,
        )
        relative_start = (start_sample - state.source_base_sample_48k) // 960
        relative_end = (end_sample - state.source_base_sample_48k) // 960
        counts = _DirectionCountSeries(
            state.direction_spool.series(relative_start, relative_end),
            start_sample=start_sample,
            maximum_count=state.speaker_count,
        )
        return Layer4LongAudioInput(
            asset_id=(
                f"{state.identity[0]}:epoch{state.identity[1]}:track{state.identity[2]}:"
                f"realtime-start{start_sample}-end{end_sample}"
            ),
            sha256=(
                sha256
                if sha256 is not None
                else state.source_spool.digest(
                    start_sample - state.source_base_sample_48k,
                    end_sample - state.source_base_sample_48k,
                )
            ),
            session_id=state.identity[0],
            stream_epoch=state.identity[1],
            track_id=state.identity[2],
            theta_deg=state.inputs[-1].theta_deg,
            start_sample=start_sample,
            sample_rate=48_000,
            waveform=waveform,
            l2_direction_counts=counts,
        )

    @staticmethod
    def _published_source_digest(
        state: _TrackState,
        start_sample: int,
        end_sample: int,
    ) -> str:
        """Hash only source samples newly exposed by the stable watermark."""

        if (
            state.published_source_start_sample_48k != start_sample
            or state.published_source_end_sample_48k is None
            or state.published_source_end_sample_48k > end_sample
        ):
            state.published_source_hasher = hashlib.sha256()
            state.published_source_start_sample_48k = start_sample
            state.published_source_end_sample_48k = start_sample
        cursor = int(state.published_source_end_sample_48k)
        if end_sample > cursor:
            state.source_spool.update_digest(
                state.published_source_hasher,  # type: ignore[arg-type]
                cursor - state.source_base_sample_48k,
                end_sample - state.source_base_sample_48k,
            )
            state.published_source_end_sample_48k = end_sample
        return state.published_source_hasher.copy().hexdigest()  # type: ignore[attr-defined]

    def _scores(
        self,
        source: Layer4LongAudioInput,
        outputs: tuple[Layer4StreamOutputChunk, ...],
    ) -> tuple[float, ...]:
        if len(outputs) == 1:
            return (1.0,)
        request_id = outputs[0].request_id or "realtime-l4"
        candidates = Layer4CandidatePair(
            request_id=request_id,
            model_id=str(getattr(self.backend, "model_id", "mossformer2")),
            model_revision=str(getattr(self.backend, "model_revision", "realtime")),
            sample_rate=16_000,
            sources=(outputs[0].waveform_16k, outputs[1].waveform_16k),
        )
        selected = self.matcher.select(
            parent=source,
            reference_16k=self.resampler.to_16k(source.waveform),
            candidates=candidates,
        )
        return tuple(float(value) for value in selected.candidate_scores)

    @staticmethod
    def _decision(state: _TrackState, source: Layer4LongAudioInput) -> SpeakerCountDecision:
        return SpeakerCountDecision(
            source.asset_id,
            state.speaker_count,  # type: ignore[arg-type]
            1.0,
            "l2_direction_count_streaming_max_v1",
            {"streaming": True, "aggregation": "monotonic min(2, maximum)"},
        )

    def _update_dnsmos(self, branch: _BranchState, *, final: bool) -> None:
        if not branch.audio_samples_16k or (final and branch.final_dnsmos is not None):
            return
        started = perf_counter()

        def score_window(end: int) -> None:
            start = max(0, end - _DNSMOS_WINDOW_SAMPLES_16K)
            waveform = branch.audio_range(start, end)
            if len(waveform) < _DNSMOS_WINDOW_SAMPLES_16K:
                waveform = np.ascontiguousarray(
                    np.resize(waveform, _DNSMOS_WINDOW_SAMPLES_16K),
                    dtype=np.float32,
                )
            values = tuple(
                float(value) for value in self.quality_scorer.score(waveform)
            )
            _mos_score(values)
            branch.dnsmos_sum += np.asarray(values, dtype=np.float64)
            branch.dnsmos_sample_count += 1
            branch.last_dnsmos_end_sample_16k = end

        while (
            not final
            and branch.audio_samples_16k >= branch.next_dnsmos_sample_16k
        ):
            score_window(branch.next_dnsmos_sample_16k)
            branch.next_dnsmos_sample_16k += self.dnsmos_interval_samples_16k
        if final:
            # DNSMOS affects ranking metadata, not audio correctness. Reuse
            # periodic evidence and score only the unobserved tail so final
            # sealing never blocks on a complete-branch quality-only rerun.
            if branch.last_dnsmos_end_sample_16k != branch.audio_samples_16k:
                score_window(branch.audio_samples_16k)
            branch.final_dnsmos = tuple(
                float(value)
                for value in branch.dnsmos_sum / branch.dnsmos_sample_count
            )  # type: ignore[assignment]
        self._stage_seconds["dnsmos"] += perf_counter() - started

    def _advance_l5(self, state: _TrackState, *, final: bool) -> None:
        l5_started = perf_counter()
        pending: list[
            tuple[_BranchState, int, int, Layer4ProcessedAudio]
        ] = []
        source_cache: dict[tuple[int, int], Layer4LongAudioInput] = {}
        for branch in state.branches.values():
            stable_end = (
                branch.audio_samples_16k
                if final
                else max(
                    branch.l5_samples_16k,
                    branch.audio_samples_16k - self.l5_context_samples_16k,
                )
            )
            if stable_end <= branch.l5_samples_16k:
                continue
            context_start = max(
                0, branch.l5_samples_16k - self.l5_context_samples_16k,
            )
            waveform = branch.audio_range(context_start, branch.audio_samples_16k)
            source_start = branch.start_sample_48k + context_start * 3
            source_end = branch.start_sample_48k + branch.audio_samples_16k * 3
            source_key = (source_start, source_end)
            source = source_cache.get(source_key)
            if source is None:
                source = self._source_range(state, source_start, source_end)
                source_cache[source_key] = source
            decision = self._decision(state, source)
            kind = (
                "merged"
                if state.speaker_count == 1
                else f"candidate_{branch.stable_branch_id}"
            )
            output_id = (
                f"{source.asset_id}:l4:realtime:l5-context:"
                f"stable{branch.stable_branch_id}"
            )
            dnsmos = branch.dnsmos
            processed = Layer4ProcessedAudio(
                request_id=f"{source.asset_id}:realtime-l5-context",
                source=source,
                speaker_count=decision,
                path=(
                    "single_speaker_bypass"
                    if state.speaker_count == 1
                    else "two_speaker_separation"
                ),
                selected=None,
                output_asset_id=output_id,
                output_sha256=_digest(waveform),
                waveform_16k=waveform,
                metadata={
                    "backend": str(getattr(self.backend, "backend", "mossformer2_ss_16k")),
                    "model_id": str(getattr(self.backend, "model_id", "mossformer2")),
                    "model_revision": str(getattr(self.backend, "model_revision", "realtime")),
                    "stable_branch_id": branch.stable_branch_id,
                    "candidate_index": branch.stable_branch_id,
                    "candidate_match_score": branch.match_score,
                    "pcm16_peak_safety_gain": branch.minimum_peak_gain,
                    "dnsmos_sig": dnsmos[0],
                    "dnsmos_bak": dnsmos[1],
                    "dnsmos_ovrl": dnsmos[2],
                    "mos_score": _mos_score(dnsmos),
                    "realtime_l5_context_samples_16k": self.l5_context_samples_16k,
                    "realtime_provisional": True,
                },
                output_kind=kind,  # type: ignore[arg-type]
            )
            pending.append((branch, stable_end, context_start, processed))
        if not pending:
            self._stage_seconds["l5"] += perf_counter() - l5_started
            return
        l5_values = self._l5_adapter.process_l5_sealed(
            tuple(item[3] for item in pending)
        )
        for (branch, stable_end, context_start, _processed), l5 in zip(
            pending, l5_values, strict=True,
        ):
            first_frame = (branch.l5_samples_16k - context_start) // 320
            last_frame = (stable_end - context_start) // 320
            probabilities = np.ascontiguousarray(
                np.asarray(l5.l5_probabilities_20ms[first_frame:last_frame], dtype=np.float32),
            )
            decisions = np.ascontiguousarray(
                np.asarray(l5.l5_is_voice_20ms[first_frame:last_frame], dtype=bool),
            )
            expected_frames = (stable_end - branch.l5_samples_16k) // 320
            if len(probabilities) != expected_frames or len(decisions) != expected_frames:
                raise RuntimeError("realtime L5 context crop changed the authoritative timeline")
            probability_start, probability_end = branch.probability_spool.append(probabilities)
            decision_start, decision_end = branch.decision_spool.append(decisions)
            expected_start = branch.l5_samples_16k // 320
            if (
                probability_start != expected_start
                or decision_start != expected_start
                or probability_end != decision_end
            ):
                raise RuntimeError("realtime L5 frame spools lost alignment")
            branch.observe_probabilities(probabilities)
            branch.l5_samples_16k = stable_end
            branch.l5_model_id = l5.l5_model_id
            branch.l5_metadata = {
                key: value
                for key, value in l5.metadata.items()
                if key != "output_waveform_16k"
            }
        self._stage_seconds["l5"] += perf_counter() - l5_started

    def _process_output_group(
        self,
        state: _TrackState,
        outputs: tuple[Layer4StreamOutputChunk, ...],
    ) -> None:
        l4_started = perf_counter()
        dnsmos_before = self._stage_seconds["dnsmos"]
        if not outputs:
            return
        start = outputs[0].start_sample_48k
        end = outputs[0].end_sample_48k
        if any(
            (item.start_sample_48k, item.end_sample_48k, item.commit_id) !=
            (start, end, outputs[0].commit_id)
            for item in outputs
        ):
            raise ValueError("one realtime L4 commit must align every branch")
        if len(outputs) != state.speaker_count:
            raise ValueError("realtime L4 commit did not return every expected branch")
        source = self._source_range(state, start, end)
        scores = self._scores(source, outputs)
        for output, match_score in zip(outputs, scores, strict=True):
            waveform = np.ascontiguousarray(output.waveform_16k, dtype=np.float32)
            peak = float(np.max(np.abs(waveform)))
            peak_gain = 1.0
            ceiling = 32767.0 / 32768.0
            if peak > ceiling:
                peak_gain = ceiling / peak
                waveform = np.ascontiguousarray(waveform * np.float32(peak_gain), dtype=np.float32)
            branch = state.branches.get(output.branch_id)
            if branch is None:
                branch = _BranchState(
                    output.branch_id,
                    start,
                    self._storage.create_spool(
                        f"track_{state.identity[2]}_branch_{output.branch_id}"
                    ),
                    self._storage.create_frame_spool(
                        f"track_{state.identity[2]}_branch_{output.branch_id}_probability",
                        dtype=np.dtype(np.float32),
                    ),
                    self._storage.create_frame_spool(
                        f"track_{state.identity[2]}_branch_{output.branch_id}_decision",
                        dtype=np.dtype(np.bool_),
                    ),
                )
                state.branches[output.branch_id] = branch
            expected_start = branch.start_sample_48k + branch.audio_samples_16k * 3
            if start != expected_start:
                raise ValueError("realtime stable branch archive contains a timeline gap")
            branch.append_audio(waveform)
            branch.match_weighted_sum += match_score * len(waveform)
            branch.match_weight_samples += len(waveform)
            branch.commit_count += 1
            branch.minimum_peak_gain = min(branch.minimum_peak_gain, peak_gain)
            self._update_dnsmos(branch, final=False)
        self._stage_seconds["l4"] += max(
            0.0,
            perf_counter()
            - l4_started
            - (self._stage_seconds["dnsmos"] - dnsmos_before),
        )
        state.committed_end_sample_48k = end
        if state.preview_start_sample_48k is None:
            state.preview_start_sample_48k = start

    def _consume_outputs(
        self,
        state: _TrackState,
        outputs: tuple[Layer4StreamOutputChunk, ...],
    ) -> None:
        grouped: dict[int, list[Layer4StreamOutputChunk]] = {}
        for output in outputs:
            grouped.setdefault(output.commit_id, []).append(output)
        for commit_id in sorted(grouped):
            ordered = tuple(sorted(grouped[commit_id], key=lambda item: item.branch_id))
            self._process_output_group(state, ordered)

    def _stream_input(
        self,
        state: _TrackState,
        source: _SourcePart,
        *,
        is_final: bool,
    ) -> None:
        source_value = self._source_range(state, source.start_sample, source.end_sample)
        item = Layer4StreamInputChunk(
            source.session_id,
            source.stream_epoch,
            source.track_id,
            state.speaker_count,  # type: ignore[arg-type]
            source.start_sample,
            source.theta_deg,
            np.ascontiguousarray(source_value.waveform, dtype=np.float32),
            is_final=is_final,
        )
        l4_started = perf_counter()
        outputs = state.session.push(item)
        self._stage_seconds["l4"] += perf_counter() - l4_started
        self._consume_outputs(state, outputs)
        self._advance_l5(state, final=is_final)
        if is_final:
            for branch in state.branches.values():
                self._update_dnsmos(branch, final=True)

    def _upgrade_and_replay(self, state: _TrackState, *, final_last: bool) -> None:
        original_start = state.original_start_sample_48k
        replay_samples = state.inputs[-1].end_sample - original_start
        state.speaker_count = 2
        state.session = self._new_session(2)
        for branch in state.branches.values():
            self._storage.release_spool(branch.audio_spool)
            self._storage.release_spool(branch.probability_spool)
            self._storage.release_spool(branch.decision_spool)
        state.branches.clear()
        state.committed_end_sample_48k = None
        self._topology_revision += 1
        if replay_samples > self.max_replay_samples_48k:
            latest = state.inputs[-1]
            state.preview_degraded_reason = "one_to_two_replay_limit_exceeded"
            state.replay_dropped_samples_48k = latest.start_sample - original_start
            state.preview_start_sample_48k = latest.start_sample
            latest_audio = state.source_spool.view(
                latest.start_sample - state.source_base_sample_48k,
                latest.end_sample - state.source_base_sample_48k,
            )
            self._storage.release_spool(state.source_spool)
            state.source_spool = self._storage.create_spool(
                f"track_{state.identity[2]}_source_replay"
            )
            state.source_spool.append(latest_audio)
            self._storage.release_spool(state.direction_spool)
            state.direction_spool = self._storage.create_u8_timeline(
                f"track_{state.identity[2]}_directions_replay"
            )
            for sample, count in latest.l2_direction_counts:
                state.direction_spool.set(
                    (sample - latest.start_sample) // 960 - 1,
                    count,
                )
            state.source_base_sample_48k = latest.start_sample
            state.inputs = deque((latest,))
            replay_inputs = (latest,)
        else:
            state.preview_start_sample_48k = original_start
            replay_inputs = tuple(state.inputs)
        for index, source in enumerate(replay_inputs):
            self._stream_input(
                state,
                source,
                is_final=final_last and index == len(replay_inputs) - 1,
            )

    def push(
        self,
        source: Layer4LongAudioInput,
        *,
        is_final_chunk: bool = False,
    ) -> RealtimePostprocessingSnapshot | None:
        if self._finalized:
            raise RuntimeError("realtime L4/L5/L6 processor is already finalized")
        if self._session_id is None:
            self._session_id = source.session_id
        elif source.session_id != self._session_id:
            raise ValueError("one realtime L4/L5/L6 processor cannot mix capture sessions")
        key = (source.session_id, source.stream_epoch, source.track_id)
        count = self._speaker_count(source)
        source_part = _SourcePart.from_input(source)
        state = self._tracks.get(key)
        if state is None:
            state = _TrackState(
                count,
                self._new_session(count),
                self._storage.create_spool(f"track_{source.track_id}_source"),
                self._storage.create_u8_timeline(f"track_{source.track_id}_directions"),
                source.start_sample,
                source.start_sample,
                key,
            )
            state.preview_start_sample_48k = source.start_sample
            self._tracks[key] = state
            self._topology_revision += 1
        elif state.finalized:
            raise RuntimeError("realtime L4 track received audio after per-track finalization")
        elif state.inputs:
            self._validate_next_source(state, source_part)
        spool_start, _ = state.source_spool.append(
            np.ascontiguousarray(source.waveform, dtype=np.float32)
        )
        expected_spool_start = source.start_sample - state.source_base_sample_48k
        if spool_start != expected_spool_start:
            raise RuntimeError("realtime source spool lost its timeline position")
        for sample, direction_count in source.l2_direction_counts:
            state.direction_spool.set(
                (sample - state.source_base_sample_48k) // 960 - 1,
                direction_count,
            )
        state.inputs.append(source_part)
        while (
            len(state.inputs) > 1
            and state.inputs[-1].end_sample - state.inputs[0].start_sample
            > self.max_replay_samples_48k
        ):
            state.inputs.popleft()
        self._processed_blocks += 1
        if count > state.speaker_count:
            self._upgrade_and_replay(state, final_last=is_final_chunk)
        else:
            self._stream_input(state, source_part, is_final=is_final_chunk)
        if is_final_chunk:
            state.finalized = True
            self._close_track_writers(state)
        return self._snapshot(is_final=False)

    @staticmethod
    def _l5_end_sample_48k(state: _TrackState) -> int | None:
        if not state.branches or any(not branch.l5_samples_16k for branch in state.branches.values()):
            return None
        return min(
            branch.start_sample_48k + branch.l5_samples_16k * 3
            for branch in state.branches.values()
        )

    def _cumulative_track(
        self,
        state: _TrackState,
        end_sample_48k: int,
    ) -> tuple[tuple[Layer4ProcessedAudio, ...], tuple[Layer4OfflineResult, ...]]:
        if state.committed_end_sample_48k is None or state.preview_start_sample_48k is None:
            return (), ()
        start_sample_48k = state.preview_start_sample_48k
        if end_sample_48k <= start_sample_48k or (end_sample_48k - start_sample_48k) % 3:
            return (), ()
        source_hash = self._published_source_digest(
            state,
            start_sample_48k,
            end_sample_48k,
        )
        source = self._source_range(
            state,
            start_sample_48k,
            end_sample_48k,
            sha256=source_hash,
        )
        decision = self._decision(state, source)
        samples_16k = (end_sample_48k - start_sample_48k) // 3
        ranked = sorted(
            state.branches.values(),
            key=lambda branch: (-branch.match_score, branch.stable_branch_id),
        )
        ranked_scores = tuple(branch.match_score for branch in ranked)
        scores_by_stable_branch = tuple(
            (branch.stable_branch_id, branch.match_score)
            for branch in sorted(state.branches.values(), key=lambda item: item.stable_branch_id)
        )
        processed_values: list[Layer4ProcessedAudio] = []
        l5_values: list[Layer4OfflineResult] = []
        for rank, branch in enumerate(ranked):
            waveform = branch.audio_range(0, samples_16k)
            probabilities = branch.probabilities(samples_16k)
            decisions = branch.decisions(samples_16k)
            dnsmos = branch.dnsmos
            mos = _mos_score(dnsmos)
            metadata = {
                "backend": str(getattr(self.backend, "backend", "mossformer2_ss_16k")),
                "model_id": str(getattr(self.backend, "model_id", "mossformer2")),
                "model_revision": str(getattr(self.backend, "model_revision", "realtime")),
                "candidate_index": branch.stable_branch_id,
                "stable_branch_id": branch.stable_branch_id,
                "candidate_rank": rank,
                "candidate_match_score": branch.match_score,
                "candidate_scores": ranked_scores,
                "candidate_scores_by_stable_branch": scores_by_stable_branch,
                "candidate_match_score_aggregation": "duration_weighted_commits_v1",
                "matching_algorithm": self.matcher.algorithm_version,
                "merge_candidates": False,
                "resampler": self.resampler.algorithm_version,
                "pcm16_peak_safety_gain": branch.minimum_peak_gain,
                "dnsmos_sig": dnsmos[0],
                "dnsmos_bak": dnsmos[1],
                "dnsmos_ovrl": dnsmos[2],
                "dnsmos_periodic_sample_count": branch.dnsmos_sample_count,
                "dnsmos_scope": (
                    "periodic_30s_plus_final_tail_9_02s"
                    if branch.final_dnsmos is not None
                    else "periodic_30s_9_02s_windows"
                ),
                "dnsmos_complete_branch": False,
                "dnsmos_finalized_without_full_rerun": branch.final_dnsmos is not None,
                "mos_score": mos,
                "realtime_commit_count": branch.commit_count,
                "realtime_mf2_request_count": state.session.model_request_count,
                "realtime_revision": self._revision + 1,
                "realtime_provisional": True,
                "realtime_l4_valid_through_sample_48k": (
                    branch.start_sample_48k + branch.audio_samples_16k * 3
                ),
                "realtime_l5_valid_through_sample_48k": end_sample_48k,
                "realtime_l5_right_hold_samples_16k": (
                    0 if branch.l5_samples_16k == branch.audio_samples_16k
                    else self.l5_context_samples_16k
                ),
                "realtime_preview_degraded": state.preview_degraded_reason is not None,
                "realtime_preview_degraded_reason": state.preview_degraded_reason,
                "realtime_replay_dropped_samples_48k": state.replay_dropped_samples_48k,
                "realtime_track_final": state.finalized,
            }
            kind = "merged" if state.speaker_count == 1 else f"candidate_{rank}"
            output_id = (
                f"{source.asset_id}:l4:realtime:rank{rank}:"
                f"stable{branch.stable_branch_id}"
            )
            output_hash = branch.published_audio_digest(samples_16k)
            processed = Layer4ProcessedAudio(
                request_id=f"{source.asset_id}:realtime",
                source=source,
                speaker_count=decision,
                path=(
                    "single_speaker_bypass"
                    if state.speaker_count == 1
                    else "two_speaker_separation"
                ),
                selected=None,
                output_asset_id=output_id,
                output_sha256=output_hash,
                waveform_16k=waveform,
                metadata=metadata,
                output_kind=kind,  # type: ignore[arg-type]
            )
            l5_metadata = {
                **metadata,
                **branch.l5_metadata,
                "output_waveform_16k": processed.waveform_16k,
                "l5_frame_count": len(probabilities),
                "realtime_revision": self._revision + 1,
                "realtime_provisional": True,
            }
            summary = branch.l5_summary_probability
            l5 = Layer4OfflineResult(
                request_id=processed.request_id,
                source=source,
                speaker_count=decision,
                path=processed.path,
                selected=None,
                l5_probability=summary,
                l5_is_voice=bool(summary >= float(getattr(self.layer5, "threshold", 0.7))),
                l5_model_id=branch.l5_model_id or "realtime-l5",
                l5_probabilities_20ms=probabilities,
                l5_is_voice_20ms=decisions,
                output_asset_id=output_id,
                output_sha256=output_hash,
                metadata=l5_metadata,
                output_kind=kind,  # type: ignore[arg-type]
            )
            processed_values.append(processed)
            l5_values.append(l5)
        return tuple(processed_values), tuple(l5_values)

    @property
    def retained_state_samples(self) -> dict[str, int]:
        """Small audit surface for proving retained state grows only with audio evidence."""

        return {
            "source_48k": sum(
                state.source_spool.length_samples for state in self._tracks.values()
            ),
            "branch_16k": sum(
                branch.audio_samples_16k
                for state in self._tracks.values()
                for branch in state.branches.values()
            ),
            "l5_frames": sum(
                branch.l5_samples_16k // 320
                for state in self._tracks.values()
                for branch in state.branches.values()
            ),
            "retained_commit_dtos": 0,
            "resident_audio_samples": 0,
            "disk_audio_bytes": sum(
                state.source_spool.disk_bytes
                + state.direction_spool.disk_bytes
                + sum(
                    branch.audio_spool.disk_bytes
                    + branch.probability_spool.disk_bytes
                    + branch.decision_spool.disk_bytes
                    for branch in state.branches.values()
                )
                for state in self._tracks.values()
            ),
        }

    def _snapshot(self, *, is_final: bool) -> RealtimePostprocessingSnapshot | None:
        snapshot_started = perf_counter()
        l6_before = self._stage_seconds["l6"]

        def no_snapshot() -> None:
            self._stage_seconds["snapshot"] += perf_counter() - snapshot_started
            return None

        if self._session_id is None:
            return no_snapshot()
        state_ends = tuple(
            sorted(
                (
                    (state, int(end))
                    for state in self._tracks.values()
                    if (end := self._l5_end_sample_48k(state)) is not None
                    and state.preview_start_sample_48k is not None
                    and int(end) > int(state.preview_start_sample_48k)
                ),
                key=lambda item: item[0].identity,
            )
        )
        if not state_ends:
            return no_snapshot()
        ready_keys = tuple(state.identity for state, _ in state_ends)
        signature: tuple[object, ...] = (
            self._topology_revision,
            tuple(
                (
                    state.identity,
                    end,
                    state.speaker_count,
                    tuple(sorted(state.branches)),
                    state.finalized,
                )
                for state, end in state_ends
            ),
        )
        if not is_final and signature == self._last_published_signature:
            return no_snapshot()
        watermark = max(end for _, end in state_ends)
        preview_starts = tuple(
            int(state.preview_start_sample_48k) for state, _ in state_ends
        )
        processed: list[Layer4ProcessedAudio] = []
        l5_results: list[Layer4OfflineResult] = []
        for state, end in state_ends:
            track_processed, track_l5 = self._cumulative_track(state, end)
            processed.extend(track_processed)
            l5_results.extend(track_l5)
        if not l5_results:
            return no_snapshot()
        last_speaker_count = int(getattr(self._last_l6_result, "speaker_count", 0))
        last_l6_duration = (
            0
            if self._last_l6_watermark_48k is None
            else self._last_l6_watermark_48k - min(preview_starts)
        )
        current_duration = watermark - min(preview_starts)
        run_l6 = (
            is_final
            or self._last_l6_result is None
            or self._last_l6_topology_revision != self._topology_revision
            or self._last_l6_ready_keys != ready_keys
            or (
                self._last_l6_watermark_48k is not None
                and watermark - self._last_l6_watermark_48k >= self.l6_interval_samples_48k
            )
            or (
                last_speaker_count == 0
                and self._last_l6_watermark_48k is not None
                and last_l6_duration < 2 * 48_000 <= current_duration
            )
        )
        l6_state_reset = False
        if run_l6:
            l6_started = perf_counter()
            self._last_l6_result = self.layer6.process_streaming(
                tuple(l5_results),
                final=is_final,
                finalized_track_keys=frozenset(
                    state.identity for state, _ in state_ends if state.finalized
                ),
            )
            self._stage_seconds["l6"] += perf_counter() - l6_started
            self._last_l6_watermark_48k = watermark
            self._last_l6_topology_revision = self._topology_revision
            self._last_l6_ready_keys = ready_keys
        assert self._last_l6_result is not None and self._last_l6_watermark_48k is not None
        self._revision += 1
        retained = self.retained_state_samples
        track_watermarks = {
            f"{state.identity[0]}:{state.identity[1]}:{state.identity[2]}": end
            for state, end in state_ends
        }
        l6_result = replace(self._last_l6_result, metadata={
            **dict(getattr(self._last_l6_result, "metadata")),
            "realtime_revision": self._revision,
            "realtime_provisional": True,
            "canonical": False,
            "is_final": is_final,
            "finality_scope": "realtime_preview_tail_flushed" if is_final else "realtime_preview",
            "realtime_tail_flushed": is_final,
            "realtime_l6_reused": not run_l6,
            "realtime_l6_state_reset": l6_state_reset,
            "realtime_l6_valid_through_sample_48k": self._last_l6_watermark_48k,
            "realtime_l5_valid_through_sample_48k": watermark,
            "realtime_l5_track_watermarks_48k": track_watermarks,
            "realtime_pending_track_count": len(self._tracks) - len(state_ends),
            "realtime_l6_interval_samples_48k": self.l6_interval_samples_48k,
            "cached_voiceprint_segments": self.cached_embedder.cached_segments,
            "retained_source_samples_48k": retained["source_48k"],
            "retained_branch_samples_16k": retained["branch_16k"],
            "retained_l5_frames": retained["l5_frames"],
            "retained_commit_dtos": retained["retained_commit_dtos"],
        })
        self._last_published_signature = signature
        self._stage_seconds["snapshot"] += max(
            0.0,
            perf_counter()
            - snapshot_started
            - (self._stage_seconds["l6"] - l6_before),
        )
        return RealtimePostprocessingSnapshot(
            session_id=self._session_id,
            revision=self._revision,
            is_final=is_final,
            valid_through_sample_48k=watermark,
            processed_blocks=self._processed_blocks,
            l4_processed=tuple(processed),
            l5_results=tuple(l5_results),
            l6_result=l6_result,
            stage_durations_seconds=tuple(self._stage_seconds.items()),
        )

    def _finalize_state(self, state: _TrackState) -> None:
        if state.finalized:
            return
        if not state.session.closed:
            l4_started = perf_counter()
            outputs = state.session.flush()
            self._stage_seconds["l4"] += perf_counter() - l4_started
            self._consume_outputs(state, outputs)
        self._advance_l5(state, final=True)
        for branch in state.branches.values():
            self._update_dnsmos(branch, final=True)
        state.finalized = True
        self._close_track_writers(state)

    @staticmethod
    def _close_track_writers(state: _TrackState) -> None:
        """Close terminal mutable handles while retaining disk-backed reads."""

        state.source_spool.close()
        state.direction_spool.close()
        for branch in state.branches.values():
            branch.audio_spool.close()
            branch.probability_spool.close()
            branch.decision_spool.close()

    def finalize_track(
        self,
        identity: tuple[str, int, int],
    ) -> RealtimePostprocessingSnapshot | None:
        """Flush one ended ID without closing the capture-wide processor."""

        if self._finalized:
            raise RuntimeError("realtime L4/L5/L6 processor is already finalized")
        state = self._tracks.get(identity)
        if state is None:
            raise ValueError("realtime L4 track finalization identity is unknown")
        self._finalize_state(state)
        return self._snapshot(is_final=False)

    def finalize(self) -> RealtimePostprocessingSnapshot | None:
        if self._finalized:
            raise RuntimeError("realtime L4/L5/L6 processor is already finalized")
        for state in self._tracks.values():
            self._finalize_state(state)
        self._finalized = True
        try:
            return self._snapshot(is_final=True)
        finally:
            reset_streaming = getattr(self.layer6, "reset_streaming", None)
            if callable(reset_streaming):
                reset_streaming()
            self.cached_embedder.clear()
            self._tracks.clear()
            self._last_l6_result = None
            self._storage.retire()

    def abort(self) -> None:
        """Discard provisional L6 identity state without flushing model tails."""

        reset_streaming = getattr(self.layer6, "reset_streaming", None)
        if callable(reset_streaming):
            reset_streaming()
        if self._finalized:
            return
        self.cached_embedder.clear()
        self._last_l6_result = None
        self._last_l6_watermark_48k = None
        self._last_l6_topology_revision = -1
        self._last_l6_ready_keys = ()
        self._last_published_signature = None
        self._tracks.clear()
        self._storage.retire()
        self._finalized = True
