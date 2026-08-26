from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Mapping, Protocol

import numpy as np

from common.disk_audio import DiskAudioStore, DiskFloat32Spool
from layer4_speech_separation import Layer4OfflineResult

from .contracts import (
    Layer6Fragment,
    Layer6Result,
    Layer6SpeakerAudio,
    speaker_label,
)
from .models import CampPlusEmbedder
from .matching import TrackMatchFeatures, hungarian_track_features
from .multistage import MultiStageSnapshot, MultiStageVoiceprintClusterer, SegmentEvidence


_MINIMUM_VOICEPRINT_SAMPLES = 8_000
_VOICEPRINT_SEGMENT_SAMPLES = 32_000
_MINIMUM_TRACK_MATCH_COUNT = 1
_TRACK_MATCH_COVERAGE = 0.30
_OUTLIER_MAD_SCALE = 2.5
_OUTLIER_MINIMUM_MARGIN = 0.05
_STREAMING_FRAME_READ_COUNT = 500  # 10 s of 20 ms frames; independent of L4 chunks.
_STREAMING_OUTPUT_INITIAL_CAPACITY_SAMPLES = 60 * 16_000


class Layer6Configuration(Protocol):
    maximum_speakers: int
    speaker_similarity_threshold: float
    secondary_candidate_match_gap_max: float
    secondary_candidate_match_min: float
    secondary_candidate_mos_min: float
    maximum_internal_silence_ms: int
    clustering_backend: str
    multistage_fallback_distance: float
    multistage_l: int
    multistage_u1: int
    multistage_u2: int


@dataclass(slots=True)
class _VoiceprintAudio:
    result: Layer4OfflineResult
    branch_index: int
    match_score: float
    mos_score: float
    waveform: np.ndarray
    embedding: np.ndarray
    all_segment_embeddings: np.ndarray
    segment_embeddings: np.ndarray
    segment_sample_counts: tuple[int, ...]
    voice_sample_count: int
    segment_count: int
    retained_segment_count: int

    @property
    def audio_id(self) -> str:
        return self.result.output_asset_id

    @property
    def start_sample_48k(self) -> int:
        return self.result.source.start_sample

    @property
    def end_sample_48k(self) -> int:
        return self.result.source.end_sample


@dataclass(slots=True)
class _StreamingVoiceprintState:
    """Incremental voice-only evidence for one immutable L4 stable branch."""

    session_id: str
    stream_epoch: int
    track_id: int
    stable_branch_id: int
    generation: int
    start_sample_48k: int
    result: Layer4OfflineResult
    branch_index: int
    match_score: float
    mos_score: float
    consumed_frames: int = 0
    pending_voice: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32),
    )
    voice_sample_count: int = 0
    segment_sample_counts: list[int] = field(default_factory=list)
    embedding_sum: np.ndarray | None = None
    finalized: bool = False

    @property
    def identity(self) -> tuple[str, int, int, int]:
        return (
            self.session_id,
            self.stream_epoch,
            self.track_id,
            self.stable_branch_id,
        )

    @property
    def track_identity(self) -> tuple[str, int, int]:
        return self.session_id, self.stream_epoch, self.track_id

    @property
    def track_key(self) -> str:
        base = (
            f"{self.session_id}:epoch{self.stream_epoch}:"
            f"track{self.track_id}:stable{self.stable_branch_id}"
        )
        return base if self.generation == 0 else f"{base}:generation{self.generation}"

    @property
    def segment_count(self) -> int:
        return len(self.segment_sample_counts)

    @property
    def embedding(self) -> np.ndarray:
        if self.embedding_sum is None or not self.segment_sample_counts:
            raise RuntimeError("streaming L6 track has no speaker embedding")
        value = np.asarray(self.embedding_sum, dtype=np.float32)
        value = value / max(float(np.linalg.norm(value)), 1e-12)
        return np.ascontiguousarray(value, dtype=np.float32)


@dataclass(slots=True)
class _StreamingAdvance:
    state: _StreamingVoiceprintState
    result: Layer4OfflineResult
    branch_index: int
    match_score: float
    mos_score: float
    consumed_frames: int
    pending_voice: np.ndarray
    added_voice_samples: int
    segments: tuple[np.ndarray, ...]
    finalized: bool


@dataclass(slots=True)
class _StreamingSpeakerMaterializer:
    """One disk-backed, silence-compressed provisional speaker WAV."""

    store: DiskAudioStore
    spool: DiskFloat32Spool
    capacity_samples: int
    logical_samples: int
    recording_start_sample_48k: int
    processed_end_sample_48k: int
    pending_silence_frames: int = 0
    has_active_audio: bool = False
    priority_order: tuple[tuple[str, int, int, int], ...] = ()
    fragment_signatures: dict[tuple[str, int, int, int], tuple[object, ...]] = field(
        default_factory=dict,
    )
    fragment_end_samples_48k: dict[tuple[str, int, int, int], int] = field(
        default_factory=dict,
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
    return float(np.dot(left, right) / denominator)


def _score(metadata: Mapping[str, object], name: str, low: float, high: float) -> float:
    try:
        value = float(metadata[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"L6 requires L4 metadata {name}") from exc
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"L6 metadata {name} must be between {low} and {high}")
    return value


def _retain_consistent_segments(embeddings: np.ndarray) -> np.ndarray:
    """Drop 2 s embeddings whose within-track centrality is a robust low outlier."""

    if len(embeddings) <= 2:
        return embeddings
    similarities = embeddings @ embeddings.T
    centrality = np.asarray([
        np.median(np.delete(similarities[index], index))
        for index in range(len(embeddings))
    ], dtype=np.float32)
    center = float(np.median(centrality))
    mad = float(np.median(np.abs(centrality - center)))
    cutoff = center - max(_OUTLIER_MINIMUM_MARGIN, _OUTLIER_MAD_SCALE * mad)
    keep = centrality >= cutoff
    if int(np.count_nonzero(keep)) < 2:
        keep[np.argsort(centrality)[-2:]] = True
    return np.ascontiguousarray(embeddings[keep], dtype=np.float32)


def _track_match_features(
    left: np.ndarray,
    right: np.ndarray,
    threshold: float = 0.62,
) -> TrackMatchFeatures:
    return hungarian_track_features(
        left,
        right,
        threshold=threshold,
        minimum_match_count=_MINIMUM_TRACK_MATCH_COUNT,
        required_coverage=_TRACK_MATCH_COVERAGE,
    )


def _track_similarity(left: np.ndarray, right: np.ndarray) -> tuple[float, int, int]:
    """Compatibility wrapper returning the Hungarian decision score and counts."""

    features = _track_match_features(left, right)
    return features.decision_score, features.matched_count, features.required_count


def _pairwise_similarities(
    items: tuple[_VoiceprintAudio, ...],
    threshold: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[tuple[dict[str, float | int] | None, ...], ...],
]:
    matrix = np.eye(len(items), dtype=np.float32)
    matched_counts = np.zeros((len(items), len(items)), dtype=np.int32)
    required_counts = np.zeros((len(items), len(items)), dtype=np.int32)
    features: list[list[dict[str, float | int] | None]] = [
        [None] * len(items) for _ in items
    ]
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            pair = _track_match_features(
                items[left].segment_embeddings,
                items[right].segment_embeddings,
                threshold,
            )
            matrix[left, right] = matrix[right, left] = pair.decision_score
            matched_counts[left, right] = matched_counts[right, left] = pair.matched_count
            required_counts[left, right] = required_counts[right, left] = pair.required_count
            features[left][right] = features[right][left] = pair.as_dict()
    return matrix, matched_counts, required_counts, tuple(tuple(row) for row in features)


def _cluster(
    similarities: np.ndarray,
    threshold: float,
    maximum_speakers: int,
) -> tuple[int, ...]:
    """Complete-link AHC over multi-segment evidence for each selected L4 track."""

    clusters: list[list[int]] = [[index] for index in range(len(similarities))]

    def complete_link_similarity(left: list[int], right: list[int]) -> float:
        return float(np.min([
            similarities[a, b]
            for a in left
            for b in right
        ]))

    while len(clusters) > 1:
        options = [
            (complete_link_similarity(clusters[left], clusters[right]), left, right)
            for left in range(len(clusters))
            for right in range(left + 1, len(clusters))
        ]
        score, left, right = max(options)
        if len(clusters) <= maximum_speakers and score < threshold:
            break
        clusters[left].extend(clusters.pop(right))

    assignments = [-1] * len(similarities)
    for cluster_index, members in enumerate(clusters):
        for member in members:
            assignments[member] = cluster_index
    return tuple(assignments)


def _compress_silence(
    waveform: np.ndarray,
    occupied: np.ndarray,
    maximum_silence_frames: int,
) -> np.ndarray:
    """Trim edge silence and cap every internal silent run at the configured length."""

    active = np.flatnonzero(occupied)
    if not len(active):
        return np.empty(0, dtype=np.float32)
    first = int(active[0])
    last = int(active[-1]) + 1
    audio_frames = waveform.reshape(-1, 320)[first:last]
    active_frames = occupied[first:last]
    parts: list[np.ndarray] = []
    cursor = 0
    while cursor < len(active_frames):
        state = bool(active_frames[cursor])
        end = cursor + 1
        while end < len(active_frames) and bool(active_frames[end]) == state:
            end += 1
        if state:
            parts.append(audio_frames[cursor:end])
        else:
            keep = min(end - cursor, maximum_silence_frames)
            parts.append(np.zeros((keep, 320), dtype=np.float32))
        cursor = end
    return np.ascontiguousarray(np.concatenate(parts).reshape(-1), dtype=np.float32)


class OfflineLayer6Pipeline:
    """Offline complete-track voiceprint clustering and MOS-based merge."""

    def __init__(
        self,
        embedder: CampPlusEmbedder,
        config: Layer6Configuration,
        *,
        spool_min_free_bytes: int = 0,
    ) -> None:
        if type(spool_min_free_bytes) is not int or spool_min_free_bytes < 0:
            raise ValueError("L6 spool free-space reserve must be a non-negative integer")
        self.embedder = embedder
        self.config = config
        self._streaming_spool_min_free_bytes = spool_min_free_bytes
        self._streaming_clusterer: MultiStageVoiceprintClusterer | None = None
        self._streaming_session_id: str | None = None
        self._streaming_identity: tuple[str, int] | None = None
        self._streaming_states: dict[
            tuple[str, int, int, int], _StreamingVoiceprintState
        ] = {}
        self._streaming_snapshot: MultiStageSnapshot | None = None
        self._streaming_last_result: Layer6Result | None = None
        self._streaming_active_keys: frozenset[
            tuple[str, int, int, int]
        ] = frozenset()
        self._streaming_materializers: dict[
            int, _StreamingSpeakerMaterializer
        ] = {}
        self._streaming_materialization_dirty = False
        self._streaming_pending_changed_speaker_ids: set[int] = set()
        self._streaming_pending_append_speaker_ids: set[int] = set()
        self._streaming_retry_evidence: dict[str, SegmentEvidence] = {}

    @staticmethod
    def _validate_results(
        results: tuple[Layer4OfflineResult, ...],
    ) -> tuple[Layer4OfflineResult, ...]:
        results = tuple(results)
        if not results:
            raise ValueError("L6 requires completed L4/L5 results")
        if len({item.source.session_id for item in results}) != 1:
            raise ValueError("one L6 job cannot mix capture sessions")
        if any(
            item.output_kind == "merged" and item.path != "single_speaker_bypass"
            for item in results
        ):
            raise ValueError(
                "L6 requires unmerged L4 A/B outputs or single-speaker bypasses"
            )
        return results

    def _select_tracks(
        self,
        results: tuple[Layer4OfflineResult, ...],
    ) -> tuple[tuple[Layer4OfflineResult, int, float, float], ...]:
        groups: dict[tuple[str, int, str], dict[int, Layer4OfflineResult]] = {}
        for result in results:
            branch = 0 if result.output_kind == "merged" else int(result.output_kind[-1])
            key = (
                result.source.session_id,
                result.source.stream_epoch,
                result.source.asset_id,
            )
            branches = groups.setdefault(key, {})
            if branch in branches:
                raise ValueError("L6 received a duplicate L4 A/B branch")
            branches[branch] = result

        selected: list[tuple[Layer4OfflineResult, int, float, float]] = []
        for branches in groups.values():
            if 0 not in branches:
                raise ValueError("L6 requires an A track for every L4 source")
            a = branches[0]
            a_match = (
                1.0
                if a.output_kind == "merged"
                else _score(a.metadata, "candidate_match_score", 0.0, 1.0)
            )
            a_mos = _score(a.metadata, "mos_score", 0.0, 1.0)
            selected.append((a, 0, a_match, a_mos))
            b = branches.get(1)
            if b is None:
                continue
            b_match = _score(b.metadata, "candidate_match_score", 0.0, 1.0)
            b_mos = _score(b.metadata, "mos_score", 0.0, 1.0)
            if (
                abs(a_match - b_match)
                <= self.config.secondary_candidate_match_gap_max + 1e-12
                and b_match > self.config.secondary_candidate_match_min
                and b_mos > self.config.secondary_candidate_mos_min
            ):
                selected.append((b, 1, b_match, b_mos))
        return tuple(selected)

    def _voiceprint_audio(
        self,
        result: Layer4OfflineResult,
        branch_index: int,
        match_score: float,
        mos_score: float,
        *,
        include_partial_segment: bool,
    ) -> _VoiceprintAudio | None:
        waveform = np.asarray(result.metadata.get("output_waveform_16k"))
        if (
            waveform.dtype != np.float32
            or waveform.ndim != 1
            or not waveform.flags.c_contiguous
            or len(waveform) != len(result.l5_probabilities_20ms) * 320
        ):
            raise ValueError("L6 requires each L4 result to retain aligned 16 kHz audio")
        voice_mask = np.asarray(result.l5_is_voice_20ms, dtype=bool)
        voice_waveform = np.ascontiguousarray(
            waveform.reshape(-1, 320)[voice_mask].reshape(-1),
            dtype=np.float32,
        )
        if len(voice_waveform) < _MINIMUM_VOICEPRINT_SAMPLES:
            return None
        segments = []
        segment_sample_counts: list[int] = []
        for start in range(0, len(voice_waveform), _VOICEPRINT_SEGMENT_SAMPLES):
            segment = np.ascontiguousarray(
                voice_waveform[start:start + _VOICEPRINT_SEGMENT_SAMPLES],
                dtype=np.float32,
            )
            if len(segment) < _MINIMUM_VOICEPRINT_SAMPLES:
                continue
            if len(segment) < _VOICEPRINT_SEGMENT_SAMPLES and not include_partial_segment:
                continue
            segments.append(segment)
            segment_sample_counts.append(len(segment))
        if not segments:
            return None
        batch_embed = getattr(self.embedder, "embed_batch", None)
        if callable(batch_embed):
            segment_embeddings = tuple(batch_embed(tuple(segments)))
        else:
            segment_embeddings = tuple(self.embedder.embed(segment) for segment in segments)
        all_embeddings = np.ascontiguousarray(np.stack(segment_embeddings), dtype=np.float32)
        retained_embeddings = _retain_consistent_segments(all_embeddings)
        embedding = np.mean(retained_embeddings, axis=0)
        embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
        embedding = np.ascontiguousarray(embedding, dtype=np.float32)
        return _VoiceprintAudio(
            result=result,
            branch_index=branch_index,
            match_score=match_score,
            mos_score=mos_score,
            waveform=np.ascontiguousarray(waveform, dtype=np.float32),
            embedding=embedding,
            all_segment_embeddings=all_embeddings,
            segment_embeddings=retained_embeddings,
            segment_sample_counts=tuple(segment_sample_counts),
            voice_sample_count=len(voice_waveform),
            segment_count=len(all_embeddings),
            retained_segment_count=len(retained_embeddings),
        )

    @staticmethod
    def _multistage_evidence(
        items: tuple[_VoiceprintAudio, ...],
    ) -> tuple[SegmentEvidence, ...]:
        evidence: list[tuple[tuple[int, int, int, int], SegmentEvidence]] = []
        for item in items:
            source = item.result.source
            track_key = (
                f"{source.session_id}:epoch{source.stream_epoch}:"
                f"track{source.track_id}:branch{item.branch_index}"
            )
            for segment_index, (embedding, sample_count) in enumerate(zip(
                item.all_segment_embeddings,
                item.segment_sample_counts,
                strict=True,
            )):
                evidence_id = f"{track_key}:segment{segment_index}"
                order = (
                    item.start_sample_48k,
                    source.track_id,
                    item.branch_index,
                    segment_index,
                )
                evidence.append((order, SegmentEvidence(
                    evidence_id,
                    track_key,
                    embedding,
                    sample_count,
                )))
        return tuple(value for _, value in sorted(evidence, key=lambda pair: pair[0]))

    @staticmethod
    def _multistage_track_assignments(
        items: tuple[_VoiceprintAudio, ...],
        snapshot: MultiStageSnapshot,
    ) -> tuple[int, ...]:
        assignments: list[int] = []
        for item in items:
            source = item.result.source
            track_key = (
                f"{source.session_id}:epoch{source.stream_epoch}:"
                f"track{source.track_id}:branch{item.branch_index}"
            )
            weighted: dict[int, int] = {}
            first_index: dict[int, int] = {}
            for segment_index, sample_count in enumerate(item.segment_sample_counts):
                label = snapshot.labels_by_evidence_id[f"{track_key}:segment{segment_index}"]
                weighted[label] = weighted.get(label, 0) + sample_count
                first_index.setdefault(label, segment_index)
            assignments.append(max(
                weighted,
                key=lambda label: (weighted[label], -first_index[label], -label),
            ))
        return tuple(assignments)

    @staticmethod
    def _streaming_stable_branch_id(
        result: Layer4OfflineResult,
        branch_index: int,
    ) -> int:
        raw = result.metadata.get("stable_branch_id", branch_index)
        if isinstance(raw, bool):
            raise ValueError("streaming L6 stable branch ID must be 0 or 1")
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("streaming L6 stable branch ID must be 0 or 1") from exc
        if value not in {0, 1}:
            raise ValueError("streaming L6 stable branch ID must be 0 or 1")
        return value

    @staticmethod
    def _streaming_waveform(result: Layer4OfflineResult) -> tuple[object, np.ndarray]:
        raw = result.metadata.get("output_waveform_16k")
        waveform = np.asarray(raw)
        frame_count = len(result.l5_is_voice_20ms)
        if (
            waveform.dtype != np.float32
            or waveform.ndim != 1
            or not waveform.flags.c_contiguous
            or len(waveform) != frame_count * 320
            or len(result.l5_probabilities_20ms) != frame_count
        ):
            raise ValueError("L6 requires each L4 result to retain aligned 16 kHz audio")
        return raw, waveform

    def _plan_streaming_advance(
        self,
        state: _StreamingVoiceprintState,
        result: Layer4OfflineResult,
        branch_index: int,
        match_score: float,
        mos_score: float,
        *,
        finalize_track: bool,
    ) -> _StreamingAdvance:
        if result.source.start_sample != state.start_sample_48k:
            raise ValueError(
                "streaming L6 cumulative track start changed; reset streaming state"
            )
        _, waveform = self._streaming_waveform(result)
        frame_count = len(result.l5_is_voice_20ms)
        if frame_count < state.consumed_frames:
            raise ValueError(
                "streaming L6 cumulative L5 watermark moved backwards; reset streaming state"
            )
        if state.finalized and frame_count != state.consumed_frames:
            raise ValueError("streaming L6 finalized track received additional audio")

        pending = np.ascontiguousarray(state.pending_voice, dtype=np.float32)
        segments: list[np.ndarray] = []
        added_voice_samples = 0
        frames = waveform.reshape(-1, 320)
        for first in range(
            state.consumed_frames,
            frame_count,
            _STREAMING_FRAME_READ_COUNT,
        ):
            last = min(frame_count, first + _STREAMING_FRAME_READ_COUNT)
            decisions = np.asarray(
                result.l5_is_voice_20ms[first:last],
                dtype=bool,
            )
            if decisions.shape != (last - first,):
                raise ValueError("streaming L6 decision range is not frame aligned")
            if not np.any(decisions):
                continue
            active = np.ascontiguousarray(
                frames[first:last][decisions].reshape(-1),
                dtype=np.float32,
            )
            added_voice_samples += len(active)
            combined = (
                np.ascontiguousarray(np.concatenate((pending, active)), dtype=np.float32)
                if len(pending)
                else active
            )
            complete = len(combined) // _VOICEPRINT_SEGMENT_SAMPLES
            for index in range(complete):
                start = index * _VOICEPRINT_SEGMENT_SAMPLES
                segments.append(np.ascontiguousarray(
                    combined[start:start + _VOICEPRINT_SEGMENT_SAMPLES],
                    dtype=np.float32,
                ))
            pending = np.ascontiguousarray(
                combined[complete * _VOICEPRINT_SEGMENT_SAMPLES:],
                dtype=np.float32,
            )

        finalized = state.finalized
        if finalize_track and not finalized:
            if len(pending) >= _MINIMUM_VOICEPRINT_SAMPLES:
                segments.append(np.ascontiguousarray(pending, dtype=np.float32))
            pending = np.empty(0, dtype=np.float32)
            finalized = True
        if len(pending) >= _VOICEPRINT_SEGMENT_SAMPLES:
            raise RuntimeError("streaming L6 retained more than one evidence segment")

        return _StreamingAdvance(
            state,
            result,
            branch_index,
            match_score,
            mos_score,
            frame_count,
            pending,
            added_voice_samples,
            tuple(segments),
            finalized,
        )

    @staticmethod
    def _streaming_raw_assignment(
        state: _StreamingVoiceprintState,
        snapshot: MultiStageSnapshot,
    ) -> int:
        assignment = snapshot.assignments_by_track_key.get(state.track_key)
        if assignment is not None:
            return int(assignment)
        weighted: dict[int, int] = {}
        first_index: dict[int, int] = {}
        for index, sample_count in enumerate(state.segment_sample_counts):
            evidence_id = f"{state.track_key}:segment{index}"
            label = snapshot.labels_by_evidence_id[evidence_id]
            weighted[label] = weighted.get(label, 0) + sample_count
            first_index.setdefault(label, index)
        if not weighted:
            raise RuntimeError("streaming L6 track assignment has no evidence")
        return max(
            weighted,
            key=lambda label: (weighted[label], -first_index[label], -label),
        )

    @staticmethod
    def _limit_streaming_assignments(
        states: tuple[_StreamingVoiceprintState, ...],
        assignments: tuple[int, ...],
        maximum_speakers: int,
    ) -> tuple[int, ...]:
        """Defensively cap fake/custom MultiStage backends at the runtime limit."""

        limited = list(assignments)
        while len(set(limited)) > maximum_speakers:
            labels = tuple(sorted(set(limited)))
            centroids: dict[int, np.ndarray] = {}
            for label in labels:
                members = [
                    state.embedding
                    for state, assigned in zip(states, limited, strict=True)
                    if assigned == label
                ]
                centroid = np.mean(members, axis=0)
                centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
                centroids[label] = centroid
            target, source = max(
                (
                    (
                        _cosine(centroids[left], centroids[right]),
                        -left,
                        -right,
                        left,
                        right,
                    )
                    for index, left in enumerate(labels)
                    for right in labels[index + 1:]
                ),
            )[3:]
            limited = [target if value == source else value for value in limited]
        return tuple(limited)

    @staticmethod
    def _streaming_output_capacity(required_samples: int) -> int:
        capacity = _STREAMING_OUTPUT_INITIAL_CAPACITY_SAMPLES
        while capacity < required_samples:
            capacity *= 2
        return capacity

    def _new_streaming_materializer(
        self,
        speaker_id: int,
        recording_start: int,
        recording_end: int,
    ) -> _StreamingSpeakerMaterializer:
        maximum_output_samples = (recording_end - recording_start) // 3
        capacity = self._streaming_output_capacity(maximum_output_samples)
        store = DiskAudioStore(
            prefix=f"micarray_l6_speaker_{speaker_id}_",
            minimum_free_bytes=self._streaming_spool_min_free_bytes,
        )
        spool = store.create_spool(f"speaker_{speaker_id}_compressed")
        try:
            spool.ensure_length(capacity)
        except BaseException:
            store.retire()
            raise
        return _StreamingSpeakerMaterializer(
            store,
            spool,
            capacity,
            0,
            recording_start,
            recording_start,
        )

    def _grow_streaming_materializer(
        self,
        materializer: _StreamingSpeakerMaterializer,
        required_samples: int,
    ) -> None:
        if required_samples <= materializer.capacity_samples:
            return
        capacity = self._streaming_output_capacity(required_samples)
        store = DiskAudioStore(
            prefix="micarray_l6_speaker_growth_",
            minimum_free_bytes=self._streaming_spool_min_free_bytes,
        )
        spool = store.create_spool("speaker_compressed")
        try:
            if materializer.logical_samples:
                materializer.spool.copy_range_to(
                    0,
                    materializer.logical_samples,
                    spool,
                )
            spool.ensure_length(capacity)
        except BaseException:
            store.retire()
            raise
        materializer.store = store
        materializer.spool = spool
        materializer.capacity_samples = capacity

    def _write_streaming_audio(
        self,
        materializer: _StreamingSpeakerMaterializer,
        waveform: np.ndarray,
    ) -> None:
        value = np.ascontiguousarray(waveform, dtype=np.float32)
        if not len(value):
            return
        end = materializer.logical_samples + len(value)
        self._grow_streaming_materializer(materializer, end)
        materializer.spool.write_at(materializer.logical_samples, value)
        materializer.logical_samples = end

    def _write_streaming_silence(
        self,
        materializer: _StreamingSpeakerMaterializer,
        frame_count: int,
    ) -> None:
        remaining = frame_count
        while remaining:
            count = min(remaining, _STREAMING_FRAME_READ_COUNT)
            self._write_streaming_audio(
                materializer,
                np.zeros(count * 320, dtype=np.float32),
            )
            remaining -= count

    @staticmethod
    def _streaming_fragment_signature(
        state: _StreamingVoiceprintState,
        _fragment: Layer6Fragment,
    ) -> tuple[object, ...]:
        return (state.start_sample_48k,)

    @staticmethod
    def _streaming_fragment_priority(
        item: tuple[_StreamingVoiceprintState, Layer6Fragment],
    ) -> tuple[object, ...]:
        _state, fragment = item
        return (
            -fragment.mos_score,
            -fragment.match_score,
            -fragment.speaker_similarity,
            fragment.branch_index,
            fragment.fragment_id,
        )

    def _streaming_materializer_requires_rebuild(
        self,
        materializer: _StreamingSpeakerMaterializer,
        items: tuple[tuple[_StreamingVoiceprintState, Layer6Fragment], ...],
        materialization_end: int,
    ) -> bool:
        if materialization_end < materializer.processed_end_sample_48k:
            return True
        signatures = {
            state.identity: self._streaming_fragment_signature(state, fragment)
            for state, fragment in items
        }
        priority_order = tuple(
            state.identity
            for state, _ in sorted(items, key=self._streaming_fragment_priority)
        )
        previous_keys = set(materializer.fragment_signatures)
        current_keys = set(signatures)
        removed = previous_keys.difference(current_keys)
        added = current_keys.difference(previous_keys)
        if removed or any(
            state.start_sample_48k < materializer.processed_end_sample_48k
            for state, _ in items
            if state.identity in added
        ):
            return True
        if any(
            materializer.fragment_signatures[key] != signatures[key]
            for key in previous_keys.intersection(current_keys)
        ):
            return True
        surviving_priority_order = tuple(
            identity for identity in priority_order if identity in previous_keys
        )
        if surviving_priority_order != materializer.priority_order:
            return True
        for state, _ in items:
            old_end = materializer.fragment_end_samples_48k.get(
                state.identity,
                state.start_sample_48k,
            )
            current_end = state.result.source.end_sample
            if current_end < old_end:
                return True
            if (
                current_end > old_end
                and old_end < materializer.processed_end_sample_48k
            ):
                return True
        return False

    @staticmethod
    def _streaming_materialization_end(
        items: tuple[tuple[_StreamingVoiceprintState, Layer6Fragment], ...],
        recording_end: int,
    ) -> int:
        """Hold the provisional tail behind every unfinished same-speaker track.

        This makes normal asynchronous watermarks append-only: a faster track's
        unpublished tail is mixed only after slower tracks have caught up or
        finalized, so late cumulative frames never force a rebuild from zero.
        """

        unfinished_ends = tuple(
            state.result.source.end_sample
            for state, _ in items
            if not state.finalized
        )
        return min(unfinished_ends, default=recording_end)

    @staticmethod
    def _streaming_mix_chunk(
        items: tuple[tuple[_StreamingVoiceprintState, Layer6Fragment], ...],
        chunk_start_sample_48k: int,
        chunk_end_sample_48k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        frame_count = (chunk_end_sample_48k - chunk_start_sample_48k) // 960
        waveform = np.zeros((frame_count, 320), dtype=np.float32)
        occupied = np.zeros(frame_count, dtype=bool)
        for _, fragment in items:
            overlap_start = max(chunk_start_sample_48k, fragment.start_sample_48k)
            overlap_end = min(chunk_end_sample_48k, fragment.end_sample_48k)
            if overlap_end <= overlap_start:
                continue
            source_first = (overlap_start - fragment.start_sample_48k) // 960
            source_last = (overlap_end - fragment.start_sample_48k) // 960
            target_first = (overlap_start - chunk_start_sample_48k) // 960
            target_last = target_first + source_last - source_first
            decisions = np.asarray(
                fragment.voice_is_active_20ms[source_first:source_last],
                dtype=bool,
            )
            available = np.logical_and(
                decisions,
                np.logical_not(occupied[target_first:target_last]),
            )
            if not np.any(available):
                continue
            source_frames = np.asarray(fragment.waveform_16k).reshape(-1, 320)
            target_frames = waveform[target_first:target_last]
            target_frames[available] = source_frames[source_first:source_last][available]
            occupied[target_first:target_last][available] = True
        return waveform, occupied

    def _streaming_compress_chunk(
        self,
        materializer: _StreamingSpeakerMaterializer,
        waveform: np.ndarray,
        occupied: np.ndarray,
    ) -> None:
        maximum_silence_frames = self.config.maximum_internal_silence_ms // 20
        cursor = 0
        while cursor < len(occupied):
            active = bool(occupied[cursor])
            end = cursor + 1
            while end < len(occupied) and bool(occupied[end]) == active:
                end += 1
            if active:
                if not materializer.has_active_audio:
                    materializer.has_active_audio = True
                elif materializer.pending_silence_frames:
                    self._write_streaming_silence(
                        materializer,
                        min(
                            materializer.pending_silence_frames,
                            maximum_silence_frames,
                        ),
                    )
                materializer.pending_silence_frames = 0
                self._write_streaming_audio(
                    materializer,
                    waveform[cursor:end].reshape(-1),
                )
            elif materializer.has_active_audio:
                materializer.pending_silence_frames += end - cursor
            cursor = end

    def _advance_streaming_materializer(
        self,
        materializer: _StreamingSpeakerMaterializer,
        items: tuple[tuple[_StreamingVoiceprintState, Layer6Fragment], ...],
        end_sample_48k: int,
    ) -> None:
        checkpoint = (
            materializer.logical_samples,
            materializer.processed_end_sample_48k,
            materializer.pending_silence_frames,
            materializer.has_active_audio,
        )
        priorities = tuple(sorted(items, key=self._streaming_fragment_priority))
        start = materializer.processed_end_sample_48k
        try:
            for chunk_start in range(
                start,
                end_sample_48k,
                _STREAMING_FRAME_READ_COUNT * 960,
            ):
                chunk_end = min(
                    end_sample_48k,
                    chunk_start + _STREAMING_FRAME_READ_COUNT * 960,
                )
                waveform, occupied = self._streaming_mix_chunk(
                    priorities,
                    chunk_start,
                    chunk_end,
                )
                self._streaming_compress_chunk(materializer, waveform, occupied)
            materializer.processed_end_sample_48k = end_sample_48k
        except BaseException:
            (
                materializer.logical_samples,
                materializer.processed_end_sample_48k,
                materializer.pending_silence_frames,
                materializer.has_active_audio,
            ) = checkpoint
            raise

    def _materialize_streaming_outputs(
        self,
        states: tuple[_StreamingVoiceprintState, ...],
        fragments: tuple[Layer6Fragment, ...],
        speaker_ids: tuple[int, ...],
        recording_start: int,
        recording_end: int,
    ) -> tuple[
        tuple[Layer6SpeakerAudio, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        grouped: dict[
            int, list[tuple[_StreamingVoiceprintState, Layer6Fragment]]
        ] = {}
        for state, fragment, speaker_id in zip(
            states,
            fragments,
            speaker_ids,
            strict=True,
        ):
            grouped.setdefault(speaker_id, []).append((state, fragment))

        previous_materializers = self._streaming_materializers
        candidate_materializers: dict[int, _StreamingSpeakerMaterializer] = {}
        previous_store_ids = {
            id(materializer.store) for materializer in previous_materializers.values()
        }
        changed: set[int] = set()
        append_only: set[int] = set()
        outputs: list[Layer6SpeakerAudio] = []
        prepared_candidates: list[_StreamingSpeakerMaterializer] = []
        try:
            for speaker_id in sorted(grouped):
                items = tuple(grouped[speaker_id])
                materialization_end = self._streaming_materialization_end(
                    items,
                    recording_end,
                )
                previous = previous_materializers.get(speaker_id)
                rebuild = previous is not None and (
                    self._streaming_materializer_requires_rebuild(
                        previous,
                        items,
                        materialization_end,
                    )
                )
                if previous is None or rebuild:
                    candidate = self._new_streaming_materializer(
                        speaker_id,
                        recording_start,
                        recording_end,
                    )
                    prepared_candidates.append(candidate)
                    self._advance_streaming_materializer(
                        candidate,
                        items,
                        materialization_end,
                    )
                    if previous is None:
                        append_only.add(speaker_id)
                    else:
                        changed.add(speaker_id)
                else:
                    candidate = copy(previous)
                    candidate.fragment_signatures = dict(previous.fragment_signatures)
                    candidate.fragment_end_samples_48k = dict(
                        previous.fragment_end_samples_48k
                    )
                    prepared_candidates.append(candidate)
                    self._advance_streaming_materializer(
                        candidate,
                        items,
                        materialization_end,
                    )
                    if candidate.logical_samples > previous.logical_samples:
                        append_only.add(speaker_id)

                candidate.recording_start_sample_48k = recording_start
                candidate.fragment_signatures = {
                    state.identity: self._streaming_fragment_signature(state, fragment)
                    for state, fragment in items
                }
                candidate.priority_order = tuple(
                    state.identity
                    for state, _ in sorted(items, key=self._streaming_fragment_priority)
                )
                candidate.fragment_end_samples_48k = {
                    state.identity: state.result.source.end_sample
                    for state, _ in items
                }
                if not candidate.logical_samples:
                    raise ValueError("L6 voiceprint cluster contains no active audio")
                candidate_materializers[speaker_id] = candidate
                outputs.append(Layer6SpeakerAudio(
                    speaker_id,
                    speaker_label(speaker_id),
                    16_000,
                    recording_start,
                    recording_end,
                    candidate.spool.view(0, candidate.logical_samples),
                    tuple(sorted({fragment.source_track_id for _, fragment in items})),
                    tuple(fragment.fragment_id for _, fragment in items),
                    float(np.mean([fragment.mos_score for _, fragment in items])),
                ))
        except BaseException:
            for candidate in prepared_candidates:
                if id(candidate.store) not in previous_store_ids:
                    candidate.store.retire()
            raise

        removed = set(previous_materializers).difference(grouped)
        changed.update(removed)
        append_only.difference_update(changed)
        self._streaming_materializers = candidate_materializers
        current_store_ids = {
            id(materializer.store) for materializer in candidate_materializers.values()
        }
        for previous in previous_materializers.values():
            if id(previous.store) not in current_store_ids:
                previous.store.retire()
        return (
            tuple(outputs),
            tuple(sorted(changed)),
            tuple(sorted(append_only)),
        )

    def _build_streaming_result(
        self,
        results: tuple[Layer4OfflineResult, ...],
        states: tuple[_StreamingVoiceprintState, ...],
        snapshot: MultiStageSnapshot,
        *,
        final: bool,
        new_evidence_count: int,
        new_voice_samples: int,
        started: float,
    ) -> Layer6Result:
        session_id = results[0].source.session_id
        recording_start = min(item.source.start_sample for item in results)
        recording_end = max(item.source.end_sample for item in results)
        if (
            (recording_end - recording_start) % 960
            or any((item.source.start_sample - recording_start) % 960 for item in results)
        ):
            raise ValueError("L6 source timelines must align to absolute 20 ms frames")

        voiceprints = tuple(state for state in states if state.segment_count)
        raw_assignments = tuple(
            self._streaming_raw_assignment(state, snapshot)
            for state in voiceprints
        )
        maximum_speakers = min(5, max(1, int(self.config.maximum_speakers)))
        raw_assignments = self._limit_streaming_assignments(
            voiceprints,
            raw_assignments,
            maximum_speakers,
        ) if voiceprints else ()
        order = tuple(sorted(set(raw_assignments), key=lambda cluster: min(
            (
                state.start_sample_48k,
                state.track_id,
                state.stable_branch_id,
            )
            for state, assignment in zip(voiceprints, raw_assignments, strict=True)
            if assignment == cluster
        )))
        remap = {raw: index + 1 for index, raw in enumerate(order)}
        speaker_ids = tuple(remap[raw] for raw in raw_assignments)

        centroids: dict[int, np.ndarray] = {}
        for speaker_id in range(1, len(order) + 1):
            centroid = np.mean([
                state.embedding
                for state, assigned in zip(voiceprints, speaker_ids, strict=True)
                if assigned == speaker_id
            ], axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            centroids[speaker_id] = centroid

        assignments: list[Layer6Fragment] = []
        for state, speaker_id in zip(voiceprints, speaker_ids, strict=True):
            raw_waveform, _ = self._streaming_waveform(state.result)
            assignments.append(Layer6Fragment(
                state.result.output_asset_id,
                state.result.source.asset_id,
                state.result.source.track_id,
                state.result.source.theta_deg,
                state.branch_index,
                state.branch_index == 0,
                state.result.source.start_sample,
                state.result.source.end_sample,
                raw_waveform,  # type: ignore[arg-type]
                state.result.l5_probabilities_20ms,
                state.result.l5_is_voice_20ms,
                state.embedding,
                speaker_id,
                state.match_score,
                state.mos_score,
                float(np.clip(
                    _cosine(state.embedding, centroids[speaker_id]),
                    0.0,
                    1.0,
                )),
            ))

        fragment_values = tuple(assignments)
        (
            outputs,
            changed_speaker_ids,
            append_only_speaker_ids,
        ) = self._materialize_streaming_outputs(
            voiceprints,
            fragment_values,
            speaker_ids,
            recording_start,
            recording_end,
        )
        if len(outputs) != len(order):
            raise ValueError("L6 voiceprint cluster contains no active audio")

        extracted_ids = tuple(state.result.output_asset_id for state in voiceprints)
        insufficient_ids = tuple(
            state.result.output_asset_id for state in states if not state.segment_count
        )
        metadata = {
            "algorithm": "campplus_2s_multistage_disk_incremental_v3",
            "multistage": {
                "evidence_count": snapshot.evidence_count,
                "cluster_count": min(len(order), maximum_speakers),
                "stage": snapshot.stage,
            },
            "recording_start_sample_48k": recording_start,
            "recording_end_sample_48k": recording_end,
            "secondary_candidate_gate": {
                "match_gap_max": self.config.secondary_candidate_match_gap_max,
                "match_min": self.config.secondary_candidate_match_min,
                "mos_min": self.config.secondary_candidate_mos_min,
            },
            "maximum_internal_silence_ms": self.config.maximum_internal_silence_ms,
            "extracted_audio_ids": extracted_ids,
            "voiceprint_voice_sample_counts": {
                state.result.output_asset_id: state.voice_sample_count
                for state in voiceprints
            },
            "voiceprint_segment_counts": {
                state.result.output_asset_id: state.segment_count
                for state in voiceprints
            },
            "voiceprint_retained_segment_counts": {
                state.result.output_asset_id: state.segment_count
                for state in voiceprints
            },
            "insufficient_voice_audio_ids": insufficient_ids,
            "silent_voiceprint_audio_ids": (),
            "pairwise_audio_ids": (),
            "pairwise_similarity_matrix": (),
            "pairwise_matched_segment_counts": (),
            "pairwise_required_segment_counts": (),
            "pairwise_match_features": (),
            "pairwise_diagnostics_available": False,
            "pairwise_skipped_reason": "multistage_incremental",
            "voiceprint_audio_ids": {
                speaker_id: tuple(
                    state.result.output_asset_id
                    for state, assigned in zip(voiceprints, speaker_ids, strict=True)
                    if assigned == speaker_id
                )
                for speaker_id in range(1, len(order) + 1)
            },
            "speaker_similarity_threshold": self.config.speaker_similarity_threshold,
            "streaming_incremental": True,
            "streaming_final": bool(final),
            "streaming_new_evidence_count": new_evidence_count,
            "streaming_new_voice_samples_16k": new_voice_samples,
            "incremental_changed_speaker_ids": changed_speaker_ids,
            "incremental_append_only_speaker_ids": append_only_speaker_ids,
            "streaming_output_storage": "disk_float32_spool",
            "streaming_materializer_chunk_frames_20ms": (
                _STREAMING_FRAME_READ_COUNT
            ),
            "streaming_materialized_samples_16k": {
                speaker_id: self._streaming_materializers[speaker_id].logical_samples
                for speaker_id in sorted(self._streaming_materializers)
            },
            "streaming_materialized_through_sample_48k": {
                speaker_id: (
                    self._streaming_materializers[speaker_id].processed_end_sample_48k
                )
                for speaker_id in sorted(self._streaming_materializers)
            },
            "streaming_pending_voice_samples_16k": {
                state.track_key: len(state.pending_voice) for state in states
            },
            "streaming_consumed_frames_20ms": {
                state.track_key: state.consumed_frames for state in states
            },
            "elapsed_ms": (perf_counter() - started) * 1_000.0,
        }
        return Layer6Result(
            session_id,
            len(outputs),
            outputs,
            fragment_values,
            metadata,
        )

    def reset_streaming(self) -> None:
        materializers = tuple(self._streaming_materializers.values())
        self._streaming_materializers.clear()
        for materializer in materializers:
            materializer.store.retire()
        self._streaming_clusterer = None
        self._streaming_session_id = None
        self._streaming_identity = None
        self._streaming_states.clear()
        self._streaming_snapshot = None
        self._streaming_last_result = None
        self._streaming_active_keys = frozenset()
        self._streaming_materialization_dirty = False
        self._streaming_pending_changed_speaker_ids.clear()
        self._streaming_pending_append_speaker_ids.clear()
        self._streaming_retry_evidence.clear()

    @staticmethod
    def _project_streaming_state(
        advance: _StreamingAdvance,
        evidence: tuple[SegmentEvidence, ...],
    ) -> _StreamingVoiceprintState:
        source = advance.state
        embedding_sum = (
            None
            if source.embedding_sum is None
            else np.array(source.embedding_sum, dtype=np.float32, copy=True)
        )
        segment_sample_counts = list(source.segment_sample_counts)
        for item in evidence:
            if embedding_sum is None:
                embedding_sum = np.zeros_like(item.embedding, dtype=np.float32)
            embedding_sum += item.embedding
            segment_sample_counts.append(item.weight_samples_16k)
        return _StreamingVoiceprintState(
            source.session_id,
            source.stream_epoch,
            source.track_id,
            source.stable_branch_id,
            source.generation,
            source.start_sample_48k,
            advance.result,
            advance.branch_index,
            advance.match_score,
            advance.mos_score,
            advance.consumed_frames,
            advance.pending_voice,
            source.voice_sample_count + advance.added_voice_samples,
            segment_sample_counts,
            embedding_sum,
            advance.finalized,
        )

    def process_streaming(
        self,
        results: tuple[Layer4OfflineResult, ...],
        *,
        final: bool = False,
        finalized_track_keys: frozenset[tuple[str, int, int]] = frozenset(),
    ) -> Layer6Result:
        backend = str(getattr(self.config, "clustering_backend", "complete_link"))
        if backend != "multistage":
            raise RuntimeError(
                "incremental streaming L6 requires the multistage clustering backend"
            )
        started = perf_counter()
        results = self._validate_results(results)
        session_id = results[0].source.session_id
        epochs = {item.source.stream_epoch for item in results}
        if len(epochs) != 1:
            raise ValueError("one streaming L6 update cannot mix stream epochs")
        identity = session_id, next(iter(epochs))
        if self._streaming_identity is not None and self._streaming_identity != identity:
            self.reset_streaming()
        if self._streaming_clusterer is None:
            self._streaming_clusterer = MultiStageVoiceprintClusterer(self.config)
            self._streaming_session_id = session_id
            self._streaming_identity = identity
            self._streaming_snapshot = self._streaming_clusterer.update(())

        selected = self._select_tracks(results)
        advances: list[_StreamingAdvance] = []
        current_keys: set[tuple[str, int, int, int]] = set()
        for result, branch_index, match_score, mos_score in selected:
            stable_branch_id = self._streaming_stable_branch_id(result, branch_index)
            key = (
                result.source.session_id,
                result.source.stream_epoch,
                result.source.track_id,
                stable_branch_id,
            )
            if key in current_keys:
                raise ValueError("streaming L6 received a duplicate stable branch")
            current_keys.add(key)
            state = self._streaming_states.get(key)
            if state is None or state.start_sample_48k != result.source.start_sample:
                generation = 0 if state is None else state.generation + 1
                state = _StreamingVoiceprintState(
                    result.source.session_id,
                    result.source.stream_epoch,
                    result.source.track_id,
                    stable_branch_id,
                    generation,
                    result.source.start_sample,
                    result,
                    branch_index,
                    match_score,
                    mos_score,
                )
            finalize_track = final or state.track_identity in finalized_track_keys
            advance = self._plan_streaming_advance(
                state,
                result,
                branch_index,
                match_score,
                mos_score,
                finalize_track=finalize_track,
            )
            advances.append(advance)

        segment_requests: list[
            tuple[tuple[int, int, int, int], _StreamingAdvance, int, np.ndarray]
        ] = []
        for advance in advances:
            first_segment = advance.state.segment_count
            for offset, segment in enumerate(advance.segments):
                segment_requests.append((
                    (
                        advance.state.start_sample_48k,
                        advance.state.track_id,
                        advance.state.stable_branch_id,
                        first_segment + offset,
                    ),
                    advance,
                    first_segment + offset,
                    segment,
                ))
        segment_requests.sort(key=lambda item: item[0])

        embedded_by_state: dict[
            tuple[str, int, int, int], list[SegmentEvidence]
        ] = {}
        if segment_requests:
            evidence_by_request: list[SegmentEvidence | None] = []
            missing_requests: list[
                tuple[int, tuple[tuple[int, int, int, int], _StreamingAdvance, int, np.ndarray]]
            ] = []
            for request_index, request in enumerate(segment_requests):
                _, advance, segment_index, _ = request
                evidence_id = f"{advance.state.track_key}:segment{segment_index}"
                cached = self._streaming_retry_evidence.get(evidence_id)
                evidence_by_request.append(cached)
                if cached is None:
                    missing_requests.append((request_index, request))
            waveforms = tuple(request[3] for _, request in missing_requests)
            batch_embed = getattr(self.embedder, "embed_batch", None)
            embeddings = (
                tuple(batch_embed(waveforms))
                if waveforms and callable(batch_embed)
                else tuple(self.embedder.embed(value) for value in waveforms)
            )
            if len(embeddings) != len(missing_requests):
                raise RuntimeError("CAMPPlus did not return every streaming embedding")
            for (request_index, request), embedding in zip(
                missing_requests,
                embeddings,
                strict=True,
            ):
                _, advance, segment_index, segment = request
                item = SegmentEvidence(
                    f"{advance.state.track_key}:segment{segment_index}",
                    advance.state.track_key,
                    np.ascontiguousarray(embedding, dtype=np.float32),
                    len(segment),
                )
                evidence_by_request[request_index] = item
                self._streaming_retry_evidence[item.evidence_id] = item
            evidence = tuple(
                item for item in evidence_by_request if item is not None
            )
            if len(evidence) != len(segment_requests):
                raise RuntimeError("streaming L6 evidence preparation is incomplete")
            for request, item in zip(segment_requests, evidence, strict=True):
                _, advance, _, _ = request
                embedded_by_state.setdefault(advance.state.identity, []).append(item)
            assert self._streaming_clusterer is not None
            self._streaming_snapshot = self._streaming_clusterer.update(evidence)

        current_states = tuple(
            self._project_streaming_state(
                advance,
                tuple(embedded_by_state.get(advance.state.identity, ())),
            )
            for advance in advances
        )

        assert self._streaming_snapshot is not None
        active_key_set = frozenset(current_keys)
        new_voice_samples = sum(item.added_voice_samples for item in advances)
        try:
            value = self._build_streaming_result(
                results,
                current_states,
                self._streaming_snapshot,
                final=final,
                new_evidence_count=len(segment_requests),
                new_voice_samples=new_voice_samples,
                started=started,
            )
        except BaseException:
            self._streaming_materialization_dirty = True
            raise
        self._streaming_materialization_dirty = False
        self._streaming_pending_changed_speaker_ids.clear()
        self._streaming_pending_append_speaker_ids.clear()
        self._streaming_retry_evidence.clear()
        for state in current_states:
            self._streaming_states[state.identity] = state
        self._streaming_active_keys = active_key_set
        self._streaming_last_result = value
        return value

    @staticmethod
    def _cluster_order(
        items: tuple[_VoiceprintAudio, ...],
        assignments: tuple[int, ...],
        included_clusters: set[int],
    ) -> tuple[int, ...]:
        return tuple(sorted(included_clusters, key=lambda cluster: min(
            (
                items[index].start_sample_48k,
                items[index].result.source.track_id,
                items[index].branch_index,
            )
            for index, assignment in enumerate(assignments)
            if assignment == cluster
        )))

    def process(self, results: tuple[Layer4OfflineResult, ...]) -> Layer6Result:
        backend = str(getattr(self.config, "clustering_backend", "complete_link"))
        multistage = (
            MultiStageVoiceprintClusterer(self.config)
            if backend == "multistage"
            else None
        )
        return self._process(
            results,
            multistage=multistage,
            include_partial_segment=True,
            partial_segment_track_keys=frozenset(),
        )

    def _process(
        self,
        results: tuple[Layer4OfflineResult, ...],
        *,
        multistage: MultiStageVoiceprintClusterer | None,
        include_partial_segment: bool,
        partial_segment_track_keys: frozenset[tuple[str, int, int]],
    ) -> Layer6Result:
        started = perf_counter()
        results = self._validate_results(results)
        session_id = results[0].source.session_id
        recording_start = min(item.source.start_sample for item in results)
        recording_end = max(item.source.end_sample for item in results)
        if (
            (recording_end - recording_start) % 960
            or any((item.source.start_sample - recording_start) % 960 for item in results)
        ):
            raise ValueError("L6 source timelines must align to absolute 20 ms frames")
        selected = self._select_tracks(results)
        extracted: list[_VoiceprintAudio] = []
        insufficient_voice_audio_ids: list[str] = []
        for result, branch, match_score, mos_score in selected:
            voiceprint = self._voiceprint_audio(
                result,
                branch,
                match_score,
                mos_score,
                include_partial_segment=(
                    include_partial_segment
                    or (
                        result.source.session_id,
                        result.source.stream_epoch,
                        result.source.track_id,
                    ) in partial_segment_track_keys
                ),
            )
            if voiceprint is None:
                insufficient_voice_audio_ids.append(result.output_asset_id)
            else:
                extracted.append(voiceprint)
        voiceprints = tuple(extracted)

        similarities, matched_counts, required_counts, match_features = _pairwise_similarities(
            voiceprints, self.config.speaker_similarity_threshold,
        )
        multistage_snapshot: MultiStageSnapshot | None = None
        if multistage is None:
            raw_assignments = _cluster(
                similarities,
                self.config.speaker_similarity_threshold,
                self.config.maximum_speakers,
            )
            algorithm = "campplus_2s_hungarian_features_complete_link_experiment_v6"
        else:
            multistage_snapshot = multistage.update(self._multistage_evidence(voiceprints))
            raw_assignments = self._multistage_track_assignments(
                voiceprints,
                multistage_snapshot,
            ) if voiceprints else ()
            algorithm = "campplus_2s_multistage_streaming_v1"
        active_clusters = {
            raw
            for item, raw in zip(voiceprints, raw_assignments)
            if any(item.result.l5_is_voice_20ms)
        }
        order = self._cluster_order(voiceprints, raw_assignments, active_clusters)
        if not order:
            matrix = tuple(tuple(float(value) for value in row) for row in similarities)
            return Layer6Result(session_id, 0, (), (), {
                "algorithm": algorithm,
                "multistage": (
                    None if multistage_snapshot is None else {
                        "evidence_count": multistage_snapshot.evidence_count,
                        "cluster_count": multistage_snapshot.cluster_count,
                        "stage": multistage_snapshot.stage,
                    }
                ),
                "recording_start_sample_48k": recording_start,
                "recording_end_sample_48k": recording_end,
                "extracted_audio_ids": tuple(item.audio_id for item in voiceprints),
                "pairwise_audio_ids": tuple(item.audio_id for item in voiceprints),
                "pairwise_similarity_matrix": matrix,
                "pairwise_matched_segment_counts": tuple(
                    tuple(int(value) for value in row) for row in matched_counts
                ),
                "pairwise_required_segment_counts": tuple(
                    tuple(int(value) for value in row) for row in required_counts
                ),
                "pairwise_match_features": match_features,
                "voiceprint_voice_sample_counts": {
                    item.audio_id: item.voice_sample_count for item in voiceprints
                },
                "voiceprint_segment_counts": {
                    item.audio_id: item.segment_count for item in voiceprints
                },
                "voiceprint_retained_segment_counts": {
                    item.audio_id: item.retained_segment_count for item in voiceprints
                },
                "silent_voiceprint_audio_ids": tuple(item.audio_id for item in voiceprints),
                "insufficient_voice_audio_ids": tuple(insufficient_voice_audio_ids),
                "elapsed_ms": (perf_counter() - started) * 1_000.0,
            })
        remap = {raw: index + 1 for index, raw in enumerate(order)}
        speaker_ids = tuple(remap.get(raw, 0) for raw in raw_assignments)
        centroids: dict[int, np.ndarray] = {}
        for speaker_id in range(1, len(order) + 1):
            centroid = np.mean([
                item.embedding
                for item, assigned in zip(voiceprints, speaker_ids)
                if assigned == speaker_id
            ], axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            centroids[speaker_id] = centroid

        assignments: list[Layer6Fragment] = []
        for item, speaker_id in zip(voiceprints, speaker_ids):
            if speaker_id == 0:
                continue
            probabilities = tuple(float(value) for value in item.result.l5_probabilities_20ms)
            speaker_similarity = float(np.clip(
                _cosine(item.embedding, centroids[speaker_id]), 0.0, 1.0,
            ))
            assignments.append(Layer6Fragment(
                item.audio_id,
                item.result.source.asset_id,
                item.result.source.track_id,
                item.result.source.theta_deg,
                item.branch_index,
                item.branch_index == 0,
                item.start_sample_48k,
                item.end_sample_48k,
                item.waveform,
                probabilities,
                tuple(item.result.l5_is_voice_20ms),
                item.embedding,
                speaker_id,
                item.match_score,
                item.mos_score,
                speaker_similarity,
            ))

        outputs = tuple(
            output
            for speaker_id in range(1, len(order) + 1)
            if (output := self._merge_speaker(
                speaker_id,
                tuple(assignments),
                recording_start,
                recording_end,
            )) is not None
        )
        if len(outputs) != len(order):
            raise ValueError("L6 voiceprint cluster contains no active audio")
        matrix = tuple(tuple(float(value) for value in row) for row in similarities)
        return Layer6Result(session_id, len(outputs), outputs, tuple(assignments), {
            "algorithm": algorithm,
            "multistage": (
                None if multistage_snapshot is None else {
                    "evidence_count": multistage_snapshot.evidence_count,
                    "cluster_count": multistage_snapshot.cluster_count,
                    "stage": multistage_snapshot.stage,
                }
            ),
            "recording_start_sample_48k": recording_start,
            "recording_end_sample_48k": recording_end,
            "secondary_candidate_gate": {
                "match_gap_max": self.config.secondary_candidate_match_gap_max,
                "match_min": self.config.secondary_candidate_match_min,
                "mos_min": self.config.secondary_candidate_mos_min,
            },
            "maximum_internal_silence_ms": self.config.maximum_internal_silence_ms,
            "extracted_audio_ids": tuple(item.audio_id for item in voiceprints),
            "voiceprint_voice_sample_counts": {
                item.audio_id: item.voice_sample_count for item in voiceprints
            },
            "voiceprint_segment_counts": {
                item.audio_id: item.segment_count for item in voiceprints
            },
            "voiceprint_retained_segment_counts": {
                item.audio_id: item.retained_segment_count for item in voiceprints
            },
            "insufficient_voice_audio_ids": tuple(insufficient_voice_audio_ids),
            "silent_voiceprint_audio_ids": tuple(
                item.audio_id
                for item, speaker_id in zip(voiceprints, speaker_ids)
                if speaker_id == 0
            ),
            "pairwise_audio_ids": tuple(item.audio_id for item in voiceprints),
            "pairwise_similarity_matrix": matrix,
            "pairwise_matched_segment_counts": tuple(
                tuple(int(value) for value in row) for row in matched_counts
            ),
            "pairwise_required_segment_counts": tuple(
                tuple(int(value) for value in row) for row in required_counts
            ),
            "pairwise_match_features": match_features,
            "voiceprint_audio_ids": {
                speaker_id: tuple(
                    item.audio_id
                    for item, assigned in zip(voiceprints, speaker_ids)
                    if assigned == speaker_id
                )
                for speaker_id in range(1, len(order) + 1)
            },
            "speaker_similarity_threshold": self.config.speaker_similarity_threshold,
            "elapsed_ms": (perf_counter() - started) * 1_000.0,
        })

    def _merge_speaker(
        self,
        speaker_id: int,
        assignments: tuple[Layer6Fragment, ...],
        recording_start: int,
        recording_end: int,
    ) -> Layer6SpeakerAudio | None:
        items = tuple(item for item in assignments if item.speaker_id == speaker_id)
        frame_count = (recording_end - recording_start) // 960
        waveform = np.zeros(frame_count * 320, dtype=np.float32)
        occupied = np.zeros(frame_count, dtype=bool)
        priorities = sorted(
            items,
            key=lambda item: (
                -item.mos_score,
                -item.match_score,
                -item.speaker_similarity,
                item.branch_index,
                item.fragment_id,
            ),
        )
        for item in priorities:
            first_output_frame = (item.start_sample_48k - recording_start) // 960
            frames = item.waveform_16k.reshape(-1, 320)
            for source_frame, active in enumerate(item.voice_is_active_20ms):
                output_frame = first_output_frame + source_frame
                if not active or occupied[output_frame]:
                    continue
                target = output_frame * 320
                waveform[target:target + 320] = frames[source_frame]
                occupied[output_frame] = True
        compressed = _compress_silence(
            waveform,
            occupied,
            self.config.maximum_internal_silence_ms // 20,
        )
        if not len(compressed):
            return None
        return Layer6SpeakerAudio(
            speaker_id,
            speaker_label(speaker_id),
            16_000,
            recording_start,
            recording_end,
            compressed,
            tuple(sorted({item.source_track_id for item in items})),
            tuple(item.fragment_id for item in items),
            float(np.mean([item.mos_score for item in items])),
        )
