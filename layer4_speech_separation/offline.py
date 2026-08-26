from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Literal, Mapping
from uuid import uuid4
import wave

import numpy as np

from layer5_voice_classifier import (
    Layer5AudioSegment,
    Layer5Engine,
    Layer5LongAudioResult,
)
from layer5_voice_classifier.gain_compensation import (
    InputGainCompensationDiagnostic,
    InputGainCompensationSettings,
)

from .contracts import (
    Layer4LongAudioInput,
    Layer4OfflineResult,
    Layer4ProcessedAudio,
    Layer4PrimarySelection,
    Layer4SeparationRequest,
    SpeakerCountDecision,
)
from .interfaces import AudioQualityScorer, Layer4SeparationBackend, SpeakerCountClassifier
from .matching import BandMagnitudeMatcher
from .resampling import Layer4Resampler


_PERSIST_HASH_CHUNK_BYTES = 4 * 1024 * 1024
_PERSIST_WAV_CHUNK_SAMPLES = 10 * 16_000
_PERSIST_INLINE_FRAME_LIMIT = 4_096
_PERSIST_FRAME_CHUNK_ITEMS = 65_536


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while payload := source.read(_PERSIST_HASH_CHUNK_BYTES):
            digest.update(payload)
    return digest.hexdigest()


def _write_pcm16_wav(final: Path, waveform: np.ndarray) -> str:
    """Write one 16 kHz WAV with bounded conversion memory and return its file hash."""

    partial = final.with_suffix(".wav.partial")
    with wave.open(str(partial), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        for first in range(0, len(waveform), _PERSIST_WAV_CHUNK_SAMPLES):
            last = min(len(waveform), first + _PERSIST_WAV_CHUNK_SAMPLES)
            chunk = np.asarray(waveform[first:last], dtype=np.float32)
            pcm = np.clip(
                np.rint(chunk * 32768.0), -32768, 32767,
            ).astype("<i2")
            writer.writeframesraw(memoryview(pcm).cast("B"))
    output_sha256 = _sha256_file(partial)
    os.replace(partial, final)
    return output_sha256


def _write_frame_sidecar(
    output_root: Path,
    *,
    stem: str,
    field: str,
    values: object,
    dtype: np.dtype,
) -> dict[str, object]:
    """Atomically persist one long frame sequence without materializing it in full."""

    count = len(values)  # type: ignore[arg-type]
    final = output_root / f"{stem}.{field}.bin"
    partial = final.with_suffix(final.suffix + ".partial")
    digest = hashlib.sha256()
    with partial.open("wb") as stream:
        for first in range(0, count, _PERSIST_FRAME_CHUNK_ITEMS):
            last = min(count, first + _PERSIST_FRAME_CHUNK_ITEMS)
            chunk = np.ascontiguousarray(
                values[first:last],  # type: ignore[index]
                dtype=dtype,
            )
            payload = memoryview(chunk).cast("B")
            digest.update(payload)
            written = 0
            while written < len(payload):
                count_written = stream.write(payload[written:])
                if count_written is None or count_written <= 0:
                    raise OSError("offline L5 sidecar ended during a short write")
                written += count_written
    os.replace(partial, final)
    return {
        "storage": "binary_sidecar_v1",
        "path": final.relative_to(output_root).as_posix(),
        "dtype": dtype.str,
        "frame_count": count,
        "sha256": digest.hexdigest(),
    }


def _persist_frame_sequence(
    output_root: Path,
    *,
    stem: str,
    field: str,
    values: object,
    dtype: np.dtype,
) -> object:
    count = len(values)  # type: ignore[arg-type]
    if count <= _PERSIST_INLINE_FRAME_LIMIT:
        if dtype == np.dtype(np.bool_):
            return tuple(bool(value) for value in values)  # type: ignore[union-attr]
        return tuple(float(value) for value in values)  # type: ignore[union-attr]
    return _write_frame_sidecar(
        output_root,
        stem=stem,
        field=field,
        values=values,
        dtype=dtype,
    )


def _read_mono_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2 or reader.getframerate() != 48_000:
            raise ValueError(f"offline L4 source is not 48 kHz mono PCM16: {path.name}")
        value = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    return np.ascontiguousarray(value.astype(np.float32) / 32768.0)


def load_sealed_l3_tracks(session_root: str | Path) -> tuple[Layer4LongAudioInput, ...]:
    """Load and stitch hash-verified canonical ID streams from a completed session."""

    root = Path(session_root).resolve()
    manifest = json.loads((root / "session_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "audio_session_v2" or manifest.get("status") not in {
        "complete", "result_incomplete",
    }:
        raise ValueError("offline L4 accepts only finalized audio_session_v2 sessions")
    rows: list[dict[str, object]] = []
    for chunk in manifest.get("chunks", ()):
        for asset in chunk.get("assets", ()):
            if asset.get("kind") != "enhanced_audio":
                continue
            if asset.get("stream_kind") != "id_continuous_gain_compensated":
                continue
            relative = asset.get("path")
            if not isinstance(relative, str) or not relative:
                raise ValueError("sealed L3 asset path is missing")
            path = (root / relative).resolve(strict=True)
            if root != path and root not in path.parents:
                raise ValueError("sealed L3 asset path escapes its session")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != asset.get("sha256"):
                raise ValueError(f"sealed L3 asset hash mismatch: {relative}")
            rows.append({**asset, "path_object": path})
    rows.sort(key=lambda x: (int(x["stream_epoch"]), int(x["track_id"]), int(x["start_sample"])))
    decisions: dict[int, list[tuple[int, int]]] = {}
    for chunk in manifest.get("chunks", ()):
        for asset in chunk.get("assets", ()):
            if asset.get("kind") != "results":
                continue
            relative = asset.get("path")
            path = (root / str(relative)).resolve(strict=True)
            if root != path and root not in path.parents:
                raise ValueError("result asset path escapes its session")
            if hashlib.sha256(path.read_bytes()).hexdigest() != asset.get("sha256"):
                raise ValueError("result asset hash mismatch")
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get("record_type") != "decision":
                        continue
                    candidates = tuple(row.get("candidates", ()))
                    if len(candidates) > 2:
                        raise ValueError("offline L4 supports at most two L2 directions")
                    decisions.setdefault(int(row["stream_epoch"]), []).append(
                        (int(row["decision_sample"]), len(candidates))
                    )

    outputs: list[Layer4LongAudioInput] = []
    group: list[dict[str, object]] = []

    def flush() -> None:
        if not group:
            return
        waveforms = tuple(_read_mono_pcm16(item["path_object"]) for item in group)  # type: ignore[arg-type]
        waveform = np.ascontiguousarray(np.concatenate(waveforms), dtype=np.float32)
        first, last = group[0], group[-1]
        if int(last["end_sample"]) - int(first["start_sample"]) != len(waveform):
            raise ValueError("sealed L3 asset timeline does not match its samples")
        digest = _sha256_bytes(waveform.tobytes())
        outputs.append(Layer4LongAudioInput(
            asset_id=(
                f"{manifest['session_id']}:epoch{int(first['stream_epoch'])}:"
                f"track{int(first['track_id'])}:start{int(first['start_sample'])}"
            ),
            sha256=digest,
            session_id=str(manifest["session_id"]),
            stream_epoch=int(first["stream_epoch"]),
            track_id=int(first["track_id"]),
            theta_deg=float(last["theta_deg"]),
            start_sample=int(first["start_sample"]),
            sample_rate=48_000,
            waveform=waveform,
            l2_direction_counts=tuple(
                (sample, count)
                for sample, count in decisions.get(int(first["stream_epoch"]), ())
                if int(first["start_sample"]) <= sample <= int(last["end_sample"])
            ),
        ))
        group.clear()

    for row in rows:
        if group and (
            int(row["stream_epoch"]), int(row["track_id"]), int(row["start_sample"])
        ) != (
            int(group[-1]["stream_epoch"]), int(group[-1]["track_id"]), int(group[-1]["end_sample"])
        ):
            flush()
        group.append(row)
    flush()
    return tuple(outputs)


class OfflineLayer4Pipeline:
    """Synchronous, user-triggered L4/L5 pipeline for already sealed sessions."""

    def __init__(
        self,
        *,
        speaker_counter: SpeakerCountClassifier,
        backends: Mapping[str, Layer4SeparationBackend],
        layer5: Layer5Engine,
        quality_scorer: AudioQualityScorer,
        default_backend: str,
        resampler: Layer4Resampler | None = None,
        matcher: BandMagnitudeMatcher | None = None,
    ) -> None:
        if default_backend not in backends:
            raise ValueError("default Layer 4 backend is not configured")
        self.speaker_counter = speaker_counter
        self.backends = dict(backends)
        self.layer5 = layer5
        self.quality_scorer = quality_scorer
        self.default_backend = default_backend
        self.resampler = resampler or Layer4Resampler()
        self.matcher = matcher or BandMagnitudeMatcher()

    def process_l4(
        self, source: Layer4LongAudioInput, *, request_id: str | None = None,
    ) -> Layer4ProcessedAudio:
        """Run resampling, speaker routing, separation and matching, but not L5."""

        request_id = request_id or str(uuid4())
        started = perf_counter()
        reference_16k = self.resampler.to_16k(source.waveform)
        count = self.speaker_counter.classify(source)
        selected: Layer4PrimarySelection | None = None
        if count.speaker_count == 1:
            output_16k = reference_16k
            path = "single_speaker_bypass"
            model_metadata: dict[str, object] = {}
        else:
            request = Layer4SeparationRequest(
                request_id, source, count, self.default_backend  # type: ignore[arg-type]
            )
            backend = self.backends[request.backend]
            candidates = backend.separate(request_id, reference_16k)
            selected = self.matcher.select(
                parent=source, reference_16k=reference_16k, candidates=candidates,
            )
            output_16k = selected.waveform
            path = "two_speaker_separation"
            model_metadata = {
                "backend": request.backend,
                "model_id": candidates.model_id,
                "model_revision": candidates.model_revision,
                "candidate_scores": selected.candidate_scores,
                "selected_source_index": selected.selected_source_index,
                "used_reference_fallback": selected.used_reference_fallback,
                "fallback_reason": selected.fallback_reason,
            }
        return self._build_processed(
            source=source,
            request_id=request_id,
            speaker_count=count,
            path=path,
            selected=selected,
            output_16k=output_16k,
            model_metadata=model_metadata,
            started=started,
        )

    def process_l4_unmerged(
        self, source: Layer4LongAudioInput, *, request_id: str | None = None,
    ) -> tuple[Layer4ProcessedAudio, ...]:
        """Return both separator outputs, ordered by their L3 matching score."""

        request_id = request_id or str(uuid4())
        started = perf_counter()
        reference_16k = self.resampler.to_16k(source.waveform)
        count = self.speaker_counter.classify(source)
        if count.speaker_count == 1:
            return (self._build_processed(
                source=source,
                request_id=request_id,
                speaker_count=count,
                path="single_speaker_bypass",
                selected=None,
                output_16k=reference_16k,
                model_metadata={},
                started=started,
            ),)

        request = Layer4SeparationRequest(
            request_id, source, count, self.default_backend  # type: ignore[arg-type]
        )
        backend = self.backends[request.backend]
        candidates = backend.separate(request_id, reference_16k)
        scored = self.matcher.select(
            parent=source,
            reference_16k=reference_16k,
            candidates=candidates,
        )
        ranked_indices = tuple(sorted(
            range(len(candidates.sources)),
            key=lambda index: (-scored.candidate_scores[index], index),
        ))
        return tuple(
            self._build_processed(
                source=source,
                request_id=request_id,
                speaker_count=count,
                path="two_speaker_separation",
                selected=None,
                output_16k=candidates.sources[candidate_index],
                model_metadata={
                    "backend": request.backend,
                    "model_id": candidates.model_id,
                    "model_revision": candidates.model_revision,
                    "candidate_index": candidate_index,
                    "candidate_rank": rank,
                    "candidate_match_score": scored.candidate_scores[candidate_index],
                    "candidate_scores": scored.candidate_scores,
                    "matching_algorithm": scored.matching_algorithm,
                    "merge_candidates": False,
                },
                started=started,
                output_kind=("candidate_0", "candidate_1")[rank],
            )
            for rank, candidate_index in enumerate(ranked_indices)
        )

    def _build_processed(
        self,
        *,
        source: Layer4LongAudioInput,
        request_id: str,
        speaker_count: SpeakerCountDecision,
        path: Literal["single_speaker_bypass", "two_speaker_separation"],
        selected: Layer4PrimarySelection | None,
        output_16k: np.ndarray,
        model_metadata: Mapping[str, object],
        started: float,
        output_kind: Literal["merged", "candidate_0", "candidate_1"] = "merged",
    ) -> Layer4ProcessedAudio:
        expected = len(source.waveform) // 3
        if len(output_16k) < expected:
            output_16k = np.pad(output_16k, (0, expected - len(output_16k)))
        output_16k = np.ascontiguousarray(output_16k[:expected], dtype=np.float32)
        peak = float(np.max(np.abs(output_16k)))
        pcm16_ceiling = 32767.0 / 32768.0
        peak_safety_gain = 1.0
        if peak > pcm16_ceiling:
            peak_safety_gain = pcm16_ceiling / peak
            output_16k = np.ascontiguousarray(
                output_16k * np.float32(peak_safety_gain), dtype=np.float32,
            )
        dnsmos_sig, dnsmos_bak, dnsmos_ovrl = self.quality_scorer.score(output_16k)
        dnsmos_values = tuple(float(value) for value in (
            dnsmos_sig, dnsmos_bak, dnsmos_ovrl,
        ))
        if any(not np.isfinite(value) or not 1.0 <= value <= 5.0 for value in dnsmos_values):
            raise ValueError("L4 DNSMOS scores must be between 1 and 5")
        dnsmos_sig, dnsmos_bak, dnsmos_ovrl = dnsmos_values
        mos_score = float(np.clip(
            ((0.25 * dnsmos_sig + 0.25 * dnsmos_bak + 0.50 * dnsmos_ovrl) - 1.0)
            / 4.0,
            0.0,
            1.0,
        ))
        output_hash = _sha256_bytes(output_16k.tobytes())
        return Layer4ProcessedAudio(
            request_id=request_id,
            source=source,
            speaker_count=speaker_count,
            path=path,
            selected=selected,
            output_asset_id=(
                f"{source.asset_id}:l4:{request_id}"
                if output_kind == "merged"
                else f"{source.asset_id}:l4:{request_id}:{output_kind}"
            ),
            output_sha256=output_hash,
            waveform_16k=output_16k,
            metadata={
                **model_metadata,
                "resampler": self.resampler.algorithm_version,
                "matching_algorithm": (
                    model_metadata.get("matching_algorithm")
                    if selected is None
                    else selected.matching_algorithm
                ),
                "pcm16_peak_safety_gain": peak_safety_gain,
                "dnsmos_sig": dnsmos_sig,
                "dnsmos_bak": dnsmos_bak,
                "dnsmos_ovrl": dnsmos_ovrl,
                "mos_score": mos_score,
                "l4_elapsed_ms": (perf_counter() - started) * 1_000.0,
            },
            output_kind=output_kind,
        )

    def _prepare_l5(
        self,
        processed: Layer4ProcessedAudio,
    ) -> tuple[np.ndarray, Layer5AudioSegment]:
        output_16k = processed.waveform_16k
        source = processed.source
        # Hub archives are already gain-compensated. L5 must not scan or copy a
        # multi-hour immutable waveform merely to record an all-zero disabled
        # gain diagnostic.
        segment_count = len(output_16k) // 320
        if segment_count * 320 != len(output_16k):
            raise ValueError("L4 output must contain complete 16 kHz 20 ms hops")
        gain_settings = replace(
            getattr(
                self.layer5,
                "input_gain_compensation",
                InputGainCompensationSettings(),
            ),
            enabled=False,
        )
        gain_diagnostic = InputGainCompensationDiagnostic(
            algorithm_version=f"{gain_settings.algorithm_version}:trusted_disabled_v1",
            enabled=False,
            segments=(),
            max_applied_gain_db=0.0,
            mean_applied_gain_db=0.0,
            compensated_segment_count=0,
            peak_protection_trigger_count=0,
            final_gain_db=0.0,
        )
        segment = Layer5AudioSegment(
            source.session_id, source.stream_epoch, source.end_sample // 960,
            source.end_sample // 3, source.theta_deg, 16_000, output_16k,
            track_id=source.track_id,
            effective_start_sample=source.start_sample // 3,
            effective_end_sample=source.end_sample // 3,
            gain_compensated=True,
            gain_compensation_diagnostic=gain_diagnostic,
        )
        return output_16k, segment

    @staticmethod
    def _complete_l5(
        processed: Layer4ProcessedAudio,
        output_16k: np.ndarray,
        l5: Layer5LongAudioResult,
        elapsed_ms: float,
        batch_track_count: int,
    ) -> Layer4OfflineResult:
        source = processed.source
        return Layer4OfflineResult(
            request_id=processed.request_id,
            source=source,
            speaker_count=processed.speaker_count,
            path=processed.path,
            selected=processed.selected,
            l5_probability=l5.summary_probability,
            l5_is_voice=l5.summary_is_voice,
            l5_model_id=l5.model_id,
            l5_probabilities_20ms=l5.probabilities_20ms,
            l5_is_voice_20ms=l5.is_voice_20ms,
            output_asset_id=processed.output_asset_id,
            # L5 receives the immutable L4 output with gain compensation
            # explicitly disabled, so its authoritative audio hash is already
            # available and need not be recomputed for every progressive view.
            output_sha256=processed.output_sha256,
            metadata={
                **processed.metadata,
                "l5_elapsed_ms": elapsed_ms,
                "l5_batch_track_count": batch_track_count,
                "l5_threshold": l5.threshold,
                "l5_frame_shift_ms": 20,
                "l5_frame_count": len(l5.probabilities_20ms),
                "l5_model_metadata": dict(l5.metadata),
                "output_waveform_16k": output_16k,
            },
            output_kind=processed.output_kind,
        )

    def process_l5(self, processed: Layer4ProcessedAudio) -> Layer4OfflineResult:
        """Run L5 only after the caller explicitly sends completed L4 audio."""

        return self.process_l5_sealed((processed,))[0]

    def process(self, source: Layer4LongAudioInput, *, request_id: str | None = None) -> Layer4OfflineResult:
        return self.process_l5(self.process_l4(source, request_id=request_id))

    def process_l4_sealed(
        self,
        sources: tuple[Layer4LongAudioInput, ...],
        *,
        merge_candidates: bool = True,
    ) -> tuple[Layer4ProcessedAudio, ...]:
        sources = self._validate_sources(sources)
        if type(merge_candidates) is not bool:
            raise ValueError("merge_candidates must be bool")
        if merge_candidates:
            return tuple(self.process_l4(source) for source in sources)
        return tuple(
            output
            for source in sources
            for output in self.process_l4_unmerged(source)
        )

    def process_l5_sealed(
        self, processed: tuple[Layer4ProcessedAudio, ...],
    ) -> tuple[Layer4OfflineResult, ...]:
        processed = tuple(processed)
        if not processed:
            raise ValueError("offline L5 requires at least one completed L4 track")
        sessions = {item.source.session_id for item in processed}
        if len(sessions) != 1:
            raise ValueError("one offline L5 job cannot mix capture sessions")
        l5_started = perf_counter()
        prepared = tuple(self._prepare_l5(item) for item in processed)
        segments = tuple(item[1] for item in prepared)
        batch = getattr(self.layer5, "process_long_audio_batch_20ms", None)
        l5_results = (
            tuple(batch(segments))
            if callable(batch)
            else tuple(self.layer5.process_long_audio_20ms(item) for item in segments)
        )
        if len(l5_results) != len(processed):
            raise RuntimeError("offline L5 batch did not return every completed L4 track")
        l5_elapsed_ms = (perf_counter() - l5_started) * 1_000.0
        return tuple(
            self._complete_l5(
                item,
                prepared_item[0],
                l5,
                l5_elapsed_ms,
                len(processed),
            )
            for item, prepared_item, l5 in zip(
                processed, prepared, l5_results, strict=True,
            )
        )

    @staticmethod
    def _validate_sources(
        sources: tuple[Layer4LongAudioInput, ...],
    ) -> tuple[Layer4LongAudioInput, ...]:
        sources = tuple(sources)
        if not sources:
            raise ValueError("offline L4 requires at least one sealed Hub track")
        sessions = {item.session_id for item in sources}
        if len(sessions) != 1:
            raise ValueError("one offline L4 job cannot mix capture sessions")
        return sources

    def process_sealed(
        self, sources: tuple[Layer4LongAudioInput, ...],
    ) -> tuple[Layer4OfflineResult, ...]:
        return self.process_l5_sealed(self.process_l4_sealed(sources))


def persist_offline_results(
    session_root: str | Path,
    results: tuple[Layer4OfflineResult, ...],
) -> Path:
    """Atomically publish WAVs and an auditable job manifest under the session."""

    root = Path(session_root).resolve()
    manifest = json.loads((root / "session_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") not in {"complete", "result_incomplete"}:
        raise ValueError("offline results may only be attached to a finalized session")
    job_id = str(uuid4())
    output_root = root / "offline_l4" / job_id
    output_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    for result in results:
        waveform = np.asarray(result.metadata["output_waveform_16k"])
        if (
            waveform.ndim != 1
            or waveform.dtype != np.dtype(np.float32)
            or not waveform.flags.c_contiguous
            or len(waveform) != len(result.l5_probabilities_20ms) * 320
        ):
            raise ValueError(
                "offline persistence requires aligned C-contiguous float32 16 kHz audio"
            )
        candidate_suffix = "" if result.output_kind == "merged" else f"_{result.output_kind}"
        name = (
            f"epoch{result.source.stream_epoch:03d}_track{result.source.track_id:06d}"
            f"{candidate_suffix}.wav"
        )
        final = output_root / name
        output_sha256 = _write_pcm16_wav(final, waveform)
        probability_payload = _persist_frame_sequence(
            output_root,
            stem=final.stem,
            field="l5_probabilities_20ms",
            values=result.l5_probabilities_20ms,
            dtype=np.dtype("<f4"),
        )
        decision_payload = _persist_frame_sequence(
            output_root,
            stem=final.stem,
            field="l5_is_voice_20ms",
            values=result.l5_is_voice_20ms,
            dtype=np.dtype(np.bool_),
        )
        metadata = {k: v for k, v in result.metadata.items() if k != "output_waveform_16k"}
        rows.append({
            "request_id": result.request_id,
            "source_asset_id": result.source.asset_id,
            "source_sha256": result.source.sha256,
            "stream_epoch": result.source.stream_epoch,
            "track_id": result.source.track_id,
            "theta_deg": result.source.theta_deg,
            "speaker_count": {
                "asset_id": result.speaker_count.asset_id,
                "speaker_count": result.speaker_count.speaker_count,
                "confidence": result.speaker_count.confidence,
                "classifier_id": result.speaker_count.classifier_id,
                "metadata": dict(result.speaker_count.metadata),
            },
            "path": result.path,
            "output_kind": result.output_kind,
            "selection": None if result.selected is None else {
                "selected_source_index": result.selected.selected_source_index,
                "candidate_scores": result.selected.candidate_scores,
                "score_margin": result.selected.score_margin,
                "matching_algorithm": result.selected.matching_algorithm,
            },
            "l5_probability": result.l5_probability,
            "l5_is_voice": result.l5_is_voice,
            "l5_model_id": result.l5_model_id,
            "l5_probabilities_20ms": probability_payload,
            "l5_is_voice_20ms": decision_payload,
            "output_path": name,
            "output_sha256": output_sha256,
            "metadata": metadata,
        })
    payload = {
        "schema_version": "offline_l4_job_v2",
        "job_id": job_id,
        "session_id": manifest["session_id"],
        "result_count": len(rows),
        "results": rows,
    }
    target = output_root / "manifest.json"
    temporary = output_root / "manifest.json.partial"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return target
