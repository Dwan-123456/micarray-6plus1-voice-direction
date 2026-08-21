from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Mapping
from uuid import uuid4
import wave

import numpy as np

from layer5_voice_classifier import Layer5AudioSegment, Layer5Engine
from layer5_voice_classifier.gain_compensation import (
    InputGainCompensationSettings,
    compensate_l5_input,
)

from .contracts import (
    Layer4LongAudioInput,
    Layer4OfflineResult,
    Layer4PrimarySelection,
    Layer4SeparationRequest,
)
from .interfaces import Layer4SeparationBackend, SpeakerCountClassifier
from .matching import BandMagnitudeMatcher
from .resampling import Layer4Resampler


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        default_backend: str,
        resampler: Layer4Resampler | None = None,
        matcher: BandMagnitudeMatcher | None = None,
    ) -> None:
        if default_backend not in backends:
            raise ValueError("default Layer 4 backend is not configured")
        self.speaker_counter = speaker_counter
        self.backends = dict(backends)
        self.layer5 = layer5
        self.default_backend = default_backend
        self.resampler = resampler or Layer4Resampler()
        self.matcher = matcher or BandMagnitudeMatcher()

    def process(self, source: Layer4LongAudioInput, *, request_id: str | None = None) -> Layer4OfflineResult:
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
            }
        output_48k = self.resampler.to_48k(output_16k)
        expected = len(source.waveform)
        if len(output_48k) < expected:
            output_48k = np.pad(output_48k, (0, expected - len(output_48k)))
        output_48k = np.ascontiguousarray(output_48k[:expected], dtype=np.float32)
        # Hub archives are already gain-compensated.  Produce the normal L5
        # diagnostic with compensation disabled so L5 never applies gain twice.
        segment_count = len(output_48k) // 960
        if segment_count * 960 != len(output_48k):
            raise ValueError("sealed Hub audio must contain complete 20 ms hops")
        output_48k, gain_diagnostic = compensate_l5_input(
            output_48k,
            (None,) * segment_count,
            replace(
                getattr(
                    self.layer5,
                    "input_gain_compensation",
                    InputGainCompensationSettings(),
                ),
                enabled=False,
            ),
            segment_count=segment_count,
        )
        l5 = self.layer5.process((Layer5AudioSegment(
            source.session_id, source.stream_epoch, source.end_sample // 960,
            source.end_sample, source.theta_deg, 48_000, output_48k,
            track_id=source.track_id,
            effective_start_sample=source.start_sample,
            effective_end_sample=source.end_sample,
            gain_compensated=True,
            gain_compensation_diagnostic=gain_diagnostic,
        ),))
        detection = l5.detections[0]
        output_hash = _sha256_bytes(output_48k.tobytes())
        return Layer4OfflineResult(
            request_id=request_id,
            source=source,
            speaker_count=count,
            path=path,  # type: ignore[arg-type]
            selected=selected,
            l5_probability=detection.probability,
            l5_is_voice=detection.is_voice,
            l5_model_id=detection.model_id,
            output_asset_id=f"{source.asset_id}:l4:{request_id}",
            output_sha256=output_hash,
            metadata={
                **model_metadata,
                "resampler": self.resampler.algorithm_version,
                "matching_algorithm": None if selected is None else selected.matching_algorithm,
                "elapsed_ms": (perf_counter() - started) * 1_000.0,
                "output_waveform_48k": output_48k,
            },
        )

    def process_sealed(
        self, sources: tuple[Layer4LongAudioInput, ...],
    ) -> tuple[Layer4OfflineResult, ...]:
        sources = tuple(sources)
        if not sources:
            raise ValueError("offline L4 requires at least one sealed Hub track")
        sessions = {item.session_id for item in sources}
        if len(sessions) != 1:
            raise ValueError("one offline L4 job cannot mix capture sessions")
        return tuple(self.process(source) for source in sources)


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
        waveform = np.asarray(result.metadata["output_waveform_48k"], dtype=np.float32)
        name = f"epoch{result.source.stream_epoch:03d}_track{result.source.track_id:06d}.wav"
        final = output_root / name
        partial = final.with_suffix(".wav.partial")
        with wave.open(str(partial), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(48_000)
            pcm = np.clip(np.rint(waveform * 32768.0), -32768, 32767).astype("<i2")
            writer.writeframes(pcm.tobytes())
        os.replace(partial, final)
        metadata = {k: v for k, v in result.metadata.items() if k != "output_waveform_48k"}
        rows.append({
            "request_id": result.request_id,
            "source_asset_id": result.source.asset_id,
            "source_sha256": result.source.sha256,
            "stream_epoch": result.source.stream_epoch,
            "track_id": result.source.track_id,
            "theta_deg": result.source.theta_deg,
            "speaker_count": asdict(result.speaker_count),
            "path": result.path,
            "selection": None if result.selected is None else {
                "selected_source_index": result.selected.selected_source_index,
                "candidate_scores": result.selected.candidate_scores,
                "score_margin": result.selected.score_margin,
                "matching_algorithm": result.selected.matching_algorithm,
            },
            "l5_probability": result.l5_probability,
            "l5_is_voice": result.l5_is_voice,
            "l5_model_id": result.l5_model_id,
            "output_path": name,
            "output_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
            "metadata": metadata,
        })
    payload = {
        "schema_version": "offline_l4_job_v1",
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
