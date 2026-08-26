from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import struct
import tempfile
import wave

import numpy as np

from layer4_speech_separation import Layer4OfflineResult, Layer4ProcessedAudio
from track_audio_stream import TrackVoiceAnnotation

from .contracts import TrackedAudioSnapshot


@dataclass(slots=True)
class _StoredTrack:
    processed: Layer4ProcessedAudio
    path: Path
    annotations: list[TrackVoiceAnnotation | None]
    preview_id: int
    display_label: str | None
    stable_key: tuple[object, ...]
    audio_samples_16k: int = 0
    envelope_hops: int = 0
    envelope_bin_hops: int = 1
    envelope_peaks: list[float] = field(default_factory=list)
    l5_applied_hops: int = 0


class OfflineLayer4UiStore:
    """Owns L4 WAV previews and their later L5 colour annotations."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="micarray_dev_ui_l4_")
        self._tracks: dict[int, _StoredTrack] = {}

    def close(self) -> None:
        self._tracks.clear()
        self._temporary.cleanup()

    def clear(self) -> None:
        self.close()
        self._temporary = tempfile.TemporaryDirectory(prefix="micarray_dev_ui_l4_")

    @staticmethod
    def _pcm16(waveform: np.ndarray) -> bytes:
        return np.clip(np.rint(waveform * 32768.0), -32768, 32767).astype("<i2").tobytes()

    @classmethod
    def _write_waveform(cls, path: Path, waveform: np.ndarray) -> None:
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16_000)
            for start in range(0, len(waveform), cls._PCM_CHUNK_SAMPLES):
                writer.writeframes(
                    cls._pcm16(waveform[start:start + cls._PCM_CHUNK_SAMPLES])
                )

    @classmethod
    def _append_waveform(cls, path: Path, waveform: np.ndarray) -> None:
        if not len(waveform):
            return
        payload = cls._pcm16(waveform)
        with wave.open(str(path), "rb") as reader:
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != 2
                or reader.getframerate() != 16_000
            ):
                raise ValueError("existing L4 preview WAV format changed")
        with path.open("r+b") as stream:
            header = stream.read(44)
            if len(header) != 44 or header[:4] != b"RIFF" or header[8:12] != b"WAVE" or header[36:40] != b"data":
                raise ValueError("existing L4 preview WAV header is unsupported")
            stream.seek(0, 2)
            stream.write(payload)
            total_bytes = stream.tell()
            stream.seek(4)
            stream.write(struct.pack("<I", total_bytes - 8))
            stream.seek(40)
            stream.write(struct.pack("<I", total_bytes - 44))
            stream.flush()

    @staticmethod
    def _stronger_annotation(
        left: TrackVoiceAnnotation | None,
        right: TrackVoiceAnnotation | None,
    ) -> TrackVoiceAnnotation | None:
        if left is None:
            return right
        if right is None:
            return left
        return right if right.probability > left.probability else left

    @classmethod
    def _compact_envelope(cls, stored: _StoredTrack) -> None:
        while len(stored.envelope_peaks) > cls._MAX_WAVEFORM_BINS:
            stored.envelope_peaks = [
                max(stored.envelope_peaks[index:index + 2])
                for index in range(0, len(stored.envelope_peaks), 2)
            ]
            stored.annotations = [
                cls._stronger_annotation(
                    stored.annotations[index],
                    stored.annotations[index + 1]
                    if index + 1 < len(stored.annotations)
                    else None,
                )
                for index in range(0, len(stored.annotations), 2)
            ]
            stored.envelope_bin_hops *= 2

    @classmethod
    def _append_envelope(cls, stored: _StoredTrack, waveform: np.ndarray) -> None:
        value = np.asarray(waveform, dtype=np.float32)
        if len(value) % 320:
            raise ValueError("L4 preview audio must align to 20 ms")
        for start in range(0, len(value), cls._PCM_CHUNK_SAMPLES):
            chunk = value[start:start + cls._PCM_CHUNK_SAMPLES]
            if len(chunk) % 320:
                # The configured chunk size is hop-aligned; this protects a
                # future change from splitting one envelope frame.
                raise ValueError("L4 preview envelope chunk lost hop alignment")
            peaks = np.max(np.abs(chunk.reshape(-1, 320)), axis=1)
            for raw_peak in peaks:
                bin_index = stored.envelope_hops // stored.envelope_bin_hops
                peak = float(raw_peak)
                if bin_index == len(stored.envelope_peaks):
                    stored.envelope_peaks.append(peak)
                    stored.annotations.append(None)
                elif bin_index == len(stored.envelope_peaks) - 1:
                    stored.envelope_peaks[bin_index] = max(
                        stored.envelope_peaks[bin_index], peak,
                    )
                else:
                    raise RuntimeError("L4 preview envelope append position regressed")
                stored.envelope_hops += 1
                cls._compact_envelope(stored)

    @staticmethod
    def _stable_key(item: Layer4ProcessedAudio, preview_id: int) -> tuple[object, ...]:
        return (
            item.source.session_id,
            item.source.stream_epoch,
            item.source.track_id,
            item.source.start_sample,
            item.output_kind,
            int(item.metadata.get("stable_branch_id", item.metadata.get("candidate_index", 0))),
            preview_id,
        )

    def set_processed(self, values: tuple[Layer4ProcessedAudio, ...]) -> None:
        values = tuple(values)
        parent_counts = Counter(item.source.track_id for item in values)
        for parent_track_id, count in parent_counts.items():
            if count <= 1:
                continue
            kinds = {
                item.output_kind
                for item in values
                if item.source.track_id == parent_track_id
            }
            if count != 2 or kinds != {"candidate_0", "candidate_1"}:
                raise ValueError("duplicate L4 parent IDs require one A/B candidate pair")
        next_preview_id = max((item.source.track_id for item in values), default=0) + 1
        updated: dict[int, _StoredTrack] = {}
        for item in values:
            parent_track_id = item.source.track_id
            mos_score = float(item.metadata["mos_score"])
            if not np.isfinite(mos_score) or not 0.0 <= mos_score <= 1.0:
                raise ValueError("L4 MOS score must be between 0 and 1")
            if parent_counts[parent_track_id] == 1:
                preview_id = parent_track_id
                display_label = (
                    f"{parent_track_id} · {item.source.theta_deg:.1f}°"
                    f" · MOS {mos_score:.3f}"
                )
            else:
                preview_id = next_preview_id
                next_preview_id += 1
                suffix = "A" if item.output_kind == "candidate_0" else "B"
                score = float(item.metadata["candidate_match_score"])
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError("L4 candidate matching score must be between 0 and 1")
                display_label = (
                    f"{parent_track_id}{suffix} · 匹配度 {score:.3f}"
                    f" · MOS {mos_score:.3f}"
                )
            path = Path(self._temporary.name) / (
                f"l4_track_{parent_track_id:06d}_{item.output_kind}.wav"
            )
            stable_key = self._stable_key(item, preview_id)
            stored = self._tracks.get(preview_id)
            append_only = bool(item.metadata.get("realtime_provisional", False))
            can_append = (
                append_only
                and stored is not None
                and stored.stable_key == stable_key
                and path == stored.path
                and len(item.waveform_16k) >= stored.audio_samples_16k
            )
            if can_append:
                tail = item.waveform_16k[stored.audio_samples_16k:]
                self._append_waveform(path, tail)
                self._append_envelope(stored, tail)
                stored.processed = item
                stored.display_label = display_label
                stored.audio_samples_16k = len(item.waveform_16k)
            else:
                self._write_waveform(path, item.waveform_16k)
                stored = _StoredTrack(
                    item,
                    path,
                    [],
                    preview_id,
                    display_label,
                    stable_key,
                    audio_samples_16k=len(item.waveform_16k),
                )
                self._append_envelope(stored, item.waveform_16k)
            updated[preview_id] = stored
        active_paths = {stored.path for stored in updated.values()}
        for preview_id, stale in self._tracks.items():
            if preview_id not in updated and stale.path not in active_paths:
                stale.path.unlink(missing_ok=True)
        self._tracks = updated

    def apply_l5(self, results: tuple[Layer4OfflineResult, ...]) -> None:
        for result in results:
            matches = tuple(
                stored
                for stored in self._tracks.values()
                if stored.processed.output_asset_id == result.output_asset_id
            )
            if len(matches) != 1:
                raise ValueError("L5 result does not match the displayed L4 audio")
            stored = matches[0]
            track_id = stored.preview_id
            frame_count = len(result.l5_probabilities_20ms)
            if frame_count != stored.envelope_hops:
                raise ValueError("L5 20 ms output does not match the displayed L4 audio")
            if stored.l5_applied_hops > frame_count:
                stored.annotations = [None] * len(stored.envelope_peaks)
                stored.l5_applied_hops = 0
            threshold = float(result.metadata["l5_threshold"])
            for first in range(stored.l5_applied_hops, frame_count, 4_096):
                last = min(frame_count, first + 4_096)
                probabilities = result.l5_probabilities_20ms[first:last]
                decisions = result.l5_is_voice_20ms[first:last]
                for offset, (raw_probability, raw_is_voice) in enumerate(
                    zip(probabilities, decisions, strict=True)
                ):
                    index = first + offset
                    probability = float(raw_probability)
                    annotation = TrackVoiceAnnotation(
                        result.source.session_id,
                        result.source.stream_epoch,
                        result.source.end_sample // 960,
                        result.source.end_sample,
                        track_id,
                        result.source.start_sample + index * 960,
                        result.source.start_sample + (index + 1) * 960,
                        probability,
                        bool(raw_is_voice),
                        result.l5_model_id,
                        threshold,
                    )
                    bin_index = index // stored.envelope_bin_hops
                    if bin_index >= len(stored.annotations):
                        raise ValueError("L5 annotation exceeds the compact preview envelope")
                    stored.annotations[bin_index] = self._stronger_annotation(
                        stored.annotations[bin_index], annotation,
                    )
            stored.l5_applied_hops = frame_count

    def audio_path(self, track_id: int) -> Path | None:
        stored = self._tracks.get(int(track_id))
        return None if stored is None else stored.path

    def snapshots(self) -> tuple[TrackedAudioSnapshot, ...]:
        outputs = []
        for track_id, stored in self._tracks.items():
            item = stored.processed
            outputs.append(TrackedAudioSnapshot(
                item.source.session_id,
                item.source.stream_epoch,
                track_id,
                "ended",
                item.source.theta_deg,
                1.0,
                len(item.waveform_16k) * 3,
                waveform_envelope=tuple(stored.envelope_peaks),
                voice_annotations_20ms=tuple(stored.annotations),
                display_label=stored.display_label,
                parent_track_id=item.source.track_id,
            ))
        return tuple(sorted(outputs, key=lambda item: item.track_id))
    _MAX_WAVEFORM_BINS = 4_096
    _PCM_CHUNK_SAMPLES = 160_000
