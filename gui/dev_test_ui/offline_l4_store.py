from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    annotations: tuple[TrackVoiceAnnotation | None, ...]


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

    def set_processed(self, values: tuple[Layer4ProcessedAudio, ...]) -> None:
        self.clear()
        for item in values:
            track_id = item.source.track_id
            if track_id in self._tracks:
                raise ValueError("L4 UI output track IDs must be unique")
            path = Path(self._temporary.name) / f"l4_track_{track_id:06d}.wav"
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16_000)
                writer.writeframes(self._pcm16(item.waveform_16k))
            hops = len(item.waveform_16k) // 320
            self._tracks[track_id] = _StoredTrack(item, path, (None,) * hops)

    def apply_l5(self, results: tuple[Layer4OfflineResult, ...]) -> None:
        for result in results:
            track_id = result.source.track_id
            stored = self._tracks.get(track_id)
            if stored is None or stored.processed.output_asset_id != result.output_asset_id:
                raise ValueError("L5 result does not match the displayed L4 audio")
            annotations = tuple(
                TrackVoiceAnnotation(
                    result.source.session_id,
                    result.source.stream_epoch,
                    result.source.end_sample // 960,
                    result.source.end_sample,
                    track_id,
                    result.source.start_sample + index * 960,
                    result.source.start_sample + (index + 1) * 960,
                    probability,
                    is_voice,
                    result.l5_model_id,
                    float(result.metadata["l5_threshold"]),
                )
                for index, (probability, is_voice) in enumerate(
                    zip(
                        result.l5_probabilities_20ms,
                        result.l5_is_voice_20ms,
                        strict=True,
                    )
                )
            )
            if len(annotations) != len(stored.annotations):
                raise ValueError("L5 20 ms output does not match the displayed L4 audio")
            stored.annotations = annotations

    def audio_path(self, track_id: int) -> Path | None:
        stored = self._tracks.get(int(track_id))
        return None if stored is None else stored.path

    def snapshots(self) -> tuple[TrackedAudioSnapshot, ...]:
        outputs = []
        for track_id, stored in self._tracks.items():
            item = stored.processed
            hops = item.waveform_16k.reshape(-1, 320)
            envelope = tuple(float(value) for value in np.max(np.abs(hops), axis=1))
            outputs.append(TrackedAudioSnapshot(
                item.source.session_id,
                item.source.stream_epoch,
                track_id,
                "ended",
                item.source.theta_deg,
                1.0,
                len(item.waveform_16k) * 3,
                waveform_envelope=envelope,
                voice_annotations_20ms=stored.annotations,
            ))
        return tuple(sorted(outputs, key=lambda item: item.track_id))
