from __future__ import annotations

from pathlib import Path
import tempfile
import wave

import numpy as np

from layer6_speaker_consolidation import Layer6Result

from .contracts import TrackedAudioSnapshot


class OfflineLayer6UiStore:
    """Temporary Test UI WAV previews for manually produced L6 speakers."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="micarray_dev_ui_l6_")
        self._paths: dict[int, Path] = {}
        self._snapshots: tuple[TrackedAudioSnapshot, ...] = ()

    def close(self) -> None:
        self._paths.clear()
        self._snapshots = ()
        self._temporary.cleanup()

    def clear(self) -> None:
        self.close()
        self._temporary = tempfile.TemporaryDirectory(prefix="micarray_dev_ui_l6_")

    def set_result(self, result: Layer6Result) -> None:
        self.clear()
        if not result.outputs:
            self._snapshots = ()
            return
        timeline_start = min(output.start_sample_48k for output in result.outputs)
        timeline_end = max(output.end_sample_48k for output in result.outputs)
        timeline_samples_16k = (timeline_end - timeline_start) // 3
        snapshots = []
        for output in result.outputs:
            timeline = np.zeros(timeline_samples_16k, dtype=np.float32)
            offset = (output.start_sample_48k - timeline_start) // 3
            timeline[offset:offset + len(output.waveform_16k)] = output.waveform_16k
            path = Path(self._temporary.name) / f"l6_id_{output.speaker_id}.wav"
            pcm = np.clip(np.rint(timeline * 32768.0), -32768, 32767).astype("<i2")
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16_000)
                writer.writeframes(pcm.tobytes())
            self._paths[output.speaker_id] = path
            envelope = tuple(float(value) for value in np.max(
                np.abs(timeline.reshape(-1, 320)), axis=1,
            ))
            ids = ",".join(str(value) for value in output.source_track_ids)
            snapshots.append(TrackedAudioSnapshot(
                result.session_id, 0, output.speaker_id, "ended", 0.0,
                output.mean_quality, timeline_end - timeline_start,
                waveform_envelope=envelope,
                display_label=(
                    f"L6 ID {output.speaker_id} · {output.label} · "
                    f"来源L2 ID {ids} · Q{output.mean_quality:.2f}"
                ),
                parent_track_id=output.source_track_ids[0],
            ))
        self._snapshots = tuple(snapshots)

    def audio_path(self, speaker_id: int) -> Path | None:
        return self._paths.get(int(speaker_id))

    def snapshots(self) -> tuple[TrackedAudioSnapshot, ...]:
        return self._snapshots
