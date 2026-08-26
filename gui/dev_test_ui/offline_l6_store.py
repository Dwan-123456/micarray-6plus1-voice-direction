from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import struct
import tempfile
import wave

import numpy as np

from layer6_speaker_consolidation import Layer6Result

from .contracts import TrackedAudioSnapshot


@dataclass(slots=True)
class _StoredSpeaker:
    path: Path
    samples_16k: int = 0
    envelope_hops: int = 0
    envelope_bin_hops: int = 1
    envelope_peaks: list[float] = field(default_factory=list)


class OfflineLayer6UiStore:
    """Temporary Test UI WAV previews for manually produced L6 speakers."""

    _MAX_WAVEFORM_BINS = 4_096
    _PCM_CHUNK_SAMPLES = 160_000

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="micarray_dev_ui_l6_")
        self._paths: dict[int, Path] = {}
        self._stored: dict[int, _StoredSpeaker] = {}
        self._snapshots: tuple[TrackedAudioSnapshot, ...] = ()
        self._session_id: str | None = None

    def close(self) -> None:
        self._paths.clear()
        self._stored.clear()
        self._snapshots = ()
        self._session_id = None
        self._temporary.cleanup()

    def clear(self) -> None:
        self.close()
        self._temporary = tempfile.TemporaryDirectory(prefix="micarray_dev_ui_l6_")

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
                raise ValueError("existing L6 preview WAV format changed")
        with path.open("r+b") as stream:
            header = stream.read(44)
            if len(header) != 44 or header[:4] != b"RIFF" or header[8:12] != b"WAVE" or header[36:40] != b"data":
                raise ValueError("existing L6 preview WAV header is unsupported")
            stream.seek(0, 2)
            stream.write(payload)
            total_bytes = stream.tell()
            stream.seek(4)
            stream.write(struct.pack("<I", total_bytes - 8))
            stream.seek(40)
            stream.write(struct.pack("<I", total_bytes - 44))
            stream.flush()

    @classmethod
    def _append_envelope(cls, stored: _StoredSpeaker, waveform: np.ndarray) -> None:
        value = np.asarray(waveform, dtype=np.float32)
        if len(value) % 320:
            raise ValueError("L6 preview audio must align to 20 ms")
        for start in range(0, len(value), cls._PCM_CHUNK_SAMPLES):
            chunk = value[start:start + cls._PCM_CHUNK_SAMPLES]
            peaks = np.max(np.abs(chunk.reshape(-1, 320)), axis=1)
            for raw_peak in peaks:
                bin_index = stored.envelope_hops // stored.envelope_bin_hops
                peak = float(raw_peak)
                if bin_index == len(stored.envelope_peaks):
                    stored.envelope_peaks.append(peak)
                elif bin_index == len(stored.envelope_peaks) - 1:
                    stored.envelope_peaks[bin_index] = max(
                        stored.envelope_peaks[bin_index], peak,
                    )
                else:
                    raise RuntimeError("L6 preview envelope append position regressed")
                stored.envelope_hops += 1
                if len(stored.envelope_peaks) > cls._MAX_WAVEFORM_BINS:
                    stored.envelope_peaks = [
                        max(stored.envelope_peaks[index:index + 2])
                        for index in range(0, len(stored.envelope_peaks), 2)
                    ]
                    stored.envelope_bin_hops *= 2

    def set_result(self, result: Layer6Result) -> None:
        if self._session_id is not None and self._session_id != result.session_id:
            self.clear()
        self._session_id = result.session_id
        if not result.outputs:
            for stored in self._stored.values():
                stored.path.unlink(missing_ok=True)
            self._paths.clear()
            self._stored.clear()
            self._snapshots = ()
            return
        metadata = dict(result.metadata)
        incremental_contract = (
            "incremental_changed_speaker_ids" in metadata
            and "incremental_append_only_speaker_ids" in metadata
        )
        changed = {
            int(value) for value in metadata.get("incremental_changed_speaker_ids", ())
        }
        append_only = {
            int(value) for value in metadata.get("incremental_append_only_speaker_ids", ())
        }
        snapshots = []
        updated: dict[int, _StoredSpeaker] = {}
        for output in result.outputs:
            timeline = np.asarray(output.waveform_16k, dtype=np.float32)
            path = Path(self._temporary.name) / f"l6_id_{output.speaker_id}.wav"
            stored = self._stored.get(output.speaker_id)
            can_append = (
                incremental_contract
                and output.speaker_id in append_only
                and output.speaker_id not in changed
                and stored is not None
                and len(timeline) >= stored.samples_16k
            )
            unchanged = (
                incremental_contract
                and output.speaker_id not in changed
                and output.speaker_id not in append_only
                and stored is not None
                and len(timeline) == stored.samples_16k
            )
            if can_append:
                tail = timeline[stored.samples_16k:]
                self._append_waveform(path, tail)
                self._append_envelope(stored, tail)
                stored.samples_16k = len(timeline)
            elif not unchanged:
                self._write_waveform(path, timeline)
                stored = _StoredSpeaker(path, samples_16k=len(timeline))
                self._append_envelope(stored, timeline)
            assert stored is not None
            updated[output.speaker_id] = stored
            self._paths[output.speaker_id] = path
            ids = ",".join(str(value) for value in output.source_track_ids)
            snapshots.append(TrackedAudioSnapshot(
                result.session_id, 0, output.speaker_id, "ended", 0.0,
                output.mean_quality, len(timeline) * 3,
                waveform_envelope=tuple(stored.envelope_peaks),
                display_label=(
                    f"声纹 {output.speaker_id} · {output.label} · "
                    f"关联音轨 {len(output.fragment_ids)} · "
                    f"来源L2 ID {ids} · MOS {output.mean_quality:.2f}"
                ),
                parent_track_id=output.source_track_ids[0],
            ))
        active_paths = {stored.path for stored in updated.values()}
        for speaker_id, stale in self._stored.items():
            if speaker_id not in updated and stale.path not in active_paths:
                stale.path.unlink(missing_ok=True)
                self._paths.pop(speaker_id, None)
        self._stored = updated
        self._snapshots = tuple(snapshots)

    def audio_path(self, speaker_id: int) -> Path | None:
        return self._paths.get(int(speaker_id))

    def snapshots(self) -> tuple[TrackedAudioSnapshot, ...]:
        return self._snapshots
