from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
import wave
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from common.data_types import IngestedAudioBlock
from layer1_input.pcm import pcm16_bytes

from .catalog import Catalog
from .contracts import RecordingMetadata
from .corpus_store import CorpusStore
from .manifests import sha256_file, utc_now
from .wizard import WizardInput, validate_wizard


def _synchronized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class WizardPhase(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class WizardStatus:
    phase: WizardPhase
    sample_count: int
    required_samples: int
    message: str
    recording_id: str | None = None

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / 48_000


class _RawInputWriter:
    """Stream the exact host-side microphone inputs without retaining audio in RAM."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        self.audio_partial = root / "native_8ch.wav.partial"
        self.audio_path = root / "native_8ch.wav"
        self.hotmap_partial = root / "hotmaps.jsonl.partial"
        self.hotmap_path = root / "hotmaps.jsonl"
        self._audio: wave.Wave_write | None = None
        self._hotmaps: TextIO | None = None
        self._hotmap_sequences: set[tuple[int, int]] = set()
        self.sample_count = 0
        self.hotmap_count = 0
        self.closed = False

    def _open_audio(self) -> wave.Wave_write:
        if self._audio is None:
            writer = wave.open(str(self.audio_partial), "wb")
            writer.setnchannels(8)
            writer.setsampwidth(2)
            writer.setframerate(48_000)
            self._audio = writer
        return self._audio

    def _write_hotmap(self, block: IngestedAudioBlock, playback_sample: int, hotmap: object) -> None:
        if hotmap is None:
            return
        sequence_id = int(getattr(hotmap, "sequence_id", block.sequence_id))
        key = (int(block.stream_epoch), sequence_id)
        if key in self._hotmap_sequences:
            return
        matrix = np.asarray(getattr(hotmap, "matrix", hotmap))
        if matrix.shape != (16, 16):
            raise ValueError("热力图必须是16×16")
        if self._hotmaps is None:
            self._hotmaps = self.hotmap_partial.open("w", encoding="utf-8", newline="\n")
        self._hotmaps.write(
            json.dumps(
                {
                    "schema_version": "recorded_hotmap_v1",
                    "sequence_id": sequence_id,
                    "timestamp": float(getattr(hotmap, "timestamp", block.timestamp)),
                    "received_at": getattr(hotmap, "received_at", None),
                    "source_stream_epoch": int(block.stream_epoch),
                    "source_start_sample": int(block.start_sample),
                    "playback_sample": max(0, int(playback_sample)),
                    "matrix": np.asarray(matrix, dtype=np.uint8).tolist(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self._hotmap_sequences.add(key)
        self.hotmap_count += 1

    def append(self, block: IngestedAudioBlock, hotmap_frames: tuple[object, ...] = ()) -> None:
        if self.closed:
            raise RuntimeError("录音写入器已经关闭")
        native = block.native_samples
        if native is None or native.shape[1] != 8:
            raise ValueError("专用测试录音要求设备原始8通道输入")
        playback_sample = self.sample_count
        self._open_audio().writeframesraw(pcm16_bytes(native))
        frames = hotmap_frames or (() if block.hotmap is None else (block.hotmap,))
        for hotmap in frames:
            timestamp = float(getattr(hotmap, "timestamp", block.timestamp))
            aligned_sample = playback_sample + round((timestamp - block.timestamp) * 48_000)
            aligned_sample = min(playback_sample + len(native) - 1, max(playback_sample, aligned_sample))
            self._write_hotmap(block, aligned_sample, hotmap)
        self.sample_count += len(native)

    @staticmethod
    def _sync(path: Path) -> None:
        with path.open("rb+") as source:
            os.fsync(source.fileno())

    def close(self) -> None:
        if self.closed:
            return
        if self._audio is not None:
            self._audio.close()
            self._audio = None
            self._sync(self.audio_partial)
        if self._hotmaps is not None:
            self._hotmaps.flush()
            os.fsync(self._hotmaps.fileno())
            self._hotmaps.close()
            self._hotmaps = None
        self.closed = True

    def finalize(self) -> list[dict[str, Any]]:
        self.close()
        if self.sample_count <= 0 or not self.audio_partial.is_file():
            raise RuntimeError("没有可保存的8通道音频")
        self.audio_partial.replace(self.audio_path)
        assets: list[dict[str, Any]] = [
            {
                "kind": "native_8ch",
                "path": self.audio_path.name,
                "sha256": sha256_file(self.audio_path),
                "sample_count": self.sample_count,
                "channel_count": 8,
                "sample_rate": 48_000,
                "dtype": "int16",
            }
        ]
        if self.hotmap_partial.is_file():
            self.hotmap_partial.replace(self.hotmap_path)
            assets.append(
                {
                    "kind": "cdc_hotmaps",
                    "path": self.hotmap_path.name,
                    "sha256": sha256_file(self.hotmap_path),
                    "frame_count": self.hotmap_count,
                }
            )
        return assets

    def quarantine(self, data_root: Path) -> Path | None:
        self.close()
        if not self.root.exists():
            return None
        target = data_root / "quarantine" / "dedicated_recordings" / self.root.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target = target.with_name(f"{target.name}-{uuid.uuid4().hex[:8]}")
        shutil.move(str(self.root), str(target))
        return target


class DedicatedRecordingController:
    """Operator-controlled formal recording on the authoritative ingest timeline."""

    def __init__(self, data_root: str | Path, catalog: Catalog):
        self.data_root = Path(data_root)
        self.catalog = catalog
        self._lock = threading.RLock()
        self.phase = WizardPhase.IDLE
        self.input: WizardInput | None = None
        self._writer: _RawInputWriter | None = None
        self._recording_root: Path | None = None
        self._session_id: str | None = None
        self._recording_id: str | None = None
        self._stream_epoch: int | None = None
        self._last_sample: int | None = None
        self._total_samples = 0
        self._recorded_intervals: list[dict[str, int]] = []
        self._resume_pending = False
        self._error_reason: str | None = None

    def _clear_capture(self) -> None:
        self._writer = None
        self._recording_root = None
        self._session_id = None
        self._recording_id = None
        self._stream_epoch = None
        self._last_sample = None
        self._total_samples = 0
        self._recorded_intervals.clear()
        self._resume_pending = False

    @_synchronized
    def begin(self, data: WizardInput) -> WizardStatus:
        errors = validate_wizard(data)
        if errors:
            raise ValueError("；".join(errors))
        if self.phase in {WizardPhase.RECORDING, WizardPhase.PAUSED, WizardPhase.FINALIZING}:
            raise RuntimeError("当前测试录音尚未结束")
        self.input = data
        self._error_reason = None
        self._clear_capture()
        recording_id = str(uuid.uuid4())
        self._recording_root = (
            self.data_root / "test_corpus" / data.dataset_id / "recordings" / recording_id
        )
        self._writer = _RawInputWriter(self._recording_root)
        self._resume_pending = True
        self.phase = WizardPhase.RECORDING
        return self.status()

    @_synchronized
    def append(self, block: IngestedAudioBlock, hotmap_frames: tuple[object, ...] = ()) -> WizardStatus:
        if self.phase != WizardPhase.RECORDING:
            return self.status()
        if self._session_id is None:
            self._session_id = block.session_id
            self._stream_epoch = block.stream_epoch
        if block.session_id != self._session_id or block.stream_epoch != self._stream_epoch:
            self.phase = WizardPhase.ERROR
            self._error_reason = "录制期间采集会话发生变化，请重新开始"
            raise ValueError(self._error_reason)
        if self._last_sample is not None and block.start_sample != self._last_sample and not self._resume_pending:
            self.phase = WizardPhase.ERROR
            self._error_reason = "录制期间音频时间轴不连续，请检查设备后重新开始"
            raise ValueError(self._error_reason)

        if self._writer is None:
            raise RuntimeError("录音写入器不可用")
        self._writer.append(block, hotmap_frames)
        if self._resume_pending or not self._recorded_intervals:
            self._recorded_intervals.append(
                {"stream_epoch": block.stream_epoch, "start_sample": block.start_sample, "end_sample": block.end_sample}
            )
        else:
            self._recorded_intervals[-1]["end_sample"] = block.end_sample
        self._resume_pending = False
        self._last_sample = block.end_sample
        self._total_samples += len(block.samples)
        return self.status()

    @_synchronized
    def pause(self) -> WizardStatus:
        if self.phase != WizardPhase.RECORDING:
            raise RuntimeError("当前状态不能暂停")
        self.phase = WizardPhase.PAUSED
        return self.status()

    @_synchronized
    def resume(self) -> WizardStatus:
        if self.phase != WizardPhase.PAUSED:
            raise RuntimeError("当前状态不能继续录制")
        self._resume_pending = True
        self.phase = WizardPhase.RECORDING
        return self.status()

    @_synchronized
    def finish(self) -> WizardStatus:
        if self.phase not in {WizardPhase.RECORDING, WizardPhase.PAUSED}:
            raise RuntimeError("当前没有可结束的测试录音")
        if self._total_samples <= 0:
            raise RuntimeError("尚未录到有效音频，不能保存")
        self.phase = WizardPhase.FINALIZING
        return self.status()

    def finish_target(self) -> WizardStatus:
        """Compatibility alias for callers from the former guided flow."""
        return self.finish()

    @_synchronized
    def finalize(self) -> str:
        if self.phase != WizardPhase.FINALIZING or self.input is None:
            raise RuntimeError("请先结束录音，再进行保存")
        if self._writer is None or self._recording_root is None:
            raise RuntimeError("录音写入器不可用")
        assets = self._writer.finalize()
        data = self.input
        metadata = RecordingMetadata(
            dataset_id=data.dataset_id,
            source_type="dedicated",
            capture_time_utc=utc_now(),
            environment_id=data.environment_id,
            room_id=data.room_id,
            array_pose_id=data.array_pose_id,
            source_count=data.source_count,
            source_categories=data.source_categories,
            rights={
                "consent_status": data.consent_status,
                "license_id": data.license_id,
                "allowed_uses": list(data.allowed_uses),
                "expires_at_utc": data.expires_at_utc,
            },
            known_theta_degrees=data.theta_degrees or None,
            distance_m=data.distance_m or None,
            speaker_ids_anonymous=data.speaker_ids,
            language_tags=data.language_tags,
            notes=data.notes,
            display_name=data.recording_name.strip(),
        )
        store = CorpusStore(self.data_root, catalog=self.catalog)
        self._recording_id = store.register_raw_recording(
            self._recording_root,
            metadata,
            lineage={
                "session_id": self._session_id,
                "recorded_intervals": tuple(dict(item) for item in self._recorded_intervals),
                "captured_sample_count": self._total_samples,
            },
            assets=assets,
            duration_samples=self._total_samples,
        )
        self.phase = WizardPhase.COMPLETE
        return self._recording_id

    @_synchronized
    def abort(self, reason: str = "录制已中止，请重新开始") -> WizardStatus:
        if self.phase in {WizardPhase.RECORDING, WizardPhase.PAUSED, WizardPhase.FINALIZING, WizardPhase.ERROR}:
            if self._writer is not None:
                self._writer.quarantine(self.data_root)
            self.phase = WizardPhase.ERROR
            self._error_reason = str(reason).strip() or "录制已中止，请重新开始"
            self._clear_capture()
        return self.status()

    @_synchronized
    def reset(self) -> WizardStatus:
        if self._writer is not None and self.phase in {
            WizardPhase.RECORDING,
            WizardPhase.PAUSED,
            WizardPhase.FINALIZING,
            WizardPhase.ERROR,
        }:
            self._writer.quarantine(self.data_root)
        self.phase = WizardPhase.IDLE
        self.input = None
        self._error_reason = None
        self._clear_capture()
        return self.status()

    @_synchronized
    def status(self) -> WizardStatus:
        duration = self._total_samples / 48_000
        messages = {
            WizardPhase.IDLE: "等待填写信息并开始录音",
            WizardPhase.RECORDING: f"正在录音 · 已录 {duration:.1f} 秒",
            WizardPhase.PAUSED: f"录音已暂停 · 已录 {duration:.1f} 秒",
            WizardPhase.FINALIZING: f"录音结束 · 正在保存 {duration:.1f} 秒音频并自动检查",
            WizardPhase.COMPLETE: f"保存和自动质量检查完成 · 共 {duration:.1f} 秒",
            WizardPhase.ERROR: self._error_reason or "录制失败，请重新开始",
        }
        return WizardStatus(self.phase, self._total_samples, 0, messages[self.phase], self._recording_id)
