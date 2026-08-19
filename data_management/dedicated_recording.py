from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import Path

import numpy as np

from common.data_types import IngestedAudioBlock

from .catalog import Catalog
from .contracts import RecordingMetadata
from .corpus_store import CorpusStore
from .manifests import utc_now
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


class DedicatedRecordingController:
    """Operator-controlled formal recording on the authoritative ingest timeline."""

    def __init__(self, data_root: str | Path, catalog: Catalog):
        self.data_root = Path(data_root)
        self.catalog = catalog
        self._lock = threading.RLock()
        self.phase = WizardPhase.IDLE
        self.input: WizardInput | None = None
        self._native: list[np.ndarray] = []
        self._physical: list[np.ndarray] = []
        self._session_id: str | None = None
        self._recording_id: str | None = None
        self._stream_epoch: int | None = None
        self._last_sample: int | None = None
        self._total_samples = 0
        self._recorded_intervals: list[dict[str, int]] = []
        self._resume_pending = False
        self._error_reason: str | None = None

    def _clear_capture(self) -> None:
        self._native.clear()
        self._physical.clear()
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
        self._resume_pending = True
        self.phase = WizardPhase.RECORDING
        return self.status()

    @_synchronized
    def append(self, block: IngestedAudioBlock) -> WizardStatus:
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

        # Formal ingest is logical 8ch; dedicated test assets remain the seven
        # independent physical microphones and must exclude HardwareMix.
        self._physical.append(block.samples[:, :7].copy())
        if block.native_samples is not None:
            self._native.append(block.native_samples.copy())
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
        physical = np.concatenate(self._physical)
        native = np.concatenate(self._native) if self._native and sum(map(len, self._native)) == len(physical) else None
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
        )
        store = CorpusStore(self.data_root, catalog=self.catalog)
        recording_id = str(uuid.uuid4())
        root = store._root(data.dataset_id, recording_id)
        payload = {name: getattr(metadata, name) for name in metadata.__slots__}
        store._rights(payload)
        self._recording_id = store._save(
            root,
            native,
            physical,
            payload,
            {
                "session_id": self._session_id,
                "recorded_intervals": tuple(dict(item) for item in self._recorded_intervals),
                "captured_sample_count": self._total_samples,
            },
        )
        self.phase = WizardPhase.COMPLETE
        return self._recording_id

    @_synchronized
    def abort(self, reason: str = "录制已中止，请重新开始") -> WizardStatus:
        if self.phase in {WizardPhase.RECORDING, WizardPhase.PAUSED, WizardPhase.FINALIZING, WizardPhase.ERROR}:
            self.phase = WizardPhase.ERROR
            self._error_reason = str(reason).strip() or "录制已中止，请重新开始"
            self._clear_capture()
        return self.status()

    @_synchronized
    def reset(self) -> WizardStatus:
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
