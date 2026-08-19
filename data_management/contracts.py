from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    config_hash: str
    calibration_hash: str
    geometry_version: str = "r6plus1_mic_face_ccw_v1"
    git_commit: str | None = None
    runtime: Mapping[str, Any] = field(default_factory=dict)
    algorithm_versions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecordingMetadata:
    dataset_id: str
    source_type: str
    capture_time_utc: str
    environment_id: str
    room_id: str
    array_pose_id: str
    source_count: int
    source_categories: tuple[str, ...]
    rights: Mapping[str, Any]
    known_theta_degrees: tuple[float, ...] | None = None
    distance_m: tuple[float, ...] | None = None
    speaker_ids_anonymous: tuple[str, ...] = ()
    language_tags: tuple[str, ...] = ()
    snr_db: float | None = None
    notes: str = ""
    display_name: str = ""
    source_movements: tuple[str, ...] = ()
    noise_source: str = ""


@dataclass(frozen=True, slots=True)
class Annotation:
    annotation_id: str
    recording_id: str
    start_sample: int
    end_sample: int
    label_type: str
    label: str
    theta_deg: float | None
    confidence: float
    annotator: str
    annotation_version: str

    def __post_init__(self) -> None:
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("标注结束时间必须大于开始时间")
        if not 0 <= self.confidence <= 1:
            raise ValueError("标注置信度必须位于0到1之间")


@dataclass(frozen=True, slots=True)
class ResultWatermark:
    session_id: str
    stream_epoch: int
    sample: int
    dropped_windows: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    doa_range: tuple[int, int]
    context_range: tuple[int, int]
    status: str
    candidates: tuple[Mapping[str, Any], ...] = ()
    detections: tuple[Mapping[str, Any], ...] = ()
    voice_direction_count: int = 0
    diagnostics: tuple[str, ...] = ()
    processing_latency_ms: float = 0.0
    raw_scores: np.ndarray | None = field(default=None, compare=False)
    normalized_scores: np.ndarray | None = field(default=None, compare=False)
    gate_decision: Mapping[str, Any] | None = field(default=None, compare=False)
    search_diagnostics: Mapping[str, Any] | None = field(default=None, compare=False)
    enhanced_audio: tuple[Mapping[str, Any], ...] = field(default=(), compare=False)
    enhanced_waveforms: tuple[np.ndarray, ...] = field(default=(), compare=False)
    l4_result: Mapping[str, Any] | None = field(default=None, compare=False)
    # v3 keeps the established payload while making staged-runtime terminal
    # states explicit.  Missing audio/detections no longer ambiguously mean
    # either "pending", "skipped" or "failed".
    stage_statuses: Mapping[str, str] = field(default_factory=dict, compare=False)
    stage_timings_ms: Mapping[str, float] = field(default_factory=dict, compare=False)
    stage_queue_wait_ms: Mapping[str, float] = field(default_factory=dict, compare=False)
    terminal_reason: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.status not in {"ok", "degraded", "error"}:
            raise ValueError("未知DecisionRecord状态")
        if min(self.stream_epoch, self.window_id, self.decision_sample) < 0:
            raise ValueError("DecisionRecord索引不能为负")
        for name in ("raw_scores", "normalized_scores"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=np.float32)
                if array.shape != (360,) or not np.isfinite(array).all():
                    raise ValueError(f"{name}必须是finite float32[360]")
                object.__setattr__(self, name, array.copy())
        if len(self.enhanced_audio) != len(self.enhanced_waveforms):
            raise ValueError("增强音频元数据与波形数量不一致")
        allowed_stage_states = {
            "completed", "skipped", "failed", "timed_out", "dropped", "cancelled",
        }
        statuses = {str(key): str(value) for key, value in self.stage_statuses.items()}
        if any(key not in {"l2", "l3", "l4"} for key in statuses):
            raise ValueError("DecisionRecord阶段名称无效")
        if any(value not in allowed_stage_states for value in statuses.values()):
            raise ValueError("DecisionRecord阶段状态无效")
        severe_states = {"failed", "timed_out", "dropped", "cancelled"}
        if statuses and any(value in severe_states for value in statuses.values()):
            if self.status != "error":
                raise ValueError("阶段失败、超时、丢弃或取消时DecisionRecord必须为error")
        if self.status in {"ok", "degraded"}:
            if len(self.candidates) != len(self.detections):
                raise ValueError("正式成功结果必须为每个候选提供一个同序检测")
            for candidate, detection in zip(self.candidates, self.detections, strict=True):
                candidate_theta = float(candidate["theta_deg"])
                detection_theta = float(detection["theta_deg"])
                if not np.isclose(candidate_theta, detection_theta, atol=1e-6, rtol=0.0):
                    raise ValueError("候选方向与检测方向必须同序对齐")
        detected_voice_count = sum(bool(item.get("is_voice", False)) for item in self.detections)
        if self.voice_direction_count != detected_voice_count:
            raise ValueError("voice_direction_count必须等于正式检测中的人声方向数")
        timings = {str(key): float(value) for key, value in self.stage_timings_ms.items()}
        waits = {str(key): float(value) for key, value in self.stage_queue_wait_ms.items()}
        if any(not np.isfinite(value) or value < 0.0 for value in (*timings.values(), *waits.values())):
            raise ValueError("DecisionRecord阶段耗时必须为有限非负值")
        waveforms = []
        for value in self.enhanced_waveforms:
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (15_360,) or not np.isfinite(array).all():
                raise ValueError("增强音频必须是finite float32[15360]")
            waveforms.append(array.copy())
        object.__setattr__(self, "enhanced_audio", tuple(dict(item) for item in self.enhanced_audio))
        object.__setattr__(self, "enhanced_waveforms", tuple(waveforms))
        object.__setattr__(self, "stage_statuses", statuses)
        object.__setattr__(self, "stage_timings_ms", timings)
        object.__setattr__(self, "stage_queue_wait_ms", waits)


def public_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    result: dict[str, Any] = {}
    for name in dir(value):
        if not name.startswith("_"):
            item = getattr(value, name)
            if not callable(item):
                result[name] = item
    return result
