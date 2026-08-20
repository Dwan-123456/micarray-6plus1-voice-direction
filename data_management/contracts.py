from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping

import numpy as np



@dataclass(frozen=True, slots=True)
class SessionMetadata:
    config_hash: str
    calibration_hash: str
    config_revision: int = 0
    calibration_revision: int = 0
    calibration_version: str = "unversioned"
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
    schema_version: str = "decision_record_v4"
    music_algorithm_version: str | None = None
    model_order: Mapping[str, Any] | None = field(default=None, compare=False)
    music_diagnostics: Mapping[str, Any] | None = field(default=None, compare=False)
    active_tracks: tuple[Mapping[str, Any], ...] = field(default=(), compare=False)
    kalman_applied: bool = False
    config_revision: int = 0
    config_hash: str = ""
    calibration_revision: int = 0
    calibration_version: str = "unversioned"
    calibration_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version not in {"decision_record_v3", "decision_record_v4"}:
            raise ValueError("DecisionRecord schema必须是v3或v4")
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
        if min(self.config_revision, self.calibration_revision) < 0:
            raise ValueError("配置和校准revision不能为负数")
        if type(self.kalman_applied) is not bool:
            raise TypeError("kalman_applied必须是bool")
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
        if self.schema_version == "decision_record_v4":
            self._validate_v4_track_alignment()
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
            if array.shape not in {(3_840,), (7_680,)} or not np.isfinite(array).all():
                raise ValueError("增强音频必须是finite float32[3840或7680]")
            waveforms.append(array.copy())
        object.__setattr__(self, "enhanced_audio", tuple(dict(item) for item in self.enhanced_audio))
        object.__setattr__(self, "candidates", tuple(dict(item) for item in self.candidates))
        object.__setattr__(self, "detections", tuple(dict(item) for item in self.detections))
        object.__setattr__(self, "active_tracks", tuple(dict(item) for item in self.active_tracks))
        object.__setattr__(self, "enhanced_waveforms", tuple(waveforms))
        if self.model_order is not None:
            object.__setattr__(self, "model_order", dict(self.model_order))
        if self.music_diagnostics is not None:
            object.__setattr__(self, "music_diagnostics", dict(self.music_diagnostics))
        object.__setattr__(self, "stage_statuses", statuses)
        object.__setattr__(self, "stage_timings_ms", timings)
        object.__setattr__(self, "stage_queue_wait_ms", waits)

    @staticmethod
    def _track_ids(items: tuple[Mapping[str, Any], ...], name: str) -> tuple[int, ...]:
        if not items:
            return ()
        values = tuple(item.get("track_id") for item in items)
        # During branch integration an empty ID set is allowed only as an
        # explicit no-ID record.  Once one item carries an ID, the whole
        # aligned batch must carry unique positive IDs.
        if all(value is None for value in values):
            return ()
        if any(type(value) is not int or int(value) <= 0 for value in values):
            raise ValueError(f"{name}必须全部包含正整数track_id")
        result = tuple(int(value) for value in values)
        if len(set(result)) != len(result):
            raise ValueError(f"{name}的track_id不能重复")
        return result

    def _validate_v4_track_alignment(self) -> None:
        candidate_ids = self._track_ids(self.candidates, "L2候选")
        enhanced_ids = self._track_ids(self.enhanced_audio, "L3增强音频")
        detection_ids = self._track_ids(self.detections, "L4检测")
        active_ids = self._track_ids(self.active_tracks, "active_tracks")
        if candidate_ids:
            if enhanced_ids and enhanced_ids != candidate_ids:
                raise ValueError("L3 track_id必须与L2同序对齐")
            if detection_ids and detection_ids != candidate_ids:
                raise ValueError("L4 track_id必须与L2同序对齐")
        elif enhanced_ids or detection_ids:
            raise ValueError("L3/L4存在track_id时L2候选也必须包含track_id")
        if active_ids and candidate_ids and not set(candidate_ids).issubset(active_ids):
            raise ValueError("本窗L2方向必须包含在active_tracks中")
        for item in (*self.candidates, *self.active_tracks):
            theta = item.get("theta_deg")
            measured = item.get("measured_theta_deg")
            if theta is not None and (not np.isfinite(theta) or not 0 <= float(theta) < 360):
                raise ValueError("输出角必须位于[0,360)")
            if measured is not None and (not np.isfinite(measured) or not 0 <= float(measured) < 360):
                raise ValueError("观测角必须位于[0,360)")


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
