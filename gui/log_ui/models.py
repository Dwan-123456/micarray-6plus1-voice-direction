from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


class Availability(StrEnum):
    AVAILABLE = "available"
    NOT_PROVIDED = "interface_not_provided"
    NOT_RECORDED = "not_recorded"
    NOT_SEALED = "not_sealed"
    VALIDATION_FAILED = "validation_failed"
    UNSUPPORTED_SCHEMA = "unsupported_schema"


class StageState(StrEnum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    DROPPED = "dropped"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @classmethod
    def from_public(cls, value: object) -> StageState:
        try:
            return cls(str(value).casefold())
        except ValueError:
            return cls.UNKNOWN


@dataclass(frozen=True, order=True, slots=True)
class WindowKey:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    sessions: bool
    decisions: bool
    tracks: bool
    track_timeline: bool
    track_audio: bool
    session_audio: bool
    runtime_status: bool = False

    @property
    def offline_review(self) -> bool:
        return self.sessions and self.decisions

    def labels(self) -> tuple[str, ...]:
        names = (
            ("sessions", self.sessions),
            ("decisions", self.decisions),
            ("tracks", self.tracks),
            ("track_timeline", self.track_timeline),
            ("track_audio", self.track_audio),
            ("session_audio", self.session_audio),
            ("runtime_status", self.runtime_status),
        )
        return tuple(name for name, enabled in names if enabled)


@dataclass(frozen=True, slots=True)
class StageObservation:
    state: StageState = StageState.UNKNOWN
    compute_ms: float | None = None
    queue_wait_ms: float | None = None
    end_to_end_ms: float | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class TrackObservation:
    session_id: str
    stream_epoch: int
    track_id: int
    window_id: int
    decision_sample: int
    measured_theta_deg: float | None
    theta_deg: float | None
    state: str | None
    is_observed: bool | None
    is_new_track: bool | None
    l5_probability: float | None
    is_voice: bool | None
    enhanced_asset: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class WindowObservation:
    key: WindowKey
    schema_version: str
    sample_range: tuple[int, int] | None
    stages: Mapping[str, StageObservation]
    candidates: tuple[Mapping[str, Any], ...]
    active_tracks: tuple[Mapping[str, Any], ...]
    detections: tuple[Mapping[str, Any], ...]
    enhanced_assets: tuple[Mapping[str, Any], ...]
    gate: Mapping[str, Any] | None
    model_order: Mapping[str, Any] | None
    music_diagnostics: Mapping[str, Any] | None
    normalized_scores: tuple[float, ...] | None
    terminal_reason: str | None
    raw_public: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Anomaly:
    category: str
    message: str
    key: WindowKey | None = None
    track_key: tuple[str, int, int] | None = None


@dataclass(frozen=True, slots=True)
class SessionDescriptor:
    session_id: str
    status: str
    schema_version: str
    started_at: str | None
    ended_at: str | None
    mode: str | None
    sample_rate: int | None
    duration_seconds: float | None
    project_version: str | None
    algorithm_version: str | None
    config_hash: str | None
    calibration_hash: str | None
    data_integrity: str | None
    capabilities: CapabilitySet
    availability: Availability = Availability.AVAILABLE
    warning: str | None = None
    raw_public: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class SessionReadModel:
    descriptor: SessionDescriptor
    windows: tuple[WindowObservation, ...]
    tracks: tuple[TrackObservation, ...]
    anomalies: tuple[Anomaly, ...]
    decision_availability: Availability = Availability.AVAILABLE

    def window(self, key: WindowKey) -> WindowObservation | None:
        return next((item for item in self.windows if item.key == key), None)
