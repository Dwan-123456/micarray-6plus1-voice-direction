from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from common.data_types import DecisionWindow
from common.window_key import WindowKey

if TYPE_CHECKING:
    from layer2_source_detection.pipeline import Layer2PipelineResult
    from layer3_direction_signal.interface import Layer3Output
    from layer4_voice_classifier.contracts import Layer4Result
else:
    Layer2PipelineResult = object
    Layer3Output = object
    Layer4Result = object


def _freeze_config_value(value: object) -> object:
    """Recursively detach the small, JSON-like runtime configuration snapshot."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_config_value(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_config_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_config_value(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    raise TypeError(
        "processing config snapshots may contain only mappings, sequences, sets, bytes, and scalar values"
    )


@dataclass(frozen=True, slots=True)
class ProcessingConfigSnapshot:
    """Immutable configuration captured when a window enters the processing graph."""

    revision: int
    config_hash: str
    geometry_version: str
    audio_mode: str
    values: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("config revision cannot be negative")
        if not self.config_hash or not self.geometry_version or not self.audio_mode:
            raise ValueError("config hash, geometry version, and audio mode cannot be empty")
        if not isinstance(self.values, Mapping):
            raise TypeError("config snapshot values must be a mapping")
        object.__setattr__(self, "values", _freeze_config_value(self.values))


@dataclass(frozen=True, slots=True)
class WindowWorkItem:
    """One immutable window and the exact settings that must process it."""

    key: WindowKey
    window: DecisionWindow
    config: ProcessingConfigSnapshot
    accepted_monotonic_ns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.key, WindowKey) or not isinstance(self.window, DecisionWindow):
            raise TypeError("WindowWorkItem requires a WindowKey and DecisionWindow")
        if self.key != WindowKey.from_window(self.window):
            raise ValueError("WindowWorkItem key must exactly match its DecisionWindow")
        if not isinstance(self.config, ProcessingConfigSnapshot):
            raise TypeError("WindowWorkItem config must be a ProcessingConfigSnapshot")
        if self.accepted_monotonic_ns < 0:
            raise ValueError("accepted_monotonic_ns cannot be negative")


class StageState(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    DROPPED = "dropped"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self is not StageState.PENDING


StageOutputT = TypeVar("StageOutputT")


def _identity_from_object(value: object) -> WindowKey | None:
    names = ("session_id", "stream_epoch", "window_id", "decision_sample")
    if all(hasattr(value, name) for name in names):
        try:
            return WindowKey(*(getattr(value, name) for name in names))
        except (TypeError, ValueError):
            return None
    return None


def _payload_identities(value: object) -> tuple[WindowKey, ...]:
    direct = _identity_from_object(value)
    if direct is not None:
        return (direct,)
    identities: list[WindowKey] = []
    for name in ("gate_decision", "spatial_response"):
        child = getattr(value, name, None)
        identity = _identity_from_object(child) if child is not None else None
        if identity is not None:
            identities.append(identity)
    for name in ("candidates", "directions", "active_tracks", "enhanced_audio", "detections"):
        for child in getattr(value, name, ()):
            identity = _identity_from_object(child)
            if identity is not None:
                identities.append(identity)
    return tuple(identities)


@dataclass(frozen=True, slots=True)
class StageResult(Generic[StageOutputT]):
    """Terminal or pending state for one processing stage and one exact window."""

    stage_name: ClassVar[str] = "stage"

    key: WindowKey
    state: StageState
    output: StageOutputT | None = None
    started_monotonic_ns: int = 0
    finished_monotonic_ns: int = 0
    reason: str | None = None
    error: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, WindowKey):
            raise TypeError("stage result key must be a WindowKey")
        try:
            state = StageState(self.state)
        except ValueError as exc:
            raise ValueError(f"invalid stage state: {self.state}") from exc
        object.__setattr__(self, "state", state)
        if min(self.started_monotonic_ns, self.finished_monotonic_ns) < 0:
            raise ValueError("stage timestamps cannot be negative")
        if state is StageState.PENDING:
            if self.output is not None or self.finished_monotonic_ns or self.reason or self.error:
                raise ValueError("pending stage cannot contain output or terminal metadata")
        else:
            if self.finished_monotonic_ns < self.started_monotonic_ns:
                raise ValueError("finished_monotonic_ns cannot precede started_monotonic_ns")
            if state is StageState.COMPLETED:
                if self.output is None or self.error is not None:
                    raise ValueError("completed stage requires output and cannot contain an error")
            elif self.output is not None:
                raise ValueError("non-completed terminal stage cannot contain output")
            if state is StageState.FAILED:
                if not self.error:
                    raise ValueError("failed stage requires an error")
            elif self.error is not None:
                raise ValueError("only a failed stage may contain an error")
            if state in {
                StageState.SKIPPED,
                StageState.TIMED_OUT,
                StageState.DROPPED,
                StageState.CANCELLED,
            } and not self.reason:
                raise ValueError(f"{state.value} stage requires a reason")
        identities = () if self.output is None else _payload_identities(self.output)
        if any(identity != self.key for identity in identities):
            raise ValueError("stage output identity does not match its WindowKey")
        object.__setattr__(self, "diagnostics", tuple(str(item) for item in self.diagnostics))

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    @property
    def duration_ms(self) -> float | None:
        if not self.is_terminal:
            return None
        return (self.finished_monotonic_ns - self.started_monotonic_ns) / 1_000_000.0

    @classmethod
    def pending(cls, key: WindowKey, *, started_monotonic_ns: int = 0) -> StageResult[StageOutputT]:
        return cls(key=key, state=StageState.PENDING, started_monotonic_ns=started_monotonic_ns)

    @classmethod
    def completed(
        cls,
        key: WindowKey,
        output: StageOutputT,
        *,
        started_monotonic_ns: int = 0,
        finished_monotonic_ns: int | None = None,
        diagnostics: tuple[str, ...] = (),
    ) -> StageResult[StageOutputT]:
        return cls(
            key=key,
            state=StageState.COMPLETED,
            output=output,
            started_monotonic_ns=started_monotonic_ns,
            finished_monotonic_ns=time.monotonic_ns() if finished_monotonic_ns is None else finished_monotonic_ns,
            diagnostics=diagnostics,
        )

    @classmethod
    def terminal(
        cls,
        key: WindowKey,
        state: StageState,
        reason: str,
        *,
        started_monotonic_ns: int = 0,
        finished_monotonic_ns: int | None = None,
        error: str | None = None,
        diagnostics: tuple[str, ...] = (),
    ) -> StageResult[StageOutputT]:
        if state in {StageState.PENDING, StageState.COMPLETED}:
            raise ValueError("terminal() requires a non-completed terminal state")
        return cls(
            key=key,
            state=state,
            started_monotonic_ns=started_monotonic_ns,
            finished_monotonic_ns=time.monotonic_ns() if finished_monotonic_ns is None else finished_monotonic_ns,
            reason=reason,
            error=error,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True, slots=True)
class L2StageResult(StageResult[Layer2PipelineResult]):
    stage_name: ClassVar[str] = "l2"


@dataclass(frozen=True, slots=True)
class L3StageResult(StageResult[Layer3Output]):
    stage_name: ClassVar[str] = "l3"


@dataclass(frozen=True, slots=True)
class L4StageResult(StageResult[Layer4Result]):
    stage_name: ClassVar[str] = "l4"


TerminalStageResult = L2StageResult | L3StageResult | L4StageResult


def _joined_state(results: tuple[TerminalStageResult, ...]) -> StageState:
    states = {item.state for item in results}
    for state in (
        StageState.DROPPED,
        StageState.CANCELLED,
        StageState.TIMED_OUT,
        StageState.FAILED,
    ):
        if state in states:
            return state
    return StageState.COMPLETED


@dataclass(frozen=True, slots=True)
class JoinedWindowResult:
    """All terminal stage states for exactly one authoritative window."""

    work_item: WindowWorkItem
    l2: L2StageResult
    l3: L3StageResult
    l4: L4StageResult
    terminal_reason: str
    completed_monotonic_ns: int
    state: StageState = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.work_item, WindowWorkItem):
            raise TypeError("joined result requires a WindowWorkItem")
        results: tuple[TerminalStageResult, ...] = (self.l2, self.l3, self.l4)
        if any(not item.is_terminal for item in results):
            raise ValueError("joined result requires terminal L2, L3, and L4 results")
        if any(item.key != self.work_item.key for item in results):
            raise ValueError("all joined stages must belong to the same WindowKey")
        if self.completed_monotonic_ns < 0:
            raise ValueError("completed_monotonic_ns cannot be negative")
        state = _joined_state(results)
        if not self.terminal_reason:
            reason = next(
                (item.error or item.reason for item in results if item.error or item.reason),
                "pipeline_completed",
            )
            object.__setattr__(self, "terminal_reason", reason)
        object.__setattr__(self, "state", state)

    @property
    def key(self) -> WindowKey:
        return self.work_item.key
