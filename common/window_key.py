from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_types import DecisionWindow


@dataclass(frozen=True, slots=True)
class WindowKey:
    """Identity of one authoritative 20 ms decision on the ingest timeline."""

    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("WindowKey session_id cannot be empty")
        if min(self.stream_epoch, self.window_id, self.decision_sample) < 0:
            raise ValueError("WindowKey numeric fields cannot be negative")

    @classmethod
    def from_window(cls, window: DecisionWindow) -> WindowKey:
        return cls(window.session_id, window.stream_epoch, window.window_id, window.decision_sample)

    @property
    def stream_key(self) -> tuple[str, int]:
        return self.session_id, self.stream_epoch

    @property
    def timeline_order(self) -> tuple[str, int, int, int]:
        return self.session_id, self.stream_epoch, self.decision_sample, self.window_id
