from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import monotonic


@dataclass(frozen=True, slots=True)
class SourceCountSnapshot:
    """Latest-only count result; intentionally exposes no direction or score."""

    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    source_count: int | None
    published_monotonic: float

    def __post_init__(self) -> None:
        if not self.session_id or min(self.stream_epoch, self.window_id, self.decision_sample) < 0:
            raise ValueError("source-count snapshot identity is invalid")
        if self.source_count is not None and (
            type(self.source_count) is not int or self.source_count not in {0, 1, 2}
        ):
            raise ValueError("source count must be 0, 1, 2, or None while warming")
        if not isfinite(self.published_monotonic) or self.published_monotonic < 0:
            raise ValueError("source-count publication time must be finite and non-negative")

    @property
    def age_ms(self) -> float:
        return max(0.0, (monotonic() - self.published_monotonic) * 1_000.0)
