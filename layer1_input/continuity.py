from __future__ import annotations

from dataclasses import dataclass

from .interface import DecodedAudio, InputHealthEvent


@dataclass(frozen=True, slots=True)
class ContinuityDecision:
    reset: bool
    reason: str | None = None


def continuity_decision(
    previous: DecodedAudio | None,
    current: DecodedAudio,
    *,
    has_health_event: bool = False,
    timestamp_tolerance_ms: float = 5.0,
) -> ContinuityDecision:
    if has_health_event:
        return ContinuityDecision(True, "health_event")
    if previous is None:
        return ContinuityDecision(False)
    if current.sample_rate != previous.sample_rate:
        return ContinuityDecision(True, "sample_rate_change")
    if current.sequence_id != previous.sequence_id + 1:
        return ContinuityDecision(True, "sequence_gap")
    expected = previous.timestamp + previous.frame_count / previous.sample_rate
    if abs(current.timestamp - expected) > timestamp_tolerance_ms / 1000.0:
        return ContinuityDecision(True, "timestamp_gap")
    return ContinuityDecision(False)


class CalibrationContinuityGuard:
    """Resets only Layer-1 calibration state; never assigns timeline IDs."""

    def __init__(self, calibrator: object, *, timestamp_tolerance_ms: float = 5.0):
        self.calibrator = calibrator
        self.timestamp_tolerance_ms = timestamp_tolerance_ms
        self._previous: DecodedAudio | None = None
        self._pending_health = False

    def start(self) -> None:
        self.calibrator.reset()
        self._previous = None
        self._pending_health = False

    def notify(self, _event: InputHealthEvent) -> None:
        self._pending_health = True

    def process(self, frame: DecodedAudio) -> DecodedAudio:
        decision = continuity_decision(
            self._previous, frame,
            has_health_event=self._pending_health,
            timestamp_tolerance_ms=self.timestamp_tolerance_ms,
        )
        if decision.reset:
            self.calibrator.reset()
        output = self.calibrator.process(frame)
        self._previous = frame
        self._pending_health = False
        return output
