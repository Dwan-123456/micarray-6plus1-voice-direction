from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from common.data_types import DecisionWindow


class SourceProbabilityState(str, Enum):
    READY = "ready"
    WARMING_UP = "warming_up"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class SourceProbability20ms:
    """One upstream array-source probability on the authoritative sample axis."""

    session_id: str
    stream_epoch: int
    start_sample: int
    end_sample: int
    probability: float | None
    state: SourceProbabilityState
    reason: str

    def __post_init__(self) -> None:
        if not self.session_id or min(self.stream_epoch, self.start_sample) < 0:
            raise ValueError("20 ms source probability identity is invalid")
        if self.end_sample - self.start_sample != 960:
            raise ValueError("source probability interval must contain exactly 960 samples")
        if not self.reason:
            raise ValueError("source probability reason cannot be empty")
        if self.state is SourceProbabilityState.READY:
            if self.probability is None or not np.isfinite(self.probability):
                raise ValueError("ready source probability must be finite")
            if not 0.0 <= self.probability <= 1.0:
                raise ValueError("source probability must be in [0,1]")
            object.__setattr__(self, "probability", float(self.probability))
        elif self.probability is not None:
            raise ValueError("non-ready source probability cannot publish a formal value")


class ProbabilityGateState(str, Enum):
    WARMING_UP = "warming_up"
    UNAVAILABLE = "unavailable"
    OPEN = "open"
    CLOSED = "closed"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ProbabilityGateDecision:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    backend: str
    state: ProbabilityGateState
    probability_previous_20ms: float | None
    probability_current_20ms: float | None
    probability_20ms: float | None
    threshold: float
    config_revision: int
    sound_present: bool
    reason: str
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.session_id or min(
            self.stream_epoch, self.window_id, self.decision_sample, self.config_revision
        ) < 0:
            raise ValueError("probability Gate identity is invalid")
        if not self.backend or not self.reason:
            raise ValueError("probability Gate backend and reason cannot be empty")
        if not np.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("probability Gate threshold must be finite and in [0,1]")
        values = (
            self.probability_previous_20ms,
            self.probability_current_20ms,
            self.probability_20ms,
        )
        if any(value is not None and (not np.isfinite(value) or not 0.0 <= value <= 1.0) for value in values):
            raise ValueError("probability Gate values must be finite and in [0,1]")
        formal = self.state in {ProbabilityGateState.OPEN, ProbabilityGateState.CLOSED}
        if formal != (
            self.probability_current_20ms is not None
            and self.probability_20ms is not None
        ):
            raise ValueError("only a formal Gate decision may publish current probability")
        if formal and self.probability_current_20ms != self.probability_20ms:
            raise ValueError("Gate probability must equal the current 20 ms probability")
        if self.state is ProbabilityGateState.OPEN and not self.sound_present:
            raise ValueError("open probability Gate must report sound_present=True")
        if self.state is not ProbabilityGateState.OPEN and self.sound_present:
            raise ValueError("non-open probability Gate must report sound_present=False")
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def allow_srp(self) -> bool:
        return self.state is ProbabilityGateState.OPEN


class ProbabilityGate:
    """Apply an inclusive threshold to the current aligned 20 ms probability."""

    backend = "current_20ms_v1"

    @staticmethod
    def _blocked(
        window: DecisionWindow,
        state: ProbabilityGateState,
        threshold: float,
        config_revision: int,
        reason: str,
        diagnostics: tuple[str, ...] = (),
    ) -> ProbabilityGateDecision:
        return ProbabilityGateDecision(
            window.session_id,
            window.stream_epoch,
            window.window_id,
            window.decision_sample,
            ProbabilityGate.backend,
            state,
            None,
            None,
            None,
            threshold,
            config_revision,
            False,
            reason,
            diagnostics,
        )

    def evaluate(
        self,
        window: DecisionWindow,
        probabilities: tuple[SourceProbability20ms, ...],
        *,
        threshold: float,
        config_revision: int,
    ) -> ProbabilityGateDecision:
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("probability Gate threshold must be finite and in [0,1]")
        if type(config_revision) is not int or config_revision < 0:
            raise ValueError("probability Gate config revision must be a non-negative int")
        probabilities = tuple(probabilities)
        if len(probabilities) not in {1, 2}:
            return self._blocked(
                window,
                ProbabilityGateState.UNAVAILABLE,
                threshold,
                config_revision,
                "requires_current_aligned_20ms_probability",
            )

        current_item = probabilities[-1]
        current_bounds = (window.doa_start_sample + 960, window.doa_end_sample)
        if (
            (current_item.session_id, current_item.stream_epoch)
            != (window.session_id, window.stream_epoch)
            or (current_item.start_sample, current_item.end_sample) != current_bounds
        ):
            return self._blocked(
                window,
                ProbabilityGateState.INVALID,
                threshold,
                config_revision,
                "current_probability_identity_or_interval_mismatch",
            )

        if current_item.state is SourceProbabilityState.WARMING_UP:
            return self._blocked(
                window,
                ProbabilityGateState.WARMING_UP,
                threshold,
                config_revision,
                "current_probability_warming_up",
                (f"current_reason={current_item.reason}",),
            )
        if current_item.state is SourceProbabilityState.INVALID:
            return self._blocked(
                window,
                ProbabilityGateState.INVALID,
                threshold,
                config_revision,
                "current_probability_invalid",
                (f"current_reason={current_item.reason}",),
            )
        if current_item.state is not SourceProbabilityState.READY:
            return self._blocked(
                window,
                ProbabilityGateState.UNAVAILABLE,
                threshold,
                config_revision,
                "current_probability_unavailable",
                (f"current_reason={current_item.reason}",),
            )

        previous = None
        if len(probabilities) == 2:
            previous_item = probabilities[0]
            previous_bounds = (window.doa_start_sample, window.doa_start_sample + 960)
            if (
                (previous_item.session_id, previous_item.stream_epoch)
                == (window.session_id, window.stream_epoch)
                and (previous_item.start_sample, previous_item.end_sample) == previous_bounds
                and previous_item.state is SourceProbabilityState.READY
            ):
                previous = previous_item.probability
        current = current_item.probability
        assert current is not None
        is_open = current >= threshold
        diagnostics = tuple(filter(None, (
            None if previous is None else f"probability_previous_20ms={previous:.6f}",
            f"probability_current_20ms={current:.6f}",
            f"probability_20ms={current:.6f}",
            f"gate_threshold={threshold:.6f}",
            f"gate_config_revision={config_revision}",
        )))
        return ProbabilityGateDecision(
            window.session_id,
            window.stream_epoch,
            window.window_id,
            window.decision_sample,
            self.backend,
            ProbabilityGateState.OPEN if is_open else ProbabilityGateState.CLOSED,
            previous,
            current,
            current,
            threshold,
            config_revision,
            is_open,
            "probability_at_or_above_threshold" if is_open else "probability_below_threshold",
            diagnostics,
        )
