from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdaptiveRateSnapshot:
    period_ms: int
    stride: int
    last_overload_reason: str | None
    healthy_elapsed_ms: int


class AdaptiveRateController:
    """Step a costly 20 ms stage down while preserving a 20 ms output clock."""

    def __init__(
        self,
        *,
        base_period_ms: int = 20,
        maximum_period_ms: int = 200,
        overload_threshold_ms: float = 20.0,
        recovery_threshold_ms: float = 12.0,
        recovery_stable_ms: int = 5_000,
    ) -> None:
        if base_period_ms <= 0 or maximum_period_ms < base_period_ms:
            raise ValueError("adaptive periods must be positive and ordered")
        if maximum_period_ms % base_period_ms:
            raise ValueError("maximum adaptive period must be a multiple of the base period")
        if not 0.0 < recovery_threshold_ms < overload_threshold_ms:
            raise ValueError("adaptive recovery threshold must be below overload threshold")
        if recovery_stable_ms <= 0:
            raise ValueError("adaptive recovery duration must be positive")
        self.base_period_ms = int(base_period_ms)
        self.maximum_period_ms = int(maximum_period_ms)
        self.overload_threshold_ms = float(overload_threshold_ms)
        self.recovery_threshold_ms = float(recovery_threshold_ms)
        self.recovery_stable_ms = int(recovery_stable_ms)
        self.reset()

    def reset(self) -> None:
        self._period_ms = self.base_period_ms
        self._windows_until_compute = 0
        self._healthy_elapsed_ms = 0
        self._last_overload_reason: str | None = None

    @property
    def period_ms(self) -> int:
        return self._period_ms

    @property
    def stride(self) -> int:
        return self._period_ms // self.base_period_ms

    @property
    def snapshot(self) -> AdaptiveRateSnapshot:
        return AdaptiveRateSnapshot(
            self._period_ms,
            self.stride,
            self._last_overload_reason,
            self._healthy_elapsed_ms,
        )

    def should_compute(self, *, force: bool = False) -> bool:
        if force or self._windows_until_compute <= 0:
            self._windows_until_compute = self.stride - 1
            return True
        self._windows_until_compute -= 1
        return False

    def observe_compute(self, *, queue_wait_ms: float, stage_ms: dict[str, float]) -> None:
        queue_wait = max(0.0, float(queue_wait_ms))
        stage_metrics = {
            name: max(0.0, float(value)) for name, value in stage_ms.items()
        }
        metrics = {"queue_wait": queue_wait, **stage_metrics}
        # Queue age is always judged against the 20 ms output clock.  Compute
        # stages are judged against their current scheduled period: a 24 ms
        # calculation is overloaded at 20 ms, but sustainable at 40 ms and
        # must not cascade all the way to the 200 ms ceiling.
        overloaded = []
        if queue_wait > self.overload_threshold_ms:
            overloaded.append(("queue_wait", queue_wait))
        overloaded.extend(
            (name, value)
            for name, value in stage_metrics.items()
            if value > self._period_ms
        )
        if overloaded:
            name, value = max(overloaded, key=lambda item: item[1])
            self.force_overload(f"{name}:{value:.2f}ms")
            return

        next_period_ms = max(self.base_period_ms, self._period_ms - self.base_period_ms)
        recovery_headroom_ms = self.base_period_ms - self.recovery_threshold_ms
        recovery_limit_ms = max(
            self.recovery_threshold_ms,
            next_period_ms - recovery_headroom_ms,
        )
        if metrics and max(metrics.values()) <= recovery_limit_ms:
            self._healthy_elapsed_ms += self._period_ms
        else:
            self._healthy_elapsed_ms = 0
        if self._period_ms > self.base_period_ms and self._healthy_elapsed_ms >= self.recovery_stable_ms:
            self._period_ms -= self.base_period_ms
            self._healthy_elapsed_ms = 0
            self._last_overload_reason = None
            self._windows_until_compute = min(self._windows_until_compute, self.stride - 1)

    def force_overload(self, reason: str) -> None:
        if not reason:
            raise ValueError("adaptive overload reason cannot be empty")
        self._last_overload_reason = str(reason)
        self._healthy_elapsed_ms = 0
        self._period_ms = min(
            self.maximum_period_ms,
            self._period_ms + self.base_period_ms,
        )
        self._windows_until_compute = self.stride - 1
