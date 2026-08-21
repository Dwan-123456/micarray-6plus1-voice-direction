from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .models import Availability, SessionReadModel, StageState


@dataclass(frozen=True, slots=True)
class Percentiles:
    p50: float | None
    p95: float | None
    p99: float | None
    n: int
    missing: int

    @property
    def missing_rate(self) -> float:
        total = self.n + self.missing
        return self.missing / total if total else 1.0


@dataclass(frozen=True, slots=True)
class StageStatistics:
    counts: Mapping[StageState, int]
    applicable: int
    missing: int
    completed_hz: float | None
    duration_seconds: float | None
    compute: Percentiles
    queue_wait: Percentiles
    end_to_end: Percentiles


@dataclass(frozen=True, slots=True)
class SessionStatistics:
    total_windows: int
    epochs: int
    stage: Mapping[str, StageStatistics]
    track_count: int | None
    voice_count: int | None
    non_voice_count: int | None
    anomaly_count: int
    availability: Availability


class StatisticsEngine:
    """Pure statistics over public normalized records; never fills missing data."""

    def calculate(self, session: SessionReadModel) -> SessionStatistics:
        windows = session.windows
        duration = self._duration_seconds(session)
        stage_names = tuple(windows[0].stages) if windows else ("l1", "gate", "l2", "l3", "l5", "commit")
        stages: dict[str, StageStatistics] = {}
        for name in stage_names:
            observations = [item.stages[name] for item in windows]
            known = [item for item in observations if item.state != StageState.UNKNOWN]
            counts = {state: sum(item.state == state for item in known) for state in StageState if state != StageState.UNKNOWN}
            completed = counts.get(StageState.COMPLETED, 0)
            stages[name] = StageStatistics(
                counts=counts,
                applicable=len(known),
                missing=len(observations) - len(known),
                completed_hz=(completed / duration if duration is not None else None),
                duration_seconds=duration,
                compute=self._percentiles([item.compute_ms for item in observations]),
                queue_wait=self._percentiles([item.queue_wait_ms for item in observations]),
                end_to_end=self._percentiles([item.end_to_end_ms for item in observations]),
            )
        track_keys = {(item.session_id, item.stream_epoch, item.track_id) for item in session.tracks}
        decisions = [item.is_voice for item in session.tracks if item.is_voice is not None]
        track_count = len(track_keys) if session.decision_availability == Availability.AVAILABLE else None
        return SessionStatistics(
            total_windows=len(windows),
            epochs=len({item.key.stream_epoch for item in windows}),
            stage=stages,
            track_count=track_count,
            voice_count=sum(decisions) if decisions else None,
            non_voice_count=sum(not item for item in decisions) if decisions else None,
            anomaly_count=len(session.anomalies),
            availability=session.decision_availability,
        )

    @staticmethod
    def _duration_seconds(session: SessionReadModel) -> float | None:
        sample_rate = session.descriptor.sample_rate
        if sample_rate is None or sample_rate <= 0:
            return None
        by_epoch: dict[int, list[tuple[int, int]]] = {}
        for window in session.windows:
            if window.sample_range is not None:
                by_epoch.setdefault(window.key.stream_epoch, []).append(window.sample_range)
        if not by_epoch:
            return None
        samples = sum(max(end for _, end in ranges) - min(start for start, _ in ranges) for ranges in by_epoch.values())
        return samples / sample_rate if samples > 0 else None

    @staticmethod
    def _percentiles(values: list[float | None]) -> Percentiles:
        available = np.asarray([item for item in values if item is not None], dtype=np.float64)
        missing = len(values) - len(available)
        if not len(available):
            return Percentiles(None, None, None, 0, missing)
        result = np.percentile(available, [50, 95, 99])
        return Percentiles(float(result[0]), float(result[1]), float(result[2]), len(available), missing)
