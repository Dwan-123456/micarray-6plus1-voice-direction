from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.data_types import CandidateDirection, TrackedDirection


def _residual_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


_delta = _residual_deg


@dataclass(frozen=True, slots=True)
class GlobalTrackerConfig:
    association_gate_deg: float = 45.0
    max_velocity_dps: float = 60.0
    confirmation_observations: int = 2
    confirmation_window_samples: int = 9_600
    coasting_ttl_samples: int = 48_000
    miss_cost: float = 1.0
    birth_cost: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.association_gate_deg <= 180 or not 0 < self.max_velocity_dps <= 360:
            raise ValueError("global tracker angular limits are invalid")
        if min(
            self.confirmation_observations, self.confirmation_window_samples,
            self.coasting_ttl_samples,
        ) <= 0 or min(self.miss_cost, self.birth_cost) <= 0:
            raise ValueError("global tracker lifecycle/cost values must be positive")


@dataclass(slots=True)
class _Track:
    track_id: int
    unwrapped_theta: float
    velocity_dps: float
    first_seen: int
    last_observed: int
    observations: int
    raw_score: float
    normalized_score: float
    confirmed: bool = False
    filtered_theta: float | None = None
    filtered_velocity_dps: float = 0.0


class GlobalDirectionTracker:
    """Deterministic sample-time lifecycle with session-scoped non-reused IDs."""

    backend = "global_assignment_v1"

    def __init__(self, config: GlobalTrackerConfig | None = None) -> None:
        self.config = config or GlobalTrackerConfig()
        self._session_id: str | None = None
        self._stream_epoch: int | None = None
        self._next_by_session: dict[str, int] = {}
        self._tracks: dict[int, _Track] = {}
        self._last_sample: int | None = None
        self.last_assignments: tuple[int, ...] = ()
        self.last_assignment_is_new: tuple[bool, ...] = ()

    def reset(self, *, preserve_session_counters: bool = False) -> None:
        self._session_id = None
        self._stream_epoch = None
        self._tracks.clear()
        self._last_sample = None
        self.last_assignments = ()
        self.last_assignment_is_new = ()
        if not preserve_session_counters:
            self._next_by_session.clear()

    def prepare_stream(self, session_id: str, stream_epoch: int) -> None:
        if session_id != self._session_id:
            self._session_id = session_id
            self._stream_epoch = stream_epoch
            self._tracks.clear()
            self._last_sample = None
            self._next_by_session.setdefault(session_id, 1)
        elif stream_epoch != self._stream_epoch:
            self._stream_epoch = stream_epoch
            self._tracks.clear()
            self._last_sample = None
        self.last_assignments = ()
        self.last_assignment_is_new = ()

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def _new_id(self, session_id: str) -> int:
        track_id = self._next_by_session.setdefault(session_id, 1)
        self._next_by_session[session_id] = track_id + 1
        return track_id

    def update(
        self,
        session_id: str,
        stream_epoch: int,
        decision_sample: int,
        candidates: tuple[CandidateDirection, ...],
        *,
        window_id: int,
        doa_start_sample: int,
        doa_end_sample: int | None = None,
        kalman_enabled: bool = False,
        q_scale: float = 1.0,
        r_scale: float = 1.0,
    ) -> tuple[tuple[TrackedDirection, ...], tuple[TrackedDirection, ...]]:
        self.prepare_stream(session_id, stream_epoch)
        if self._last_sample is not None and decision_sample <= self._last_sample:
            raise ValueError("direction tracking sample must advance")
        doa_end_sample = decision_sample if doa_end_sample is None else doa_end_sample
        ttl = self.config.coasting_ttl_samples
        for track_id, track in tuple(self._tracks.items()):
            if decision_sample - track.last_observed > ttl:
                del self._tracks[track_id]
        candidates = tuple(candidates)
        track_ids = tuple(sorted(self._tracks))
        assigned: dict[int, int] = {}
        if track_ids and candidates:
            rows, columns = len(track_ids), len(candidates)
            size = rows + columns
            cost = np.full((size, size), 1.0e6, dtype=np.float64)
            for row, track_id in enumerate(track_ids):
                track = self._tracks[track_id]
                dt = max(0.0, (decision_sample - (self._last_sample or decision_sample)) / 48_000.0)
                predicted = track.unwrapped_theta + track.velocity_dps * dt
                gate = min(180.0, self.config.association_gate_deg + self.config.max_velocity_dps * dt)
                for column, candidate in enumerate(candidates):
                    distance = abs(_delta(candidate.theta_deg, predicted))
                    if distance <= gate:
                        # Normalized association cost is comparable with the
                        # configured miss+birth alternative.  Raw degrees here
                        # would otherwise create a new ID for almost every
                        # residual larger than two degrees.
                        quality_penalty = 0.2 * (1.0 - candidate.normalized_score)
                        # The tiny stable term makes an exactly tied matrix
                        # deterministic without affecting any physical cost.
                        tie_break = (row * (columns + 1) + column) * 1.0e-12
                        cost[row, column] = (
                            distance / max(gate, 1.0e-9) + quality_penalty + tie_break
                        )
                cost[row, columns + row] = self.config.miss_cost
            for column in range(columns):
                cost[rows + column, column] = self.config.birth_cost
            cost[rows:, columns:] = 0.0
            selected_rows, selected_columns = linear_sum_assignment(cost)
            for row, column in zip(selected_rows, selected_columns, strict=True):
                if row < rows and column < columns and cost[row, column] < 1.0e5:
                    assigned[column] = track_ids[row]
        directions: list[TrackedDirection] = []
        assignment_ids: list[int] = []
        new_flags: list[bool] = []
        for rank, candidate in enumerate(candidates, start=1):
            index = rank - 1
            track_id = assigned.get(index)
            is_new = track_id is None
            if is_new:
                track_id = self._new_id(session_id)
                track = _Track(
                    track_id, candidate.theta_deg, 0.0, decision_sample,
                    decision_sample, 1, candidate.raw_score, candidate.normalized_score,
                )
                self._tracks[track_id] = track
            else:
                track = self._tracks[track_id]
                elapsed = max(1, decision_sample - track.last_observed)
                previous_unwrapped = track.unwrapped_theta
                unwrapped_measurement = previous_unwrapped + _delta(candidate.theta_deg, previous_unwrapped)
                measured_velocity = (unwrapped_measurement - previous_unwrapped) * 48_000.0 / elapsed
                measured_velocity = float(np.clip(
                    measured_velocity, -self.config.max_velocity_dps, self.config.max_velocity_dps
                ))
                # Geometric association state is updated independently of the
                # optional display/output smoother.
                track.unwrapped_theta = unwrapped_measurement
                track.velocity_dps = measured_velocity
                track.last_observed = decision_sample
                track.observations += 1
                track.raw_score = candidate.raw_score
                track.normalized_score = candidate.normalized_score
            if kalman_enabled:
                if track.filtered_theta is None:
                    track.filtered_theta = track.unwrapped_theta
                    track.filtered_velocity_dps = track.velocity_dps
                else:
                    alpha = float(np.clip(q_scale / (q_scale + r_scale), 0.02, 0.98))
                    predicted_filter = track.filtered_theta + track.filtered_velocity_dps * (elapsed / 48_000.0)
                    track.filtered_theta = predicted_filter + alpha * _delta(
                        candidate.theta_deg, predicted_filter
                    )
                    track.filtered_velocity_dps = (
                        (1.0 - alpha) * track.filtered_velocity_dps + alpha * track.velocity_dps
                    )
                output_theta = float(track.filtered_theta % 360.0)
            else:
                track.filtered_theta = None
                track.filtered_velocity_dps = 0.0
                output_theta = float(track.unwrapped_theta % 360.0)
            if (
                not track.confirmed
                and track.observations >= self.config.confirmation_observations
                and decision_sample - track.first_seen <= self.config.confirmation_window_samples
            ):
                track.confirmed = True
            state = "confirmed" if track.confirmed else "tentative"
            directions.append(TrackedDirection(
                session_id, stream_epoch, candidate.window_id, decision_sample,
                candidate.doa_start_sample, candidate.doa_end_sample,
                track_id, rank, candidate.theta_deg, output_theta,
                candidate.raw_score, candidate.normalized_score, state,
                True, is_new, track.first_seen, track.last_observed, 0, kalman_enabled,
            ))
            assignment_ids.append(track_id)
            new_flags.append(is_new)
        active: list[TrackedDirection] = list(directions)
        observed_ids = set(assignment_ids)
        for track_id, track in sorted(self._tracks.items()):
            if track_id in observed_ids:
                continue
            missed = decision_sample - track.last_observed
            theta = (track.unwrapped_theta + track.velocity_dps * missed / 48_000.0) % 360.0
            active.append(TrackedDirection(
                session_id, stream_epoch,
                window_id,
                decision_sample,
                doa_start_sample, doa_end_sample,
                track_id, len(active) + 1, None, theta,
                track.raw_score, track.normalized_score, "coasting", False, False,
                track.first_seen, track.last_observed, missed, kalman_enabled,
            ))
        self._last_sample = decision_sample
        self.last_assignments = tuple(assignment_ids)
        self.last_assignment_is_new = tuple(new_flags)
        return tuple(directions), tuple(active)
