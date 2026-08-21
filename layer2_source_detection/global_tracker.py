from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.data_types import CandidateDirection, TrackedDirection


def _residual_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


_delta = _residual_deg


def _circular_mean_deg(values: list[float]) -> float | None:
    if not values:
        return None
    radians = np.deg2rad(np.asarray(values, dtype=np.float64))
    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))
    if np.hypot(mean_sin, mean_cos) <= 1.0e-12:
        return None
    return float(np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360.0)


@dataclass(frozen=True, slots=True)
class GlobalTrackerConfig:
    association_gate_deg: float = 50.0
    association_gate_base_deg: float = 20.0
    association_gate_growth_dps: float = 15.0
    max_velocity_dps: float = 60.0
    confirmation_observations: int = 3
    confirmation_window_samples: int = 9_600
    coasting_ttl_samples: int = 2 * 48_000
    miss_cost: float = 1.0
    birth_cost: float = 1.0
    stationary_history_samples: int = 3 * 48_000
    stationary_inlier_ratio: float = 0.70
    stationary_inlier_tolerance_deg: float = 15.0
    stationary_outlier_window_samples: int = 48_000
    stationary_outlier_tolerance_deg: float = 20.0
    stationary_exit_observations: int = 4
    kalman_process_angle_std_deg: float = 1.5
    kalman_process_velocity_std_dps: float = 25.0
    kalman_measurement_std_deg: float = 5.0
    kalman_velocity_half_life_seconds: float = 0.5
    kalman_prediction_freeze_std_deg: float = float("inf")

    def __post_init__(self) -> None:
        if (
            not 0 < self.association_gate_base_deg <= self.association_gate_deg <= 180
            or not 0 <= self.association_gate_growth_dps <= 360
            or not 0 < self.max_velocity_dps <= 360
        ):
            raise ValueError("global tracker angular limits are invalid")
        if min(
            self.confirmation_observations, self.confirmation_window_samples,
            self.coasting_ttl_samples, self.stationary_history_samples,
            self.stationary_outlier_window_samples, self.stationary_exit_observations,
        ) <= 0 or min(self.miss_cost, self.birth_cost) <= 0:
            raise ValueError("global tracker lifecycle/cost values must be positive")
        if not 0 <= self.stationary_inlier_ratio <= 1:
            raise ValueError("stationary inlier ratio must be in [0,1]")
        if not (
            0 < self.stationary_inlier_tolerance_deg
            < self.stationary_outlier_tolerance_deg <= 180
        ):
            raise ValueError("stationary angular limits are invalid")
        if min(
            self.kalman_process_angle_std_deg,
            self.kalman_process_velocity_std_dps,
            self.kalman_measurement_std_deg,
            self.kalman_velocity_half_life_seconds,
            self.kalman_prediction_freeze_std_deg,
        ) <= 0:
            raise ValueError("Kalman model values must be positive")


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
    confirmation_samples: list[int] = field(default_factory=list)
    confirmed: bool = False
    filtered_theta: float | None = None
    filtered_velocity_dps: float = 0.0
    last_voice_sample: int | None = None
    last_voice_probability: float | None = None
    voice_confirmation_samples: set[int] = field(default_factory=set)
    coasting_voice_expiry_sample: int | None = None
    noise_interference: bool = False
    noise_voice_recovery_samples: list[int] = field(default_factory=list)
    stationary_angle_history: list[tuple[int, float]] = field(default_factory=list)
    stationary_locked: bool = False
    stationary_theta: float | None = None
    stationary_outlier_samples: list[int] = field(default_factory=list)
    kalman_covariance: np.ndarray | None = None
    kalman_last_sample: int | None = None
    kalman_last_trusted_theta: float | None = None
    existence_probability: float = 0.55


class GlobalDirectionTracker:
    """Deterministic sample-time lifecycle with session-scoped non-reused IDs."""

    backend = "global_assignment_v1"
    voice_absence_noise_samples = 3 * 48_000
    minimum_voice_confirmations = 2
    max_active_tracks = 4

    def __init__(self, config: GlobalTrackerConfig | None = None, *, association_backend: str = "hungarian") -> None:
        if association_backend not in {"hungarian", "lmb_jpda"}:
            raise ValueError("unsupported direction association backend")
        self.config = config or GlobalTrackerConfig()
        self.association_backend = association_backend
        self.backend = "lmb_jpda_v1" if association_backend == "lmb_jpda" else "global_assignment_v1"
        self._session_id: str | None = None
        self._stream_epoch: int | None = None
        self._next_by_session: dict[str, int] = {}
        self._tracks: dict[int, _Track] = {}
        self._last_sample: int | None = None
        self.last_assignments: tuple[int, ...] = ()
        self.last_assignment_is_new: tuple[bool, ...] = ()
        self.last_association_probabilities: tuple[float, ...] = ()

    def _jpda_assign(
        self,
        track_ids: tuple[int, ...],
        candidates: tuple[CandidateDirection, ...],
        predicted_by_track: dict[int, float],
        decision_sample: int,
    ) -> dict[int, int]:
        """Exact bounded JPDA hypothesis sum, followed by deterministic MAP extraction."""
        rows, columns = len(track_ids), len(candidates)
        likelihood = np.zeros((rows, columns), dtype=np.float64)
        for row, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            gate = self._association_gate(track, decision_sample)
            for column, candidate in enumerate(candidates):
                distance = abs(_delta(candidate.theta_deg, predicted_by_track[track_id]))
                if distance <= gate:
                    sigma = max(3.0, gate / 2.5)
                    likelihood[row, column] = (
                        np.exp(-0.5 * (distance / sigma) ** 2)
                        * max(1.0e-4, candidate.normalized_score)
                    )
        hypotheses: list[tuple[float, tuple[int | None, ...]]] = []

        def visit(row: int, used: frozenset[int], assignment: tuple[int | None, ...], weight: float) -> None:
            if row == rows:
                unused = columns - len(used)
                hypotheses.append((weight * (0.35 ** unused), assignment))
                return
            visit(row + 1, used, assignment + (None,), weight * 0.25)
            for column in range(columns):
                value = likelihood[row, column]
                if column not in used and value > 0.0:
                    visit(row + 1, used | {column}, assignment + (column,), weight * value)

        visit(0, frozenset(), (), 1.0)
        normalizer = sum(weight for weight, _ in hypotheses)
        marginals = np.zeros_like(likelihood)
        if normalizer > 0.0:
            for weight, hypothesis in hypotheses:
                for row, column in enumerate(hypothesis):
                    if column is not None:
                        marginals[row, column] += weight / normalizer
        assigned: dict[int, int] = {}
        probabilities: list[float] = []
        if rows and columns:
            selected_rows, selected_columns = linear_sum_assignment(-marginals)
            for row, column in zip(selected_rows, selected_columns, strict=True):
                probability = float(marginals[row, column])
                if probability >= 0.20:
                    assigned[int(column)] = track_ids[int(row)]
                    probabilities.append(probability)
        self.last_association_probabilities = tuple(probabilities)
        matched_ids = set(assigned.values())
        for track_id in track_ids:
            track = self._tracks[track_id]
            if track_id in matched_ids:
                probability = max(
                    marginals[track_ids.index(track_id), column]
                    for column, assigned_id in assigned.items() if assigned_id == track_id
                )
                track.existence_probability = min(0.995, 0.75 * track.existence_probability + 0.25 * probability)
            else:
                track.existence_probability *= 0.92
        return assigned

    def reset(self, *, preserve_session_counters: bool = False) -> None:
        self._session_id = None
        self._stream_epoch = None
        self._tracks.clear()
        self._last_sample = None
        self.last_assignments = ()
        self.last_assignment_is_new = ()
        self.last_association_probabilities = ()
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
        self.last_association_probabilities = ()

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def has_live_tracks(
        self, session_id: str, stream_epoch: int, decision_sample: int
    ) -> bool:
        """Return whether this window starts with at least one unexpired ID.

        The check uses the same absolute-sample TTL as ``update``.  It also
        applies stream lifecycle changes before the probability Gate is
        evaluated, so an ID from an old epoch can never force the new epoch
        open.
        """

        self.prepare_stream(session_id, stream_epoch)
        self._expire_tracks(decision_sample)
        self._refresh_noise_labels(decision_sample)
        return bool(self._tracks)

    def voice_confirmed_track_ids(
        self, session_id: str, stream_epoch: int, decision_sample: int
    ) -> frozenset[int]:
        """Return live tracking-confirmed IDs with positive L5 voice evidence.

        A direction observation alone is deliberately insufficient here.  The
        returned IDs are the only tracks allowed to force the probability Gate
        open or publish a missing-observation/coasting direction to L3.
        """

        self.prepare_stream(session_id, stream_epoch)
        self._expire_tracks(decision_sample)
        self._refresh_noise_labels(decision_sample)
        return frozenset(
            track_id
            for track_id, track in self._tracks.items()
            if track.confirmed
            and len(track.voice_confirmation_samples) >= self.minimum_voice_confirmations
            and not track.noise_interference
        )

    def _expire_tracks(self, decision_sample: int) -> None:
        ttl = self.config.coasting_ttl_samples
        for track_id, track in tuple(self._tracks.items()):
            expiry_sample = track.last_observed + ttl
            if track.coasting_voice_expiry_sample is not None:
                expiry_sample = max(expiry_sample, track.coasting_voice_expiry_sample)
            if decision_sample > expiry_sample:
                del self._tracks[track_id]

    def _refresh_noise_labels(self, decision_sample: int) -> None:
        for track in self._tracks.values():
            semantic_anchor = (
                track.first_seen if track.last_voice_sample is None else track.last_voice_sample
            )
            if decision_sample - semantic_anchor >= self.voice_absence_noise_samples:
                track.noise_interference = True

    def apply_voice_feedback(
        self,
        session_id: str,
        stream_epoch: int,
        decision_sample: int,
        track_id: int,
        probability: float,
        is_voice: bool,
    ) -> bool:
        """Apply one delayed, authoritative L5 result to an existing ID."""

        if (session_id, stream_epoch) != (self._session_id, self._stream_epoch):
            return False
        track = self._tracks.get(track_id)
        if track is None or decision_sample < track.first_seen:
            return False
        probability = float(probability)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("L5 voice feedback probability must be in [0,1]")
        track.last_voice_probability = probability
        if is_voice:
            if len(track.voice_confirmation_samples) < self.minimum_voice_confirmations:
                track.voice_confirmation_samples.add(decision_sample)
            if (
                track.confirmed
                and len(track.voice_confirmation_samples) >= self.minimum_voice_confirmations
                and decision_sample > track.last_observed
            ):
                renewed_expiry = decision_sample + self.config.coasting_ttl_samples
                if track.coasting_voice_expiry_sample is None:
                    track.coasting_voice_expiry_sample = renewed_expiry
                else:
                    track.coasting_voice_expiry_sample = max(
                        track.coasting_voice_expiry_sample, renewed_expiry
                    )
            track.last_voice_sample = max(
                decision_sample,
                decision_sample if track.last_voice_sample is None else track.last_voice_sample,
            )
            if track.noise_interference:
                nearby_non_noise = any(
                    other_id != track_id
                    and not other.noise_interference
                    and abs(_delta(track.unwrapped_theta, other.unwrapped_theta))
                    <= self.config.association_gate_deg
                    for other_id, other in self._tracks.items()
                )
                if nearby_non_noise:
                    track.noise_voice_recovery_samples.clear()
                else:
                    cutoff = decision_sample - self.voice_absence_noise_samples
                    track.noise_voice_recovery_samples[:] = [
                        sample for sample in track.noise_voice_recovery_samples
                        if sample >= cutoff
                    ]
                    if (
                        not track.noise_voice_recovery_samples
                        or decision_sample > track.noise_voice_recovery_samples[-1]
                    ):
                        track.noise_voice_recovery_samples.append(decision_sample)
                    if len(track.noise_voice_recovery_samples) >= 5:
                        track.noise_interference = False
                        track.noise_voice_recovery_samples.clear()
        return True

    @staticmethod
    def _update_observation(track: _Track, candidate: CandidateDirection, decision_sample: int,
                            max_velocity_dps: float) -> int:
        elapsed = max(1, decision_sample - track.last_observed)
        previous_unwrapped = track.unwrapped_theta
        unwrapped_measurement = previous_unwrapped + _delta(candidate.theta_deg, previous_unwrapped)
        measured_velocity = (unwrapped_measurement - previous_unwrapped) * 48_000.0 / elapsed
        track.unwrapped_theta = unwrapped_measurement
        track.velocity_dps = float(np.clip(measured_velocity, -max_velocity_dps, max_velocity_dps))
        track.last_observed = decision_sample
        track.observations += 1
        track.raw_score = candidate.raw_score
        track.normalized_score = candidate.normalized_score
        return elapsed

    @staticmethod
    def _clear_stationary_state(track: _Track, *, clear_history: bool) -> None:
        track.stationary_locked = False
        track.stationary_theta = None
        track.stationary_outlier_samples.clear()
        if clear_history:
            track.stationary_angle_history.clear()

    def _append_stationary_history(
        self, track: _Track, decision_sample: int, theta_deg: float,
    ) -> None:
        cutoff = decision_sample - self.config.stationary_history_samples
        track.stationary_angle_history[:] = [
            item for item in track.stationary_angle_history if item[0] >= cutoff
        ]
        if (
            not track.stationary_angle_history
            or track.stationary_angle_history[-1][0] != decision_sample
        ):
            track.stationary_angle_history.append((decision_sample, float(theta_deg) % 360.0))

    def _maybe_lock_stationary(self, track: _Track, decision_sample: int) -> None:
        if (
            track.stationary_locked
            or not track.confirmed
            or decision_sample - track.first_seen < self.config.stationary_history_samples
            or len(track.stationary_angle_history) < self.config.confirmation_observations
        ):
            return
        mean = _circular_mean_deg([item[1] for item in track.stationary_angle_history])
        if mean is None:
            return
        inliers = sum(
            abs(_delta(theta, mean)) <= self.config.stationary_inlier_tolerance_deg
            for _, theta in track.stationary_angle_history
        )
        if inliers / len(track.stationary_angle_history) < self.config.stationary_inlier_ratio:
            return
        track.stationary_locked = True
        track.stationary_theta = mean
        track.stationary_outlier_samples.clear()
        track.unwrapped_theta += _delta(mean, track.unwrapped_theta)
        track.velocity_dps = 0.0

    def _update_locked_stationary(
        self, track: _Track, candidate: CandidateDirection, decision_sample: int,
    ) -> bool:
        """Return True when the measurement is consumed without normal motion update."""

        mean = track.stationary_theta
        if not track.stationary_locked or mean is None:
            return False
        outlier_cutoff = decision_sample - self.config.stationary_outlier_window_samples
        track.stationary_outlier_samples[:] = [
            sample for sample in track.stationary_outlier_samples if sample >= outlier_cutoff
        ]
        if abs(_delta(candidate.theta_deg, mean)) > self.config.stationary_outlier_tolerance_deg:
            if (
                not track.stationary_outlier_samples
                or track.stationary_outlier_samples[-1] != decision_sample
            ):
                track.stationary_outlier_samples.append(decision_sample)
            if len(track.stationary_outlier_samples) >= self.config.stationary_exit_observations:
                self._clear_stationary_state(track, clear_history=True)
                self._update_observation(
                    track, candidate, decision_sample, self.config.max_velocity_dps
                )
                self._append_stationary_history(track, decision_sample, candidate.theta_deg)
                return True
            # Up to three one-second outliers refresh the ID observation lease
            # but cannot move its association anchor or published angle.
            track.last_observed = decision_sample
            track.observations += 1
            track.raw_score = candidate.raw_score
            track.normalized_score = candidate.normalized_score
            track.velocity_dps = 0.0
            return True
        self._append_stationary_history(track, decision_sample, candidate.theta_deg)
        updated_mean = _circular_mean_deg(
            [item[1] for item in track.stationary_angle_history]
        )
        if updated_mean is not None:
            track.stationary_theta = updated_mean
            track.unwrapped_theta += _delta(updated_mean, track.unwrapped_theta)
        track.last_observed = decision_sample
        track.observations += 1
        track.raw_score = candidate.raw_score
        track.normalized_score = candidate.normalized_score
        track.velocity_dps = 0.0
        return True

    def _update_confirmation(self, track: _Track, decision_sample: int) -> None:
        """Confirm from observations in the latest absolute-sample window.

        ``first_seen`` is permanent public identity metadata, so it cannot also
        serve as a retry deadline.  Pruning the private observation history
        lets a tentative track start a fresh confirmation opportunity after an
        early sparse observation without changing its authoritative ID.
        """

        if track.confirmed:
            return
        cutoff = decision_sample - self.config.confirmation_window_samples
        track.confirmation_samples[:] = [
            sample for sample in track.confirmation_samples if sample >= cutoff
        ]
        if (
            not track.confirmation_samples
            or track.confirmation_samples[-1] != decision_sample
        ):
            track.confirmation_samples.append(decision_sample)
        if len(track.confirmation_samples) >= self.config.confirmation_observations:
            track.confirmed = True
            track.confirmation_samples.clear()

    def _association_gate(self, track: _Track, decision_sample: int) -> float:
        """Angle gate derived from time since this ID's last real observation."""

        if not track.confirmed:
            return self.config.association_gate_base_deg
        missed_seconds = max(0, decision_sample - track.last_observed) / 48_000.0
        return min(
            self.config.association_gate_deg,
            self.config.association_gate_base_deg
            + self.config.association_gate_growth_dps * missed_seconds,
        )

    def _damped_transition(self, dt: float) -> tuple[np.ndarray, float]:
        half_life = self.config.kalman_velocity_half_life_seconds
        gamma = float(2.0 ** (-dt / half_life))
        velocity_time = half_life / np.log(2.0) * (1.0 - gamma)
        return np.asarray(((1.0, velocity_time), (0.0, gamma)), dtype=np.float64), gamma

    def _raw_forecast(self, track: _Track, decision_sample: int) -> float:
        dt = max(0, decision_sample - track.last_observed) / 48_000.0
        transition, _ = self._damped_transition(dt)
        return float((transition @ np.asarray((track.unwrapped_theta, track.velocity_dps)))[0])

    def _kalman_initialize(self, track: _Track, decision_sample: int) -> None:
        track.filtered_theta = track.unwrapped_theta
        track.filtered_velocity_dps = float(np.clip(
            track.velocity_dps, -self.config.max_velocity_dps, self.config.max_velocity_dps
        ))
        track.kalman_covariance = np.diag((
            self.config.kalman_measurement_std_deg**2,
            self.config.kalman_process_velocity_std_dps**2,
        )).astype(np.float64)
        track.kalman_last_sample = decision_sample
        track.kalman_last_trusted_theta = track.filtered_theta

    def _kalman_predict(self, track: _Track, decision_sample: int, q_scale: float) -> None:
        if track.filtered_theta is None or track.kalman_covariance is None:
            self._kalman_initialize(track, decision_sample)
            return
        assert track.kalman_last_sample is not None
        delta = decision_sample - track.kalman_last_sample
        if delta < 0:
            raise ValueError("Kalman sample time cannot move backwards")
        if delta == 0:
            return
        dt = delta / 48_000.0
        transition, _ = self._damped_transition(dt)
        vector = transition @ np.asarray(
            (track.filtered_theta, track.filtered_velocity_dps), dtype=np.float64
        )
        process = np.diag((
            self.config.kalman_process_angle_std_deg**2,
            self.config.kalman_process_velocity_std_dps**2,
        )) * max(dt, 0.02) * q_scale
        covariance = transition @ track.kalman_covariance @ transition.T + process
        if not np.isfinite(vector).all() or not np.isfinite(covariance).all():
            raise ValueError("non-finite damped circular Kalman prediction")
        track.filtered_theta = float(vector[0])
        track.filtered_velocity_dps = float(np.clip(
            vector[1], -self.config.max_velocity_dps, self.config.max_velocity_dps
        ))
        track.kalman_covariance = covariance
        track.kalman_last_sample = decision_sample

    def _kalman_forecast(self, track: _Track, decision_sample: int) -> float:
        if (
            track.filtered_theta is None
            or track.kalman_last_sample is None
            or decision_sample < track.kalman_last_sample
        ):
            return self._raw_forecast(track, decision_sample)
        dt = (decision_sample - track.kalman_last_sample) / 48_000.0
        transition, _ = self._damped_transition(dt)
        return float((transition @ np.asarray(
            (track.filtered_theta, track.filtered_velocity_dps), dtype=np.float64
        ))[0])

    def _kalman_correct(
        self,
        track: _Track,
        theta_deg: float,
        r_scale: float,
        measurement_confidence: float,
    ) -> None:
        assert track.filtered_theta is not None and track.kalman_covariance is not None
        measurement = track.filtered_theta + _delta(theta_deg, track.filtered_theta)
        h = np.asarray((1.0, 0.0), dtype=np.float64)
        measurement_variance = (
            self.config.kalman_measurement_std_deg**2
            * r_scale
            / measurement_confidence
        )
        innovation_variance = float(h @ track.kalman_covariance @ h + measurement_variance)
        if not np.isfinite(innovation_variance) or innovation_variance <= 0:
            raise ValueError("invalid damped circular Kalman innovation variance")
        gain = track.kalman_covariance @ h / innovation_variance
        vector = np.asarray(
            (track.filtered_theta, track.filtered_velocity_dps), dtype=np.float64
        )
        vector += gain * (measurement - track.filtered_theta)
        identity_minus_kh = np.eye(2) - np.outer(gain, h)
        covariance = (
            identity_minus_kh @ track.kalman_covariance @ identity_minus_kh.T
            + np.outer(gain, gain) * measurement_variance
        )
        track.filtered_theta = float(vector[0])
        track.filtered_velocity_dps = float(np.clip(
            vector[1], -self.config.max_velocity_dps, self.config.max_velocity_dps
        ))
        track.kalman_covariance = covariance
        track.kalman_last_trusted_theta = track.filtered_theta

    def _kalman_output_theta(self, track: _Track) -> float:
        assert track.filtered_theta is not None
        covariance = track.kalman_covariance
        if (
            covariance is not None
            and np.sqrt(max(0.0, float(covariance[0, 0])))
            > self.config.kalman_prediction_freeze_std_deg
            and track.kalman_last_trusted_theta is not None
        ):
            return float(track.kalman_last_trusted_theta % 360.0)
        return float(track.filtered_theta % 360.0)

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
        allow_births: bool = True,
    ) -> tuple[tuple[TrackedDirection, ...], tuple[TrackedDirection, ...]]:
        self.prepare_stream(session_id, stream_epoch)
        if self._last_sample is not None and decision_sample <= self._last_sample:
            raise ValueError("direction tracking sample must advance")
        doa_end_sample = decision_sample if doa_end_sample is None else doa_end_sample
        self._expire_tracks(decision_sample)
        self._refresh_noise_labels(decision_sample)
        if kalman_enabled:
            for track in self._tracks.values():
                self._clear_stationary_state(track, clear_history=True)
        candidates = tuple(candidates)
        # Noise markers are excluded from the exclusive Hungarian rows so a
        # nearby normal ID can never be merged into a stationary noise ID.
        track_ids = tuple(sorted(
            track_id for track_id, track in self._tracks.items()
            if not track.noise_interference
        ))
        assigned: dict[int, int] = {}
        predicted_by_track: dict[int, float] = {}
        if track_ids and candidates:
            rows, columns = len(track_ids), len(candidates)
            size = rows + columns
            cost = np.full((size, size), 1.0e6, dtype=np.float64)
            for row, track_id in enumerate(track_ids):
                track = self._tracks[track_id]
                predicted = (
                    self._kalman_forecast(track, decision_sample)
                    if kalman_enabled
                    else self._raw_forecast(track, decision_sample)
                )
                predicted_by_track[track_id] = predicted
                gate = self._association_gate(track, decision_sample)
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
            if self.association_backend == "lmb_jpda":
                assigned = self._jpda_assign(
                    track_ids, candidates, predicted_by_track, decision_sample
                )
        noise_ids = tuple(sorted(
            track_id for track_id, track in self._tracks.items()
            if track.noise_interference
        ))
        used_noise_ids: set[int] = set()
        suppressed_birth_indices: set[int] = set()
        for index, candidate in enumerate(candidates):
            if index in assigned:
                continue
            # A second peak beside an existing ordinary ID is evidence for the
            # same direction track, not permission to create a duplicate ID.
            # Keep this guard narrower than the normal association gate so a
            # genuinely separate source can still be born outside 20 degrees.
            normal_nearby = any(
                abs(_delta(candidate.theta_deg, predicted_by_track[track_id]))
                <= self.config.association_gate_base_deg
                for track_id in track_ids
            )
            if normal_nearby:
                suppressed_birth_indices.add(index)
                continue
            viable_noise = tuple(
                (abs(_delta(candidate.theta_deg, self._tracks[track_id].unwrapped_theta)), track_id)
                for track_id in noise_ids
                if abs(_delta(candidate.theta_deg, self._tracks[track_id].unwrapped_theta))
                <= self.config.association_gate_deg
            )
            if viable_noise:
                unused_noise = tuple(
                    item for item in viable_noise if item[1] not in used_noise_ids
                )
                if unused_noise:
                    noise_id = min(unused_noise)[1]
                    assigned[index] = noise_id
                    used_noise_ids.add(noise_id)

        birth_indices = [
            index for index in range(len(candidates))
            if index not in assigned
            and index not in suppressed_birth_indices
            and allow_births
        ]
        required_slots = max(
            0, len(self._tracks) + len(birth_indices) - self.max_active_tracks
        )
        protected_ids = set(assigned.values())
        victims = sorted(
            (track for track in self._tracks.values() if track.track_id not in protected_ids),
            key=lambda track: (
                0 if track.noise_interference else 1,
                0 if track.last_voice_sample is None else 1,
                0 if not track.confirmed else 1,
                track.last_observed,
                track.normalized_score,
                track.track_id,
            ),
        )
        for victim in victims[:required_slots]:
            del self._tracks[victim.track_id]
        available_births = max(0, self.max_active_tracks - len(self._tracks))
        allowed_births = set(sorted(
            birth_indices,
            key=lambda index: (
                -candidates[index].normalized_score,
                candidates[index].theta_deg,
                index,
            ),
        )[:available_births])
        directions: list[TrackedDirection] = []
        assignment_ids: list[int] = []
        new_flags: list[bool] = []
        for index, candidate in enumerate(candidates):
            track_id = assigned.get(index)
            is_new = track_id is None
            missed_before = 0
            if is_new:
                if index not in allowed_births:
                    continue
                track_id = self._new_id(session_id)
                track = _Track(
                    track_id, candidate.theta_deg, 0.0, decision_sample,
                    decision_sample, 1, candidate.raw_score, candidate.normalized_score,
                )
                self._tracks[track_id] = track
                if not kalman_enabled:
                    self._append_stationary_history(
                        track, decision_sample, candidate.theta_deg
                    )
            else:
                track = self._tracks[track_id]
                missed_before = max(0, decision_sample - track.last_observed)
                if not kalman_enabled and track.stationary_locked:
                    self._update_locked_stationary(track, candidate, decision_sample)
                else:
                    self._update_observation(
                        track, candidate, decision_sample, self.config.max_velocity_dps
                    )
                    if not kalman_enabled:
                        self._append_stationary_history(
                            track, decision_sample, candidate.theta_deg
                        )
            self._update_confirmation(track, decision_sample)
            if not kalman_enabled:
                self._maybe_lock_stationary(track, decision_sample)
            if kalman_enabled:
                if track.filtered_theta is None:
                    self._kalman_initialize(track, decision_sample)
                else:
                    self._kalman_predict(track, decision_sample, q_scale)
                    reacquisition_confidence = 1.0 + min(
                        1.0, missed_before / max(1, self.config.coasting_ttl_samples)
                    )
                    self._kalman_correct(
                        track, candidate.theta_deg, r_scale, reacquisition_confidence
                    )
                output_theta = self._kalman_output_theta(track)
            else:
                track.filtered_theta = None
                track.filtered_velocity_dps = 0.0
                track.kalman_covariance = None
                track.kalman_last_sample = None
                track.kalman_last_trusted_theta = None
                output_theta = float(
                    (track.stationary_theta if track.stationary_locked else track.unwrapped_theta)
                    % 360.0
                )
            state = "confirmed" if track.confirmed else "tentative"
            rank = len(directions) + 1
            directions.append(TrackedDirection(
                session_id, stream_epoch, candidate.window_id, decision_sample,
                candidate.doa_start_sample, candidate.doa_end_sample,
                track_id, rank, candidate.theta_deg, output_theta,
                candidate.raw_score, candidate.normalized_score, state,
                True, is_new, track.first_seen, track.last_observed, 0, kalman_enabled,
                track.noise_interference,
            ))
            assignment_ids.append(track_id)
            new_flags.append(is_new)
        active: list[TrackedDirection] = list(directions)
        observed_ids = set(assignment_ids)
        for track_id, track in sorted(self._tracks.items()):
            if track_id in observed_ids:
                continue
            missed = decision_sample - track.last_observed
            if kalman_enabled:
                self._kalman_predict(track, decision_sample, q_scale)
                theta = self._kalman_output_theta(track)
            else:
                # ID lifetime and association remain active with Kalman OFF,
                # but the published/coasting angle is a zero-order hold of
                # the last real observation.  Raw velocity may still be used
                # internally by the global assignment and is never exposed
                # as an OFF-mode angle prediction.
                track.filtered_theta = None
                track.filtered_velocity_dps = 0.0
                track.kalman_covariance = None
                track.kalman_last_sample = None
                track.kalman_last_trusted_theta = None
                theta = track.unwrapped_theta % 360.0
            active.append(TrackedDirection(
                session_id, stream_epoch,
                window_id,
                decision_sample,
                doa_start_sample, doa_end_sample,
                track_id, len(active) + 1, None, theta,
                track.raw_score, track.normalized_score,
                "coasting" if track.confirmed else "tentative", False, False,
                track.first_seen, track.last_observed, missed, kalman_enabled,
                track.noise_interference,
            ))
        self._last_sample = decision_sample
        self.last_assignments = tuple(assignment_ids)
        self.last_assignment_is_new = tuple(new_flags)
        return tuple(directions), tuple(active)
