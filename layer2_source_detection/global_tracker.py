from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, log, sqrt

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.data_types import CandidateDirection, TrackedDirection


SAMPLE_RATE = 48_000


def _delta(theta: float, reference: float) -> float:
    """Shortest signed circular displacement from reference to theta."""

    return ((float(theta) - float(reference) + 180.0) % 360.0) - 180.0


def _nearest_unwrapped(theta: float, reference: float) -> float:
    return float(reference) + _delta(theta, reference)


@dataclass(frozen=True, slots=True)
class GlobalTrackerConfig:
    backend: str = "circular_imm_jpda_v1"
    association_gate_deg: float = 50.0
    association_chi2: float = 20.0
    max_velocity_dps: float = 60.0
    confirmation_observations: int = 3
    confirmation_window_samples: int = 200 * 48
    tentative_ttl_samples: int = 500 * 48
    coasting_ttl_samples: int = 2_000 * 48
    probability_detect: float = 0.85
    probability_track: float = 0.80
    probability_new: float = 0.10
    probability_false: float = 0.10
    minimum_association_probability: float = 0.20
    minimum_birth_probability: float = 0.45
    confirmation_existence_probability: float = 0.70
    deletion_existence_probability: float = 0.05
    # 0.97 retention per 20 ms, expressed on the absolute one-second timeline.
    survival_probability_per_second: float = 0.97**50
    measurement_std_deg: float = 5.0
    stationary_angle_std_deg: float = 0.35
    stationary_velocity_std_dps: float = 3.0
    stationary_velocity_half_life_seconds: float = 0.15
    moving_angle_std_deg: float = 1.25
    moving_velocity_std_dps: float = 15.0
    moving_velocity_half_life_seconds: float = 0.50
    stationary_to_moving_probability: float = 0.02
    moving_to_stationary_probability: float = 0.05
    prediction_freeze_std_deg: float = 25.0
    duplicate_birth_guard_deg: float = 15.0
    max_active_tracks: int = 4

    def __post_init__(self) -> None:
        if self.backend != "circular_imm_jpda_v1":
            raise ValueError("unsupported global tracker backend")
        if not 0 < self.association_gate_deg <= 180 or self.association_chi2 <= 0:
            raise ValueError("invalid circular association gate")
        if not 0 < self.max_velocity_dps <= 360:
            raise ValueError("invalid maximum direction velocity")
        if min(
            self.confirmation_observations,
            self.confirmation_window_samples,
            self.tentative_ttl_samples,
            self.coasting_ttl_samples,
            self.max_active_tracks,
        ) <= 0:
            raise ValueError("invalid tracker lifecycle configuration")
        probabilities = (
            self.probability_detect,
            self.probability_track,
            self.probability_new,
            self.probability_false,
            self.minimum_association_probability,
            self.minimum_birth_probability,
            self.confirmation_existence_probability,
            self.deletion_existence_probability,
            self.survival_probability_per_second,
            self.stationary_to_moving_probability,
            self.moving_to_stationary_probability,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("tracker probabilities must be in [0,1]")
        if abs(
            self.probability_track + self.probability_new + self.probability_false - 1.0
        ) > 1.0e-9:
            raise ValueError("track/new/false probabilities must sum to one")
        if min(
            self.measurement_std_deg,
            self.stationary_angle_std_deg,
            self.stationary_velocity_std_dps,
            self.stationary_velocity_half_life_seconds,
            self.moving_angle_std_deg,
            self.moving_velocity_std_dps,
            self.moving_velocity_half_life_seconds,
            self.prediction_freeze_std_deg,
            self.duplicate_birth_guard_deg,
        ) <= 0:
            raise ValueError("invalid IMM model configuration")


@dataclass(slots=True)
class _ModelState:
    mean: np.ndarray
    covariance: np.ndarray


@dataclass(slots=True)
class _Track:
    track_id: int
    first_seen: int
    last_observed: int
    models: list[_ModelState]
    model_probabilities: np.ndarray
    existence_probability: float
    raw_score: float
    normalized_score: float
    confirmation_samples: list[int] = field(default_factory=list)
    confirmed: bool = False
    observations: int = 1
    last_output_theta: float = 0.0


@dataclass(frozen=True, slots=True)
class TrackerDiagnostics:
    backend: str
    joint_hypotheses: int
    association_probabilities: tuple[tuple[float, ...], ...]
    new_probabilities: tuple[float, ...]
    false_probabilities: tuple[float, ...]
    existence_probabilities: tuple[tuple[int, float], ...]
    model_probabilities: tuple[tuple[int, float, float], ...]


class GlobalDirectionTracker:
    """Circular IMM-JPDA direction tracker using the 48 kHz sample timeline.

    IDs describe direction trajectories, not speaker identities. L4 feedback is
    accepted for interface compatibility but is deliberately not consumed by
    this tracker version.
    """

    backend = "circular_imm_jpda_v1"

    def __init__(self, config: GlobalTrackerConfig | None = None) -> None:
        self.config = config or GlobalTrackerConfig()
        self._session_id: str | None = None
        self._stream_epoch: int | None = None
        self._next_by_session: dict[str, int] = {}
        self._tracks: dict[int, _Track] = {}
        self._last_sample: int | None = None
        self.last_assignments: tuple[int, ...] = ()
        self.last_assignment_is_new: tuple[bool, ...] = ()
        self.last_association_probabilities: tuple[float, ...] = ()
        self.last_diagnostics = TrackerDiagnostics(self.backend, 0, (), (), (), (), ())
        self._feedback_audit: list[tuple[str, int, int, int, float, bool]] = []

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def reset(self, *, preserve_session_counters: bool = False) -> None:
        self._session_id = None
        self._stream_epoch = None
        self._tracks.clear()
        self._last_sample = None
        self.last_assignments = ()
        self.last_assignment_is_new = ()
        self.last_association_probabilities = ()
        self.last_diagnostics = TrackerDiagnostics(self.backend, 0, (), (), (), (), ())
        self._feedback_audit.clear()
        if not preserve_session_counters:
            self._next_by_session.clear()

    def prepare_stream(self, session_id: str, stream_epoch: int) -> None:
        if not session_id or stream_epoch < 0:
            raise ValueError("invalid direction stream identity")
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

    def _new_id(self, session_id: str) -> int:
        result = self._next_by_session.setdefault(session_id, 1)
        self._next_by_session[session_id] = result + 1
        return result

    def apply_voice_feedback(
        self,
        session_id: str,
        stream_epoch: int,
        decision_sample: int,
        track_id: int,
        probability: float,
        is_voice: bool,
    ) -> bool:
        """Preserve the L4 feedback branch without influencing tracking state."""

        if (
            (session_id, stream_epoch) != (self._session_id, self._stream_epoch)
            or decision_sample < 0
            or track_id not in self._tracks
            or not isfinite(probability)
            or not 0 <= probability <= 1
            or type(is_voice) is not bool
        ):
            return False
        self._feedback_audit.append(
            (session_id, stream_epoch, decision_sample, track_id, float(probability), is_voice)
        )
        del self._feedback_audit[:-64]
        return True

    def voice_confirmed_track_ids(
        self, session_id: str, stream_epoch: int, decision_sample: int
    ) -> frozenset[int]:
        self.prepare_stream(session_id, stream_epoch)
        self._expire_tracks(decision_sample)
        return frozenset()

    def has_live_tracks(self, session_id: str, stream_epoch: int, decision_sample: int) -> bool:
        self.prepare_stream(session_id, stream_epoch)
        self._expire_tracks(decision_sample)
        return bool(self._tracks)

    def _combined(self, track: _Track) -> tuple[np.ndarray, np.ndarray]:
        probability = track.model_probabilities / np.sum(track.model_probabilities)
        mean = sum(
            (probability[index] * model.mean for index, model in enumerate(track.models)),
            start=np.zeros(2, dtype=np.float64),
        )
        covariance = np.zeros((2, 2), dtype=np.float64)
        for weight, model in zip(probability, track.models, strict=True):
            difference = model.mean - mean
            covariance += weight * (model.covariance + np.outer(difference, difference))
        return mean, covariance

    def _transition(self, dt: float, half_life: float) -> np.ndarray:
        gamma = 2.0 ** (-dt / half_life)
        integrated = half_life / log(2.0) * (1.0 - gamma)
        return np.asarray(((1.0, integrated), (0.0, gamma)), dtype=np.float64)

    def _model_matrix(self) -> np.ndarray:
        cfg = self.config
        return np.asarray(
            (
                (1.0 - cfg.stationary_to_moving_probability, cfg.stationary_to_moving_probability),
                (cfg.moving_to_stationary_probability, 1.0 - cfg.moving_to_stationary_probability),
            ),
            dtype=np.float64,
        )

    def _predict_track(self, track: _Track, decision_sample: int) -> None:
        previous_sample = track.last_observed if self._last_sample is None else self._last_sample
        delta_samples = decision_sample - previous_sample
        if delta_samples < 0:
            raise ValueError("direction tracking sample cannot move backwards")
        if delta_samples == 0:
            return
        dt = delta_samples / SAMPLE_RATE
        transition_probabilities = self._model_matrix()
        prior = track.model_probabilities
        destination = prior @ transition_probabilities
        mixed: list[_ModelState] = []
        for target in range(2):
            weights = prior * transition_probabilities[:, target] / max(destination[target], 1.0e-12)
            mean = sum(
                (weights[source] * track.models[source].mean for source in range(2)),
                start=np.zeros(2, dtype=np.float64),
            )
            covariance = np.zeros((2, 2), dtype=np.float64)
            for source in range(2):
                difference = track.models[source].mean - mean
                covariance += weights[source] * (
                    track.models[source].covariance + np.outer(difference, difference)
                )
            mixed.append(_ModelState(mean, covariance))
        model_parameters = (
            (
                self.config.stationary_velocity_half_life_seconds,
                self.config.stationary_angle_std_deg,
                self.config.stationary_velocity_std_dps,
            ),
            (
                self.config.moving_velocity_half_life_seconds,
                self.config.moving_angle_std_deg,
                self.config.moving_velocity_std_dps,
            ),
        )
        for index, (half_life, angle_std, velocity_std) in enumerate(model_parameters):
            transition = self._transition(dt, half_life)
            process = np.diag((angle_std**2, velocity_std**2)) * max(dt, 0.02)
            mean = transition @ mixed[index].mean
            mean[1] = np.clip(mean[1], -self.config.max_velocity_dps, self.config.max_velocity_dps)
            covariance = transition @ mixed[index].covariance @ transition.T + process
            track.models[index] = _ModelState(mean, covariance)
        track.model_probabilities = destination / np.sum(destination)
        track.existence_probability *= self.config.survival_probability_per_second**dt
        self._rebase(track)

    def _likelihood(self, track: _Track, candidate: CandidateDirection) -> tuple[float, np.ndarray]:
        likelihoods = np.zeros(2, dtype=np.float64)
        measurement_variance = self.config.measurement_std_deg**2
        for index, model in enumerate(track.models):
            innovation = _delta(candidate.theta_deg, model.mean[0])
            variance = float(model.covariance[0, 0] + measurement_variance)
            if (
                abs(innovation) <= self.config.association_gate_deg
                and innovation**2 / max(variance, 1.0e-12) <= self.config.association_chi2
            ):
                # A dimensionless gated likelihood is required here because
                # JPDA also compares it with explicit new/false class priors.
                # Using the Gaussian density (1/degree) would make every valid
                # five-degree observation look less likely than a false alarm.
                likelihoods[index] = exp(-0.5 * innovation**2 / variance)
        mixture = float(track.model_probabilities @ likelihoods)
        return mixture * max(0.05, candidate.normalized_score), likelihoods

    def _joint_probabilities(
        self, tracks: tuple[_Track, ...], candidates: tuple[CandidateDirection, ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[np.ndarray]]:
        rows, columns = len(tracks), len(candidates)
        likelihood = np.zeros((rows, columns), dtype=np.float64)
        model_likelihoods: list[np.ndarray] = []
        for row, track in enumerate(tracks):
            row_models = np.zeros((columns, 2), dtype=np.float64)
            for column, candidate in enumerate(candidates):
                likelihood[row, column], row_models[column] = self._likelihood(track, candidate)
            model_likelihoods.append(row_models)

        hypotheses: list[tuple[float, tuple[int | None, ...], frozenset[int]]] = []

        def visit(row: int, used: frozenset[int], assignment: tuple[int | None, ...], weight: float) -> None:
            if row == rows:
                for column, candidate in enumerate(candidates):
                    if column not in used:
                        quality = max(0.05, candidate.normalized_score)
                        weight *= self.config.probability_new * quality + self.config.probability_false
                hypotheses.append((weight, assignment, used))
                return
            track = tracks[row]
            missed = max(
                1.0e-12,
                1.0 - track.existence_probability * self.config.probability_detect,
            )
            visit(row + 1, used, assignment + (None,), weight * missed)
            for column in range(columns):
                if column in used or likelihood[row, column] <= 0:
                    continue
                assigned = (
                    self.config.probability_track
                    * track.existence_probability
                    * self.config.probability_detect
                    * likelihood[row, column]
                )
                visit(row + 1, used | {column}, assignment + (column,), weight * assigned)

        visit(0, frozenset(), (), 1.0)
        total = sum(weight for weight, _, _ in hypotheses)
        association = np.zeros((rows, columns), dtype=np.float64)
        new_probability = np.zeros(columns, dtype=np.float64)
        false_probability = np.zeros(columns, dtype=np.float64)
        if total <= 0 or not isfinite(total):
            return association, new_probability, np.ones(columns), len(hypotheses), model_likelihoods
        for weight, assignment, used in hypotheses:
            posterior = weight / total
            for row, column in enumerate(assignment):
                if column is not None:
                    association[row, column] += posterior
            for column, candidate in enumerate(candidates):
                if column in used:
                    continue
                quality = max(0.05, candidate.normalized_score)
                denominator = self.config.probability_new * quality + self.config.probability_false
                new_probability[column] += posterior * self.config.probability_new * quality / denominator
                false_probability[column] += posterior * self.config.probability_false / denominator
        return association, new_probability, false_probability, len(hypotheses), model_likelihoods

    def _rescue_nearby_associations(
        self,
        tracks: tuple[_Track, ...],
        candidates: tuple[CandidateDirection, ...],
        association: np.ndarray,
        observed_track_by_candidate: dict[int, int],
        observed_candidate_by_track: dict[int, int],
    ) -> None:
        """Keep a nearby observation on an existing ID when JPDA is uncertain.

        The probabilistic gate is intentionally allowed to reject an abrupt
        observation.  That rejection must not create a second ID inside the
        physical 50-degree association region, though.  Rescue assignments are
        one-to-one and still enter the IMM measurement update, so the published
        angle is the filtered posterior rather than the raw candidate angle.
        """

        available_rows = [
            row for row, track in enumerate(tracks)
            if track.track_id not in observed_candidate_by_track
        ]
        available_columns = [
            column for column in range(len(candidates))
            if column not in observed_track_by_candidate
        ]
        if not available_rows or not available_columns:
            return

        invalid_cost = self.config.association_gate_deg + 1.0
        costs = np.full(
            (len(available_rows), len(available_columns)), invalid_cost, dtype=np.float64
        )
        for local_row, row in enumerate(available_rows):
            predicted_theta = self._combined(tracks[row])[0][0]
            for local_column, column in enumerate(available_columns):
                distance = abs(_delta(candidates[column].theta_deg, predicted_theta))
                if distance <= self.config.association_gate_deg:
                    costs[local_row, local_column] = distance

        rescue_rows, rescue_columns = linear_sum_assignment(costs)
        for local_row, local_column in zip(rescue_rows, rescue_columns, strict=True):
            if costs[local_row, local_column] > self.config.association_gate_deg:
                continue
            row = available_rows[int(local_row)]
            column = available_columns[int(local_column)]
            track_id = tracks[row].track_id
            # Replace the weak JPDA row/column with a deterministic measurement
            # association. _update_track below applies the ordinary IMM/Kalman
            # gain; it never copies this raw candidate directly to theta_deg.
            association[row, :] = 0.0
            association[:, column] = 0.0
            association[row, column] = 1.0
            observed_track_by_candidate[column] = track_id
            observed_candidate_by_track[track_id] = column

    def _update_track(
        self,
        track: _Track,
        candidates: tuple[CandidateDirection, ...],
        beta: np.ndarray,
        model_likelihoods: np.ndarray,
        decision_sample: int,
        observed: bool,
    ) -> None:
        detected_probability = float(np.sum(beta))
        missed_probability = max(0.0, 1.0 - detected_probability)
        measurement_variance = self.config.measurement_std_deg**2
        prior_model_probability = track.model_probabilities.copy()
        model_evidence = np.full(2, missed_probability, dtype=np.float64)
        for model_index, model in enumerate(track.models):
            innovations = np.asarray(
                [_delta(candidate.theta_deg, model.mean[0]) for candidate in candidates],
                dtype=np.float64,
            )
            variance = float(model.covariance[0, 0] + measurement_variance)
            gain = model.covariance[:, 0] / max(variance, 1.0e-12)
            weighted_innovation = float(beta @ innovations) if len(beta) else 0.0
            corrected_mean = model.mean + gain * weighted_innovation
            corrected_mean[1] = np.clip(
                corrected_mean[1], -self.config.max_velocity_dps, self.config.max_velocity_dps
            )
            corrected_covariance = model.covariance - np.outer(gain, model.covariance[0])
            spread = float(beta @ (innovations**2)) - weighted_innovation**2 if len(beta) else 0.0
            posterior_covariance = (
                missed_probability * model.covariance
                + detected_probability * corrected_covariance
                + np.outer(gain, gain) * max(0.0, spread)
            )
            model.mean = corrected_mean
            model.covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
            if len(beta):
                model_evidence[model_index] += float(beta @ model_likelihoods[:, model_index])
        posterior_models = prior_model_probability * np.maximum(model_evidence, 1.0e-12)
        track.model_probabilities = posterior_models / np.sum(posterior_models)

        prior_existence = track.existence_probability
        if track.confirmed and not observed:
            # Prediction already applies the configured time-based survival
            # decay. Do not additionally apply the per-window Bayesian miss
            # collapse: confirmed tracks must remain recoverable throughout
            # their two-second absolute-sample coasting lease.
            track.existence_probability = prior_existence
        else:
            missed_existence = (
                prior_existence * (1.0 - self.config.probability_detect)
                / max(1.0 - prior_existence * self.config.probability_detect, 1.0e-12)
            )
            posterior_existence = min(
                0.999, detected_probability + missed_probability * missed_existence
            )
            track.existence_probability = (
                max(prior_existence, posterior_existence)
                if track.confirmed and observed
                else posterior_existence
            )
        if detected_probability > 0 and candidates:
            weights = beta / detected_probability
            anchor, _ = self._combined(track)
            measured = anchor[0] + float(
                weights @ np.asarray([_delta(item.theta_deg, anchor[0]) for item in candidates])
            )
            track.last_output_theta = measured
            track.raw_score = float(weights @ np.asarray([item.raw_score for item in candidates]))
            track.normalized_score = float(
                weights @ np.asarray([item.normalized_score for item in candidates])
            )
        self._rebase(track)

    def _rebase(self, track: _Track) -> None:
        combined, _ = self._combined(track)
        offset = 360.0 * round(combined[0] / 360.0)
        if offset:
            for model in track.models:
                model.mean[0] -= offset
            track.last_output_theta -= offset

    def _birth(self, session_id: str, decision_sample: int, candidate: CandidateDirection) -> _Track:
        theta = float(candidate.theta_deg)
        covariance = np.diag((self.config.measurement_std_deg**2, 20.0**2)).astype(np.float64)
        return _Track(
            self._new_id(session_id),
            decision_sample,
            decision_sample,
            [
                _ModelState(np.asarray((theta, 0.0), dtype=np.float64), covariance.copy()),
                _ModelState(np.asarray((theta, 0.0), dtype=np.float64), covariance.copy()),
            ],
            np.asarray((0.75, 0.25), dtype=np.float64),
            max(0.55, self.config.minimum_birth_probability),
            candidate.raw_score,
            candidate.normalized_score,
            [decision_sample],
            False,
            1,
            theta,
        )

    def _expire_tracks(self, decision_sample: int) -> None:
        for track_id, track in tuple(self._tracks.items()):
            age = decision_sample - track.last_observed
            ttl = self.config.coasting_ttl_samples if track.confirmed else self.config.tentative_ttl_samples
            ttl_expired = age >= ttl
            probability_expired = (
                not track.confirmed
                and track.existence_probability < self.config.deletion_existence_probability
            )
            if ttl_expired or probability_expired:
                del self._tracks[track_id]

    def _confirm(self, track: _Track, decision_sample: int, observed: bool) -> None:
        cutoff = decision_sample - self.config.confirmation_window_samples
        track.confirmation_samples[:] = [sample for sample in track.confirmation_samples if sample >= cutoff]
        if observed and (
            not track.confirmation_samples or track.confirmation_samples[-1] != decision_sample
        ):
            track.confirmation_samples.append(decision_sample)
            track.observations += 1
        if (
            len(track.confirmation_samples) >= self.config.confirmation_observations
            and track.existence_probability >= self.config.confirmation_existence_probability
        ):
            track.confirmed = True

    def _make_direction(
        self,
        track: _Track,
        decision_sample: int,
        window_id: int,
        doa_start_sample: int,
        doa_end_sample: int,
        rank: int,
        measured_theta: float | None,
        is_new: bool,
    ) -> TrackedDirection:
        mean, covariance = self._combined(track)
        output_theta = float(mean[0] % 360.0)
        if (
            measured_theta is None
            and sqrt(max(0.0, float(covariance[0, 0]))) > self.config.prediction_freeze_std_deg
        ):
            output_theta = float(track.last_output_theta % 360.0)
        elif measured_theta is not None:
            track.last_output_theta = mean[0]
        state = "confirmed" if measured_theta is not None else "coasting"
        if not track.confirmed:
            state = "tentative"
        return TrackedDirection(
            self._session_id or "",
            self._stream_epoch or 0,
            window_id,
            decision_sample,
            doa_start_sample,
            doa_end_sample,
            track.track_id,
            rank,
            None if measured_theta is None else float(measured_theta % 360.0),
            output_theta,
            track.raw_score,
            float(np.clip(track.normalized_score, 0.0, 1.0)),
            state,
            measured_theta is not None,
            is_new,
            track.first_seen,
            track.last_observed,
            decision_sample - track.last_observed,
            True,
            False,
        )

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
        allow_births: bool = True,
    ) -> tuple[tuple[TrackedDirection, ...], tuple[TrackedDirection, ...]]:
        self.prepare_stream(session_id, stream_epoch)
        if self._last_sample is not None and decision_sample <= self._last_sample:
            raise ValueError("direction tracking sample must advance monotonically")
        doa_end_sample = decision_sample if doa_end_sample is None else doa_end_sample
        candidates = tuple(candidates)
        self._expire_tracks(decision_sample)
        tracks = tuple(self._tracks[track_id] for track_id in sorted(self._tracks))
        for track in tracks:
            self._predict_track(track, decision_sample)
        association, p_new, p_false, hypothesis_count, model_likelihoods = self._joint_probabilities(
            tracks, candidates
        )

        observed_track_by_candidate: dict[int, int] = {}
        observed_candidate_by_track: dict[int, int] = {}
        if len(tracks) and len(candidates):
            rows, columns = linear_sum_assignment(-association)
            for row, column in zip(rows, columns, strict=True):
                probability = float(association[row, column])
                if probability >= self.config.minimum_association_probability:
                    observed_track_by_candidate[int(column)] = tracks[int(row)].track_id
                    observed_candidate_by_track[tracks[int(row)].track_id] = int(column)
        self._rescue_nearby_associations(
            tracks,
            candidates,
            association,
            observed_track_by_candidate,
            observed_candidate_by_track,
        )

        for row, track in enumerate(tracks):
            observed = track.track_id in observed_candidate_by_track
            self._update_track(
                track,
                candidates,
                association[row],
                model_likelihoods[row],
                decision_sample,
                observed,
            )
            if observed:
                track.last_observed = decision_sample
            self._confirm(track, decision_sample, observed)

        new_track_ids: set[int] = set()
        for column, candidate in enumerate(candidates):
            if column in observed_track_by_candidate or not allow_births:
                continue
            nearest = min(
                (
                    abs(_delta(candidate.theta_deg, self._combined(track)[0][0]))
                    for track in self._tracks.values()
                ),
                default=float("inf"),
            )
            birth_exclusion_deg = max(
                self.config.duplicate_birth_guard_deg,
                self.config.association_gate_deg,
            )
            if (
                p_new[column] < self.config.minimum_birth_probability
                or nearest <= birth_exclusion_deg
            ):
                continue
            if len(self._tracks) >= self.config.max_active_tracks:
                victims = sorted(
                    (track for track in self._tracks.values() if not track.confirmed),
                    key=lambda item: (item.existence_probability, item.last_observed, item.track_id),
                )
                if not victims:
                    continue
                del self._tracks[victims[0].track_id]
            track = self._birth(session_id, decision_sample, candidate)
            self._tracks[track.track_id] = track
            new_track_ids.add(track.track_id)
            observed_track_by_candidate[column] = track.track_id
            observed_candidate_by_track[track.track_id] = column

        self._expire_tracks(decision_sample)
        active_items: list[TrackedDirection] = []
        observed_items: list[TrackedDirection] = []
        ordered_tracks = sorted(
            self._tracks.values(),
            key=lambda item: (-int(item.confirmed), -item.existence_probability, item.track_id),
        )
        for rank, track in enumerate(ordered_tracks, 1):
            column = observed_candidate_by_track.get(track.track_id)
            measured = None if column is None else candidates[column].theta_deg
            item = self._make_direction(
                track,
                decision_sample,
                window_id,
                doa_start_sample,
                doa_end_sample,
                rank,
                measured,
                track.track_id in new_track_ids,
            )
            active_items.append(item)
            if measured is not None:
                observed_items.append(item)

        self.last_assignments = tuple(
            observed_track_by_candidate[column]
            for column in sorted(observed_track_by_candidate)
        )
        self.last_assignment_is_new = tuple(
            track_id in new_track_ids for track_id in self.last_assignments
        )
        self.last_association_probabilities = tuple(
            float(np.max(association[:, column])) if association.shape[0] else 0.0
            for column in sorted(observed_track_by_candidate)
        )
        self.last_diagnostics = TrackerDiagnostics(
            self.backend,
            hypothesis_count,
            tuple(tuple(float(value) for value in row) for row in association),
            tuple(float(value) for value in p_new),
            tuple(float(value) for value in p_false),
            tuple((track.track_id, track.existence_probability) for track in ordered_tracks),
            tuple(
                (track.track_id, float(track.model_probabilities[0]), float(track.model_probabilities[1]))
                for track in ordered_tracks
            ),
        )
        self._last_sample = decision_sample
        return tuple(observed_items), tuple(active_items)
