from __future__ import annotations

from dataclasses import dataclass, field

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
    confirmation_observations: int = 6
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
    confirmation_samples: list[int] = field(default_factory=list)
    confirmed: bool = False
    filtered_theta: float | None = None
    filtered_velocity_dps: float = 0.0
    last_voice_sample: int | None = None
    last_voice_probability: float | None = None
    noise_interference: bool = False
    noise_voice_recovery_samples: list[int] = field(default_factory=list)


class GlobalDirectionTracker:
    """Deterministic sample-time lifecycle with session-scoped non-reused IDs."""

    backend = "global_assignment_v1"
    voice_absence_noise_samples = 3 * 48_000
    max_active_tracks = 4

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
        """Return live tracking-confirmed IDs with positive L4 voice evidence.

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
            and track.last_voice_sample is not None
            and not track.noise_interference
        )

    def _expire_tracks(self, decision_sample: int) -> None:
        ttl = self.config.coasting_ttl_samples
        for track_id, track in tuple(self._tracks.items()):
            if decision_sample - track.last_observed > ttl:
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
        """Apply one delayed, authoritative L4 result to an existing ID."""

        if (session_id, stream_epoch) != (self._session_id, self._stream_epoch):
            return False
        track = self._tracks.get(track_id)
        if track is None or decision_sample < track.first_seen:
            return False
        probability = float(probability)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("L4 voice feedback probability must be in [0,1]")
        track.last_voice_probability = probability
        if is_voice:
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
        candidates = tuple(candidates)
        # Noise markers are excluded from the exclusive Hungarian rows so a
        # nearby normal ID can never be merged into a stationary noise ID.
        track_ids = tuple(sorted(
            track_id for track_id, track in self._tracks.items()
            if not track.noise_interference
        ))
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
        noise_ids = tuple(sorted(
            track_id for track_id, track in self._tracks.items()
            if track.noise_interference
        ))
        used_noise_ids: set[int] = set()
        for index, candidate in enumerate(candidates):
            if index in assigned:
                continue
            normal_nearby = any(
                abs(_delta(candidate.theta_deg, self._tracks[track_id].unwrapped_theta))
                <= self.config.association_gate_deg
                for track_id in track_ids
            )
            if normal_nearby:
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
            if is_new:
                if index not in allowed_births:
                    continue
                track_id = self._new_id(session_id)
                track = _Track(
                    track_id, candidate.theta_deg, 0.0, decision_sample,
                    decision_sample, 1, candidate.raw_score, candidate.normalized_score,
                )
                self._tracks[track_id] = track
            else:
                track = self._tracks[track_id]
                elapsed = self._update_observation(
                    track, candidate, decision_sample, self.config.max_velocity_dps
                )
            self._update_confirmation(track, decision_sample)
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
                # Prediction is an explicit Kalman-only output feature.  If
                # Kalman was enabled after the last observation, initialize
                # its forecast from the last real ID position instead of
                # inventing a discontinuity or changing the trajectory ID.
                base_theta = (
                    track.unwrapped_theta
                    if track.filtered_theta is None
                    else track.filtered_theta
                )
                forecast_velocity = (
                    track.velocity_dps
                    if track.filtered_theta is None
                    else track.filtered_velocity_dps
                )
                theta = (base_theta + forecast_velocity * missed / 48_000.0) % 360.0
            else:
                # ID lifetime and association remain active with Kalman OFF,
                # but the published/coasting angle is a zero-order hold of
                # the last real observation.  Raw velocity may still be used
                # internally by the global assignment and is never exposed
                # as an OFF-mode angle prediction.
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
