from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from common.config import ProjectConfig
from common.data_types import CandidateDirection


_SAMPLE_RATE = 48_000
_HOP_SAMPLES = 960
_COLLISION_EPS_DEG = 1.0e-6


def circular_delta_deg(measured_deg: float, reference_deg: float) -> float:
    """Signed shortest delta ``measured-reference`` in [-180, 180)."""
    return (float(measured_deg) - float(reference_deg) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True, slots=True)
class DirectionSmoothingConfig:
    enabled: bool = True
    backend: str = "circular_kalman_v1"
    association_gate_deg: float = 20.0
    process_angle_std_deg: float = 1.5
    process_velocity_std_dps: float = 25.0
    measurement_std_deg: float = 5.0
    max_missed_windows: int = 25

    @classmethod
    def from_project(cls, config: ProjectConfig) -> "DirectionSmoothingConfig":
        kalman = config.layer2.direction_kalman
        tracking = config.layer2.direction_id_tracking
        return cls(
            enabled=kalman.enabled and tracking.enabled,
            backend=kalman.backend,
            association_gate_deg=tracking.association_gate_deg,
            process_angle_std_deg=kalman.process_angle_std_deg,
            process_velocity_std_dps=kalman.process_velocity_std_dps,
            measurement_std_deg=kalman.measurement_std_deg,
            max_missed_windows=max(kalman.max_missed_windows, tracking.max_missed_windows),
        )

    def __post_init__(self) -> None:
        values = (
            self.association_gate_deg,
            self.process_angle_std_deg,
            self.process_velocity_std_dps,
            self.measurement_std_deg,
        )
        if self.backend != "circular_kalman_v1":
            raise ValueError(f"unsupported direction smoothing backend: {self.backend}")
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("direction smoothing parameters must be finite and positive")
        if self.association_gate_deg > 180.0 or self.max_missed_windows < 0:
            raise ValueError("direction smoothing gate or lifetime is invalid")


class DirectionSmoothingError(RuntimeError):
    """Declared numerical or temporal failure in the private L2 smoother."""


@dataclass(slots=True)
class _Track:
    track_id: int
    angle_unwrapped: float
    velocity_dps: float
    covariance: np.ndarray
    last_decision_sample: int
    hits: int = 1
    missed_windows: int = 0


class DirectionSmoother:
    """Private L2 association and circular Kalman filter.

    Track IDs never enter a public DTO. Candidate rank, scores, count and timing
    are inherited from the raw SRP result; only ``theta_deg`` is replaced.
    """

    def __init__(self, config: DirectionSmoothingConfig = DirectionSmoothingConfig()) -> None:
        self.config = config
        self._stream_key: tuple[str, int] | None = None
        self._last_decision_sample: int | None = None
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1

    @property
    def active_track_count(self) -> int:
        """Diagnostic count only; private IDs remain inaccessible."""
        return len(self._tracks)

    def reset(self) -> None:
        self._stream_key = None
        self._last_decision_sample = None
        self._tracks.clear()
        self._next_track_id = 1

    def update(
        self,
        session_id: str,
        stream_epoch: int,
        decision_sample: int,
        candidates: tuple[CandidateDirection, ...],
    ) -> tuple[CandidateDirection, ...]:
        candidates = tuple(candidates)
        self._validate_input(session_id, stream_epoch, decision_sample, candidates)
        if not self.config.enabled:
            self.reset()
            return candidates

        key = (session_id, stream_epoch)
        if key != self._stream_key:
            self.reset()
            self._stream_key = key

        elapsed_windows, dt = self._elapsed(decision_sample)
        self._predict_tracks(dt, decision_sample)
        self._expire_before_association(elapsed_windows)
        assignment = self._associate(candidates)

        output_angles: list[float] = []
        matched_track_ids = {track_id for track_id in assignment if track_id is not None}
        for track_id, track in tuple(self._tracks.items()):
            if track_id not in matched_track_ids:
                track.missed_windows += elapsed_windows
                if track.missed_windows > self.config.max_missed_windows:
                    del self._tracks[track_id]

        for candidate, track_id in zip(candidates, assignment, strict=True):
            if track_id is None:
                track = self._create_track(candidate)
                output_angles.append(float(candidate.theta_deg))
            else:
                track = self._tracks[track_id]
                self._correct(track, candidate.theta_deg)
                track.missed_windows = 0
                output_angles.append(track.angle_unwrapped % 360.0)

        output_angles = self._avoid_rank_collisions(output_angles, candidates)
        self._assert_finite_state()
        self._last_decision_sample = decision_sample
        return tuple(
            replace(candidate, theta_deg=float(theta_deg))
            for candidate, theta_deg in zip(candidates, output_angles, strict=True)
        )

    def _elapsed(self, decision_sample: int) -> tuple[int, float]:
        previous = self._last_decision_sample
        if previous is None:
            return 0, 0.0
        delta_samples = decision_sample - previous
        if delta_samples <= 0:
            raise DirectionSmoothingError("decision_sample must advance monotonically")
        elapsed_windows = max(1, int(np.ceil(delta_samples / _HOP_SAMPLES)))
        return elapsed_windows, delta_samples / _SAMPLE_RATE

    def _predict_tracks(self, dt: float, decision_sample: int) -> None:
        if dt <= 0.0:
            return
        transition = np.asarray(((1.0, dt), (0.0, 1.0)), dtype=np.float64)
        process_scale = max(dt, _HOP_SAMPLES / _SAMPLE_RATE)
        process = np.diag(
            (
                self.config.process_angle_std_deg**2,
                self.config.process_velocity_std_dps**2,
            )
        ) * process_scale
        for track in self._tracks.values():
            state = transition @ np.asarray(
                (track.angle_unwrapped, track.velocity_dps), dtype=np.float64
            )
            track.angle_unwrapped = float(state[0])
            track.velocity_dps = float(state[1])
            track.covariance = transition @ track.covariance @ transition.T + process
            track.last_decision_sample = decision_sample

    def _expire_before_association(self, elapsed_windows: int) -> None:
        skipped_windows = max(0, elapsed_windows - 1)
        for track_id, track in tuple(self._tracks.items()):
            if track.missed_windows + skipped_windows > self.config.max_missed_windows:
                del self._tracks[track_id]

    def _associate(
        self, candidates: tuple[CandidateDirection, ...]
    ) -> tuple[int | None, ...]:
        if not candidates:
            return ()
        track_ids = tuple(sorted(self._tracks))
        options: list[tuple[int | None, ...]] = []
        for candidate in candidates:
            feasible = tuple(
                track_id
                for track_id in track_ids
                if abs(
                    circular_delta_deg(
                        candidate.theta_deg,
                        self._tracks[track_id].angle_unwrapped % 360.0,
                    )
                )
                <= self.config.association_gate_deg
            )
            options.append((None, *feasible))

        best: tuple[int | None, ...] | None = None
        best_key: tuple[object, ...] | None = None
        none_rank = self._next_track_id + len(track_ids) + 1
        for assignment in product(*options):
            assigned = tuple(item for item in assignment if item is not None)
            if len(set(assigned)) != len(assigned):
                continue
            matches = len(assigned)
            cost = sum(
                abs(
                    circular_delta_deg(
                        candidates[index].theta_deg,
                        self._tracks[track_id].angle_unwrapped % 360.0,
                    )
                )
                for index, track_id in enumerate(assignment)
                if track_id is not None
            )
            key = (
                -matches,
                round(float(cost), 12),
                tuple(none_rank if item is None else item for item in assignment),
            )
            if best_key is None or key < best_key:
                best_key, best = key, assignment
        if best is None:
            raise DirectionSmoothingError("direction association failed")
        return best

    def _correct(self, track: _Track, theta_deg: float) -> None:
        measurement = track.angle_unwrapped + circular_delta_deg(
            theta_deg, track.angle_unwrapped % 360.0
        )
        observation = np.asarray((1.0, 0.0), dtype=np.float64)
        measurement_variance = self.config.measurement_std_deg**2
        innovation_variance = float(
            observation @ track.covariance @ observation + measurement_variance
        )
        if not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
            raise DirectionSmoothingError("invalid Kalman innovation variance")
        gain = track.covariance @ observation / innovation_variance
        innovation = measurement - track.angle_unwrapped
        state = np.asarray((track.angle_unwrapped, track.velocity_dps), dtype=np.float64)
        state += gain * innovation
        identity_minus_kh = np.eye(2, dtype=np.float64) - np.outer(gain, observation)
        covariance = (
            identity_minus_kh @ track.covariance @ identity_minus_kh.T
            + np.outer(gain, gain) * measurement_variance
        )
        track.angle_unwrapped, track.velocity_dps = map(float, state)
        track.covariance = covariance
        track.hits += 1

    def _create_track(self, candidate: CandidateDirection) -> _Track:
        track = _Track(
            self._next_track_id,
            float(candidate.theta_deg),
            0.0,
            np.diag(
                (
                    self.config.measurement_std_deg**2,
                    self.config.process_velocity_std_dps**2,
                )
            ).astype(np.float64),
            candidate.decision_sample,
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    @staticmethod
    def _avoid_rank_collisions(
        angles: list[float], candidates: tuple[CandidateDirection, ...]
    ) -> list[float]:
        result: list[float] = []
        for angle, candidate in zip(angles, candidates, strict=True):
            selected = float(angle) % 360.0
            if any(abs(circular_delta_deg(selected, existing)) <= _COLLISION_EPS_DEG for existing in result):
                selected = float(candidate.theta_deg)
            result.append(selected)
        return result

    def _assert_finite_state(self) -> None:
        for track in self._tracks.values():
            try:
                minimum_eigenvalue = float(
                    np.min(
                        np.linalg.eigvalsh(
                            (track.covariance + track.covariance.T) / 2.0
                        )
                    )
                )
            except np.linalg.LinAlgError as exc:
                raise DirectionSmoothingError("invalid Kalman covariance") from exc
            if (
                not np.isfinite(track.angle_unwrapped)
                or not np.isfinite(track.velocity_dps)
                or track.covariance.shape != (2, 2)
                or not np.isfinite(track.covariance).all()
                or minimum_eigenvalue < -1.0e-8
            ):
                raise DirectionSmoothingError("non-finite or invalid Kalman state")

    @staticmethod
    def _validate_input(
        session_id: str,
        stream_epoch: int,
        decision_sample: int,
        candidates: tuple[CandidateDirection, ...],
    ) -> None:
        if not session_id or stream_epoch < 0 or decision_sample < 0:
            raise ValueError("invalid direction smoothing identity")
        if len(candidates) > 2:
            raise ValueError("direction smoother accepts at most 2 candidates")
        identity = (session_id, stream_epoch, decision_sample)
        if any(
            (item.session_id, item.stream_epoch, item.decision_sample) != identity
            for item in candidates
        ):
            raise ValueError("candidate belongs to another session, epoch or decision")
