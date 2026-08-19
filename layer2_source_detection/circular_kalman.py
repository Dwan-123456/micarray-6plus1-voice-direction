from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from common.data_types import CandidateDirection

from .direction_smoothing import DirectionSmoothingError, circular_delta_deg


_SAMPLE_RATE = 48_000
_HOP_SAMPLES = 960


@dataclass(frozen=True, slots=True)
class CircularKalmanConfig:
    process_angle_std_deg: float = 1.5
    process_velocity_std_dps: float = 25.0
    measurement_std_deg: float = 5.0
    max_missed_windows: int = 150


@dataclass(slots=True)
class _FilterState:
    angle_unwrapped: float
    velocity_dps: float
    covariance: np.ndarray
    missed_windows: int = 0


class CircularKalmanFilter:
    """Circular Kalman filter keyed by IDs assigned by DirectionIdTracker."""

    backend = "circular_kalman_v1"

    def __init__(self, config: CircularKalmanConfig = CircularKalmanConfig()) -> None:
        self.config = config
        self._stream_key: tuple[str, int] | None = None
        self._last_decision_sample: int | None = None
        self._states: dict[int, _FilterState] = {}

    def reset(self) -> None:
        self._stream_key = None
        self._last_decision_sample = None
        self._states.clear()

    def retain_track_ids(self, track_ids: tuple[int, ...]) -> None:
        """Drop filter state immediately when the owning private ID expires."""
        retained = set(track_ids)
        for track_id in tuple(self._states):
            if track_id not in retained:
                del self._states[track_id]

    def update(
        self,
        session_id: str,
        stream_epoch: int,
        decision_sample: int,
        candidates: tuple[CandidateDirection, ...],
        track_ids: tuple[int, ...],
        process_noise_scale: float,
        measurement_noise_scale: float,
    ) -> tuple[CandidateDirection, ...]:
        candidates = tuple(candidates)
        track_ids = tuple(track_ids)
        if len(track_ids) != len(candidates) or len(set(track_ids)) != len(track_ids):
            raise DirectionSmoothingError("Kalman requires one unique private ID per candidate")
        if any(type(track_id) is not int or track_id <= 0 for track_id in track_ids):
            raise DirectionSmoothingError("Kalman private IDs must be positive integers")
        for name, value in (
            ("Q", process_noise_scale),
            ("R", measurement_noise_scale),
        ):
            if (
                not np.isfinite(value)
                or not 0.02 <= value <= 10.0
                or (value != 0.02 and abs(value * 10.0 - round(value * 10.0)) > 1.0e-9)
            ):
                raise DirectionSmoothingError(
                    f"Kalman {name} scale must be 0.02..10.00 in 0.1 steps (or the 0.02 minimum)"
                )
        key = (session_id, stream_epoch)
        if key != self._stream_key:
            self.reset()
            self._stream_key = key
        if self._last_decision_sample is None:
            elapsed_windows, dt = 1, 0.0
        else:
            delta = decision_sample - self._last_decision_sample
            if delta <= 0:
                raise DirectionSmoothingError("Kalman decision_sample must advance monotonically")
            elapsed_windows = max(1, int(np.ceil(delta / _HOP_SAMPLES)))
            dt = delta / _SAMPLE_RATE
        self._predict(dt, process_noise_scale)

        output: list[CandidateDirection] = []
        present_ids = set(track_ids)
        for track_id, state in tuple(self._states.items()):
            if track_id not in present_ids:
                state.missed_windows += elapsed_windows

        for track_id, candidate in zip(track_ids, candidates, strict=True):
            state = self._states.get(track_id)
            if state is None:
                self._states[track_id] = self._new_state(candidate.theta_deg)
                theta = candidate.theta_deg
            else:
                measurement_confidence = 2.0 if state.missed_windows > 0 else 1.0
                theta = self._correct(
                    state,
                    candidate.theta_deg,
                    measurement_noise_scale,
                    measurement_confidence,
                )
                state.missed_windows = 0
            output.append(replace(candidate, theta_deg=float(theta) % 360.0))
        self._last_decision_sample = decision_sample
        return tuple(output)

    def _predict(self, dt: float, process_noise_scale: float) -> None:
        if dt <= 0:
            return
        transition = np.asarray(((1.0, dt), (0.0, 1.0)), dtype=np.float64)
        process = np.diag((
            self.config.process_angle_std_deg**2,
            self.config.process_velocity_std_dps**2,
        )) * max(dt, _HOP_SAMPLES / _SAMPLE_RATE) * process_noise_scale
        for state in self._states.values():
            vector = transition @ np.asarray((state.angle_unwrapped, state.velocity_dps))
            state.angle_unwrapped, state.velocity_dps = map(float, vector)
            state.covariance = transition @ state.covariance @ transition.T + process

    def _correct(
        self,
        state: _FilterState,
        theta_deg: float,
        measurement_noise_scale: float,
        measurement_confidence: float,
    ) -> float:
        measurement = state.angle_unwrapped + circular_delta_deg(
            theta_deg, state.angle_unwrapped % 360.0
        )
        h = np.asarray((1.0, 0.0))
        measurement_variance = (
            self.config.measurement_std_deg**2
            * measurement_noise_scale
            / measurement_confidence
        )
        variance = float(h @ state.covariance @ h + measurement_variance)
        if not np.isfinite(variance) or variance <= 0:
            raise DirectionSmoothingError("invalid Kalman innovation variance")
        gain = state.covariance @ h / variance
        vector = np.asarray((state.angle_unwrapped, state.velocity_dps))
        vector += gain * (measurement - state.angle_unwrapped)
        state.angle_unwrapped, state.velocity_dps = map(float, vector)
        identity_minus_kh = np.eye(2) - np.outer(gain, h)
        state.covariance = (
            identity_minus_kh @ state.covariance @ identity_minus_kh.T
            + np.outer(gain, gain) * measurement_variance
        )
        if not np.isfinite(state.covariance).all() or not np.isfinite(vector).all():
            raise DirectionSmoothingError("non-finite Kalman state")
        return state.angle_unwrapped

    def _new_state(self, theta_deg: float) -> _FilterState:
        return _FilterState(
            float(theta_deg),
            0.0,
            np.diag((
                self.config.measurement_std_deg**2,
                self.config.process_velocity_std_dps**2,
            )).astype(np.float64),
        )

    def predicted_angles(self, track_ids: tuple[int, ...]) -> tuple[float, ...]:
        try:
            return tuple(float(self._states[track_id].angle_unwrapped % 360.0) for track_id in track_ids)
        except KeyError as exc:
            raise DirectionSmoothingError("prediction requested for unknown Kalman ID") from exc

    def forecast_angles(
        self,
        session_id: str,
        stream_epoch: int,
        decision_sample: int,
        track_ids: tuple[int, ...],
    ) -> dict[int, float]:
        """Read-only angle forecasts for ID association at a future sample."""

        track_ids = tuple(track_ids)
        if (session_id, stream_epoch) != self._stream_key or self._last_decision_sample is None:
            return {}
        delta = int(decision_sample) - self._last_decision_sample
        if delta <= 0:
            raise DirectionSmoothingError("Kalman forecast sample must advance monotonically")
        dt = delta / _SAMPLE_RATE
        try:
            return {
                track_id: float(
                    (self._states[track_id].angle_unwrapped + self._states[track_id].velocity_dps * dt)
                    % 360.0
                )
                for track_id in track_ids
            }
        except KeyError as exc:
            raise DirectionSmoothingError("forecast requested for unknown Kalman ID") from exc
