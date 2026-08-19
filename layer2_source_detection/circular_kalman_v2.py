from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from common.data_types import CandidateDirection

from .direction_smoothing import DirectionSmoothingError, circular_delta_deg


_SAMPLE_RATE = 48_000
_MIN_DT = 960 / _SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class CircularKalmanV2Config:
    process_angle_std_deg: float = 1.5
    process_velocity_std_dps: float = 25.0
    measurement_std_deg: float = 5.0
    velocity_half_life_seconds: float = 0.5
    max_velocity_dps: float = 60.0
    prediction_freeze_std_deg: float = float("inf")
    innovation_gate_deg: float = 20.0


@dataclass(slots=True)
class _State:
    angle: float
    velocity: float
    covariance: np.ndarray
    last_trustworthy_angle: float
    missed: bool = False


class CircularKalmanFilterV2:
    backend = "damped_circular_kalman_v2"

    def __init__(self, config: CircularKalmanV2Config = CircularKalmanV2Config()) -> None:
        self.config = config
        self._stream_key: tuple[str, int] | None = None
        self._last_sample: int | None = None
        self._states: dict[int, _State] = {}

    def reset(self) -> None:
        self._stream_key = None
        self._last_sample = None
        self._states.clear()

    def retain_track_ids(self, track_ids: tuple[int, ...]) -> None:
        keep = set(track_ids)
        for track_id in tuple(self._states):
            if track_id not in keep:
                del self._states[track_id]

    def update(
        self, session_id: str, stream_epoch: int, decision_sample: int,
        candidates: tuple[CandidateDirection, ...], track_ids: tuple[int, ...],
        process_noise_scale: float, measurement_noise_scale: float,
        ready_track_ids: tuple[int, ...] | None = None,
    ) -> tuple[CandidateDirection, ...]:
        if len(candidates) != len(track_ids) or len(set(track_ids)) != len(track_ids):
            raise DirectionSmoothingError("Kalman V2 requires aligned unique IDs")
        key = (session_id, stream_epoch)
        if key != self._stream_key:
            self.reset()
            self._stream_key = key
        if self._last_sample is None:
            dt = 0.0
        else:
            if decision_sample <= self._last_sample:
                raise DirectionSmoothingError("Kalman V2 sample must advance")
            dt = (decision_sample - self._last_sample) / _SAMPLE_RATE
        self._predict(dt, process_noise_scale)
        ready = set(track_ids if ready_track_ids is None else ready_track_ids)
        present = set(track_ids)
        for track_id, state in self._states.items():
            if track_id not in present:
                state.missed = True
        output: list[CandidateDirection] = []
        for candidate, track_id in zip(candidates, track_ids, strict=True):
            if track_id not in ready:
                output.append(candidate)
                continue
            state = self._states.get(track_id)
            if state is None:
                state = _State(
                    candidate.theta_deg, 0.0,
                    np.diag((self.config.measurement_std_deg ** 2,
                             self.config.process_velocity_std_dps ** 2)).astype(np.float64),
                    candidate.theta_deg,
                )
                self._states[track_id] = state
            else:
                innovation = circular_delta_deg(candidate.theta_deg, state.angle % 360.0)
                if abs(innovation) <= self.config.innovation_gate_deg:
                    confidence = 2.0 if state.missed else 1.0
                    self._correct(state, innovation, measurement_noise_scale, confidence)
                    state.last_trustworthy_angle = state.angle
            state.missed = False
            output.append(replace(candidate, theta_deg=float(state.angle % 360.0)))
        self._last_sample = decision_sample
        return tuple(output)

    def forecast_angles(
        self, session_id: str, stream_epoch: int, decision_sample: int,
        track_ids: tuple[int, ...]
    ) -> dict[int, float]:
        if (session_id, stream_epoch) != self._stream_key or self._last_sample is None:
            return {}
        dt = (decision_sample - self._last_sample) / _SAMPLE_RATE
        if dt <= 0:
            raise DirectionSmoothingError("Kalman V2 forecast sample must advance")
        return {
            track_id: float(self._forecast_state(self._states[track_id], dt)[0] % 360.0)
            for track_id in track_ids if track_id in self._states
        }

    def predicted_angles(self, track_ids: tuple[int, ...]) -> tuple[float, ...]:
        return tuple(float(self._states[track_id].angle % 360.0)
                     for track_id in track_ids if track_id in self._states)

    def _forecast_state(self, state: _State, dt: float) -> tuple[float, float, np.ndarray]:
        gamma = 2.0 ** (-dt / self.config.velocity_half_life_seconds)
        integration = self.config.velocity_half_life_seconds / np.log(2.0) * (1.0 - gamma)
        transition = np.asarray(((1.0, integration), (0.0, gamma)), dtype=np.float64)
        vector = transition @ np.asarray((state.angle, state.velocity))
        vector[1] = np.clip(vector[1], -self.config.max_velocity_dps, self.config.max_velocity_dps)
        covariance = transition @ state.covariance @ transition.T
        return float(vector[0]), float(vector[1]), covariance

    def _predict(self, dt: float, q_scale: float) -> None:
        if dt <= 0:
            return
        for state in self._states.values():
            angle, velocity, covariance = self._forecast_state(state, dt)
            process = np.diag((self.config.process_angle_std_deg ** 2,
                               self.config.process_velocity_std_dps ** 2)) * max(dt, _MIN_DT) * q_scale
            covariance = covariance + process
            if state.missed and np.sqrt(max(float(covariance[0, 0]), 0.0)) > self.config.prediction_freeze_std_deg:
                angle = state.last_trustworthy_angle
            state.angle, state.velocity, state.covariance = angle, velocity, covariance

    def _correct(self, state: _State, innovation: float, r_scale: float, confidence: float) -> None:
        h = np.asarray((1.0, 0.0))
        r = self.config.measurement_std_deg ** 2 * r_scale / confidence
        s = float(h @ state.covariance @ h + r)
        if not np.isfinite(s) or s <= 0:
            raise DirectionSmoothingError("invalid Kalman V2 innovation variance")
        gain = state.covariance @ h / s
        vector = np.asarray((state.angle, state.velocity)) + gain * innovation
        vector[1] = np.clip(vector[1], -self.config.max_velocity_dps, self.config.max_velocity_dps)
        identity_minus = np.eye(2) - np.outer(gain, h)
        covariance = identity_minus @ state.covariance @ identity_minus.T + np.outer(gain, gain) * r
        if not np.isfinite(vector).all() or not np.isfinite(covariance).all():
            raise DirectionSmoothingError("non-finite Kalman V2 state")
        state.angle, state.velocity, state.covariance = float(vector[0]), float(vector[1]), covariance
