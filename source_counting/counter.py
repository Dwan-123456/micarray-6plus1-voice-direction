from __future__ import annotations

from collections import Counter, deque
from time import monotonic, perf_counter

import numpy as np

from common.data_types import DecisionWindow
from common.geometry import MicGeometry

from .configuration import SourceCounterConfig
from .interface import SourceCountSnapshot


class IncrementalGccPhatSourceCounter:
    """Rolling 0/1/2 prominent-direction counter used between Gate and MUSIC.

    The first window builds fifteen 50%-overlapped STFT frame features.  A
    normal 20 ms successor computes only the two newly available frames and
    removes the two oldest frames.  A short queue gap is caught up from only
    the frames that were not processed previously; the full 160 ms context is
    rebuilt only for a new stream or a gap larger than the retained context.
    """

    algorithm_version = "incremental_gcc_phat_deemphasis_v1"

    def __init__(self, config: SourceCounterConfig) -> None:
        self.config = config
        self._analysis_window = np.hanning(config.win_length + 1)[:-1].astype(np.float64)
        all_frequencies = np.fft.rfftfreq(config.n_fft, 1.0 / 48_000.0)
        self._frequency_indices = np.flatnonzero(
            (all_frequencies >= config.frequency_min_hz)
            & (all_frequencies <= config.frequency_max_hz)
        )
        self._frequencies_hz = all_frequencies[self._frequency_indices]
        self._pair_left, self._pair_right = np.triu_indices(7, k=1)
        self._max_frames = 1 + (
            config.context_ms * 48 - config.win_length
        ) // config.hop_length
        self._frame_cross: deque[np.ndarray] = deque()
        self._frame_power: deque[float] = deque()
        self._cross_sum: np.ndarray | None = None
        self._power_sum = 0.0
        self._stream_key: tuple[str, int] | None = None
        self._last_sample: int | None = None
        self._geometry_key: tuple[object, ...] | None = None
        self._lags_samples: np.ndarray | None = None
        self._lag_phase: np.ndarray | None = None
        self._delay_samples: np.ndarray | None = None
        self._lag_lower: np.ndarray | None = None
        self._lag_fraction: np.ndarray | None = None
        self._raw_history: deque[int] = deque(maxlen=config.persistence_window_frames)
        self._stable_count: int | None = None
        self.last_processing_ms = 0.0
        self.last_update_kind = "idle"
        self.last_added_frames = 0
        self.last_removed_frames = 0
        self.last_raw_count: int | None = None
        self.last_rms_dbfs = -np.inf
        self.last_first_peak = 0.0
        self.last_first_peak_z = 0.0
        self.last_residual_peak = 0.0
        self.last_residual_peak_z = 0.0
        self.last_residual_ratio = 0.0
        self.last_coactive_frames = 0

    def reset(self) -> None:
        self._frame_cross.clear()
        self._frame_power.clear()
        self._cross_sum = None
        self._power_sum = 0.0
        self._stream_key = None
        self._last_sample = None
        self._raw_history.clear()
        self._stable_count = None
        self.last_processing_ms = 0.0
        self.last_update_kind = "idle"
        self.last_added_frames = 0
        self.last_removed_frames = 0
        self.last_raw_count = None
        self.last_rms_dbfs = -np.inf
        self.last_first_peak = 0.0
        self.last_first_peak_z = 0.0
        self.last_residual_peak = 0.0
        self.last_residual_peak_z = 0.0
        self.last_residual_ratio = 0.0
        self.last_coactive_frames = 0

    def _prepare_geometry(self, geometry: MicGeometry) -> None:
        key = (
            geometry.version,
            geometry.speed_of_sound_mps,
            tuple(np.asarray(geometry.positions_m).ravel()),
        )
        if key == self._geometry_key:
            return
        theta = np.deg2rad(np.arange(360, dtype=np.float64))
        directions = np.stack((np.cos(theta), np.sin(theta)), axis=1)
        delays_seconds = -(
            directions @ np.asarray(geometry.positions_m, dtype=np.float64).T
        ) / geometry.speed_of_sound_mps
        pair_delays = (
            delays_seconds[:, self._pair_left] - delays_seconds[:, self._pair_right]
        ).T
        delay_samples = pair_delays * 48_000.0
        maximum_delay = int(np.ceil(np.max(np.abs(delay_samples)))) + 2
        oversampling = self.config.lag_oversampling
        lags = np.arange(
            -maximum_delay * oversampling,
            maximum_delay * oversampling + 1,
            dtype=np.float64,
        ) / oversampling
        fractional_index = (delay_samples - lags[0]) * oversampling
        lower = np.floor(fractional_index).astype(np.int64)
        lower = np.clip(lower, 0, lags.size - 2)
        fraction = np.clip(fractional_index - lower, 0.0, 1.0)
        self._lags_samples = lags
        self._lag_phase = np.exp(
            2j * np.pi * self._frequencies_hz[:, None] * lags[None, :] / 48_000.0
        )
        self._delay_samples = delay_samples
        self._lag_lower = lower
        self._lag_fraction = fraction
        self._geometry_key = key

    def _frame_feature(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        physical = np.asarray(frame[:, :7], dtype=np.float64)
        physical = physical - physical.mean(axis=0, keepdims=True)
        spectrum = np.fft.rfft(
            physical * self._analysis_window[:, None],
            n=self.config.n_fft,
            axis=0,
        )[self._frequency_indices]
        left = spectrum[:, self._pair_left]
        right = spectrum[:, self._pair_right]
        denominator = np.abs(left) * np.abs(right)
        scale = max(float(np.max(denominator)), 1.0)
        cross = left * right.conj() / np.maximum(denominator, scale * 1.0e-12)
        power = float(np.mean(physical * physical))
        return np.asarray(cross, dtype=np.complex128), power

    def _append_frames(self, audio: np.ndarray, expected_count: int) -> int:
        added = 0
        for start in range(0, len(audio) - self.config.win_length + 1, self.config.hop_length):
            cross, power = self._frame_feature(audio[start : start + self.config.win_length])
            if self._cross_sum is None:
                self._cross_sum = np.zeros_like(cross)
            self._frame_cross.append(cross)
            self._frame_power.append(power)
            self._cross_sum += cross
            self._power_sum += power
            added += 1
        if added != expected_count:
            raise RuntimeError(f"source counter expected {expected_count} new frames, got {added}")
        return added

    def _trim_frames(self) -> int:
        removed = 0
        assert self._cross_sum is not None
        while len(self._frame_cross) > self._max_frames:
            self._cross_sum -= self._frame_cross.popleft()
            self._power_sum -= self._frame_power.popleft()
            removed += 1
        return removed

    def _rebuild(self, window: DecisionWindow) -> None:
        self._frame_cross.clear()
        self._frame_power.clear()
        self._cross_sum = None
        self._power_sum = 0.0
        self.last_added_frames = self._append_frames(window.physical_samples, self._max_frames)
        self.last_removed_frames = self._trim_frames()
        self.last_update_kind = "rebuilt"

    def _advance(self, window: DecisionWindow, decision_hops: int) -> None:
        new_frame_count = decision_hops * 2
        if new_frame_count >= self._max_frames:
            self._rebuild(window)
            return
        span = self.config.win_length + (new_frame_count - 1) * self.config.hop_length
        audio = window.physical_samples[-span:]
        self.last_added_frames = self._append_frames(audio, new_frame_count)
        self.last_removed_frames = self._trim_frames()
        self.last_update_kind = "advanced"

    @staticmethod
    def _robust_peak_z(values: np.ndarray, peak: float) -> float:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return (peak - median) / max(1.4826 * mad, 1.0e-6)

    def _sample_map(self, gcc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._lag_lower is not None and self._lag_fraction is not None
        pair = np.arange(self._pair_left.size, dtype=np.int64)[:, None]
        lower = self._lag_lower
        fraction = self._lag_fraction
        responses = (
            gcc[pair, lower] * (1.0 - fraction)
            + gcc[pair, lower + 1] * fraction
        )
        return np.mean(responses, axis=0), responses

    @staticmethod
    def _circular_distance_degrees(theta: np.ndarray, reference: float) -> np.ndarray:
        return np.abs((theta - reference + 180.0) % 360.0 - 180.0)

    def _coactive_frame_count(self, first_index: int, second_index: int) -> int:
        """Count cached frames supporting both candidates without another FFT."""

        assert self._delay_samples is not None
        candidate_delays = self._delay_samples[:, (first_index, second_index)]
        steering = np.exp(
            2j
            * np.pi
            * self._frequencies_hz[:, None, None]
            * candidate_delays[None, :, :]
            / 48_000.0
        )
        frame_cross = np.stack(tuple(self._frame_cross), axis=0)
        responses = np.real(
            np.einsum("tfp,fpk->tk", frame_cross, steering, optimize=True)
        ) / (self._frequencies_hz.size * self._pair_left.size)
        return int(
            np.count_nonzero(
                np.all(responses >= self.config.coactivity_frame_threshold, axis=1)
            )
        )

    def _raw_count(self) -> int:
        if self._cross_sum is None or not self._frame_cross:
            return 0
        assert self._lag_phase is not None and self._lags_samples is not None
        assert self._delay_samples is not None
        cross = self._cross_sum / len(self._frame_cross)
        gcc = np.real(np.einsum("fp,fl->pl", cross, self._lag_phase, optimize=True))
        gcc /= self._frequencies_hz.size
        spatial, _ = self._sample_map(gcc)
        first_index = int(np.argmax(spatial))
        first_peak = float(spatial[first_index])
        first_z = self._robust_peak_z(spatial, first_peak)
        rms_dbfs = 10.0 * np.log10(max(self._power_sum / len(self._frame_power), 1.0e-15))
        self.last_rms_dbfs = rms_dbfs
        self.last_first_peak = first_peak
        self.last_first_peak_z = first_z
        self.last_residual_peak = 0.0
        self.last_residual_peak_z = 0.0
        self.last_residual_ratio = 0.0
        self.last_coactive_frames = 0
        if (
            rms_dbfs < self.config.activity_rms_threshold_dbfs
            or first_peak < self.config.first_peak_threshold
            or first_z < self.config.first_peak_z_threshold
        ):
            return 0

        first_delays = self._delay_samples[:, first_index]
        distance = self._lags_samples[None, :] - first_delays[:, None]
        notch = 1.0 - self.config.deemphasis_strength * np.exp(
            -0.5 * (distance / self.config.deemphasis_width_samples) ** 2
        )
        residual_spatial, _ = self._sample_map(gcc * notch)
        theta = np.arange(360, dtype=np.float64)
        eligible = self._circular_distance_degrees(
            theta, float(first_index)
        ) >= self.config.min_peak_distance_deg
        original_local_peak = (spatial > np.roll(spatial, 1)) & (
            spatial >= np.roll(spatial, -1)
        )
        residual_local_peak = (residual_spatial > np.roll(residual_spatial, 1)) & (
            residual_spatial >= np.roll(residual_spatial, -1)
        )
        candidate_indices = np.flatnonzero(eligible & residual_local_peak)
        original_support = np.flatnonzero(
            eligible
            & original_local_peak
            & (spatial >= self.config.residual_peak_threshold)
        )
        if candidate_indices.size == 0 or original_support.size == 0:
            return 1
        second_index = int(
            candidate_indices[np.argmax(residual_spatial[candidate_indices])]
        )
        support_distance = self._circular_distance_degrees(
            original_support.astype(np.float64),
            float(second_index),
        )
        if float(np.min(support_distance)) > self.config.min_peak_distance_deg / 4.0:
            return 1
        second_peak = float(residual_spatial[second_index])
        finite_residual = residual_spatial[eligible]
        second_z = self._robust_peak_z(finite_residual, second_peak)
        residual_ratio = second_peak / max(first_peak, 1.0e-6)
        self.last_residual_peak = second_peak
        self.last_residual_peak_z = second_z
        self.last_residual_ratio = residual_ratio
        if (
            second_peak < self.config.residual_peak_threshold
            or second_z < self.config.residual_peak_z_threshold
            or residual_ratio < self.config.residual_ratio_threshold
        ):
            return 1
        coactive_frames = self._coactive_frame_count(first_index, second_index)
        self.last_coactive_frames = coactive_frames
        if coactive_frames >= self.config.coactivity_required_frames:
            return 2
        return 1

    def _stabilize(self, raw_count: int) -> int | None:
        self._raw_history.append(raw_count)
        votes = Counter(self._raw_history)
        winner, count = max(votes.items(), key=lambda item: (item[1], item[0] == raw_count))
        if count >= self.config.persistence_required_frames:
            self._stable_count = int(winner)
        return self._stable_count

    def process(self, window: DecisionWindow, geometry: MicGeometry) -> SourceCountSnapshot:
        started = perf_counter()
        self._prepare_geometry(geometry)
        stream_key = (window.session_id, window.stream_epoch)
        continuous_hops = None
        if self._stream_key == stream_key and self._last_sample is not None:
            delta = window.decision_sample - self._last_sample
            if delta > 0 and delta % 960 == 0:
                continuous_hops = delta // 960
        if continuous_hops is None:
            self._raw_history.clear()
            self._stable_count = None
            self._rebuild(window)
        else:
            self._advance(window, continuous_hops)
            if continuous_hops > 1:
                self._raw_history.clear()
                self._stable_count = None
        self._stream_key = stream_key
        self._last_sample = window.decision_sample
        raw_count = self._raw_count()
        self.last_raw_count = raw_count
        stable_count = self._stabilize(raw_count)
        self.last_processing_ms = (perf_counter() - started) * 1_000.0
        return SourceCountSnapshot(
            window.session_id,
            window.stream_epoch,
            window.window_id,
            window.decision_sample,
            stable_count,
            monotonic(),
        )
