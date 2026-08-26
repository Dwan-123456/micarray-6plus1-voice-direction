from __future__ import annotations

from collections import deque
from math import ceil

import numpy as np
from scipy.special import exp1

from common.config import Layer1ImcraConfig, ProjectConfig
from common.data_types import CalibrationMetadata, ImcraHopSnapshot, IngestedAudioBlock


class Layer1Imcra:
    """Seven-channel implementation of Cohen's 2003 IMCRA algorithm."""

    FEATURE_NAMES = ("noise_level_db", "signal_level_db", "snr_db", "mean_spp")

    def __init__(self, config: Layer1ImcraConfig, *, sample_rate: int = 48_000) -> None:
        if sample_rate != 48_000:
            raise ValueError("L1 IMCRA only supports 48 kHz")
        self.config, self.sample_rate, self.hop_samples = config, sample_rate, config.hop_samples
        self._all_frequencies_hz = np.fft.rfftfreq(config.n_fft, 1.0 / sample_rate).astype(np.float32)
        self._window = np.hanning(self.hop_samples + 1)[:-1].astype(np.float64)
        # Normalized three-point Hanning frequency smoother (paper Table I: w=1).
        self._frequency_kernel = np.asarray((0.25, 0.5, 0.25), dtype=np.float64)
        self._output_band = (self._all_frequencies_hz >= config.output_frequency_min_hz) & (
            self._all_frequencies_hz <= config.output_frequency_max_hz
        )
        self._gate_band = (self._all_frequencies_hz >= config.frequency_min_hz) & (
            self._all_frequencies_hz <= config.frequency_max_hz
        )
        self.frequencies_hz = self._all_frequencies_hz[self._output_band]
        self._warmup_hops = ceil(config.warmup_seconds * sample_rate / self.hop_samples)
        self._identity: tuple[str, int] | None = None
        self._calibration: CalibrationMetadata | None = None
        self._next_input_sample = self._buffer_start = 0
        self._buffer = np.empty((0, 7), dtype=np.float32)
        self._segments: deque[tuple[int, int, int]] = deque()
        self._frame_count = self._subwindow_frame = 0
        self._smoothed: np.ndarray | None = None
        self._conditional_smoothed: np.ndarray | None = None
        self._minimum: np.ndarray | None = None
        self._conditional_minimum: np.ndarray | None = None
        self._subwindow_minimum: np.ndarray | None = None
        self._conditional_subwindow_minimum: np.ndarray | None = None
        self._minimum_history: deque[np.ndarray] = deque(maxlen=config.minimum_history_subwindows)
        self._conditional_minimum_history: deque[np.ndarray] = deque(maxlen=config.minimum_history_subwindows)
        self._noise_bar: np.ndarray | None = None
        self._noise: np.ndarray | None = None
        self._noise_covariance_bar: np.ndarray | None = None
        self._noise_covariance: np.ndarray | None = None
        self._previous_gamma: np.ndarray | None = None
        self._previous_gain_h1: np.ndarray | None = None
        self._spp: np.ndarray | None = None
        self._speech_absence_probability: np.ndarray | None = None
        self._posterior_snr: np.ndarray | None = None
        self._prior_snr: np.ndarray | None = None

    @classmethod
    def from_project(cls, config: ProjectConfig) -> "Layer1Imcra":
        return cls(config.layer1_imcra, sample_rate=config.device.sample_rate)

    def reset(self) -> None:
        self._identity = None
        self._calibration = None
        self._next_input_sample = self._buffer_start = 0
        self._buffer = np.empty((0, 7), dtype=np.float32)
        self._segments.clear()
        self._frame_count = self._subwindow_frame = 0
        for name in (
            "_smoothed", "_conditional_smoothed", "_minimum", "_conditional_minimum",
            "_subwindow_minimum", "_conditional_subwindow_minimum", "_noise_bar", "_noise",
            "_noise_covariance_bar", "_noise_covariance",
            "_previous_gamma", "_previous_gain_h1", "_spp", "_speech_absence_probability",
            "_posterior_snr", "_prior_snr",
        ):
            setattr(self, name, None)
        self._minimum_history.clear()
        self._conditional_minimum_history.clear()

    def _start_epoch(self, block: IngestedAudioBlock) -> None:
        previous_identity = self._identity
        preserve_statistics = (
            previous_identity is not None
            and previous_identity[0] == block.session_id
            and self._calibration == block.calibration
            and self._smoothed is not None
        )
        if not preserve_statistics:
            self.reset()
        else:
            # A new epoch marks an explicit hole in the authoritative sample
            # axis.  Discard only transport/alignment state: accumulated IMCRA
            # noise statistics remain valid for the same physical capture
            # session and calibration.  The missing interval publishes no
            # probability; if the estimator was already ready, the first
            # recovered hop can resume a formal probability without another
            # One-second warm-up, synchronized with the adapted minimum history.
            self._buffer = np.empty((0, 7), dtype=np.float32)
            self._segments.clear()
        self._identity = (block.session_id, block.stream_epoch)
        self._calibration = block.calibration
        self._next_input_sample = self._buffer_start = block.start_sample

    def process(self, block: IngestedAudioBlock) -> tuple[ImcraHopSnapshot, ...]:
        identity = (block.session_id, block.stream_epoch)
        if identity != self._identity:
            self._start_epoch(block)
        if block.start_sample != self._next_input_sample:
            raise ValueError("L1 IMCRA received a discontinuity without a new stream epoch")
        self._buffer = np.concatenate((self._buffer, np.asarray(block.samples[:, :7], np.float32)), axis=0)
        self._segments.append((block.start_sample, block.end_sample, block.sequence_id))
        self._next_input_sample = block.end_sample
        output: list[ImcraHopSnapshot] = []
        while len(self._buffer) >= self.hop_samples:
            start, end = self._buffer_start, self._buffer_start + self.hop_samples
            sequence_ids = tuple(
                sequence_id for segment_start, segment_end, sequence_id in self._segments
                if segment_end > start and segment_start < end
            )
            output.append(self._process_hop(self._buffer[: self.hop_samples], start, end, sequence_ids))
            self._buffer, self._buffer_start = self._buffer[self.hop_samples :], end
            while self._segments and self._segments[0][1] <= end:
                self._segments.popleft()
        return tuple(output)

    def _frequency_smooth(self, values: np.ndarray) -> np.ndarray:
        padded = np.pad(values, ((0, 0), (1, 1)), mode="edge")
        return sum(weight * padded[:, offset : offset + values.shape[1]] for offset, weight in enumerate(self._frequency_kernel))

    def _conditional_frequency_smooth(self, power: np.ndarray, indicator: np.ndarray) -> np.ndarray:
        denominator = self._frequency_smooth(indicator.astype(np.float64))
        numerator = self._frequency_smooth(power * indicator)
        assert self._conditional_smoothed is not None
        return np.divide(
            numerator, denominator, out=self._conditional_smoothed.copy(),
            where=denominator > self.config.eps,
        )

    def _close_subwindow(self) -> None:
        assert self._smoothed is not None and self._conditional_smoothed is not None
        assert self._subwindow_minimum is not None and self._conditional_subwindow_minimum is not None
        self._minimum_history.append(self._subwindow_minimum.copy())
        self._conditional_minimum_history.append(self._conditional_subwindow_minimum.copy())
        self._minimum = np.minimum.reduce(tuple(self._minimum_history))
        self._conditional_minimum = np.minimum.reduce(tuple(self._conditional_minimum_history))
        self._subwindow_minimum = self._smoothed.copy()
        self._conditional_subwindow_minimum = self._conditional_smoothed.copy()
        self._subwindow_frame = 0

    def _initialize(
        self, power: np.ndarray, frequency_smoothed: np.ndarray, spatial_outer: np.ndarray,
    ) -> None:
        self._smoothed = self._conditional_smoothed = frequency_smoothed.copy()
        self._minimum = self._conditional_minimum = frequency_smoothed.copy()
        self._subwindow_minimum = self._conditional_subwindow_minimum = frequency_smoothed.copy()
        self._noise_bar = self._noise = power.copy()
        self._noise_covariance_bar = spatial_outer.copy()
        self._noise_covariance = spatial_outer.copy()
        diagonal = np.arange(7)
        self._noise_covariance[:, diagonal, diagonal] = self._noise.T
        self._previous_gamma = self._previous_gain_h1 = np.ones_like(power)
        self._spp = np.zeros_like(power)
        self._speech_absence_probability = np.ones_like(power)
        self._posterior_snr = np.ones_like(power)
        self._prior_snr = np.zeros_like(power)

    def _cohen_update(
        self, power: np.ndarray, frequency_smoothed: np.ndarray, spatial_outer: np.ndarray,
    ) -> None:
        cfg = self.config
        assert self._noise is not None and self._noise_bar is not None
        assert self._noise_covariance is not None and self._noise_covariance_bar is not None
        assert self._smoothed is not None and self._conditional_smoothed is not None
        assert self._minimum is not None and self._conditional_minimum is not None
        assert self._previous_gamma is not None and self._previous_gain_h1 is not None

        # Cohen Eqs. (3), (32), (33): posterior/prior SNR and the H1 LSA gain.
        gamma = power / np.maximum(self._noise, cfg.eps)
        xi = cfg.prior_snr_smoothing * np.square(self._previous_gain_h1) * self._previous_gamma
        xi += (1.0 - cfg.prior_snr_smoothing) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, 0.0)
        nu = gamma * xi / (1.0 + xi)
        gain_h1 = np.zeros_like(xi)
        positive = xi > cfg.eps
        gain_h1[positive] = xi[positive] / (1.0 + xi[positive]) * np.exp(
            0.5 * exp1(np.maximum(nu[positive], cfg.eps))
        )
        gain_h1 = np.clip(gain_h1, 0.0, 10.0)

        # Eqs. (14)-(16), (18), (21): first smoothing/minimum pass and rough VAD.
        self._smoothed = cfg.spectrum_smoothing * self._smoothed + (1.0 - cfg.spectrum_smoothing) * frequency_smoothed
        self._minimum = np.minimum(self._minimum, self._smoothed)
        assert self._subwindow_minimum is not None
        self._subwindow_minimum = np.minimum(self._subwindow_minimum, self._smoothed)
        first_reference = cfg.minimum_bias * np.maximum(self._minimum, cfg.eps)
        indicator = (power / first_reference < cfg.gamma0) & (self._smoothed / first_reference < cfg.zeta0)

        # Eqs. (26)-(28): conditional second smoothing/minimum pass.
        conditional_frequency = self._conditional_frequency_smooth(power, indicator)
        self._conditional_smoothed = cfg.spectrum_smoothing * self._conditional_smoothed + (
            1.0 - cfg.spectrum_smoothing
        ) * conditional_frequency
        self._conditional_minimum = np.minimum(self._conditional_minimum, self._conditional_smoothed)
        assert self._conditional_subwindow_minimum is not None
        self._conditional_subwindow_minimum = np.minimum(
            self._conditional_subwindow_minimum, self._conditional_smoothed
        )

        # Eq. (29): minima-controlled a-priori speech-absence probability q_hat.
        reference = cfg.minimum_bias * np.maximum(self._conditional_minimum, cfg.eps)
        # Cohen Eq. (28) deliberately uses first-pass S in zeta-tilde's numerator.
        gamma_min, zeta = power / reference, self._smoothed / reference
        q_hat = np.zeros_like(power)
        local_noise = zeta < cfg.zeta0
        q_hat[local_noise & (gamma_min <= 1.0)] = 1.0
        transition = local_noise & (gamma_min > 1.0) & (gamma_min < cfg.gamma1)
        q_hat[transition] = (cfg.gamma1 - gamma_min[transition]) / (cfg.gamma1 - 1.0)

        # Eq. (7): conditional speech-presence probability, evaluated in log space.
        q_safe = np.clip(q_hat, cfg.eps, 1.0 - cfg.eps)
        log_absence_odds = np.log(q_safe) - np.log1p(-q_safe) + np.log1p(xi) - nu
        spp = 1.0 / (1.0 + np.exp(np.clip(log_absence_odds, -80.0, 80.0)))
        spp = np.where(q_hat >= 1.0, 0.0, np.where(q_hat <= 0.0, 1.0, spp))

        # Eqs. (10)-(12): probability-controlled recursive average and beta correction.
        alpha_tilde = cfg.noise_smoothing + (1.0 - cfg.noise_smoothing) * spp
        self._noise_bar = alpha_tilde * self._noise_bar + (1.0 - alpha_tilde) * power
        self._noise = cfg.bias_compensation * self._noise_bar
        array_spp = np.max(spp, axis=0)
        covariance_alpha = cfg.noise_smoothing + (1.0 - cfg.noise_smoothing) * array_spp
        self._noise_covariance_bar = (
            covariance_alpha[:, None, None] * self._noise_covariance_bar
            + (1.0 - covariance_alpha)[:, None, None] * spatial_outer
        )
        self._noise_covariance = cfg.bias_compensation * self._noise_covariance_bar
        covariance_diagonal = np.maximum(
            np.real(np.diagonal(self._noise_covariance, axis1=1, axis2=2)),
            cfg.eps,
        )
        diagonal_scale = np.sqrt(self._noise.T / covariance_diagonal)
        self._noise_covariance *= (
            diagonal_scale[:, :, None] * diagonal_scale[:, None, :]
        )
        self._noise_covariance = 0.5 * (
            self._noise_covariance + self._noise_covariance.conj().transpose(0, 2, 1)
        )
        self._previous_gamma, self._previous_gain_h1 = gamma, gain_h1
        self._spp = np.clip(spp, 0.0, 1.0)
        self._speech_absence_probability = np.clip(q_hat, 0.0, 1.0)
        self._posterior_snr, self._prior_snr = gamma, xi

    def _process_hop(
        self, samples: np.ndarray, start_sample: int, end_sample: int, source_sequence_ids: tuple[int, ...]
    ) -> ImcraHopSnapshot:
        cfg = self.config
        centered = samples.astype(np.float64) - np.mean(samples, axis=0, keepdims=True)
        spectrum = np.fft.rfft(centered * self._window[:, None], n=cfg.n_fft, axis=0)
        window_energy = np.sum(self._window**2)
        normalized_spectrum = spectrum / np.sqrt(window_energy)
        spatial_outer = np.einsum(
            "fc,fd->fcd", normalized_spectrum, normalized_spectrum.conj(), optimize=True,
        )
        power = np.maximum((np.abs(spectrum) ** 2).T / window_energy, cfg.eps)
        frequency_smoothed = self._frequency_smooth(power)
        self._initialize(power, frequency_smoothed, spatial_outer) if self._smoothed is None else self._cohen_update(
            power, frequency_smoothed, spatial_outer,
        )
        self._frame_count += 1
        self._subwindow_frame += 1
        if self._subwindow_frame == cfg.minimum_subwindow_frames:
            self._close_subwindow()

        assert self._noise is not None and self._smoothed is not None and self._conditional_smoothed is not None
        assert self._minimum is not None and self._conditional_minimum is not None and self._spp is not None
        assert self._speech_absence_probability is not None and self._posterior_snr is not None
        assert self._prior_snr is not None and self._identity is not None
        assert self._noise_covariance is not None
        band_noise = np.mean(self._noise[:, self._output_band], axis=1)
        band_signal = np.mean(power[:, self._output_band], axis=1)
        noise_level_db = 10.0 * np.log10(np.maximum(band_noise, cfg.eps))
        signal_level_db = 10.0 * np.log10(np.maximum(band_signal, cfg.eps))
        probability_per_mic = np.clip(np.mean(self._spp[:, self._gate_band], axis=1), 0.0, 1.0)
        features = np.column_stack((noise_level_db, signal_level_db, signal_level_db - noise_level_db, probability_per_mic))
        ready = self._frame_count >= self._warmup_hops
        array_probability = float(np.median(probability_per_mic)) if ready else None
        return ImcraHopSnapshot(
            self._identity[0], self._identity[1], start_sample, end_sample, source_sequence_ids,
            cfg.algorithm_version, "ready" if ready else "warming_up", self.frequencies_hz,
            self._noise[:, self._output_band].astype(np.float32), self._smoothed[:, self._output_band].astype(np.float32),
            self._conditional_smoothed[:, self._output_band].astype(np.float32),
            self._minimum[:, self._output_band].astype(np.float32),
            self._conditional_minimum[:, self._output_band].astype(np.float32),
            self._spp[:, self._output_band].astype(np.float32),
            self._speech_absence_probability[:, self._output_band].astype(np.float32),
            self._posterior_snr[:, self._output_band].astype(np.float32),
            self._prior_snr[:, self._output_band].astype(np.float32),
            features.astype(np.float32), noise_level_db.astype(np.float32),
            probability_per_mic.astype(np.float32), array_probability,
            self._noise_covariance[self._output_band].astype(np.complex64),
        )
