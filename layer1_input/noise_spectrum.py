from __future__ import annotations

from collections import deque

import numpy as np

from .interface import DecodedAudio, NoiseSpectrumRecord


class DynamicNoiseSpectrumRecorder:
    """MCRA-style per-channel/per-bin PSD recorder; never modifies audio."""

    def __init__(
        self,
        *,
        n_fft: int = 2048,
        smoothing: float = 0.90,
        noise_smoothing: float = 0.80,
        minimum_history_frames: int = 75,
        presence_ratio_low: float = 1.5,
        presence_ratio_high: float = 4.0,
        floor: float = 1.0e-12,
    ) -> None:
        if n_fft <= 0 or n_fft % 2 or not 0.0 <= smoothing < 1.0:
            raise ValueError("invalid noise spectrum FFT/smoothing configuration")
        if not 0.0 <= noise_smoothing < 1.0 or minimum_history_frames <= 0:
            raise ValueError("invalid noise spectrum update configuration")
        if not 1.0 <= presence_ratio_low < presence_ratio_high or floor <= 0.0:
            raise ValueError("invalid noise spectrum presence/floor configuration")
        self.n_fft = int(n_fft)
        self.smoothing = float(smoothing)
        self.noise_smoothing = float(noise_smoothing)
        self.minimum_history_frames = int(minimum_history_frames)
        self.presence_ratio_low = float(presence_ratio_low)
        self.presence_ratio_high = float(presence_ratio_high)
        self.floor = float(floor)
        self._window_cache: dict[int, np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        self._smoothed: np.ndarray | None = None
        self._noise_psd: np.ndarray | None = None
        self._minimum_history: deque[np.ndarray] = deque(maxlen=self.minimum_history_frames)
        self._previous_sequence: int | None = None
        self._previous_timestamp: float | None = None
        self._previous_frames = 0
        self._frames_observed = 0

    def _window(self, length: int) -> np.ndarray:
        cached = self._window_cache.get(length)
        if cached is None:
            cached = np.hanning(length + 1)[:-1].astype(np.float64)
            self._window_cache[length] = cached
        return cached

    def process(self, audio: DecodedAudio) -> NoiseSpectrumRecord:
        if self._previous_sequence is not None:
            assert self._previous_timestamp is not None
            expected_timestamp = self._previous_timestamp + self._previous_frames / audio.sample_rate
            if audio.sequence_id != self._previous_sequence + 1 or abs(audio.timestamp - expected_timestamp) > 0.005:
                self.reset()
        # IMCRA is defined only for the seven calibrated physical microphones;
        # HardwareMix remains a display/recording channel and is never folded
        # into the array-source probability.
        samples = np.asarray(audio.samples[:, :7], dtype=np.float64)
        length = min(samples.shape[0], self.n_fft)
        segment = samples[-length:]
        segment = segment - segment.mean(axis=0, keepdims=True)
        window = self._window(length)
        spectrum = np.fft.rfft(segment * window[:, None], n=self.n_fft, axis=0)
        power = (np.abs(spectrum).T ** 2 / max(float(np.square(window).sum()), self.floor)).astype(np.float64)

        if self._smoothed is None:
            self._smoothed = power.copy()
            self._noise_psd = np.maximum(power, self.floor)
        else:
            self._smoothed = self.smoothing * self._smoothed + (1.0 - self.smoothing) * power

        self._minimum_history.append(self._smoothed.copy())
        minimum = np.minimum.reduce(tuple(self._minimum_history))
        ratio = self._smoothed / np.maximum(minimum, self.floor)
        presence = np.clip(
            (ratio - self.presence_ratio_low) / (self.presence_ratio_high - self.presence_ratio_low),
            0.0,
            1.0,
        )
        alpha = self.noise_smoothing + (1.0 - self.noise_smoothing) * presence
        assert self._noise_psd is not None
        self._noise_psd = np.maximum(alpha * self._noise_psd + (1.0 - alpha) * power, self.floor)

        self._frames_observed += 1
        frequencies = np.fft.rfftfreq(self.n_fft, 1.0 / audio.sample_rate)
        speech_band = (frequencies >= 500.0) & (frequencies <= 4_000.0)
        noise_level_db = 10.0 * np.log10(
            np.maximum(np.mean(self._noise_psd[:, speech_band], axis=1), self.floor)
        )
        per_mic_probability = np.clip(np.mean(presence[:, speech_band], axis=1), 0.0, 1.0)
        ready = self._frames_observed >= self.minimum_history_frames
        array_probability = float(np.median(per_mic_probability)) if ready else None

        record = NoiseSpectrumRecord(
            self._noise_psd.astype(np.float32),
            np.fft.rfftfreq(self.n_fft, 1.0 / audio.sample_rate).astype(np.float32),
            audio.sample_rate,
            self.n_fft,
            audio.sequence_id,
            audio.timestamp,
            state="ready" if ready else "warming_up",
            noise_level_db=noise_level_db.astype(np.float32),
            source_probability_per_mic=per_mic_probability.astype(np.float32),
            array_source_probability_20ms=array_probability,
        )
        self._previous_sequence = audio.sequence_id
        self._previous_timestamp = audio.timestamp
        self._previous_frames = audio.frame_count
        return record
