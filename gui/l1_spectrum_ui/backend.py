from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.data_types import IngestedAudioBlock
from gui.dev_test_ui.contracts import L1MeterSnapshot


# Logical channel 6 is the seventh physical microphone (Center); logical
# channel 7 is the hardware-generated mix and has no independent IMCRA model.
CHANNEL_NAMES = ("MIC0", "MIC1", "MIC2", "MIC3", "MIC4", "MIC5", "Center", "Mix")
CENTER_CHANNEL_INDEX = 6


def _readonly_float32(value: object) -> np.ndarray:
    raw = np.ascontiguousarray(value, dtype=np.float32)
    return np.frombuffer(raw.tobytes(), dtype=np.float32).reshape(raw.shape)


@dataclass(frozen=True, slots=True)
class L1SpectrumFrame:
    session_id: str
    stream_epoch: int
    end_sample: int
    sequence_id: int
    frequencies_hz: np.ndarray
    channel_levels_dbfs: np.ndarray
    noise_frequencies_hz: np.ndarray | None
    noise_levels_dbfs: np.ndarray | None
    meter: L1MeterSnapshot

    def __post_init__(self) -> None:
        frequencies = _readonly_float32(self.frequencies_hz)
        levels = _readonly_float32(self.channel_levels_dbfs)
        if frequencies.ndim != 1 or levels.shape != (8, frequencies.size):
            raise ValueError("L1 spectrum must be [8, frequency_bins]")
        if frequencies.size == 0 or frequencies[0] < 0.0 or frequencies[-1] > 10_000.0:
            raise ValueError("L1 spectrum frequency range must be 0..10000 Hz")
        if not np.isfinite(levels).all():
            raise ValueError("L1 spectrum levels must be finite")
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "channel_levels_dbfs", levels)

        if self.noise_frequencies_hz is None or self.noise_levels_dbfs is None:
            if self.noise_frequencies_hz is not None or self.noise_levels_dbfs is not None:
                raise ValueError("IMCRA noise frequency and level arrays must appear together")
            return
        noise_frequencies = _readonly_float32(self.noise_frequencies_hz)
        noise_levels = _readonly_float32(self.noise_levels_dbfs)
        if noise_frequencies.ndim != 1 or noise_levels.shape != (7, noise_frequencies.size):
            raise ValueError("IMCRA noise spectrum must be [7, frequency_bins]")
        if not np.isfinite(noise_levels).all():
            raise ValueError("IMCRA noise spectrum must be finite")
        object.__setattr__(self, "noise_frequencies_hz", noise_frequencies)
        object.__setattr__(self, "noise_levels_dbfs", noise_levels)


class L1SpectrumAnalyzer:
    """Produces current-audio and aligned IMCRA-noise spectra once per 20 ms hop."""

    def __init__(self, *, sample_rate: int = 48_000, hop_samples: int = 960, n_fft: int = 2_048) -> None:
        if sample_rate != 48_000 or hop_samples != 960 or n_fft != 2_048:
            raise ValueError("L1 spectrum UI requires 48 kHz, 960-sample hops and a 2048-point FFT")
        self.sample_rate = sample_rate
        self.hop_samples = hop_samples
        self.n_fft = n_fft
        self._window = np.hanning(hop_samples + 1)[:-1].astype(np.float64)
        self._amplitude_scale = 2.0 / float(np.sum(self._window))
        all_frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        self._visible = all_frequencies <= 10_000.0
        self.frequencies_hz = all_frequencies[self._visible].astype(np.float32)
        # Convert the IMCRA power normalization |X|^2/sum(w^2) back to the
        # same single-bin amplitude convention used by the current spectrum.
        self._imcra_power_scale = (
            2.0 * np.sqrt(float(np.sum(self._window**2))) / float(np.sum(self._window))
        ) ** 2

    @staticmethod
    def _dbfs_from_power(power: np.ndarray) -> np.ndarray:
        return np.maximum(10.0 * np.log10(np.maximum(power, 1.0e-12)), -120.0).astype(np.float32)

    def analyze(self, block: IngestedAudioBlock, meter: L1MeterSnapshot) -> L1SpectrumFrame:
        if block.sample_rate != self.sample_rate or block.samples.shape != (self.hop_samples, 8):
            raise ValueError("L1 spectrum UI requires one exact 20 ms [960,8] block")
        centered = block.samples.astype(np.float64) - np.mean(block.samples, axis=0, keepdims=True)
        spectrum = np.fft.rfft(centered * self._window[:, None], n=self.n_fft, axis=0)
        amplitude = np.abs(spectrum) * self._amplitude_scale
        channel_levels = self._dbfs_from_power(np.square(amplitude[self._visible].T))

        hop = block.imcra_hop
        noise_frequencies = noise_levels = None
        if hop is not None:
            noise_frequencies = hop.frequencies_hz
            noise_levels = self._dbfs_from_power(np.asarray(hop.noise_psd) * self._imcra_power_scale)
        return L1SpectrumFrame(
            block.session_id,
            block.stream_epoch,
            block.end_sample,
            block.sequence_id,
            self.frequencies_hz,
            channel_levels,
            noise_frequencies,
            noise_levels,
            meter,
        )
