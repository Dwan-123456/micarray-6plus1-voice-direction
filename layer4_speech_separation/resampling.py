from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly


class Layer4Resampler:
    """Shared deterministic polyphase adapter for the 48/16 kHz boundary."""

    algorithm_version = "scipy_resample_poly_kaiser_5p0_v1"

    @staticmethod
    def _audio(value: NDArray[np.float32]) -> NDArray[np.float32]:
        audio = np.asarray(value)
        if audio.ndim != 1 or audio.dtype != np.float32 or not audio.flags.c_contiguous:
            raise ValueError("resampler input must be C-contiguous float32 mono audio")
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("resampler input must be non-empty and finite")
        return audio

    def to_16k(self, waveform_48k: NDArray[np.float32]) -> NDArray[np.float32]:
        value = self._audio(waveform_48k)
        output = resample_poly(value, 1, 3, window=("kaiser", 5.0))
        return np.ascontiguousarray(output, dtype=np.float32)

    def to_48k(self, waveform_16k: NDArray[np.float32]) -> NDArray[np.float32]:
        value = self._audio(waveform_16k)
        output = resample_poly(value, 3, 1, window=("kaiser", 5.0))
        return np.ascontiguousarray(output, dtype=np.float32)
