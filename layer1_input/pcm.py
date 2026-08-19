from __future__ import annotations

import numpy as np


PCM16_SCALE = 32768.0
PCM16_MIN = -32768
PCM16_MAX = 32767


def float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    """Quantize normalized floating-point audio to little-endian PCM16.

    Round-to-nearest is important here: every value produced by decoding an
    original S16 sample must encode back to the same integer code, including
    the positive full-scale value 32767.
    """

    data = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError("音频包含 NaN 或 Inf，拒绝写入损坏的 PCM")
    quantized = np.rint(data * PCM16_SCALE)
    return np.clip(quantized, PCM16_MIN, PCM16_MAX).astype("<i2")


def pcm16_bytes(samples: np.ndarray) -> bytes:
    return float_to_pcm16(samples).tobytes(order="C")


__all__ = ["PCM16_MAX", "PCM16_MIN", "PCM16_SCALE", "float_to_pcm16", "pcm16_bytes"]
