from __future__ import annotations

from collections import deque

import numpy as np

from common.data_types import IngestedAudioBlock
from .contracts import L1MeterSnapshot


class L1Meter:
    """Computes eight logical-channel meters from the latest continuous 20 ms."""

    def __init__(self, window_samples: int = 960):
        self.window_samples = window_samples
        self._epoch_key: tuple[str, int] | None = None
        self._chunks: deque[np.ndarray] = deque()
        self._frames = 0

    def add(
        self, block: IngestedAudioBlock, *, light_state: str = "unknown", recording_state: str = "idle",
        pre_denoise_enabled: bool = False, pre_denoise_mean_gain_db: float = 0.0,
    ) -> L1MeterSnapshot:
        key = (block.session_id, block.stream_epoch)
        if key != self._epoch_key:
            self._chunks.clear()
            self._frames = 0
            self._epoch_key = key
        self._chunks.append(block.samples)
        self._frames += len(block.samples)
        while self._chunks and self._frames - len(self._chunks[0]) >= self.window_samples:
            self._frames -= len(self._chunks.popleft())
        data = np.concatenate(tuple(self._chunks), axis=0)[-self.window_samples :]
        rms = np.sqrt(np.mean(np.square(data, dtype=np.float64), axis=0))
        peak = np.max(np.abs(data), axis=0)
        rms_db = np.maximum(20.0 * np.log10(np.maximum(rms, 1e-6)), -120.0).astype(np.float32)
        peak_db = np.maximum(20.0 * np.log10(np.maximum(peak, 1e-6)), -120.0).astype(np.float32)
        return L1MeterSnapshot(
            block.session_id,
            block.stream_epoch,
            block.end_sample,
            block.sequence_id,
            rms_db,
            peak_db,
            peak >= 0.999,
            light_state,
            recording_state,
            block.imcra_hop,
            pre_denoise_enabled,
            pre_denoise_mean_gain_db,
        )
