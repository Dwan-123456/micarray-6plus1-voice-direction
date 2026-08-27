from __future__ import annotations

from collections import deque

import numpy as np

from common.data_types import IngestedAudioBlock
from .contracts import L1MeterSnapshot


class L1Meter:
    def __init__(self, window_samples: int = 960) -> None:
        self.window_samples = window_samples
        self._key: tuple[str, int] | None = None
        self._chunks: deque[np.ndarray] = deque()
        self._samples = 0

    def add(
        self, block: IngestedAudioBlock, *, light_state: str = "unknown",
        pre_denoise_enabled: bool = False,
        pre_denoise_mean_gain_db: float = 0.0,
    ) -> L1MeterSnapshot:
        key = (block.session_id, block.stream_epoch)
        if key != self._key:
            self._key = key
            self._chunks.clear()
            self._samples = 0
        self._chunks.append(block.samples)
        self._samples += len(block.samples)
        while self._chunks and self._samples - len(self._chunks[0]) >= self.window_samples:
            self._samples -= len(self._chunks.popleft())
        data = np.concatenate(tuple(self._chunks), axis=0)[-self.window_samples:]
        rms = np.sqrt(np.mean(np.square(data, dtype=np.float64), axis=0))
        peak = np.max(np.abs(data), axis=0)
        rms_db = np.maximum(20 * np.log10(np.maximum(rms, 1e-6)), -120).astype(np.float32)
        peak_db = np.maximum(20 * np.log10(np.maximum(peak, 1e-6)), -120).astype(np.float32)
        return L1MeterSnapshot(
            block.session_id, block.stream_epoch, block.end_sample, block.sequence_id,
            rms_db, peak_db, peak >= 0.999, light_state,
            block.imcra_hop, pre_denoise_enabled, float(pre_denoise_mean_gain_db),
            block.calibration.status, block.calibration.version, block.calibration.calibration_hash,
        )
