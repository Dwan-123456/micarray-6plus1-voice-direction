from __future__ import annotations

import numpy as np

from common.data_types import CalibrationMetadata

from .configuration import CalibrationConfig
from .interface import DecodedAudio


class ChannelCalibrator:
    def __init__(self, config: CalibrationConfig):
        config.validate()
        self._delays = np.asarray(config.delay_samples, dtype=np.int64)
        self._scale = np.asarray(config.gains, dtype=np.float32) * np.asarray(config.polarity, dtype=np.float32)
        self._max_delay = int(np.max(self._delays))
        self._history = np.zeros((self._max_delay, len(self._delays)), dtype=np.float32)
        self._metadata = config.metadata

    @property
    def metadata(self) -> CalibrationMetadata:
        return self._metadata

    def reset(self) -> None:
        self._history.fill(0.0)

    def process(self, frame: DecodedAudio) -> DecodedAudio:
        if frame.channels != 8 or len(self._delays) != 7:
            raise ValueError("L1校准要求逻辑8通道且仅配置7个物理麦")
        scaled = frame.samples[:, :7] * self._scale[None, :]
        if not self._max_delay:
            output = scaled
        else:
            combined = np.concatenate((self._history, scaled), axis=0)
            output = np.empty_like(scaled)
            for channel, delay in enumerate(self._delays):
                start = self._max_delay - int(delay)
                output[:, channel] = combined[start:start + frame.frame_count, channel]
            self._history = combined[-self._max_delay:].copy()
        logical = np.column_stack((output, frame.samples[:, 7]))
        return DecodedAudio(
            logical,
            frame.sample_rate,
            frame.sequence,
            frame.timestamp,
            native_samples=frame.native_samples,
            hotmap=frame.hotmap,
            noise_spectrum=frame.noise_spectrum,
            calibration=self._metadata,
        )
