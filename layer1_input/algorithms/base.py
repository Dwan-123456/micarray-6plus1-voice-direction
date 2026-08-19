from abc import ABC, abstractmethod

import numpy as np


class AudioDecoder(ABC):
    @abstractmethod
    def decode(self, payload: bytes, channels: int) -> np.ndarray:
        """Raw device bytes -> float32 array shaped (samples, hardware_channels)."""
        ...
