import numpy as np

from .base import AudioDecoder


class Pcm16InterleavedDecoder(AudioDecoder):
    def decode(self, payload: bytes, channels: int) -> np.ndarray:
        raw = np.frombuffer(payload, dtype="<i2")
        if raw.size % channels:
            raise ValueError("PCM 字节数无法按通道数整除")
        return raw.reshape(-1, channels).astype(np.float32) / 32768.0
