from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


class NativeChannelPlayer:
    """Stream one selected channel from a native 8-channel recording."""

    def __init__(self, volume: float = 0.5):
        self.volume = float(volume)
        self._lock = threading.RLock()
        self._wave: wave.Wave_read | None = None
        self._stream: sd.OutputStream | None = None
        self._channel = 0

    def play(self, path: str | Path, channel: int) -> None:
        if channel not in range(8):
            raise ValueError("试听通道必须位于1到8")
        self.stop()
        source = wave.open(str(path), "rb")
        if source.getnchannels() != 8 or source.getsampwidth() != 2 or source.getframerate() != 48_000:
            source.close()
            raise ValueError("试听文件必须是48 kHz原始8通道PCM16")
        with self._lock:
            self._wave = source
            self._channel = channel
        stream = sd.OutputStream(
            samplerate=48_000,
            channels=1,
            dtype="float32",
            callback=self._callback,
            finished_callback=self._finished,
        )
        with self._lock:
            self._stream = stream
        stream.start()

    def _callback(self, outdata, frames, _time, status) -> None:
        outdata.fill(0)
        if status:
            raise sd.CallbackAbort
        with self._lock:
            source = self._wave
            channel = self._channel
            if source is None:
                raise sd.CallbackStop
            payload = source.readframes(frames)
        if not payload:
            raise sd.CallbackStop
        values = np.frombuffer(payload, dtype="<i2").reshape(-1, 8)
        count = len(values)
        outdata[:count, 0] = values[:, channel].astype(np.float32) * (self.volume / 32768.0)
        if count < frames:
            raise sd.CallbackStop

    def _finished(self) -> None:
        with self._lock:
            if self._wave is not None:
                self._wave.close()
                self._wave = None

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
            source, self._wave = self._wave, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        if source is not None:
            source.close()

    close = stop


__all__ = ["NativeChannelPlayer"]
