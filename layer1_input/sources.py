from __future__ import annotations

import logging
import queue
import time
import wave
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .algorithms import AudioDecoder, Pcm16InterleavedDecoder
from .capture import AudioCapture
from .configuration import AudioConfig
from .interface import DecodedAudio
from .pcm import pcm16_bytes

logger = logging.getLogger(__name__)


def pcm16_to_float32(payload: bytes, channels: int) -> np.ndarray:
    return Pcm16InterleavedDecoder().decode(payload, channels)


def map_physical_channels(samples: np.ndarray, channel_map: tuple[int, ...]) -> np.ndarray:
    if samples.ndim != 2:
        raise ValueError("samples 必须是二维数组")
    return np.ascontiguousarray(samples[:, channel_map], dtype=np.float32)


def map_logical_channels(samples: np.ndarray, channel_map: tuple[int, ...]) -> np.ndarray:
    if (
        samples.ndim != 2
        or samples.shape[1] != 8
        or tuple(sorted(channel_map)) != tuple(range(8))
    ):
        raise ValueError("logical mapping requires native [N,8] and eight map entries")
    return np.ascontiguousarray(samples[:, channel_map], dtype=np.float32)


class AudioSource(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read(self, timeout: float | None = None) -> DecodedAudio | None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def __enter__(self) -> "AudioSource":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()


class LiveSipeedSource(AudioSource):
    def __init__(self, config: AudioConfig, capture: AudioCapture | None = None, decoder: AudioDecoder | None = None):
        self.config = config
        self.capture = capture or AudioCapture(config.device_name, config.host_api, config.sample_rate, config.device_channels, config.block_size)
        self._receiver: queue.Queue[bytes] | None = None
        self._sequence = 0
        self._sample_position = 0
        self._origin_timestamp = 0.0
        self._started_capture = False
        self.decoder = decoder or Pcm16InterleavedDecoder()
        self._pending_health_events = []
        self._last_visible_sequence: int | None = None

    def start(self) -> None:
        if self._receiver is not None:
            return
        # Main-path handoff is bounded. Capture assigns sequence IDs before
        # enqueue so a dropped item remains observable as a sequence gap.
        subscribe = getattr(self.capture, "subscribe_numbered", None)
        capacity = self.config.handoff_blocks
        self._receiver = (
            subscribe(maxsize=capacity)
            if subscribe is not None
            else self.capture.subscribe(maxsize=capacity)
        )
        self._origin_timestamp = time.monotonic()
        try:
            if not self.capture.running:
                self.capture.start()
                self._started_capture = True
        except Exception:
            self.capture.unsubscribe(self._receiver)
            self._receiver = None
            raise
        self._sequence = 0
        self._sample_position = 0
        self._pending_health_events.clear()
        self._last_visible_sequence = None
        logger.info("Layer1 live source started")

    def read(self, timeout: float | None = None) -> DecodedAudio | None:
        if self._receiver is None:
            raise RuntimeError("数据源尚未启动")
        try:
            item = self._receiver.get(timeout=timeout)
        except queue.Empty:
            return None
        if hasattr(item, "payload"):
            payload, sequence, timestamp = item.payload, item.sequence_id, item.timestamp
        else:
            payload, sequence = item, self._sequence
            timestamp = self._origin_timestamp + self._sample_position / self.config.sample_rate
        raw = self.decoder.decode(payload, self.config.device_channels)
        frame = DecodedAudio(
            map_logical_channels(raw, self.config.logical_channel_map),
            self.config.sample_rate,
            sequence,
            timestamp,
            native_samples=raw,
        )
        self._sequence += 1
        self._sample_position += raw.shape[0]
        self._last_visible_sequence = sequence
        return frame

    def take_health_events(self):
        method = getattr(self.capture, "take_health_events", None)
        if method is not None:
            self._pending_health_events.extend(method())
        ready, future = [], []
        for event in self._pending_health_events:
            first = event.first_sequence_id_after_gap
            if first is None or self._last_visible_sequence is None or first <= self._last_visible_sequence:
                ready.append(event)
            else:
                future.append(event)
        self._pending_health_events = future
        return tuple(ready)

    def stop(self) -> None:
        if self._receiver is not None:
            self.capture.unsubscribe(self._receiver)
            self._receiver = None
        if self._started_capture:
            self.capture.stop()
            self._started_capture = False


class WavAudioSource(AudioSource):
    def __init__(self, path: str | Path, block_size: int = 960, channel_map: tuple[int, ...] | None = None, realtime: bool = False):
        self.path, self.block_size, self.channel_map, self.realtime = Path(path), block_size, channel_map, realtime
        self._wav: wave.Wave_read | None = None
        self._sequence = 0
        self._start_time = 0.0
        self._exhausted = False

    def start(self) -> None:
        self._wav = wave.open(str(self.path), "rb")
        if self._wav.getsampwidth() != 2:
            raise ValueError("仅支持 16-bit PCM WAV")
        if self.channel_map and max(self.channel_map) >= self._wav.getnchannels():
            raise ValueError("WAV 通道数不足")
        self._sequence, self._start_time, self._exhausted = 0, time.monotonic(), False

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def sample_rate(self) -> int:
        if self._wav is None:
            raise RuntimeError("WAV 尚未打开")
        return self._wav.getframerate()

    def read(self, timeout: float | None = None) -> DecodedAudio | None:
        del timeout
        if self._wav is None:
            raise RuntimeError("数据源尚未启动")
        payload = self._wav.readframes(self.block_size)
        if not payload:
            self._exhausted = True
            return None
        native_samples = pcm16_to_float32(payload, self._wav.getnchannels())
        samples = native_samples
        if self.channel_map is not None:
            samples = map_logical_channels(samples, self.channel_map)
        elif samples.shape[1] == 7:
            # Legacy 6+1 replay has no HardwareMix; keep the logical contract
            # explicit with a silent eighth channel.
            samples = np.column_stack((samples, np.zeros(samples.shape[0], dtype=np.float32)))
        timestamp = self._sequence * self.block_size / self.sample_rate
        if self.realtime:
            time.sleep(max(0.0, self._start_time + timestamp - time.monotonic()))
        frame = DecodedAudio(
            samples,
            self.sample_rate,
            self._sequence,
            timestamp,
            native_samples=native_samples if self.channel_map is not None else None,
        )
        self._sequence += 1
        return frame

    def stop(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None


class MultichannelWavRecorder:
    def __init__(self, path: str | Path, sample_rate: int, channels: int = 8):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wav = wave.open(str(self.path), "wb")
        self._wav.setnchannels(channels)
        self._wav.setsampwidth(2)
        self._wav.setframerate(sample_rate)
        self.channels = channels

    def write(self, frame: DecodedAudio) -> None:
        if frame.channels != self.channels:
            raise ValueError(f"需要 {self.channels} 通道，实际 {frame.channels}")
        self._wav.writeframes(pcm16_bytes(frame.samples))

    def close(self) -> None:
        self._wav.close()

    def __enter__(self) -> "MultichannelWavRecorder":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
