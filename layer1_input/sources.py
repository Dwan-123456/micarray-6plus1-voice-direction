from __future__ import annotations

import logging
import queue
import time
from abc import ABC, abstractmethod

import numpy as np

from .algorithms import AudioDecoder, Pcm16InterleavedDecoder
from .capture import AudioCapture
from .configuration import AudioConfig
from .interface import DecodedAudio

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
