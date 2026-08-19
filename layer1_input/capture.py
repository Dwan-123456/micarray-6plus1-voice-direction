from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Any

import numpy as np

from dataclasses import dataclass
import time

from .interface import InputHealthEvent


@dataclass(frozen=True, slots=True)
class NumberedCaptureBlock:
    payload: bytes
    sequence_id: int
    timestamp: float
    frame_count: int


class AudioCapture:
    """Low-level MA-USB8 UAC capture; preserves native interleaved S16_LE bytes."""

    def __init__(self, device_name: str, host_api: str, sample_rate: int, channels: int, block_size: int):
        self.device_name, self.host_api = device_name, host_api
        self.sample_rate, self.channels, self.block_size = sample_rate, channels, block_size
        self._stream: Any | None = None
        self._device_index: int | None = None
        self._device_info: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._subscribers: set[queue.Queue[bytes]] = set()
        self._numbered_subscribers: set[queue.Queue[NumberedCaptureBlock]] = set()
        self._health_events: deque[InputHealthEvent] = deque()
        self._next_sequence_id = 0
        self._next_event_id = 0
        self._stream_origin_monotonic = 0.0
        self._recent: deque[bytes] = deque(maxlen=max(1, sample_rate // block_size * 5))
        self._levels = [0.0] * channels
        self._frames_received = 0
        self._last_error: str | None = None
        self._dropped_subscriber_blocks = 0
        self._callback_status_count = 0
        self._input_overflow_count = 0

    @property
    def running(self) -> bool:
        return self._stream is not None

    def _find_device(self, sd: Any) -> tuple[int, dict[str, Any]]:
        host_apis = sd.query_hostapis()
        candidates = []
        for index, raw in enumerate(sd.query_devices()):
            info = dict(raw)
            info["hostapi_name"] = host_apis[info["hostapi"]]["name"]
            if info.get("max_input_channels", 0) >= self.channels:
                candidates.append((index, info))
                if self.device_name.casefold() in str(info.get("name", "")).casefold() and self.host_api.casefold() in str(info["hostapi_name"]).casefold():
                    return index, info
        names = [f"{info.get('name')} [{info.get('hostapi_name')}]" for _, info in candidates]
        raise RuntimeError(f"找不到输入设备 {self.device_name!r} / {self.host_api!r}。可用设备: {names}")

    def start(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            return self._start_locked()

    def _start_locked(self) -> dict[str, Any]:
        if self.running:
            return self.status()
        import sounddevice as sd

        index, info = self._find_device(sd)
        self._device_index, self._device_info, self._last_error = index, info, None
        with self._lock:
            self._recent.clear()
            self._frames_received = 0
            self._dropped_subscriber_blocks = 0
            self._callback_status_count = 0
            self._input_overflow_count = 0
            self._health_events.clear()
            self._next_sequence_id = 0
            self._next_event_id = 0
            self._stream_origin_monotonic = time.monotonic()
        stream: Any | None = None
        try:
            stream = sd.InputStream(device=index, samplerate=self.sample_rate, channels=self.channels, dtype="int16", blocksize=self.block_size, callback=self._callback)
            stream.start()
            actual_rate = float(stream.samplerate)
            if abs(actual_rate - self.sample_rate) > 0.5:
                raise RuntimeError(
                    f"设备实际采样率 {actual_rate:g} Hz 与要求的 {self.sample_rate} Hz 不一致"
                )
            self._stream = stream
        except Exception as exc:
            self._last_error = str(exc)
            # InputStream construction may succeed even when starting it or
            # validating its negotiated sample rate fails.  Keep the original
            # exception, but always attempt both cleanup operations so no
            # PortAudio stream remains open after a failed start.
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            self._stream = None
            raise
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            stream, self._stream = self._stream, None
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
            return self.status()

    def _callback(self, indata: np.ndarray, frames: int, _time: Any, status: Any) -> None:
        with self._lock:
            sequence_id = self._next_sequence_id
            self._next_sequence_id += 1
            timestamp = self._stream_origin_monotonic + self._frames_received / self.sample_rate
        if status:
            self._last_error = str(status)
            with self._lock:
                self._callback_status_count += 1
                if bool(getattr(status, "input_overflow", False)):
                    self._input_overflow_count += 1
                    self._health_events.append(InputHealthEvent(
                        self._next_event_id, time.monotonic(), "input_overflow",
                        sequence_id - 1 if sequence_id else None, sequence_id, None, str(status),
                    ))
                    self._next_event_id += 1
        samples = np.asarray(indata, dtype="<i2")
        payload = samples.tobytes(order="C")
        levels = np.sqrt(np.mean(np.square(samples.astype(np.float32) / 32768.0), axis=0)).tolist()
        with self._lock:
            self._recent.append(payload)
            self._levels = [float(value) for value in levels]
            self._frames_received += frames
            subscribers = tuple(self._subscribers)
            numbered_subscribers = tuple(self._numbered_subscribers)
        for subscriber in subscribers:
            while True:
                try:
                    subscriber.put_nowait(payload)
                    break
                except queue.Full:
                    try:
                        subscriber.get_nowait()
                    except queue.Empty:
                        # A consumer raced us and made room after ``Full``.
                        # Retry the current block instead of losing it silently.
                        continue
                    with self._lock:
                        self._dropped_subscriber_blocks += 1
        numbered = NumberedCaptureBlock(payload, sequence_id, timestamp, frames)
        for subscriber in numbered_subscribers:
            try:
                subscriber.put_nowait(numbered)
            except queue.Full:
                try:
                    dropped = subscriber.get_nowait()
                    subscriber.put_nowait(numbered)
                except queue.Empty:
                    continue
                with self._lock:
                    self._dropped_subscriber_blocks += 1
                    self._health_events.append(InputHealthEvent(
                        self._next_event_id, time.monotonic(), "handoff_drop",
                        dropped.sequence_id - 1 if dropped.sequence_id else None,
                        dropped.sequence_id + 1, dropped.frame_count, "capture handoff queue full",
                    ))
                    self._next_event_id += 1

    def subscribe(self, maxsize: int = 10) -> queue.Queue[bytes]:
        """Subscribe to native blocks.

        Network/API consumers stay bounded by default.  The local recording
        pipeline explicitly requests ``maxsize=0`` so no capture block is
        discarded under temporary DSP backpressure.
        """
        if maxsize < 0:
            raise ValueError("subscriber maxsize 不能小于 0")
        receiver: queue.Queue[bytes] = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.add(receiver)
        return receiver

    def subscribe_numbered(self, maxsize: int = 100) -> queue.Queue[NumberedCaptureBlock]:
        if maxsize <= 0:
            raise ValueError("主链路handoff必须为正容量有界队列")
        receiver: queue.Queue[NumberedCaptureBlock] = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._numbered_subscribers.add(receiver)
        return receiver

    def unsubscribe(self, receiver: queue.Queue[bytes]) -> None:
        with self._lock:
            self._subscribers.discard(receiver)
            self._numbered_subscribers.discard(receiver)

    def take_health_events(self) -> tuple[InputHealthEvent, ...]:
        with self._lock:
            events = tuple(self._health_events)
            self._health_events.clear()
        return events

    def latest(self, blocks: int = 1) -> bytes:
        with self._lock:
            return b"".join(list(self._recent)[-max(1, min(blocks, 50)):])

    def select_channel(self, payload: bytes, channel: int) -> bytes:
        if not 0 <= channel < self.channels:
            raise ValueError(f"channel 必须在 0..{self.channels - 1} 范围内")
        samples = np.frombuffer(payload, dtype="<i2")
        return b"" if samples.size == 0 else samples.reshape(-1, self.channels)[:, channel].astype("<i2", copy=False).tobytes()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"running": self.running, "requested_device": self.device_name, "requested_host_api": self.host_api, "device_index": self._device_index, "device": self._device_info, "sample_rate": self.sample_rate, "channels": self.channels, "sample_format": "s16-le", "layout": "interleaved", "block_size_frames": self.block_size, "bytes_per_frame": self.channels * 2, "frames_received": self._frames_received, "rms_levels": self._levels, "last_error": self._last_error, "callback_status_count": self._callback_status_count, "input_overflow_count": self._input_overflow_count, "dropped_subscriber_blocks": self._dropped_subscriber_blocks}
