from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any

import numpy as np

from .interface import CdcHotmapFrame


class SerialDevice:
    """Read CDC bytes and decode official 16x16 hotmap frames."""

    HOTMAP_HEADER = b"\xff" * 16
    HOTMAP_PAYLOAD_SIZE = 256
    HOTMAP_FRAME_SIZE = 272

    def __init__(self, port: str, baudrate: int):
        self.port, self.baudrate = port, baudrate
        self._serial: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # Public lifecycle transitions may wait for the reader, so they use a
        # separate lock that the reader never needs while exiting.
        self._transition_lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._subscribers: set[queue.Queue[bytes]] = set()
        self._recent: deque[bytes] = deque(maxlen=100)
        self._bytes_received = self._bytes_sent = self._hotmap_frames = 0
        self._last_error: str | None = None
        self._frame_buffer = bytearray()
        self._hotmap: CdcHotmapFrame | None = None
        # v1.4 does not consume the legacy DPD hot-map stream.  Retain only
        # the newest frame for the compatibility accessor instead of queuing
        # every 50 Hz CDC frame forever during long live sessions.
        self._pending_hotmaps: deque[CdcHotmapFrame] = deque(maxlen=1)

    @property
    def running(self) -> bool:
        with self._lifecycle_lock:
            return (
                self._serial is not None
                and bool(self._serial.is_open)
                and self._thread is not None
                and self._thread.is_alive()
                and self._last_error is None
            )

    def start(self) -> dict[str, Any]:
        with self._transition_lock:
            with self._lifecycle_lock:
                if self.running:
                    return self.status()
                stale_port, stale_thread, stale_stop = self._serial, self._thread, self._stop
                stale_stop.set()
                self._serial = None
                self._thread = None
            if not self._wait_then_close(stale_port, stale_thread, timeout=1.0):
                message = "previous CDC reader did not stop within 1.0 s"
                with self._lifecycle_lock:
                    self._serial = stale_port
                    self._thread = stale_thread
                    self._stop = stale_stop
                    self._last_error = message
                raise RuntimeError(message)
            # A new CDC lifecycle must never expose a snapshot retained from an
            # earlier connection, including when opening the new port fails.
            with self._lock:
                self._frame_buffer.clear()
                self._hotmap = None
                self._pending_hotmaps.clear()
                self._hotmap_frames = 0
            import serial
            try:
                self._serial = serial.Serial(self.port, self.baudrate, timeout=0.1, write_timeout=1.0)
            except Exception as exc:
                self._serial = None
                self._last_error = str(exc)
                raise
            self._last_error = None
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._read_loop,
                args=(self._serial, stop_event),
                daemon=True,
            )
            self._stop = stop_event
            self._thread = thread
            try:
                thread.start()
            except Exception as exc:
                stop_event.set()
                port, self._serial = self._serial, None
                self._thread = None
                if port is not None and port.is_open:
                    port.close()
                self._last_error = str(exc)
                raise
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self._transition_lock:
            with self._lifecycle_lock:
                stop_event, thread, port = self._stop, self._thread, self._serial
                stop_event.set()
                self._thread = None
                self._serial = None
            if not self._wait_then_close(port, thread, timeout=1.0):
                with self._lifecycle_lock:
                    self._serial = port
                    self._thread = thread
                    self._stop = stop_event
                    self._last_error = "CDC reader did not stop within 1.0 s"
            return self.status()

    @staticmethod
    def _wait_then_close(port: Any | None, thread: threading.Thread | None, *, timeout: float) -> bool:
        """Wait for the reader that owns ``port`` before closing a leftover handle."""

        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=max(0.0, timeout))
        if thread is not None and thread.is_alive():
            return False
        if port is not None and port.is_open:
            port.close()
        return True

    def _read_loop(self, port: Any, stop_event: threading.Event) -> None:
        try:
            while not stop_event.is_set():
                data = port.read(port.in_waiting or 1)
                if data and not stop_event.is_set():
                    self._publish(data)
        except Exception as exc:
            if not stop_event.is_set():
                self._last_error = str(exc)
        finally:
            if port is not None and port.is_open:
                port.close()
            with self._lock:
                if not stop_event.is_set():
                    self._frame_buffer.clear()
                    self._hotmap = None
            with self._lifecycle_lock:
                if self._serial is port:
                    self._serial = None
                    if self._thread is threading.current_thread():
                        self._thread = None

    def _publish(self, data: bytes) -> None:
        with self._lock:
            self._recent.append(data)
            self._bytes_received += len(data)
            self._parse_hotmap_locked(data)
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(data)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(data)
                except queue.Empty:
                    pass

    def _parse_hotmap_locked(self, data: bytes) -> None:
        self._frame_buffer.extend(data)
        while True:
            start = self._frame_buffer.find(self.HOTMAP_HEADER)
            if start < 0:
                if len(self._frame_buffer) > 15:
                    del self._frame_buffer[:-15]
                return
            if start:
                del self._frame_buffer[:start]
            if len(self._frame_buffer) < self.HOTMAP_FRAME_SIZE:
                return
            payload = bytes(self._frame_buffer[16:self.HOTMAP_FRAME_SIZE])
            del self._frame_buffer[:self.HOTMAP_FRAME_SIZE]
            self._hotmap = CdcHotmapFrame(
                np.frombuffer(payload, dtype=np.uint8).reshape(16, 16),
                self._hotmap_frames,
                time.monotonic(),
                time.time(),
            )
            self._pending_hotmaps.append(self._hotmap)
            self._hotmap_frames += 1

    def write(self, data: bytes) -> int:
        with self._transition_lock:
            if not self.running:
                self.start()
            with self._lifecycle_lock:
                assert self._serial is not None
                count = int(self._serial.write(data))
            with self._lock:
                self._bytes_sent += count
            return count

    def subscribe(self) -> queue.Queue[bytes]:
        receiver: queue.Queue[bytes] = queue.Queue(maxsize=100)
        with self._lock:
            self._subscribers.add(receiver)
        return receiver

    def unsubscribe(self, receiver: queue.Queue[bytes]) -> None:
        with self._lock:
            self._subscribers.discard(receiver)

    def latest(self) -> bytes:
        with self._lock:
            return b"".join(self._recent)

    def latest_hotmap_frame(self) -> CdcHotmapFrame | None:
        """Return the latest immutable snapshot, or ``None`` before a frame."""
        with self._lock:
            return self._hotmap

    def take_hotmap_frames(self) -> tuple[CdcHotmapFrame, ...]:
        """Drain the newest compatibility frame without retaining a backlog."""
        with self._lock:
            frames = tuple(self._pending_hotmaps)
            self._pending_hotmaps.clear()
            return frames

    def latest_hotmap(self) -> dict[str, Any]:
        with self._lock:
            frame = self._hotmap
            return {
                "available": frame is not None,
                "width": 16,
                "height": 16,
                "frame_count": self._hotmap_frames,
                "sequence_id": None if frame is None else frame.sequence_id,
                "timestamp": None if frame is None else frame.timestamp,
                "received_at": None if frame is None else frame.received_at,
                "matrix": None if frame is None else frame.matrix.tolist(),
            }

    def status(self) -> dict[str, Any]:
        return {"running": self.running, "port": self.port, "baudrate": self.baudrate, "bytes_received": self._bytes_received, "bytes_sent": self._bytes_sent, "hotmap_frames": self._hotmap_frames, "last_error": self._last_error}
