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
        # Serialize open/write/close as one lifecycle.  The RLock lets write()
        # lazily call start() without racing /serial/start or /serial/stop.
        self._lifecycle_lock = threading.RLock()
        self._subscribers: set[queue.Queue[bytes]] = set()
        self._recent: deque[bytes] = deque(maxlen=100)
        self._bytes_received = self._bytes_sent = self._hotmap_frames = 0
        self._last_error: str | None = None
        self._frame_buffer = bytearray()
        self._hotmap: CdcHotmapFrame | None = None

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
        with self._lifecycle_lock:
            if self.running:
                return self.status()
            stale_port, stale_stop = self._serial, self._stop
            stale_stop.set()
            self._serial = None
            self._thread = None
            if stale_port is not None and stale_port.is_open:
                stale_port.close()
            # A new CDC lifecycle must never expose a snapshot retained from an
            # earlier connection, including when opening the new port fails.
            with self._lock:
                self._frame_buffer.clear()
                self._hotmap = None
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
        with self._lifecycle_lock:
            stop_event, thread, port = self._stop, self._thread, self._serial
            stop_event.set()
            self._thread = None
            self._serial = None
            if port is not None and port.is_open:
                port.close()
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=1.0)
        return self.status()

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
            self._hotmap_frames += 1

    def write(self, data: bytes) -> int:
        with self._lifecycle_lock:
            if not self.running:
                self.start()
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
