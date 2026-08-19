from __future__ import annotations

import json
import threading
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

import numpy as np

from .interface import CdcHotmapFrame


def hotmap_url_from_service_url(service_url: str) -> str:
    """Resolve Layer-1 ``/hotmap/latest`` from any endpoint on that service."""

    parsed = urlsplit(service_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Layer 1 service URL 必须是有效的 HTTP(S) 地址")
    return urlunsplit((parsed.scheme, parsed.netloc, "/hotmap/latest", "", ""))


class HttpHotmapSource:
    """Poll the Layer-1 API when another process owns the CDC serial port."""

    def __init__(self, url: str, *, poll_interval: float = 0.05, timeout: float = 0.6):
        self.url = hotmap_url_from_service_url(url)
        self.poll_interval = max(0.01, float(poll_interval))
        self.timeout = max(0.01, float(timeout))
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: CdcHotmapFrame | None = None
        self._last_error: str | None = None

    @property
    def running(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            with self._lock:
                self._latest = None
                self._last_error = None
            self._fetch_once()  # Fail synchronously so ``required`` is honoured.
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            try:
                self._thread.start()
            except Exception:
                self._thread = None
                self._stop.set()
                raise

    def stop(self) -> None:
        with self._lifecycle_lock:
            stop_event, thread = self._stop, self._thread
            stop_event.set()
            self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.timeout + self.poll_interval))

    def latest_hotmap_frame(self) -> CdcHotmapFrame | None:
        with self._lock:
            return self._latest

    def _fetch_once(self) -> None:
        with urlopen(self.url, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not payload.get("available"):
            with self._lock:
                self._latest = None
            return
        frame = CdcHotmapFrame(
            np.asarray(payload["matrix"], dtype=np.uint8),
            payload["sequence_id"],
            payload["timestamp"],
            payload.get("received_at"),
        )
        with self._lock:
            self._latest = frame
            self._last_error = None

    def _poll_loop(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                self._fetch_once()
            except Exception as exc:
                # Do not attach a stale pre-disconnect snapshot to later audio.
                # InputPipeline's required policy applies to startup.
                with self._lock:
                    self._latest = None
                    self._last_error = str(exc)
