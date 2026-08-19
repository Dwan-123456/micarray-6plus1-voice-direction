from __future__ import annotations

import logging
import threading
from typing import Protocol

from .calibration import ChannelCalibrator
from .continuity import CalibrationContinuityGuard
from .interface import CdcHotmapFrame, DecodedAudio
from .sources import AudioSource

logger = logging.getLogger(__name__)


class HotmapSource(Protocol):
    def start(self) -> object: ...
    def stop(self) -> object: ...
    def latest_hotmap_frame(self) -> CdcHotmapFrame | None: ...


class InputPipeline:
    """Layer-1 facade: audio plus the latest optional CDC Hotmap snapshot."""

    def __init__(
        self,
        source: AudioSource,
        calibrator: ChannelCalibrator,
        hotmap_source: HotmapSource | None = None,
        *,
        owns_hotmap_source: bool = True,
        hotmap_required: bool = False,
        timestamp_tolerance_ms: float = 5.0,
    ):
        self.source, self.calibrator = source, calibrator
        self.hotmap_source = hotmap_source
        self.owns_hotmap_source = owns_hotmap_source
        self.hotmap_required = hotmap_required
        self._hotmap_started = False
        self._hotmap_read_enabled = False
        self._source_started = False
        self._pending_health_events: list[object] = []
        self._lifecycle_lock = threading.RLock()
        self.continuity_guard = CalibrationContinuityGuard(
            calibrator, timestamp_tolerance_ms=timestamp_tolerance_ms
        )

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._source_started:
                return
            if self.hotmap_source is not None:
                try:
                    self.hotmap_source.start()
                    self._hotmap_started = True
                    self._hotmap_read_enabled = True
                except Exception as exc:
                    self._hotmap_started = False
                    self._hotmap_read_enabled = False
                    if self.hotmap_required:
                        raise
                    logger.warning("CDC hotmap unavailable; continuing with audio only: %s", exc)
            try:
                self.source.start()
                self._source_started = True
                self.continuity_guard.start()
            except Exception:
                if self._hotmap_started and self.owns_hotmap_source and self.hotmap_source is not None:
                    self.hotmap_source.stop()
                self._hotmap_started = False
                self._hotmap_read_enabled = False
                raise

    def read(self, timeout: float | None = None) -> DecodedAudio | None:
        audio = self.source.read(timeout)
        if audio is None:
            return None
        health_events = ()
        take_events = getattr(self.source, "take_health_events", None)
        if take_events is not None:
            health_events = tuple(take_events())
            self._pending_health_events.extend(health_events)
            for event in health_events:
                self.continuity_guard.notify(event)
        calibrated = self.continuity_guard.process(audio)
        if self._hotmap_read_enabled and self.hotmap_source is not None:
            try:
                calibrated.hotmap = self.hotmap_source.latest_hotmap_frame()
            except Exception as exc:
                if self.hotmap_required:
                    raise
                self._hotmap_read_enabled = False
                calibrated.hotmap = None
                logger.warning("CDC hotmap read failed; continuing with audio only: %s", exc)
        return calibrated

    def take_health_events(self) -> tuple[object, ...]:
        events = tuple(self._pending_health_events)
        self._pending_health_events.clear()
        return events

    def take_hotmap_frames(self) -> tuple[CdcHotmapFrame, ...]:
        method = getattr(self.hotmap_source, "take_hotmap_frames", None)
        if method is None:
            return ()
        return tuple(method())

    def stop(self) -> None:
        with self._lifecycle_lock:
            try:
                if self._source_started:
                    self.source.stop()
            finally:
                self._source_started = False
                if self._hotmap_started and self.owns_hotmap_source and self.hotmap_source is not None:
                    self.hotmap_source.stop()
                self._hotmap_started = False
                self._hotmap_read_enabled = False

    def __enter__(self) -> "InputPipeline":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()
