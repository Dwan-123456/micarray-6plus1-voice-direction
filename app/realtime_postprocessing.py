from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, replace
from time import monotonic
from typing import Callable, Protocol

from layer4_speech_separation import Layer4LongAudioInput


@dataclass(frozen=True, slots=True)
class RealtimePostprocessingSnapshot:
    """Latest replaceable L4/L5/L6 preview produced by the configurable sidecar."""

    session_id: str
    revision: int
    is_final: bool
    valid_through_sample_48k: int
    processed_blocks: int
    l4_processed: tuple[object, ...]
    l5_results: tuple[object, ...]
    l6_result: object | None
    stage_durations_seconds: tuple[tuple[str, float], ...] = ()
    queued_blocks: int = 0

    def __post_init__(self) -> None:
        if not self.session_id or self.revision <= 0:
            raise ValueError("realtime postprocessing snapshot identity is invalid")
        if self.valid_through_sample_48k < 0 or self.valid_through_sample_48k % 960:
            raise ValueError("realtime postprocessing watermark must align to 20 ms")
        if self.processed_blocks < 0 or self.queued_blocks < 0:
            raise ValueError("realtime postprocessing counters cannot be negative")
        if type(self.is_final) is not bool:
            raise ValueError("realtime postprocessing final flag must be bool")


@dataclass(frozen=True, slots=True)
class RealtimePostprocessingStatus:
    state: str
    queued_blocks: int
    submitted_blocks: int
    processed_blocks: int
    dropped_blocks: int
    latest_revision: int
    error: str | None
    model_load_seconds: float


class RealtimePostprocessor(Protocol):
    def push(
        self,
        source: Layer4LongAudioInput,
        *,
        is_final_chunk: bool = False,
    ) -> RealtimePostprocessingSnapshot | None: ...

    def finalize(self) -> RealtimePostprocessingSnapshot | None: ...

    def abort(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _BlockWork:
    source: Layer4LongAudioInput
    is_final_chunk: bool


_FINISH = object()
_ABORT = object()


class RealtimePostprocessingService:
    """Non-blocking, bounded owner of the progressive L4/L5/L6 sidecar.

    L1-L3 callers only use :meth:`submit`.  A full queue is reported as an
    explicit downstream failure and never applies backpressure to the 20 ms
    runtime graph. Final sealing validates exact reusable tracks and falls back
    only for missing or unsafe tracks.
    """

    def __init__(
        self,
        factory: Callable[[], RealtimePostprocessor],
        *,
        queue_chunks: int,
        enabled: bool = True,
    ) -> None:
        if queue_chunks <= 0:
            raise ValueError("realtime postprocessing queue must be positive")
        self._factory = factory
        self._capacity = int(queue_chunks)
        self._enabled = bool(enabled)
        self._lock = threading.RLock()
        self._mailbox: queue.Queue[object] = queue.Queue(maxsize=self._capacity)
        self.latest: queue.Queue[RealtimePostprocessingSnapshot] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._processor: RealtimePostprocessor | None = None
        self._state = "disabled" if not self._enabled else "idle"
        self._submitted_blocks = 0
        self._processed_blocks = 0
        self._dropped_blocks = 0
        self._latest_revision = 0
        self._error: str | None = None
        self._model_load_seconds = 0.0
        self._final_snapshot: RealtimePostprocessingSnapshot | None = None
        self._accepting = False
        self._finished = threading.Event()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def active(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def available_slots(self) -> int:
        """Conservative non-blocking admission budget for the single producer."""

        with self._lock:
            if (
                not self._accepting
                or self._error is not None
                or self._thread is None
                or not self._thread.is_alive()
            ):
                return 0
            return max(0, self._capacity - self._mailbox.qsize())

    @property
    def status(self) -> RealtimePostprocessingStatus:
        with self._lock:
            return RealtimePostprocessingStatus(
                self._state,
                self._mailbox.qsize(),
                self._submitted_blocks,
                self._processed_blocks,
                self._dropped_blocks,
                self._latest_revision,
                self._error,
                self._model_load_seconds,
            )

    @property
    def final_snapshot(self) -> RealtimePostprocessingSnapshot | None:
        """Retain the drained final result independently of the latest-only UI queue."""

        with self._lock:
            return self._final_snapshot

    def _clear_mailboxes(self) -> None:
        for mailbox in (self._mailbox, self.latest):
            while True:
                try:
                    mailbox.get_nowait()
                except queue.Empty:
                    break

    def start(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._clear_mailboxes()
            self._processor = None
            self._state = "waiting"
            self._submitted_blocks = 0
            self._processed_blocks = 0
            self._dropped_blocks = 0
            self._latest_revision = 0
            self._error = None
            self._model_load_seconds = 0.0
            self._final_snapshot = None
            self._accepting = True
            self._finished.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="application-runtime-layer456-stream",
                daemon=True,
            )
            self._thread.start()

    def submit(
        self,
        source: Layer4LongAudioInput,
        *,
        is_final_chunk: bool = False,
    ) -> bool:
        if not self._enabled:
            return False
        if not isinstance(source, Layer4LongAudioInput):
            raise TypeError("realtime postprocessing requires a Layer4LongAudioInput chunk")
        with self._lock:
            if (
                not self._accepting
                or self._thread is None
                or not self._thread.is_alive()
            ):
                return False
            if self._error is not None:
                self._dropped_blocks += 1
                return False
            try:
                self._mailbox.put_nowait(_BlockWork(source, bool(is_final_chunk)))
            except queue.Full:
                self._dropped_blocks += 1
                self._error = "realtime_layer456_queue_overflow"
                self._state = "failed"
                self._accepting = False
                return False
            self._submitted_blocks += 1
            self._state = "queued"
            return True

    def finish(self, *, timeout: float | None = None) -> bool:
        """Drain accepted chunks, flush model tails, and wait for completion."""

        if not self._enabled:
            return True
        with self._lock:
            worker = self._thread
            if worker is None:
                return True
            self._accepting = False
            if self._error is None:
                self._state = "finishing"
        deadline = None if timeout is None else monotonic() + float(timeout)
        while worker.is_alive():
            try:
                self._mailbox.put(_FINISH, timeout=0.05)
                break
            except queue.Full:
                if deadline is not None and monotonic() >= deadline:
                    with self._lock:
                        self._error = "realtime_layer456_finish_timeout"
                        self._state = "failed"
                    self._queue_abort_marker()
                    return False
        worker.join(
            timeout=None if deadline is None else max(0.0, deadline - monotonic())
        )
        stopped = not worker.is_alive()
        if not stopped:
            with self._lock:
                self._error = "realtime_layer456_finish_timeout"
                self._state = "failed"
            self._queue_abort_marker()
        return stopped

    def _queue_abort_marker(self) -> None:
        """Discard queued previews and guarantee the worker eventually exits."""

        while True:
            try:
                item = self._mailbox.get_nowait()
                if isinstance(item, _BlockWork):
                    with self._lock:
                        self._dropped_blocks += 1
            except queue.Empty:
                break
        try:
            self._mailbox.put_nowait(_ABORT)
        except queue.Full:
            # A concurrent worker can only have made progress; retry once
            # after removing the value that raced into the single mailbox.
            try:
                item = self._mailbox.get_nowait()
                if isinstance(item, _BlockWork):
                    with self._lock:
                        self._dropped_blocks += 1
            except queue.Empty:
                pass
            self._mailbox.put_nowait(_ABORT)

    def abort(self, *, timeout: float = 30.0) -> bool:
        """Drop waiting preview work while allowing an active model call to return."""

        if not self._enabled:
            return True
        with self._lock:
            worker = self._thread
            if worker is None:
                return True
            self._accepting = False
            self._state = "aborting"
        self._queue_abort_marker()
        worker.join(timeout=float(timeout))
        return not worker.is_alive()

    def _publish(self, snapshot: RealtimePostprocessingSnapshot) -> None:
        durations = dict(snapshot.stage_durations_seconds)
        durations["model_load"] = self._model_load_seconds
        value = replace(
            snapshot,
            queued_blocks=self._mailbox.qsize(),
            stage_durations_seconds=tuple(durations.items()),
        )
        with self._lock:
            # Once admission or drain has failed, an incomplete accepted
            # prefix must never supersede the UI with a counterfeit final.
            if self._error is not None and value.is_final:
                return
            try:
                self.latest.put_nowait(value)
            except queue.Full:
                try:
                    self.latest.get_nowait()
                except queue.Empty:
                    pass
                self.latest.put_nowait(value)
            self._processed_blocks = max(self._processed_blocks, value.processed_blocks)
            self._latest_revision = value.revision
            if value.is_final:
                self._final_snapshot = value
            if self._error is None:
                self._state = "final" if value.is_final else "running"

    def _run(self) -> None:
        finalized = False
        try:
            with self._lock:
                self._state = "loading"
            started = monotonic()
            self._processor = self._factory()
            with self._lock:
                self._model_load_seconds = monotonic() - started
                if self._error is None:
                    self._state = "waiting"
            while True:
                work = self._mailbox.get()
                if work is _ABORT:
                    processor = self._processor
                    if processor is not None:
                        processor.abort()
                    finalized = True
                    with self._lock:
                        self._state = "failed" if self._error is not None else "aborted"
                    return
                if work is _FINISH:
                    processor = self._processor
                    if processor is not None:
                        snapshot = processor.finalize()
                        finalized = True
                        if snapshot is not None:
                            self._publish(snapshot)
                    with self._lock:
                        if self._error is not None:
                            self._state = "failed"
                        elif self._state != "final":
                            self._state = "finished"
                    return
                if not isinstance(work, _BlockWork):
                    continue
                assert self._processor is not None
                snapshot = self._processor.push(
                    work.source,
                    is_final_chunk=work.is_final_chunk,
                )
                if snapshot is not None:
                    self._publish(snapshot)
        except Exception as exc:
            processor = self._processor
            if processor is not None and not finalized:
                try:
                    processor.abort()
                except Exception:
                    pass
            with self._lock:
                self._error = str(exc)
                self._state = "failed"
        finally:
            while True:
                try:
                    waiting = self._mailbox.get_nowait()
                    if isinstance(waiting, _BlockWork):
                        with self._lock:
                            self._dropped_blocks += 1
                except queue.Empty:
                    break
            with self._lock:
                self._accepting = False
                self._thread = None
                self._processor = None
            self._finished.set()
