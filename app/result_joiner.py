from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass

from .compute_cache import GpuArtifactError, artifact_byte_size
from .processing_contracts import (
    JoinedWindowResult,
    L2StageResult,
    L3StageResult,
    L5StageResult,
    StageState,
    TerminalStageResult,
    WindowKey,
    WindowWorkItem,
)


class ResultJoinerError(RuntimeError):
    pass


class TimelineGapError(ResultJoinerError):
    pass


class DuplicateWindowError(ResultJoinerError):
    pass


class UnknownWindowError(ResultJoinerError):
    pass


class DuplicateStageResultError(ResultJoinerError):
    pass


class JoinerCapacityError(ResultJoinerError):
    pass


class ResultDeliveryError(ResultJoinerError):
    pass


@dataclass(frozen=True, slots=True)
class TimelineGap:
    session_id: str
    stream_epoch: int
    previous_decision_sample: int
    next_decision_sample: int
    expected_hop_samples: int
    reason: str

    def __post_init__(self) -> None:
        if not self.session_id or min(
            self.stream_epoch, self.previous_decision_sample, self.next_decision_sample
        ) < 0:
            raise ValueError("invalid timeline gap identity")
        if self.expected_hop_samples <= 0 or not self.reason:
            raise ValueError("timeline gap requires a positive hop and explicit reason")
        if self.next_decision_sample <= self.previous_decision_sample + self.expected_hop_samples:
            raise ValueError("TimelineGap must describe at least one missing window")


@dataclass(frozen=True, slots=True)
class ResultJoinerSnapshot:
    pending_windows: int
    pending_bytes: int
    max_pending_windows: int
    max_pending_bytes: int
    joined_windows: int
    explicit_gaps: int
    committed_through: tuple[tuple[str, int, int], ...]


@dataclass(slots=True)
class _PendingWindow:
    work_item: WindowWorkItem
    byte_size: int
    l2: L2StageResult | None = None
    l3: L3StageResult | None = None
    l5: L5StageResult | None = None

    @property
    def ready(self) -> bool:
        return self.l2 is not None and self.l3 is not None and self.l5 is not None


@dataclass(slots=True)
class _Stream:
    windows: OrderedDict[WindowKey, _PendingWindow]
    last_registered_sample: int | None = None
    committed_through: int | None = None


class ResultJoiner:
    """Join independently completed stages without crossing an unreported sample gap."""

    def __init__(
        self,
        *,
        expected_hop_samples: int = 960,
        max_pending_windows: int = 64,
        max_pending_bytes: int = 256 * 1024 * 1024,
        on_joined: Callable[[JoinedWindowResult], None] | None = None,
        on_gap: Callable[[TimelineGap], None] | None = None,
    ) -> None:
        if min(expected_hop_samples, max_pending_windows, max_pending_bytes) <= 0:
            raise ValueError("joiner hop and capacity limits must be positive")
        self.expected_hop_samples = expected_hop_samples
        self.max_pending_windows = max_pending_windows
        self.max_pending_bytes = max_pending_bytes
        self._on_joined = on_joined
        self._on_gap = on_gap
        self._streams: dict[tuple[str, int], _Stream] = {}
        self._ready: deque[JoinedWindowResult] = deque()
        self._gaps: deque[TimelineGap] = deque()
        self._pending_windows = 0
        self._pending_bytes = 0
        self._joined_windows = 0
        self._explicit_gaps = 0
        self._lock = threading.RLock()

    def register(self, work_item: WindowWorkItem, *, preceding_gap_reason: str | None = None) -> None:
        if not isinstance(work_item, WindowWorkItem):
            raise TypeError("ResultJoiner.register requires a WindowWorkItem")
        work_bytes = artifact_byte_size(work_item)
        if work_bytes > self.max_pending_bytes:
            raise JoinerCapacityError("one WindowWorkItem exceeds the joiner byte limit")

        gap: TimelineGap | None = None
        with self._lock:
            key = work_item.key
            stream = self._streams.setdefault(key.stream_key, _Stream(OrderedDict()))
            if stream.committed_through is not None and key.decision_sample <= stream.committed_through:
                raise DuplicateWindowError("window is at or before the committed timeline watermark")
            if key in stream.windows:
                raise DuplicateWindowError(f"window was already registered: {key}")
            previous = stream.last_registered_sample
            if previous is not None:
                expected = previous + self.expected_hop_samples
                if key.decision_sample != expected:
                    if key.decision_sample < expected:
                        raise TimelineGapError(
                            f"non-monotonic or misaligned decision sample {key.decision_sample}; expected {expected}"
                        )
                    if not preceding_gap_reason:
                        raise TimelineGapError(
                            f"unreported timeline gap: expected {expected}, received {key.decision_sample}"
                        )
                    gap = TimelineGap(
                        key.session_id,
                        key.stream_epoch,
                        previous,
                        key.decision_sample,
                        self.expected_hop_samples,
                        preceding_gap_reason,
                    )
            elif preceding_gap_reason is not None:
                raise ValueError("the first window of an epoch cannot have a preceding joiner gap")
            if self._pending_windows >= self.max_pending_windows:
                raise JoinerCapacityError("pending-window capacity reached; caller must terminate or drain explicitly")
            if self._pending_bytes + work_bytes > self.max_pending_bytes:
                raise JoinerCapacityError("pending-byte capacity reached; caller must terminate or drain explicitly")
            stream.windows[key] = _PendingWindow(work_item, work_bytes)
            stream.last_registered_sample = key.decision_sample
            self._pending_windows += 1
            self._pending_bytes += work_bytes
            if gap is not None:
                self._explicit_gaps += 1
        if gap is not None:
            if self._on_gap is None:
                with self._lock:
                    self._gaps.append(gap)
            else:
                try:
                    # External callbacks must never run while the timeline lock
                    # is held: they may block on I/O or call back into the joiner.
                    self._on_gap(gap)
                except Exception as exc:
                    # Registration is committed, and the explicit gap remains observable for retry/audit.
                    with self._lock:
                        self._gaps.append(gap)
                    raise ResultDeliveryError("timeline-gap callback failed") from exc

    def submit(self, result: TerminalStageResult) -> None:
        if not isinstance(result, (L2StageResult, L3StageResult, L5StageResult)):
            raise TypeError("joiner accepts only L2StageResult, L3StageResult, or L5StageResult")
        if not result.is_terminal:
            raise ValueError("joiner accepts terminal stage results only")
        try:
            result_bytes = artifact_byte_size(result)
        except GpuArtifactError:
            raise
        joined: tuple[JoinedWindowResult, ...]
        with self._lock:
            stream = self._streams.get(result.key.stream_key)
            pending = None if stream is None else stream.windows.get(result.key)
            if pending is None:
                raise UnknownWindowError(f"stage result has no registered work item: {result.key}")
            attribute = result.stage_name
            if getattr(pending, attribute) is not None:
                raise DuplicateStageResultError(f"{attribute} result already published for {result.key}")
            if self._pending_bytes + result_bytes > self.max_pending_bytes:
                raise JoinerCapacityError("stage result would exceed the pending-byte limit")
            setattr(pending, attribute, result)
            pending.byte_size += result_bytes
            self._pending_bytes += result_bytes
            joined = self._flush_stream_locked(result.key.stream_key)
        self._publish_joined(joined)

    def submit_l2(self, result: L2StageResult) -> None:
        self.submit(result)

    def submit_l3(self, result: L3StageResult) -> None:
        self.submit(result)

    def submit_l5(self, result: L5StageResult) -> None:
        self.submit(result)

    def terminate_window(self, key: WindowKey, state: StageState, reason: str) -> None:
        """Explicitly terminate every not-yet-published stage, retaining completed stages."""

        if state not in {
            StageState.SKIPPED,
            StageState.FAILED,
            StageState.TIMED_OUT,
            StageState.DROPPED,
            StageState.CANCELLED,
        }:
            raise ValueError("terminate_window requires a non-completed terminal state")
        if not reason:
            raise ValueError("window termination reason cannot be empty")
        joined: tuple[JoinedWindowResult, ...]
        with self._lock:
            stream = self._streams.get(key.stream_key)
            pending = None if stream is None else stream.windows.get(key)
            if pending is None:
                raise UnknownWindowError(f"cannot terminate unknown window: {key}")
            additions: list[tuple[str, TerminalStageResult, int]] = []
            for attribute, result_type in (
                ("l2", L2StageResult),
                ("l3", L3StageResult),
                ("l5", L5StageResult),
            ):
                if getattr(pending, attribute) is None:
                    now_ns = time.monotonic_ns()
                    result = result_type.terminal(
                        key,
                        state,
                        reason,
                        started_monotonic_ns=now_ns,
                        finished_monotonic_ns=now_ns,
                        error=reason if state is StageState.FAILED else None,
                    )
                    result_bytes = artifact_byte_size(result)
                    additions.append((attribute, result, result_bytes))
            additional_bytes = sum(item[2] for item in additions)
            if self._pending_bytes + additional_bytes > self.max_pending_bytes:
                raise JoinerCapacityError("terminal markers would exceed the pending-byte limit")
            for attribute, result, result_bytes in additions:
                setattr(pending, attribute, result)
                pending.byte_size += result_bytes
                self._pending_bytes += result_bytes
            joined = self._flush_stream_locked(key.stream_key)
        self._publish_joined(joined)

    def skip_missing_downstream(self, key: WindowKey, reason: str) -> None:
        """Mark only missing L3/L5 results skipped after a valid terminal L2 result."""

        if not reason:
            raise ValueError("skip reason cannot be empty")
        joined: tuple[JoinedWindowResult, ...]
        with self._lock:
            stream = self._streams.get(key.stream_key)
            pending = None if stream is None else stream.windows.get(key)
            if pending is None:
                raise UnknownWindowError(f"cannot skip downstream for unknown window: {key}")
            if pending.l2 is None:
                raise ValueError("L2 must be terminal before downstream stages can be skipped")
            additions: list[tuple[str, TerminalStageResult, int]] = []
            for attribute, result_type in (("l3", L3StageResult), ("l5", L5StageResult)):
                if getattr(pending, attribute) is None:
                    now_ns = time.monotonic_ns()
                    result = result_type.terminal(
                        key,
                        StageState.SKIPPED,
                        reason,
                        started_monotonic_ns=now_ns,
                        finished_monotonic_ns=now_ns,
                    )
                    result_bytes = artifact_byte_size(result)
                    additions.append((attribute, result, result_bytes))
            additional_bytes = sum(item[2] for item in additions)
            if self._pending_bytes + additional_bytes > self.max_pending_bytes:
                raise JoinerCapacityError("skip markers would exceed the pending-byte limit")
            for attribute, result, result_bytes in additions:
                setattr(pending, attribute, result)
                pending.byte_size += result_bytes
                self._pending_bytes += result_bytes
            joined = self._flush_stream_locked(key.stream_key)
        self._publish_joined(joined)

    def drain_ready(self, limit: int | None = None) -> tuple[JoinedWindowResult, ...]:
        with self._lock:
            if limit is not None and limit < 0:
                raise ValueError("ready drain limit cannot be negative")
            count = len(self._ready) if limit is None else min(limit, len(self._ready))
            output = tuple(self._ready.popleft() for _ in range(count))
            return output

    def drain_gaps(self) -> tuple[TimelineGap, ...]:
        with self._lock:
            output = tuple(self._gaps)
            self._gaps.clear()
            return output

    def pending_keys(self) -> tuple[WindowKey, ...]:
        with self._lock:
            return tuple(
                key
                for stream_key in sorted(self._streams)
                for key in self._streams[stream_key].windows
            )

    def prune_completed_streams(
        self, session_id: str, *, before_epoch: int
    ) -> tuple[tuple[str, int], ...]:
        """Forget closed epoch metadata once the caller has advanced past it.

        A stream is removable only after every registered window is terminal.
        The runtime calls this after observing a newer epoch, so removing the
        old ``last_registered_sample``/watermark cannot permit an old stream to
        be reopened. Global joined/gap counters remain cumulative.
        """

        if not session_id or before_epoch < 0:
            raise ValueError("stream-history pruning requires a session and non-negative epoch")
        with self._lock:
            removable = tuple(
                stream_key
                for stream_key, stream in self._streams.items()
                if stream_key[0] == session_id
                and stream_key[1] < before_epoch
                and not stream.windows
            )
            for stream_key in removable:
                self._streams.pop(stream_key, None)
            return removable

    def close(self, *, cancel_pending: bool = False, reason: str = "joiner_shutdown") -> None:
        if cancel_pending:
            while keys := self.pending_keys():
                self.terminate_window(keys[0], StageState.CANCELLED, reason)
            return
        if self.pending_keys():
            raise ResultJoinerError("cannot close while windows remain pending")

    def snapshot(self) -> ResultJoinerSnapshot:
        with self._lock:
            committed = tuple(
                (session_id, epoch, stream.committed_through)
                for (session_id, epoch), stream in sorted(self._streams.items())
                if stream.committed_through is not None
            )
            return ResultJoinerSnapshot(
                pending_windows=self._pending_windows,
                pending_bytes=self._pending_bytes,
                max_pending_windows=self.max_pending_windows,
                max_pending_bytes=self.max_pending_bytes,
                joined_windows=self._joined_windows,
                explicit_gaps=self._explicit_gaps,
                committed_through=committed,
            )

    def _flush_stream_locked(
        self, stream_key: tuple[str, int]
    ) -> tuple[JoinedWindowResult, ...]:
        """Materialize consecutive terminal windows while holding ``_lock``.

        Delivery is intentionally deferred to :meth:`_publish_joined`, after
        the lock has been released.  This keeps slow disk/queue consumers from
        blocking stage publication or the L1 admission/drop path while owning
        the authoritative timeline lock.
        """

        stream = self._streams[stream_key]
        ready: list[JoinedWindowResult] = []
        while stream.windows:
            key = next(iter(stream.windows))
            pending = stream.windows[key]
            if not pending.ready:
                break
            assert pending.l2 is not None and pending.l3 is not None and pending.l5 is not None
            joined = JoinedWindowResult(
                work_item=pending.work_item,
                l2=pending.l2,
                l3=pending.l3,
                l5=pending.l5,
                terminal_reason="",
                completed_monotonic_ns=time.monotonic_ns(),
            )
            stream.windows.pop(key)
            stream.committed_through = key.decision_sample
            self._pending_windows -= 1
            self._pending_bytes -= pending.byte_size
            self._joined_windows += 1
            ready.append(joined)
        return tuple(ready)

    def _publish_joined(self, joined: tuple[JoinedWindowResult, ...]) -> None:
        if not joined:
            return
        callback = self._on_joined
        if callback is None:
            with self._lock:
                self._ready.extend(joined)
            return
        for index, result in enumerate(joined):
            try:
                callback(result)
            except Exception as exc:
                # Timeline advancement is already committed.  Preserve this
                # result and every following result for explicit retry via
                # drain_ready(); never silently lose a terminal window.
                with self._lock:
                    self._ready.extend(joined[index:])
                raise ResultDeliveryError(
                    f"joined-result callback failed for {result.key}"
                ) from exc
