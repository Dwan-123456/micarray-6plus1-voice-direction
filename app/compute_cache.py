from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Callable

import numpy as np

from .processing_contracts import WindowKey


class ComputeCacheError(RuntimeError):
    pass


class DuplicateArtifactError(ComputeCacheError):
    pass


class ArtifactTooLargeError(ComputeCacheError):
    pass


class RetiredWindowError(ComputeCacheError):
    pass


class GpuArtifactError(ComputeCacheError):
    pass


@dataclass(frozen=True, slots=True)
class CachePartitionLimits:
    max_windows: int
    max_bytes: int
    max_artifacts_per_window: int = 16

    def __post_init__(self) -> None:
        if min(self.max_windows, self.max_bytes, self.max_artifacts_per_window) <= 0:
            raise ValueError("all cache partition limits must be positive")


@dataclass(frozen=True, slots=True)
class EvictedArtifact:
    partition: str
    key: WindowKey
    artifact_name: str
    byte_size: int
    reason: str


@dataclass(frozen=True, slots=True)
class CachePublishResult:
    partition: str
    key: WindowKey
    artifact_name: str
    byte_size: int
    evicted: tuple[EvictedArtifact, ...]


@dataclass(frozen=True, slots=True)
class CacheSnapshot:
    partition: str
    windows: int
    artifacts: int
    current_bytes: int
    max_windows: int
    max_bytes: int
    max_artifacts_per_window: int
    evicted_windows: int
    evicted_artifacts: int
    retired_streams: int


@dataclass(slots=True)
class _Entry:
    value: object
    byte_size: int


@dataclass(slots=True)
class _WindowBucket:
    entries: OrderedDict[str, _Entry]
    published_names: set[str]


def _device_type(value: object) -> str | None:
    device = getattr(value, "device", None)
    if device is None:
        return None
    kind = getattr(device, "type", device)
    return str(kind).lower()


def artifact_byte_size(value: object) -> int:
    """Conservatively account a reachable object graph and reject any GPU payload."""

    seen: set[int] = set()

    def visit(item: object) -> int:
        if item is None or isinstance(item, (bool, int, float, complex)):
            return sys.getsizeof(item)
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)

        if bool(getattr(item, "is_cuda", False)):
            raise GpuArtifactError("GPU tensors cannot be retained in the compute cache")
        device_type = _device_type(item)
        if device_type is not None and device_type not in {"cpu", "none"}:
            raise GpuArtifactError(f"{device_type} artifacts cannot be retained in the compute cache")

        if isinstance(item, np.ndarray):
            return sys.getsizeof(item) + int(item.nbytes)
        if isinstance(item, (str, bytes, bytearray, memoryview)):
            return sys.getsizeof(item)
        if isinstance(item, Mapping):
            return sys.getsizeof(item) + sum(visit(key) + visit(child) for key, child in item.items())
        if isinstance(item, (tuple, list, set, frozenset)):
            return sys.getsizeof(item) + sum(visit(child) for child in item)

        element_size = getattr(item, "element_size", None)
        element_count = getattr(item, "nelement", None)
        if callable(element_size) and callable(element_count):
            return sys.getsizeof(item) + int(element_size()) * int(element_count())
        if is_dataclass(item) and not isinstance(item, type):
            return sys.getsizeof(item) + sum(visit(getattr(item, spec.name)) for spec in fields(item))
        values = getattr(item, "__dict__", None)
        if isinstance(values, dict):
            return sys.getsizeof(item) + visit(values)
        return sys.getsizeof(item)

    return max(1, visit(value))


class WindowArtifactStore:
    """Thread-safe bounded cache whose eviction unit is a complete timeline window."""

    def __init__(
        self,
        partition: str,
        limits: CachePartitionLimits,
        *,
        on_evict: Callable[[EvictedArtifact], None] | None = None,
    ) -> None:
        if not partition:
            raise ValueError("cache partition name cannot be empty")
        self.partition = partition
        self.limits = limits
        self._on_evict = on_evict
        self._windows: OrderedDict[WindowKey, _WindowBucket] = OrderedDict()
        self._retired_through: dict[tuple[str, int], int] = {}
        self._current_bytes = 0
        self._artifact_count = 0
        self._evicted_windows = 0
        self._evicted_artifacts = 0
        self._lock = threading.RLock()

    def publish(self, key: WindowKey, artifact_name: str, value: object) -> CachePublishResult:
        if not isinstance(key, WindowKey):
            raise TypeError("cache key must be a WindowKey")
        if not artifact_name:
            raise ValueError("artifact name cannot be empty")
        byte_size = artifact_byte_size(value)
        if byte_size > self.limits.max_bytes:
            raise ArtifactTooLargeError(
                f"artifact requires {byte_size} bytes, partition limit is {self.limits.max_bytes}"
            )

        callbacks: list[EvictedArtifact] = []
        with self._lock:
            bucket = self._windows.get(key)
            if bucket is None:
                retired = self._retired_through.get(key.stream_key, -1)
                if key.decision_sample <= retired:
                    raise RetiredWindowError(
                        f"window {key.timeline_order} was already retired through sample {retired}"
                    )
                bucket = _WindowBucket(OrderedDict(), set())
                self._windows[key] = bucket
            if artifact_name in bucket.published_names:
                raise DuplicateArtifactError(f"artifact {artifact_name!r} was already published for {key}")
            if len(bucket.published_names) >= self.limits.max_artifacts_per_window:
                if not bucket.entries and not bucket.published_names:
                    self._windows.pop(key, None)
                raise ArtifactTooLargeError("per-window artifact count limit reached")

            bucket.entries[artifact_name] = _Entry(value=value, byte_size=byte_size)
            bucket.published_names.add(artifact_name)
            self._current_bytes += byte_size
            self._artifact_count += 1
            while len(self._windows) > self.limits.max_windows or self._current_bytes > self.limits.max_bytes:
                oldest = next(iter(self._windows))
                if oldest == key and len(self._windows) == 1:
                    # The single item was checked above; this protects transactionality if accounting changes.
                    self._remove_artifact_locked(key, artifact_name, retire_empty=False)
                    raise ArtifactTooLargeError("artifact cannot fit within the cache limits")
                callbacks.extend(self._evict_window_locked(oldest, "capacity"))

        self._notify(callbacks)
        return CachePublishResult(self.partition, key, artifact_name, byte_size, tuple(callbacks))

    def get(self, key: WindowKey, artifact_name: str) -> object | None:
        with self._lock:
            bucket = self._windows.get(key)
            entry = None if bucket is None else bucket.entries.get(artifact_name)
            return None if entry is None else entry.value

    def require(self, key: WindowKey, artifact_name: str) -> object:
        value = self.get(key, artifact_name)
        if value is None:
            raise KeyError((key, artifact_name))
        return value

    def evict_artifact(self, key: WindowKey, artifact_name: str, reason: str = "explicit") -> EvictedArtifact | None:
        if not reason:
            raise ValueError("eviction reason cannot be empty")
        with self._lock:
            event = self._remove_artifact_locked(key, artifact_name, retire_empty=False, reason=reason)
        if event is not None:
            self._notify([event])
        return event

    def evict_window(self, key: WindowKey, reason: str = "explicit") -> tuple[EvictedArtifact, ...]:
        if not reason:
            raise ValueError("eviction reason cannot be empty")
        with self._lock:
            events = tuple(self._evict_window_locked(key, reason))
        self._notify(list(events))
        return events

    def evict_stream(self, session_id: str, stream_epoch: int, reason: str = "epoch_end") -> tuple[EvictedArtifact, ...]:
        if not session_id or stream_epoch < 0 or not reason:
            raise ValueError("invalid stream eviction request")
        with self._lock:
            keys = [key for key in self._windows if key.stream_key == (session_id, stream_epoch)]
            events = tuple(
                event
                for key in keys
                for event in self._evict_window_locked(key, reason)
            )
        self._notify(list(events))
        return events

    def clear(self, reason: str = "clear") -> tuple[EvictedArtifact, ...]:
        if not reason:
            raise ValueError("eviction reason cannot be empty")
        with self._lock:
            events = tuple(
                event
                for key in tuple(self._windows)
                for event in self._evict_window_locked(key, reason)
            )
        self._notify(list(events))
        return events

    def prune_stream_history(
        self, stream_keys: Iterable[tuple[str, int]]
    ) -> tuple[tuple[str, int], ...]:
        """Release retired-watermark metadata for caller-certified closed streams.

        Live buckets are never pruned.  This makes the method safe if a newer
        epoch is observed while an older epoch still has in-flight stage work.
        """

        requested = frozenset(stream_keys)
        if any(not session_id or epoch < 0 for session_id, epoch in requested):
            raise ValueError("invalid stream key for cache-history pruning")
        with self._lock:
            live_streams = {key.stream_key for key in self._windows}
            removable = requested.difference(live_streams)
            for stream_key in removable:
                self._retired_through.pop(stream_key, None)
            return tuple(sorted(removable))

    def snapshot(self) -> CacheSnapshot:
        with self._lock:
            return CacheSnapshot(
                partition=self.partition,
                windows=len(self._windows),
                artifacts=self._artifact_count,
                current_bytes=self._current_bytes,
                max_windows=self.limits.max_windows,
                max_bytes=self.limits.max_bytes,
                max_artifacts_per_window=self.limits.max_artifacts_per_window,
                evicted_windows=self._evicted_windows,
                evicted_artifacts=self._evicted_artifacts,
                retired_streams=len(self._retired_through),
            )

    def _remove_artifact_locked(
        self,
        key: WindowKey,
        artifact_name: str,
        *,
        retire_empty: bool,
        reason: str = "rollback",
    ) -> EvictedArtifact | None:
        bucket = self._windows.get(key)
        if bucket is None:
            return None
        entry = bucket.entries.pop(artifact_name, None)
        if entry is None:
            return None
        self._current_bytes -= entry.byte_size
        self._artifact_count -= 1
        self._evicted_artifacts += 1
        event = EvictedArtifact(self.partition, key, artifact_name, entry.byte_size, reason)
        if retire_empty and not bucket.entries:
            self._windows.pop(key, None)
            self._retire_locked(key)
            self._evicted_windows += 1
        return event

    def _evict_window_locked(self, key: WindowKey, reason: str) -> list[EvictedArtifact]:
        bucket = self._windows.pop(key, None)
        if bucket is None:
            return []
        events = [
            EvictedArtifact(self.partition, key, name, entry.byte_size, reason)
            for name, entry in bucket.entries.items()
        ]
        self._current_bytes -= sum(item.byte_size for item in bucket.entries.values())
        self._artifact_count -= len(bucket.entries)
        self._evicted_artifacts += len(bucket.entries)
        self._evicted_windows += 1
        self._retire_locked(key)
        return events

    def _retire_locked(self, key: WindowKey) -> None:
        previous = self._retired_through.get(key.stream_key, -1)
        self._retired_through[key.stream_key] = max(previous, key.decision_sample)

    def _notify(self, events: list[EvictedArtifact]) -> None:
        if self._on_evict is not None:
            for event in events:
                self._on_evict(event)


class ComputeCache:
    """Named hard-bounded window cache partitions for independent stage workers."""

    def __init__(
        self,
        partitions: Mapping[str, CachePartitionLimits],
        *,
        max_total_bytes: int | None = None,
        on_evict: Callable[[EvictedArtifact], None] | None = None,
    ) -> None:
        if not partitions:
            raise ValueError("at least one compute-cache partition is required")
        normalized = dict(partitions)
        if any(not name for name in normalized):
            raise ValueError("partition names cannot be empty")
        configured_bytes = sum(item.max_bytes for item in normalized.values())
        total_limit = configured_bytes if max_total_bytes is None else max_total_bytes
        if total_limit <= 0 or configured_bytes > total_limit:
            raise ValueError("sum of partition byte limits must not exceed max_total_bytes")
        self.max_total_bytes = total_limit
        self._partitions = {
            name: WindowArtifactStore(name, limits, on_evict=on_evict)
            for name, limits in normalized.items()
        }

    def partition(self, name: str) -> WindowArtifactStore:
        try:
            return self._partitions[name]
        except KeyError as exc:
            raise KeyError(f"unknown compute-cache partition: {name}") from exc

    def publish(self, partition: str, key: WindowKey, artifact_name: str, value: object) -> CachePublishResult:
        return self.partition(partition).publish(key, artifact_name, value)

    def get(self, partition: str, key: WindowKey, artifact_name: str) -> object | None:
        return self.partition(partition).get(key, artifact_name)

    def evict_window(self, key: WindowKey, reason: str = "window_complete") -> tuple[EvictedArtifact, ...]:
        return tuple(
            event
            for store in self._partitions.values()
            for event in store.evict_window(key, reason)
        )

    def evict_stream(self, session_id: str, stream_epoch: int, reason: str = "epoch_end") -> tuple[EvictedArtifact, ...]:
        return tuple(
            event
            for store in self._partitions.values()
            for event in store.evict_stream(session_id, stream_epoch, reason)
        )

    def clear(self, reason: str = "shutdown") -> tuple[EvictedArtifact, ...]:
        return tuple(event for store in self._partitions.values() for event in store.clear(reason))

    def prune_stream_history(
        self, stream_keys: Iterable[tuple[str, int]]
    ) -> tuple[tuple[str, int], ...]:
        """Release per-epoch retired metadata from every cache partition."""

        keys = tuple(stream_keys)
        safe = set(keys)
        for store in self._partitions.values():
            safe.intersection_update(store.prune_stream_history(keys))
        return tuple(sorted(safe))

    def snapshots(self) -> Mapping[str, CacheSnapshot]:
        return {name: store.snapshot() for name, store in self._partitions.items()}

    @property
    def current_bytes(self) -> int:
        return sum(item.current_bytes for item in self.snapshots().values())
