from __future__ import annotations

import hashlib
import errno
from pathlib import Path
import shutil
import tempfile
import threading
import weakref
from collections.abc import Iterator, Sequence
from typing import BinaryIO, overload

import numpy as np


class _ClosedSpoolLock:
    """Lock-free context manager for an immutable closed disk spool."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: object) -> bool:
        del exc_info
        return False


_CLOSED_SPOOL_LOCK = _ClosedSpoolLock()


def _write_all(stream: BinaryIO, payload: memoryview, *, context: str) -> None:
    """Write every byte or fail without accepting a zero-progress short write."""

    written = 0
    while written < len(payload):
        count = stream.write(payload[written:])
        if count is None or count <= 0 or count > len(payload) - written:
            raise OSError(f"{context} ended during a short write")
        written += count


def _finalize_disk_audio_view(mapping: object, owner: "DiskAudioStore") -> None:
    """Close the Windows mapping before permitting its store to be removed."""

    try:
        close = getattr(mapping, "close", None)
        if close is not None:
            close()
    except Exception:
        # Finalizers must still release the store lease.  A failed Windows
        # close is handled by the store's bounded cleanup retry below.
        pass
    finally:
        owner._release_view()


class DiskAudioView(np.memmap):
    """Trusted finite, read-only float32 audio backed by a session spool file.

    The view keeps its store alive.  Retired stores are deleted only after the
    last published DTO/view has been released, which is required on Windows
    where a mapped file cannot be removed eagerly.
    """

    _disk_audio_trusted = True

    def __new__(
        cls,
        path: Path,
        *,
        offset_bytes: int,
        samples: int,
        owner: "DiskAudioStore",
    ) -> "DiskAudioView":
        if samples <= 0 or offset_bytes < 0 or offset_bytes % np.dtype(np.float32).itemsize:
            raise ValueError("disk audio view range is invalid")
        owner._acquire_view()
        try:
            value = np.memmap.__new__(
                cls,
                path,
                dtype=np.float32,
                mode="r",
                offset=offset_bytes,
                shape=(samples,),
                order="C",
            )
        except BaseException:
            owner._release_view()
            raise
        value._disk_audio_owner = owner
        value._disk_audio_finalizer = weakref.finalize(
            value,
            _finalize_disk_audio_view,
            value._mmap,
            owner,
        )
        value.flags.writeable = False
        return value

    def __array_finalize__(self, source: object) -> None:
        self._disk_audio_owner = getattr(source, "_disk_audio_owner", None)
        self._disk_audio_finalizer = getattr(source, "_disk_audio_finalizer", None)


def is_trusted_disk_audio(value: object) -> bool:
    """Return whether validation already happened before a read-only disk write."""

    current = value
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, DiskAudioView) and bool(
            getattr(current, "_disk_audio_trusted", False)
        ):
            return True
        current = getattr(current, "base", None)
    return False


class DiskAudioStore:
    """Own append-only session audio files and defer cleanup across live views."""

    _CLEANUP_RETRY_DELAYS_SECONDS = (0.01, 0.05, 0.2, 0.5, 1.0)

    _SPACE_RECHECK_BYTES = 64 * 1024 * 1024

    def __init__(
        self,
        *,
        prefix: str = "micarray_audio_",
        minimum_free_bytes: int = 0,
    ) -> None:
        if type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
            raise ValueError("disk audio free-space reserve must be a non-negative integer")
        self.root = Path(tempfile.mkdtemp(prefix=prefix))
        self._lock = threading.RLock()
        self._spools: list[DiskFloat32Spool | DiskFrameSpool | DiskUInt8Timeline] = []
        self._next_id = 0
        self._views = 0
        self._retired = False
        self._cleaned = False
        self.minimum_free_bytes = minimum_free_bytes
        self._space_bytes_since_check = self._SPACE_RECHECK_BYTES
        self._space_last_free_bytes: int | None = None
        self._cleanup_retry_index = 0
        self._cleanup_retry_timer: threading.Timer | None = None
        try:
            self._reserve_growth(0)
        except BaseException:
            shutil.rmtree(self.root, ignore_errors=True)
            raise

    def _reserve_growth(self, additional_bytes: int) -> None:
        """Fail before state publication when a spool would consume the reserve."""

        if type(additional_bytes) is not int or additional_bytes < 0:
            raise ValueError("disk audio growth must be a non-negative integer")
        if not self.minimum_free_bytes:
            return
        with self._lock:
            projected = self._space_bytes_since_check + additional_bytes
            cached_free = self._space_last_free_bytes
            if (
                cached_free is not None
                and projected < self._SPACE_RECHECK_BYTES
                and cached_free - projected >= self.minimum_free_bytes
            ):
                self._space_bytes_since_check = projected
                return
            free_bytes = int(shutil.disk_usage(self.root).free)
            if free_bytes - additional_bytes < self.minimum_free_bytes:
                raise OSError(
                    errno.ENOSPC,
                    "disk audio spool would cross its configured free-space reserve",
                    str(self.root),
                )
            self._space_last_free_bytes = free_bytes
            self._space_bytes_since_check = additional_bytes

    @property
    def active_views(self) -> int:
        with self._lock:
            return self._views

    def create_spool(self, label: str) -> "DiskFloat32Spool":
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in label)
        with self._lock:
            if self._retired:
                raise RuntimeError("cannot create a spool in a retired disk audio store")
            self._next_id += 1
            path = self.root / f"{self._next_id:06d}_{safe or 'audio'}.f32"
            spool = DiskFloat32Spool(path, self)
            self._spools.append(spool)
            return spool

    def create_frame_spool(self, label: str, *, dtype: np.dtype) -> "DiskFrameSpool":
        value_dtype = np.dtype(dtype)
        if value_dtype not in {np.dtype(np.float32), np.dtype(np.bool_)}:
            raise ValueError("disk frame spools support only float32 and bool")
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in label)
        with self._lock:
            if self._retired:
                raise RuntimeError("cannot create a spool in a retired disk audio store")
            self._next_id += 1
            path = self.root / f"{self._next_id:06d}_{safe or 'frames'}.bin"
            spool = DiskFrameSpool(path, self, value_dtype)
            self._spools.append(spool)
            return spool

    def create_u8_timeline(self, label: str) -> "DiskUInt8Timeline":
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in label)
        with self._lock:
            if self._retired:
                raise RuntimeError("cannot create a timeline in a retired disk audio store")
            self._next_id += 1
            path = self.root / f"{self._next_id:06d}_{safe or 'timeline'}.u8"
            timeline = DiskUInt8Timeline(path, self)
            self._spools.append(timeline)
            return timeline

    def release_spool(
        self,
        spool: "DiskFloat32Spool | DiskFrameSpool | DiskUInt8Timeline",
    ) -> None:
        """Close and forget one discarded mutable spool before store retirement.

        Closing drops the Windows file object and ``RLock`` resources; removing
        a terminal spool from the owner's registry also prevents discarded IDs
        from accumulating Python wrappers. The containing temporary directory
        remains owned by this store for normal retirement cleanup.
        """

        if getattr(spool, "owner", None) is not self:
            raise ValueError("disk spool belongs to a different store")
        spool.close()
        with self._lock:
            try:
                self._spools.remove(spool)
            except ValueError:
                pass

    def _acquire_view(self) -> None:
        with self._lock:
            if self._cleaned:
                raise RuntimeError("disk audio store was already cleaned")
            self._views += 1

    def _release_view(self) -> None:
        cleanup = False
        with self._lock:
            self._views = max(0, self._views - 1)
            cleanup = self._retired and self._views == 0 and not self._cleaned
        if cleanup:
            self._cleanup()

    def retire(self) -> None:
        with self._lock:
            already_retired = self._retired
            if already_retired:
                spools = ()
                if (
                    not self._cleaned
                    and self._views == 0
                    and self._cleanup_retry_timer is None
                    and self._cleanup_retry_index
                    >= len(self._CLEANUP_RETRY_DELAYS_SECONDS)
                ):
                    # An explicit later retire() starts one fresh bounded retry
                    # cycle after a persistent Windows file lock has cleared.
                    self._cleanup_retry_index = 0
            else:
                self._retired = True
                spools = tuple(self._spools)
        for spool in spools:
            spool.close()
        with self._lock:
            cleanup = self._views == 0 and not self._cleaned
        if cleanup:
            self._cleanup()

    def _schedule_cleanup_retry_locked(self) -> None:
        if (
            self._cleanup_retry_timer is not None
            or self._cleaned
            or self._views
            or not self._retired
            or self._cleanup_retry_index
            >= len(self._CLEANUP_RETRY_DELAYS_SECONDS)
        ):
            return
        delay = self._CLEANUP_RETRY_DELAYS_SECONDS[self._cleanup_retry_index]
        self._cleanup_retry_index += 1
        timer = threading.Timer(delay, self._retry_cleanup)
        timer.daemon = True
        self._cleanup_retry_timer = timer
        timer.start()

    def _retry_cleanup(self) -> None:
        with self._lock:
            self._cleanup_retry_timer = None
            cleanup = self._retired and self._views == 0 and not self._cleaned
        if cleanup:
            try:
                self._cleanup()
            except OSError:
                # A background best-effort cleanup must not surface an
                # unhandled thread exception.  Explicit retire() still reports
                # non-permission filesystem failures to its caller.
                pass

    def _cleanup(self) -> None:
        with self._lock:
            if self._cleaned or self._views:
                return
            self._cleaned = True
        try:
            shutil.rmtree(self.root)
        except FileNotFoundError:
            pass
        except PermissionError:
            with self._lock:
                self._cleaned = False
                self._schedule_cleanup_retry_locked()
            return
        except OSError:
            with self._lock:
                self._cleaned = False
            raise
        with self._lock:
            timer = self._cleanup_retry_timer
            self._cleanup_retry_timer = None
            self._cleanup_retry_index = 0
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()

    def __del__(self) -> None:
        try:
            self.retire()
        except Exception:
            pass


class DiskFloat32Spool:
    """Random-write float32 timeline with zero-filled sparse gaps and mmap reads."""

    _COPY_CHUNK_BYTES = 4 * 1024 * 1024

    def __init__(self, path: Path, owner: DiskAudioStore) -> None:
        self.path = path
        self.owner = owner
        self._stream = path.open("w+b")
        self._lock = threading.RLock()
        self._length_samples = 0
        self._hashed_samples = 0
        self._hash = hashlib.sha256()
        self._hash_valid = True
        self._closed = False

    @property
    def length_samples(self) -> int:
        with self._lock:
            return self._length_samples

    @property
    def resident_bytes(self) -> int:
        return 0

    @property
    def disk_bytes(self) -> int:
        with self._lock:
            return self._length_samples * np.dtype(np.float32).itemsize

    def append(self, waveform: np.ndarray) -> tuple[int, int]:
        with self._lock:
            start = self._length_samples
            self.write_at(start, waveform)
            return start, start + len(waveform)

    def write_at(self, start_sample: int, waveform: np.ndarray) -> None:
        value = np.asarray(waveform)
        if (
            type(start_sample) is not int
            or start_sample < 0
            or value.ndim != 1
            or value.dtype != np.float32
            or not value.flags.c_contiguous
            or not len(value)
            or not np.isfinite(value).all()
        ):
            raise ValueError("disk audio writes require finite contiguous float32 mono audio")
        payload = memoryview(value).cast("B")
        end_sample = start_sample + len(value)
        with self._lock:
            if self._closed:
                raise RuntimeError("disk audio spool is closed")
            itemsize = np.dtype(np.float32).itemsize
            start_byte = start_sample * itemsize
            old_length_bytes = self._length_samples * itemsize
            self.owner._reserve_growth(
                max(0, start_byte + len(payload) - old_length_bytes)
            )
            overlap_bytes = max(
                0,
                min(start_byte + len(payload), old_length_bytes) - start_byte,
            )
            backup = b""
            if overlap_bytes:
                self._stream.flush()
                self._stream.seek(start_byte)
                backup = self._stream.read(overlap_bytes)
                if len(backup) != overlap_bytes:
                    raise RuntimeError("disk audio spool ended while preparing a write")
            try:
                self._stream.seek(start_byte)
                _write_all(self._stream, payload, context="disk audio write")
                self._stream.flush()
            except BaseException:
                restored = False
                try:
                    if backup:
                        self._stream.seek(start_byte)
                        _write_all(
                            self._stream,
                            memoryview(backup),
                            context="disk audio rollback",
                        )
                    self._stream.truncate(old_length_bytes)
                    self._stream.flush()
                    restored = True
                except BaseException:
                    pass
                if not restored:
                    self._hash_valid = False
                raise
            if start_sample != self._hashed_samples:
                self._hash_valid = False
            if self._hash_valid:
                self._hash.update(payload)
                self._hashed_samples = end_sample
            self._length_samples = max(self._length_samples, end_sample)

    def ensure_length(self, samples: int) -> None:
        """Extend a sparse timeline with deterministic zero-filled audio."""

        if type(samples) is not int or samples < 0:
            raise ValueError("disk audio length must be a non-negative integer")
        with self._lock:
            if self._closed:
                raise RuntimeError("disk audio spool is closed")
            if samples <= self._length_samples:
                return
            self.owner._reserve_growth(
                (samples - self._length_samples) * np.dtype(np.float32).itemsize
            )
            self._stream.truncate(samples * np.dtype(np.float32).itemsize)
            self._stream.flush()
            self._length_samples = samples
            self._hash_valid = False

    def view(self, start_sample: int = 0, end_sample: int | None = None) -> DiskAudioView:
        with self._lock:
            end = self._resolve_range_locked(
                start_sample,
                end_sample,
                invalid_message="disk audio read range is invalid",
                exceeds_message="disk audio read exceeds the written timeline",
            )
            if not self._closed:
                self._stream.flush()
            # Acquire the store lease while the spool lock still prevents a
            # concurrent retire() from closing and deleting this path.
            return DiskAudioView(
                self.path,
                offset_bytes=start_sample * np.dtype(np.float32).itemsize,
                samples=end - start_sample,
                owner=self.owner,
            )

    def _resolve_range_locked(
        self,
        start_sample: int,
        end_sample: int | None,
        *,
        invalid_message: str,
        exceeds_message: str,
    ) -> int:
        end = self._length_samples if end_sample is None else int(end_sample)
        if self._closed and not self.path.exists():
            raise RuntimeError("disk audio spool was removed")
        if type(start_sample) is not int or start_sample < 0 or end <= start_sample:
            raise ValueError(invalid_message)
        if end > self._length_samples:
            raise ValueError(exceeds_message)
        return end

    def copy_range_to(
        self,
        start_sample: int,
        end_sample: int,
        destination: "DiskFloat32Spool",
        *,
        destination_start_sample: int = 0,
    ) -> str:
        """Copy a stable range to ``destination`` and return its SHA-256.

        Only one bounded transfer buffer is resident.  The source lock remains
        held for the complete copy and digest so concurrent backfills cannot
        produce a waveform from one revision and a hash from another.
        """

        if destination is self:
            raise ValueError("disk audio range destination must be a different spool")
        if type(destination_start_sample) is not int or destination_start_sample < 0:
            raise ValueError("disk audio destination start must be a non-negative integer")
        with self._lock:
            end = self._resolve_range_locked(
                start_sample,
                end_sample,
                invalid_message="disk audio copy range is invalid",
                exceeds_message="disk audio copy exceeds the written timeline",
            )
            if not self._closed:
                self._stream.flush()
            remaining = (end - start_sample) * np.dtype(np.float32).itemsize
            destination_sample = destination_start_sample
            digest = hashlib.sha256()
            with self.path.open("rb", buffering=0) as source:
                source.seek(start_sample * np.dtype(np.float32).itemsize)
                while remaining:
                    chunk_bytes = min(remaining, self._COPY_CHUNK_BYTES)
                    payload = bytearray(chunk_bytes)
                    target = memoryview(payload)
                    copied = 0
                    while copied < chunk_bytes:
                        count = source.readinto(target[copied:])
                        if count is None or count <= 0:
                            raise RuntimeError("disk audio spool ended during range copy")
                        copied += count
                    digest.update(target)
                    values = np.frombuffer(payload, dtype=np.float32)
                    destination.write_at(destination_sample, values)
                    destination_sample += len(values)
                    remaining -= chunk_bytes
            return digest.hexdigest()

    def digest(self, start_sample: int = 0, end_sample: int | None = None) -> str:
        with self._lock:
            end = self._resolve_range_locked(
                start_sample,
                end_sample,
                invalid_message="disk audio digest range is invalid",
                exceeds_message="disk audio digest range is invalid",
            )
            if (
                start_sample == 0
                and end == self._hashed_samples
                and self._hash_valid
            ):
                return self._hash.copy().hexdigest()
            if not self._closed:
                self._stream.flush()
            digest = hashlib.sha256()
            remaining = (end - start_sample) * np.dtype(np.float32).itemsize
            with self.path.open("rb") as source:
                source.seek(start_sample * np.dtype(np.float32).itemsize)
                while remaining:
                    payload = source.read(min(remaining, self._COPY_CHUNK_BYTES))
                    if not payload:
                        raise RuntimeError("disk audio spool ended during hashing")
                    digest.update(payload)
                    remaining -= len(payload)
            return digest.hexdigest()

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            stream = self._stream
            try:
                stream.flush()
            finally:
                self._closed = True
                stream.close()
                self._stream = None  # type: ignore[assignment]
                self._lock = _CLOSED_SPOOL_LOCK  # type: ignore[assignment]


class DiskFrameSeries(Sequence[float | bool]):
    """Immutable sequence facade over compact frame metadata on disk."""

    _disk_frame_trusted = True

    def __init__(
        self,
        path: Path,
        *,
        dtype: np.dtype,
        count: int,
        owner: DiskAudioStore,
    ) -> None:
        if count < 0:
            raise ValueError("disk frame series count cannot be negative")
        self.path = path
        self.dtype = np.dtype(dtype)
        self.count = int(count)
        self._owner = owner
        owner._acquire_view()
        self._finalizer = weakref.finalize(self, owner._release_view)

    def __len__(self) -> int:
        return self.count

    def _mapped(self) -> np.memmap:
        if not self.count:
            return np.memmap(
                self.path,
                dtype=self.dtype,
                mode="r",
                shape=(0,),
            )
        return np.memmap(
            self.path,
            dtype=self.dtype,
            mode="r",
            shape=(self.count,),
        )

    @overload
    def __getitem__(self, index: int) -> float | bool: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[float | bool, ...]: ...

    def __getitem__(self, index: int | slice) -> float | bool | tuple[float | bool, ...]:
        if not self.count:
            if isinstance(index, slice):
                return ()
            raise IndexError("disk frame series index out of range")
        mapped = self._mapped()
        try:
            value = mapped[index]
            if isinstance(index, slice):
                return tuple(item.item() for item in np.asarray(value))
            return value.item()
        finally:
            mapped._mmap.close()

    def __iter__(self) -> Iterator[float | bool]:
        if not self.count:
            return
        mapped = self._mapped()
        try:
            for value in mapped:
                yield value.item()
        finally:
            mapped._mmap.close()

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        if not self.count:
            return np.empty(0, dtype=self.dtype if dtype is None else dtype)
        mapped = self._mapped()
        try:
            # NumPy callers use this for short-lived vectorized work.  Return a
            # bounded metadata copy so no untracked mmap can escape the series.
            return np.array(mapped, dtype=dtype, copy=True)
        finally:
            mapped._mmap.close()


class DiskFrameSpool:
    """Append-only compact float32/bool frame series."""

    def __init__(self, path: Path, owner: DiskAudioStore, dtype: np.dtype) -> None:
        self.path = path
        self.owner = owner
        self.dtype = np.dtype(dtype)
        self._stream = path.open("w+b")
        self._lock = threading.RLock()
        self._count = 0
        self._closed = False

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def disk_bytes(self) -> int:
        with self._lock:
            return self._count * self.dtype.itemsize

    def append(self, values: np.ndarray) -> tuple[int, int]:
        value = np.asarray(values)
        if value.ndim != 1 or value.dtype != self.dtype or not value.flags.c_contiguous:
            raise ValueError("disk frame writes require contiguous values with the configured dtype")
        if self.dtype == np.dtype(np.float32) and not np.isfinite(value).all():
            raise ValueError("disk probability frames must be finite")
        with self._lock:
            if self._closed:
                raise RuntimeError("disk frame spool is closed")
            start = self._count
            start_byte = start * self.dtype.itemsize
            self.owner._reserve_growth(len(value) * self.dtype.itemsize)
            try:
                self._stream.seek(start_byte)
                _write_all(
                    self._stream,
                    memoryview(value).cast("B"),
                    context="disk frame write",
                )
                self._stream.flush()
            except BaseException:
                try:
                    self._stream.truncate(start_byte)
                    self._stream.flush()
                except BaseException:
                    pass
                raise
            self._count += len(value)
            return start, self._count

    def series(self, count: int | None = None) -> DiskFrameSeries:
        with self._lock:
            end = self._count if count is None else int(count)
            if end < 0 or end > self._count:
                raise ValueError("disk frame view exceeds written values")
            if not self._closed:
                self._stream.flush()
            return DiskFrameSeries(
                self.path,
                dtype=self.dtype,
                count=end,
                owner=self.owner,
            )

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            stream = self._stream
            try:
                stream.flush()
            finally:
                self._closed = True
                stream.close()
                self._stream = None  # type: ignore[assignment]
                self._lock = _CLOSED_SPOOL_LOCK  # type: ignore[assignment]


class DiskUInt8Series(Sequence[int]):
    """Read-only bounded slice of a random-write uint8 disk timeline."""

    def __init__(
        self,
        path: Path,
        *,
        start: int,
        count: int,
        owner: DiskAudioStore,
    ) -> None:
        self.path = path
        self.start = int(start)
        self.count = int(count)
        self._owner = owner
        owner._acquire_view()
        self._finalizer = weakref.finalize(self, owner._release_view)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            return tuple(self)[index]
        resolved = index + self.count if index < 0 else index
        if resolved < 0 or resolved >= self.count:
            raise IndexError("disk timeline index out of range")
        with self.path.open("rb") as source:
            source.seek(self.start + resolved)
            payload = source.read(1)
        if len(payload) != 1:
            raise RuntimeError("disk timeline ended during indexed read")
        return payload[0]

    def __iter__(self) -> Iterator[int]:
        remaining = self.count
        with self.path.open("rb") as source:
            source.seek(self.start)
            while remaining:
                payload = source.read(min(remaining, 64 * 1024))
                if not payload:
                    raise RuntimeError("disk timeline ended during iteration")
                yield from payload
                remaining -= len(payload)


class DiskUInt8Timeline:
    """Sparse random-write one-byte metadata timeline."""

    def __init__(self, path: Path, owner: DiskAudioStore) -> None:
        self.path = path
        self.owner = owner
        self._stream = path.open("w+b")
        self._lock = threading.RLock()
        self._length = 0
        self._closed = False

    @property
    def disk_bytes(self) -> int:
        with self._lock:
            return self._length

    def set(self, index: int, value: int) -> None:
        if type(index) is not int or index < 0 or type(value) is not int or not 0 <= value <= 255:
            raise ValueError("uint8 timeline write is invalid")
        self.write_range(index, bytes((value,)))

    def write_range(self, start: int, values: bytes) -> None:
        """Write one contiguous metadata range with one flush and rollback."""

        if type(start) is not int or start < 0 or type(values) is not bytes or not values:
            raise ValueError("uint8 timeline range write is invalid")
        with self._lock:
            if self._closed:
                raise RuntimeError("uint8 timeline is closed")
            old_length = self._length
            end = start + len(values)
            self.owner._reserve_growth(max(0, end - old_length))
            overlap_bytes = max(0, min(end, old_length) - start)
            backup = b""
            if overlap_bytes:
                self._stream.flush()
                self._stream.seek(start)
                backup = self._stream.read(overlap_bytes)
                if len(backup) != overlap_bytes:
                    raise RuntimeError("uint8 timeline ended while preparing a write")
            try:
                if end > old_length:
                    self._stream.truncate(end)
                self._stream.seek(start)
                _write_all(
                    self._stream,
                    memoryview(values),
                    context="uint8 timeline range write",
                )
                self._stream.flush()
            except BaseException:
                try:
                    if backup:
                        self._stream.seek(start)
                        _write_all(
                            self._stream,
                            memoryview(backup),
                            context="uint8 timeline rollback",
                        )
                    self._stream.truncate(old_length)
                    self._stream.flush()
                except BaseException:
                    pass
                raise
            self._length = max(old_length, end)

    def get(self, index: int, default: int = 0) -> int:
        with self._lock:
            if index < 0 or index >= self._length:
                return default
            if not self._closed:
                self._stream.flush()
            elif not self.path.exists():
                raise RuntimeError("uint8 timeline was removed")
            with self.path.open("rb") as source:
                source.seek(index)
                payload = source.read(1)
            return default if not payload else payload[0]

    def read_range(self, start: int, end: int) -> bytes:
        """Read one bounded range with one filesystem read under the spool lock."""

        with self._lock:
            if type(start) is not int or type(end) is not int or start < 0 or end <= start:
                raise ValueError("uint8 timeline range is invalid")
            if end > self._length:
                if self._closed:
                    raise ValueError("uint8 timeline range exceeds closed data")
                self.owner._reserve_growth(end - self._length)
                self._stream.truncate(end)
                self._stream.flush()
                self._length = end
            elif not self._closed:
                self._stream.flush()
            elif not self.path.exists():
                raise RuntimeError("uint8 timeline was removed")
            with self.path.open("rb") as source:
                source.seek(start)
                payload = source.read(end - start)
            if len(payload) != end - start:
                raise RuntimeError("uint8 timeline ended during range read")
            return payload

    def series(self, start: int, end: int) -> DiskUInt8Series:
        with self._lock:
            if start < 0 or end <= start:
                raise ValueError("uint8 timeline range is invalid")
            if end > self._length:
                if self._closed:
                    raise ValueError("uint8 timeline range exceeds closed data")
                self.owner._reserve_growth(end - self._length)
                self._stream.truncate(end)
                self._stream.flush()
                self._length = end
            return DiskUInt8Series(
                self.path,
                start=start,
                count=end - start,
                owner=self.owner,
            )

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            stream = self._stream
            try:
                stream.flush()
            finally:
                self._closed = True
                stream.close()
                self._stream = None  # type: ignore[assignment]
                self._lock = _CLOSED_SPOOL_LOCK  # type: ignore[assignment]
