from __future__ import annotations

import errno
import gc
import hashlib
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import common.disk_audio as disk_audio
from common.disk_audio import DiskAudioStore


class _DelegatingStream:
    def __init__(
        self,
        wrapped,
        *,
        limit: int | None = None,
        stop_after: int | None = None,
        zero_on_call: int | None = None,
    ):
        self.wrapped = wrapped
        self.limit = limit
        self.stop_after = stop_after
        self.zero_on_call = zero_on_call
        self.write_calls = 0
        self.flush_calls = 0

    def __getattr__(self, name: str):
        return getattr(self.wrapped, name)

    def write(self, payload) -> int:
        self.write_calls += 1
        if self.stop_after is not None and self.write_calls > self.stop_after:
            return 0
        if self.zero_on_call == self.write_calls:
            return 0
        limited = payload if self.limit is None else payload[: self.limit]
        return self.wrapped.write(limited)

    def flush(self) -> None:
        self.flush_calls += 1
        self.wrapped.flush()


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        gc.collect()
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_release_spool_forgets_closed_wrapper_until_store_retirement() -> None:
    store = DiskAudioStore(prefix="test_disk_audio_release_")
    timeline = store.create_u8_timeline("discarded")
    root = store.root
    path = timeline.path
    timeline.set(0, 1)

    store.release_spool(timeline)

    assert timeline._closed is True
    assert timeline._stream is None
    assert timeline not in store._spools
    assert path.exists()

    store.retire()
    assert not root.exists()


def test_copy_range_to_is_bounded_stable_and_returns_matching_digest(monkeypatch):
    source_store = DiskAudioStore(prefix="test_disk_audio_source_")
    target_store = DiskAudioStore(prefix="test_disk_audio_target_")
    source = source_store.create_spool("source")
    target = target_store.create_spool("target")
    original = np.linspace(-0.75, 0.75, 32, dtype=np.float32)
    replacement = np.full(32, 0.25, dtype=np.float32)
    source.write_at(0, original)

    copy_entered = threading.Event()
    release_copy = threading.Event()
    writer_done = threading.Event()
    results: dict[str, str] = {}
    errors: list[BaseException] = []
    original_write = target.write_at

    def blocking_write(start_sample: int, waveform: np.ndarray) -> None:
        copy_entered.set()
        if not release_copy.wait(2.0):
            raise TimeoutError("test did not release range copy")
        original_write(start_sample, waveform)

    monkeypatch.setattr(target, "write_at", blocking_write)

    def run_copy() -> None:
        try:
            results["digest"] = source.copy_range_to(0, len(original), target)
        except BaseException as error:
            errors.append(error)

    def run_writer() -> None:
        try:
            source.write_at(0, replacement)
        except BaseException as error:
            errors.append(error)
        finally:
            writer_done.set()

    copy_thread = threading.Thread(target=run_copy)
    writer_thread = threading.Thread(target=run_writer)
    copy_thread.start()
    assert copy_entered.wait(2.0)
    writer_thread.start()
    assert not writer_done.wait(0.05)
    release_copy.set()
    copy_thread.join(2.0)
    writer_thread.join(2.0)

    assert not copy_thread.is_alive()
    assert not writer_thread.is_alive()
    assert errors == []
    expected_digest = hashlib.sha256(original.tobytes()).hexdigest()
    assert results["digest"] == expected_digest
    assert target.digest(0, len(original)) == expected_digest
    copied = target.view(0, len(original))
    np.testing.assert_array_equal(copied, original)
    assert copied.flags.writeable is False

    del copied
    source_store.retire()
    target_store.retire()


def test_copy_range_to_preserves_a_destination_prefix():
    store = DiskAudioStore(prefix="test_disk_audio_prefixed_copy_")
    source = store.create_spool("source")
    target = store.create_spool("target")
    prefix = np.array([-0.5, -0.25], dtype=np.float32)
    waveform = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    source.write_at(0, waveform)
    target.write_at(0, prefix)

    digest = source.copy_range_to(
        0,
        len(waveform),
        target,
        destination_start_sample=len(prefix),
    )

    assert digest == hashlib.sha256(waveform.tobytes()).hexdigest()
    copied = target.view(0, len(prefix) + len(waveform))
    np.testing.assert_array_equal(copied, np.concatenate((prefix, waveform)))
    with pytest.raises(ValueError, match="destination start"):
        source.copy_range_to(0, len(waveform), target, destination_start_sample=-1)
    del copied
    store.retire()


def test_digest_holds_the_spool_lock_for_the_complete_file_read(monkeypatch):
    store = DiskAudioStore(prefix="test_disk_audio_digest_")
    spool = store.create_spool("source")
    original = np.arange(16, dtype=np.float32)
    spool.write_at(0, original)
    spool.write_at(4, np.array([9.0], dtype=np.float32))

    read_entered = threading.Event()
    release_read = threading.Event()
    writer_done = threading.Event()
    errors: list[BaseException] = []
    original_path_open = Path.open

    class _BlockingReader:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self.wrapped.__exit__(*args)

        def seek(self, *args):
            return self.wrapped.seek(*args)

        def read(self, size: int = -1):
            read_entered.set()
            if not release_read.wait(2.0):
                raise TimeoutError("test did not release digest read")
            return self.wrapped.read(size)

    def patched_open(path: Path, mode="r", *args, **kwargs):
        opened = original_path_open(path, mode, *args, **kwargs)
        if path == spool.path and mode == "rb":
            return _BlockingReader(opened)
        return opened

    monkeypatch.setattr(Path, "open", patched_open)

    def run_digest() -> None:
        try:
            spool.digest(0, len(original))
        except BaseException as error:
            errors.append(error)

    def run_writer() -> None:
        try:
            spool.write_at(0, np.array([-1.0], dtype=np.float32))
        except BaseException as error:
            errors.append(error)
        finally:
            writer_done.set()

    digest_thread = threading.Thread(target=run_digest)
    writer_thread = threading.Thread(target=run_writer)
    digest_thread.start()
    assert read_entered.wait(2.0)
    writer_thread.start()
    assert not writer_done.wait(0.05)
    release_read.set()
    digest_thread.join(2.0)
    writer_thread.join(2.0)

    assert not digest_thread.is_alive()
    assert not writer_thread.is_alive()
    assert errors == []
    store.retire()


def test_uint8_read_range_uses_one_read_and_preserves_sparse_zeros(monkeypatch):
    store = DiskAudioStore(prefix="test_disk_u8_range_")
    timeline = store.create_u8_timeline("directions")
    timeline.set(0, 5)
    timeline.set(3, 9)
    read_sizes: list[int] = []
    original_path_open = Path.open

    class _CountingReader:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self.wrapped.__exit__(*args)

        def seek(self, *args):
            return self.wrapped.seek(*args)

        def read(self, size: int = -1):
            read_sizes.append(size)
            return self.wrapped.read(size)

    def patched_open(path: Path, mode="r", *args, **kwargs):
        opened = original_path_open(path, mode, *args, **kwargs)
        if path == timeline.path and mode == "rb":
            return _CountingReader(opened)
        return opened

    monkeypatch.setattr(Path, "open", patched_open)

    assert timeline.read_range(0, 4) == bytes((5, 0, 0, 9))
    assert read_sizes == [4]
    store.retire()


def test_uint8_write_range_batches_short_writes_and_rolls_back_failure():
    store = DiskAudioStore(prefix="test_disk_u8_write_range_")
    timeline = store.create_u8_timeline("directions")
    raw_stream = timeline._stream
    partial_stream = _DelegatingStream(raw_stream, limit=2)
    timeline._stream = partial_stream

    original = bytes((1, 2, 3, 4, 5))
    timeline.write_range(0, original)

    assert partial_stream.write_calls == 3
    assert partial_stream.flush_calls == 1
    assert timeline.read_range(0, len(original)) == original

    failing_stream = _DelegatingStream(raw_stream, limit=2, zero_on_call=2)
    timeline._stream = failing_stream
    with pytest.raises(OSError, match="short write"):
        timeline.write_range(1, bytes((9, 9, 9, 9)))

    assert timeline.disk_bytes == len(original)
    assert timeline.path.stat().st_size == len(original)
    assert timeline.read_range(0, len(original)) == original
    store.retire()


def test_all_disk_writers_reject_zero_progress_and_keep_logical_lengths():
    store = DiskAudioStore(prefix="test_disk_short_write_")
    audio = store.create_spool("audio")
    frames = store.create_frame_spool("frames", dtype=np.dtype(np.float32))
    timeline = store.create_u8_timeline("timeline")

    audio._stream = _DelegatingStream(audio._stream, limit=2, stop_after=1)
    frames._stream = _DelegatingStream(frames._stream, limit=2, stop_after=1)
    timeline._stream = _DelegatingStream(timeline._stream, stop_after=0)

    with pytest.raises(OSError, match="short write"):
        audio.write_at(0, np.arange(4, dtype=np.float32))
    with pytest.raises(OSError, match="short write"):
        frames.append(np.arange(4, dtype=np.float32))
    with pytest.raises(OSError, match="short write"):
        timeline.set(4, 7)

    assert audio.length_samples == 0
    assert frames.count == 0
    assert timeline.disk_bytes == 0
    assert audio.path.stat().st_size == 0
    assert frames.path.stat().st_size == 0
    assert timeline.path.stat().st_size == 0
    store.retire()


def test_positive_short_writes_are_completed_and_flushed():
    store = DiskAudioStore(prefix="test_disk_partial_write_")
    spool = store.create_spool("audio")
    wrapped = _DelegatingStream(spool._stream, limit=3)
    spool._stream = wrapped
    waveform = np.linspace(-1.0, 1.0, 8, dtype=np.float32)

    spool.write_at(0, waveform)

    assert wrapped.write_calls > 1
    assert wrapped.flush_calls >= 1
    view = spool.view(0, len(waveform))
    np.testing.assert_array_equal(view, waveform)
    del view
    store.retire()


def test_low_free_space_fails_before_audio_spool_length_advances(monkeypatch):
    free_values = iter((10_000, 100))
    monkeypatch.setattr(
        disk_audio.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=next(free_values)),
    )
    store = DiskAudioStore(
        prefix="test_disk_free_space_",
        minimum_free_bytes=500,
    )
    spool = store.create_spool("audio")
    # Force the next growth through the real disk-usage branch instead of the
    # deliberately amortized 64 MiB cached-space fast path.
    store._space_bytes_since_check = store._SPACE_RECHECK_BYTES

    with pytest.raises(OSError) as raised:
        spool.write_at(0, np.arange(4, dtype=np.float32))

    assert raised.value.errno == errno.ENOSPC
    assert spool.length_samples == 0
    assert spool.disk_bytes == 0
    assert spool.path.stat().st_size == 0
    store.retire()


def test_view_mapping_outlives_retire_and_releases_before_cleanup():
    store = DiskAudioStore(prefix="test_disk_view_lifetime_")
    root = store.root
    spool = store.create_spool("audio")
    spool.write_at(0, np.arange(8, dtype=np.float32))
    view = spool.view(0, 8)
    child = view[2:]

    store.retire()
    assert root.exists()
    assert store.active_views == 1
    del view
    gc.collect()
    assert root.exists()
    np.testing.assert_array_equal(child, np.arange(2, 8, dtype=np.float32))

    del child
    _wait_until(lambda: store.active_views == 0 and not root.exists())


def test_permission_error_cleanup_is_retried_automatically(monkeypatch):
    store = DiskAudioStore(prefix="test_disk_cleanup_retry_")
    root = store.root
    original_rmtree = disk_audio.shutil.rmtree
    calls = 0

    def flaky_rmtree(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("simulated Windows mapping delay")
        return original_rmtree(path)

    monkeypatch.setattr(disk_audio.shutil, "rmtree", flaky_rmtree)
    store.retire()

    _wait_until(lambda: calls >= 2 and not root.exists())
