from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import threading
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, TypeVar

import numpy as np

from common.data_types import IngestedAudioBlock
from layer1_input.pcm import pcm16_bytes


@dataclass(slots=True)
class _OpenSegment:
    index: int
    session_id: str
    stream_epoch: int
    start_sample: int
    end_sample: int
    first_sequence_id: int
    last_sequence_id: int
    sequence_count: int
    native: wave.Wave_write | None
    physical: wave.Wave_write
    float_raw: BinaryIO
    native_path: Path | None
    physical_path: Path
    float_path: Path
    float_raw_path: Path


@dataclass(slots=True)
class _Command:
    action: Callable[[], object]
    done: threading.Event
    result: object | None = None
    error: BaseException | None = None


T = TypeVar("T")


class ScratchRecorder:
    """Bounded, development-only scratch recorder with one writer owner."""

    STATES = {"idle", "recording", "paused", "finalizing", "complete", "error"}

    def __init__(self, current_root: str | Path, *, project_root: str | Path, queue_blocks: int = 100):
        self.project_root = Path(project_root).resolve()
        self.current_root = (
            (self.project_root / current_root).resolve()
            if not Path(current_root).is_absolute()
            else Path(current_root).resolve()
        )
        self.scratch_parent = (self.project_root / "data/dev_test_ui/scratch").resolve()
        if self.scratch_parent not in self.current_root.parents or self.current_root.name != "current":
            raise ValueError("scratch路径必须是项目data/dev_test_ui/scratch/current")
        if queue_blocks <= 0:
            raise ValueError("scratch队列容量必须为正数")
        self._state = "idle"
        self._state_lock = threading.RLock()
        self._segments: list[dict[str, object]] = []
        self._open: _OpenSegment | None = None
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=queue_blocks)
        self._worker_error: BaseException | None = None
        self._stop_requested = threading.Event()
        self._worker = threading.Thread(target=self._run, name="scratch-writer", daemon=True)
        self._worker.start()

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def queued_blocks(self) -> int:
        return self._queue.qsize()

    @property
    def last_error(self) -> str | None:
        with self._state_lock:
            return None if self._worker_error is None else str(self._worker_error)

    def _set_state(self, value: str) -> None:
        with self._state_lock:
            self._state = value

    def _run(self) -> None:
        try:
            while True:
                try:
                    kind, value = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._stop_requested.is_set():
                        break
                    continue
                try:
                    if kind == "stop":
                        break
                    if kind == "audio":
                        self._append_on_worker(value)
                    elif kind == "command":
                        command = value
                        try:
                            command.result = command.action()
                        except BaseException as exc:  # propagate command failures to the caller
                            command.error = exc
                        finally:
                            command.done.set()
                except BaseException as exc:
                    with self._state_lock:
                        self._worker_error = exc
                        self._state = "error"
                if self._stop_requested.is_set() and self._queue.empty():
                    break
        finally:
            # Even if a command could not be enqueued, the worker remains the
            # sole owner that closes any open WAV/raw handles before shutdown.
            try:
                self._close_segment()
            except BaseException as exc:
                with self._state_lock:
                    self._worker_error = self._worker_error or exc
                    self._state = "error"

    def _command(self, action: Callable[[], T], *, timeout: float = 30.0) -> T:
        if not self._worker.is_alive():
            error = RuntimeError("scratch writer已停止")
            with self._state_lock:
                self._worker_error = error
                self._state = "error"
            raise error
        command = _Command(action, threading.Event())
        try:
            self._queue.put(("command", command), timeout=2.0)
        except queue.Full as exc:
            error = RuntimeError("scratch writer命令队列已满")
            with self._state_lock:
                self._worker_error = error
                self._state = "error"
            raise error from exc
        if not command.done.wait(timeout):
            self._set_state("error")
            raise TimeoutError("scratch writer命令超时")
        if command.error is not None:
            with self._state_lock:
                self._worker_error = command.error
                self._state = "error"
            raise command.error
        return command.result  # type: ignore[return-value]

    def _prepare_empty_current(self) -> None:
        self.scratch_parent.mkdir(parents=True, exist_ok=True)
        deleting: Path | None = None
        if self.current_root.exists():
            deleting = self.scratch_parent / f"deleting-{uuid.uuid4()}"
            self.current_root.replace(deleting)
        try:
            (self.current_root / "segments").mkdir(parents=True, exist_ok=False)
        except Exception:
            if deleting is not None and deleting.exists() and not self.current_root.exists():
                deleting.replace(self.current_root)
            raise
        if deleting is not None:
            shutil.rmtree(deleting)

    def record(self) -> None:
        if self.state == "finalizing":
            raise RuntimeError("finalizing期间不能开始新录制")
        self._set_state("finalizing")

        def begin() -> None:
            self._close_segment()
            self._prepare_empty_current()
            self._segments.clear()
            with self._state_lock:
                self._worker_error = None
                self._state = "recording"

        self._command(begin)

    def pause(self) -> None:
        if self.state != "recording":
            raise RuntimeError("只有recording状态可以暂停")
        self._set_state("paused")
        self._command(self._close_segment)

    def resume(self) -> None:
        if self.state != "paused":
            raise RuntimeError("只有paused状态可以继续")
        self._set_state("recording")

    @staticmethod
    def _wav(path: Path, channels: int, sample_rate: int) -> wave.Wave_write:
        writer = wave.open(str(path) + ".partial", "wb")
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        return writer

    def _open_segment(self, block: IngestedAudioBlock) -> _OpenSegment:
        index = len(self._segments)
        stem = self.current_root / "segments" / f"segment_{index:03d}"
        native_path = stem.with_name(stem.name + "_native_8ch.wav") if block.native_samples is not None else None
        physical_path = stem.with_name(stem.name + "_physical_7ch.wav")
        float_path = stem.with_name(stem.name + "_physical_7ch_float.npy")
        float_raw_path = Path(str(float_path) + ".raw.partial")
        return _OpenSegment(
            index,
            block.session_id,
            block.stream_epoch,
            block.start_sample,
            block.start_sample,
            block.sequence_id,
            block.sequence_id,
            0,
            None if native_path is None else self._wav(native_path, 8, block.sample_rate),
            self._wav(physical_path, 7, block.sample_rate),
            float_raw_path.open("wb"),
            native_path,
            physical_path,
            float_path,
            float_raw_path,
        )

    def append(self, block: IngestedAudioBlock) -> None:
        if self.state != "recording":
            return
        try:
            self._queue.put_nowait(("audio", block))
        except queue.Full as exc:
            with self._state_lock:
                self._worker_error = RuntimeError("scratch写盘队列已满；录音已停止以保护实时采集")
                self._state = "error"
            raise RuntimeError("scratch写盘队列已满；录音已停止以保护实时采集") from exc

    def _append_on_worker(self, block: IngestedAudioBlock) -> None:
        if self._open is None:
            self._open = self._open_segment(block)
        segment = self._open
        if (block.session_id, block.stream_epoch, block.start_sample) != (
            segment.session_id,
            segment.stream_epoch,
            segment.end_sample,
        ):
            self._close_segment()
            segment = self._open_segment(block)
            self._open = segment
        if (segment.native is None) != (block.native_samples is None):
            raise ValueError("同一segment的native可用性不能变化")
        physical = block.samples[:, :7]
        segment.physical.writeframesraw(pcm16_bytes(physical))
        if segment.native is not None and block.native_samples is not None:
            segment.native.writeframesraw(pcm16_bytes(block.native_samples))
        segment.float_raw.write(memoryview(np.ascontiguousarray(physical)).cast("B"))
        segment.end_sample = block.end_sample
        segment.last_sequence_id = block.sequence_id
        segment.sequence_count += 1

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _close_segment(self) -> None:
        segment, self._open = self._open, None
        if segment is None:
            return
        frames = segment.end_sample - segment.start_sample
        try:
            segment.physical.close()
            Path(str(segment.physical_path) + ".partial").replace(segment.physical_path)
            if segment.native is not None and segment.native_path is not None:
                segment.native.close()
                Path(str(segment.native_path) + ".partial").replace(segment.native_path)
            segment.float_raw.flush()
            os.fsync(segment.float_raw.fileno())
            segment.float_raw.close()
            float_partial = Path(str(segment.float_path) + ".partial")
            mapped = np.lib.format.open_memmap(
                float_partial, mode="w+", dtype=np.float32, shape=(frames, 7), fortran_order=False
            )
            flat = mapped.reshape(-1)
            cursor = 0
            with segment.float_raw_path.open("rb") as source:
                while payload := source.read(1024 * 1024):
                    values = np.frombuffer(payload, dtype=np.float32)
                    flat[cursor : cursor + len(values)] = values
                    cursor += len(values)
            if cursor != frames * 7:
                raise OSError("scratch float32临时文件长度与sample范围不一致")
            mapped.flush()
            del flat, mapped
            float_partial.replace(segment.float_path)
            segment.float_raw_path.unlink(missing_ok=True)
            files = [segment.physical_path, segment.float_path] + (
                [] if segment.native_path is None else [segment.native_path]
            )
            self._segments.append(
                {
                    "index": segment.index,
                    "session_id": segment.session_id,
                    "stream_epoch": segment.stream_epoch,
                    "start_sample": segment.start_sample,
                    "end_sample": segment.end_sample,
                    "frame_count": frames,
                    "source_sequence_range": {
                        "first": segment.first_sequence_id,
                        "last": segment.last_sequence_id,
                        "count": segment.sequence_count,
                    },
                    "native_unavailable": segment.native_path is None,
                    "files": [
                        {"path": str(path.relative_to(self.current_root)), "sha256": self._hash(path)}
                        for path in files
                    ],
                }
            )
        except Exception:
            for writer in (segment.physical, segment.native, segment.float_raw):
                try:
                    if writer is not None:
                        writer.close()
                except Exception:
                    pass
            raise

    def finish(self) -> Path:
        if self.state not in {"recording", "paused"}:
            raise RuntimeError("当前没有可结束的录制")
        self._set_state("finalizing")

        def finalize() -> Path:
            self._close_segment()
            manifest = self.current_root / "scratch_manifest.json"
            partial = Path(str(manifest) + ".partial")
            payload = {"schema_version": "scratch_manifest_v1", "segments": self._segments}
            with partial.open("w", encoding="utf-8", newline="\n") as destination:
                json.dump(payload, destination, ensure_ascii=False, indent=2)
                destination.flush()
                os.fsync(destination.fileno())
            partial.replace(manifest)
            self._set_state("complete")
            return manifest

        return self._command(finalize)

    def close(self) -> None:
        if self.state in {"recording", "paused"}:
            self.finish()

    def _purge_scratch_files(self) -> None:
        """Delete only the Development Test UI scratch area after the writer stops."""
        expected_parent = (self.project_root / "data/dev_test_ui/scratch").resolve()
        if self.scratch_parent != expected_parent:
            raise RuntimeError("refusing to delete an unexpected scratch directory")
        if not self.scratch_parent.exists():
            return
        # Resolve again at deletion time so a replaced directory/symlink cannot
        # redirect cleanup outside data/dev_test_ui.
        if self.scratch_parent.resolve() != expected_parent:
            raise RuntimeError("refusing to delete a redirected scratch directory")
        deleting = expected_parent.parent / f"deleting-scratch-{uuid.uuid4()}"
        self.scratch_parent.replace(deleting)
        shutil.rmtree(deleting)
        self._segments.clear()
        self._set_state("idle")

    def shutdown(self, *, delete_files: bool = False) -> None:
        close_error: BaseException | None = None
        try:
            self.close()
        except BaseException as exc:
            # Still stop the writer and remove temporary files. A partially
            # finalized scratch recording is disposable by definition.
            close_error = exc
        if self._worker.is_alive():
            # A separate event avoids deadlock when the bounded command queue
            # is full.  The worker drains accepted items, closes its own file
            # handles in _run's finally block, then exits.
            self._stop_requested.set()
            self._worker.join(timeout=2.0)
        if self._worker.is_alive():
            raise RuntimeError("scratch writer did not stop; temporary files were not deleted")
        if delete_files:
            self._purge_scratch_files()
        if close_error is not None and not delete_files:
            raise close_error
