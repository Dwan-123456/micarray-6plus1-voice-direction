from __future__ import annotations

import hashlib
import io
import queue
import threading
from collections import deque
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy import signal

from common.config import Layer1SpeakerCountConfig
from common.data_types import IngestedAudioBlock
from .interface import SpeakerCountAnnotation


class SpeakerCountModel(Protocol):
    model_id: str
    model_hash: str

    def predict(self, waveform_16k: np.ndarray) -> np.ndarray: ...


class TorchScriptCountNet:
    """Verified TorchScript adapter; model classes above two are folded into P2."""

    def __init__(self, artifact: Path, *, model_id: str, expected_hash: str) -> None:
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != expected_hash:
            raise ValueError("CountNet model SHA-256 does not match configuration")
        import torch

        self.model_id, self.model_hash = model_id, actual
        self._torch = torch
        self._model = torch.jit.load(io.BytesIO(artifact.read_bytes()), map_location="cpu").eval()

    def predict(self, waveform_16k: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            output = self._model(torch.from_numpy(waveform_16k[None, :].astype(np.float32)))
        logits = np.asarray(output.detach().cpu(), dtype=np.float64).reshape(-1)
        if logits.size < 3 or not np.isfinite(logits).all():
            raise ValueError("CountNet output must contain at least three finite classes")
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return np.asarray((probabilities[0], probabilities[1], probabilities[2:].sum()), np.float32)


class AsyncSpeakerCounter:
    """Non-blocking L1 sidecar with stateful anti-alias 48→16 kHz decimation."""

    input_rate = 48_000
    output_rate = 16_000
    center_channel = 6

    def __init__(
        self,
        config: Layer1SpeakerCountConfig,
        *,
        project_root: Path,
        model: SpeakerCountModel | None = None,
        timestamp_tolerance_ms: float = 5.0,
    ) -> None:
        self.config = config
        self._enabled = bool(config.enabled)
        self._timestamp_tolerance = timestamp_tolerance_ms / 1_000.0
        self._queue: queue.Queue[IngestedAudioBlock | None] = queue.Queue(maxsize=config.queue_blocks)
        self._lock = threading.Lock()
        self._latest: SpeakerCountAnnotation | None = None
        self._generation = 0
        self._overflowed = threading.Event()
        self._reset_requested = threading.Event()
        self.dropped_blocks = 0
        self.model_error: str | None = None
        self._model = model
        self._artifact = Path(config.model_artifact)
        if not self._artifact.is_absolute():
            self._artifact = project_root / self._artifact
        if self._model is None:
            if config.model_sha256 is None:
                self.model_error = "model SHA-256 is not configured"
            elif not self._artifact.is_file():
                self.model_error = f"model asset missing: {self._artifact}"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self._enabled:
            self._ensure_thread()

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._run, name="l1-countnet-worker", daemon=True)
                self._thread.start()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, value: bool) -> bool:
        if type(value) is not bool:
            raise ValueError("L1 CountNet setting must be bool")
        with self._lock:
            if value != self._enabled:
                self._enabled = value
                self._generation += 1
                self._latest = None
                self._reset_requested.set()
                while True:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
        if value:
            self._ensure_thread()
        return value

    def submit(self, block: IngestedAudioBlock) -> bool:
        if not self.enabled:
            return False
        try:
            self._queue.put_nowait(block)
            return True
        except queue.Full:
            self.dropped_blocks += 1
            self._overflowed.set()
            return False

    def latest(self) -> SpeakerCountAnnotation | None:
        with self._lock:
            return self._latest

    def close(self, timeout: float = 1.0) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _publish(self, value: SpeakerCountAnnotation, generation: int) -> None:
        with self._lock:
            if self._enabled and generation == self._generation:
                self._latest = value

    def _ensure_model(self) -> None:
        if self._model is not None or self.model_error is not None:
            return
        try:
            assert self.config.model_sha256 is not None
            self._model = TorchScriptCountNet(
                self._artifact,
                model_id=self.config.model_id,
                expected_hash=self.config.model_sha256,
            )
        except Exception as exc:
            self.model_error = str(exc)

    def _adapt_input_level(self, context: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Match quiet calibrated-array audio to the pretrained model's level domain."""

        centered = np.asarray(context, dtype=np.float32) - np.float32(np.mean(context))
        rms = float(np.sqrt(np.mean(np.square(centered, dtype=np.float64))))
        rms_dbfs = float(20.0 * np.log10(max(rms, np.finfo(np.float32).tiny)))
        gain_db = 0.0
        if rms_dbfs >= self.config.input_level_floor_dbfs:
            gain_db = float(min(
                self.config.maximum_input_gain_db,
                max(0.0, self.config.input_level_target_dbfs - rms_dbfs),
            ))
        adapted = centered * np.float32(10.0 ** (gain_db / 20.0))
        peak = float(np.max(np.abs(adapted)))
        if peak > 1.0:
            limiter = 1.0 / peak
            adapted *= np.float32(limiter)
            gain_db += float(20.0 * np.log10(limiter))
        return np.ascontiguousarray(adapted, dtype=np.float32), rms_dbfs, gain_db

    def _run(self) -> None:
        sos = signal.butter(8, 7_200.0, btype="lowpass", fs=self.input_rate, output="sos")
        zi_template = np.zeros((sos.shape[0], 2), dtype=np.float64)
        key: tuple[str, int] | None = None
        previous: IngestedAudioBlock | None = None
        zi = zi_template.copy()
        chunks: deque[np.ndarray] = deque()
        buffered = 0
        group_start: int | None = None
        group_blocks = 0
        reset_invalid = False

        def reset(next_key: tuple[str, int] | None = None, *, invalid: bool = True) -> None:
            nonlocal key, previous, zi, buffered, group_start, group_blocks, reset_invalid
            key, previous, zi = next_key, None, zi_template.copy()
            chunks.clear()
            buffered, group_start, group_blocks = 0, None, 0
            reset_invalid = invalid

        while not self._stop.is_set():
            try:
                block = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if block is None:
                continue
            with self._lock:
                enabled, generation = self._enabled, self._generation
            if not enabled:
                reset(None)
                continue
            block_key = (block.session_id, block.stream_epoch)
            if self._reset_requested.is_set():
                reset(block_key, invalid=False)
                self._reset_requested.clear()
            continuity_error = self._overflowed.is_set() or (key is not None and key != block_key)
            self._overflowed.clear()
            if previous is not None and key == block_key:
                expected_timestamp = previous.timestamp + (previous.end_sample - previous.start_sample) / self.input_rate
                continuity_error = continuity_error or (
                    block.sequence_id != previous.sequence_id + 1
                    or block.start_sample != previous.end_sample
                    or abs(block.timestamp - expected_timestamp) > self._timestamp_tolerance
                )
            if continuity_error:
                reset(block_key)
            elif key is None:
                reset(block_key, invalid=False)
            key = block_key
            filtered, zi = signal.sosfilt(sos, np.asarray(block.samples[:, self.center_channel]), zi=zi)
            downsampled = np.ascontiguousarray(filtered[::3], dtype=np.float32)
            if len(downsampled) * 3 != len(block.samples):
                reset(block_key)
                previous = block
                continue
            chunks.append(downsampled)
            buffered += len(downsampled)
            limit = self.output_rate * self.config.context_seconds
            while chunks and buffered - len(chunks[0]) >= limit:
                buffered -= len(chunks.popleft())
            group_start = block.start_sample if group_start is None else group_start
            group_blocks += 1
            previous = block
            if group_blocks != 5:
                continue
            start_sample, end_sample = group_start, block.end_sample
            group_start, group_blocks = None, 0
            status, count, probabilities, reason = "warming_up", None, None, None
            input_rms_dbfs, input_gain_db = None, None
            if reset_invalid:
                status, reason, reset_invalid = "invalid", "stream continuity reset", False
            elif buffered >= limit:
                self._ensure_model()
                if self._model is None:
                    status, reason = "invalid", self.model_error or "CountNet model unavailable"
                else:
                    try:
                        context = np.concatenate(tuple(chunks))[-limit:]
                        context, input_rms_dbfs, input_gain_db = self._adapt_input_level(context)
                        probabilities = np.asarray(self._model.predict(context), dtype=np.float32)
                        if probabilities.shape != (3,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0):
                            raise ValueError("model probabilities must be finite non-negative [3]")
                        probabilities = probabilities / np.maximum(probabilities.sum(), 1.0e-12)
                        count, status = int(np.argmax(probabilities)), "ready"
                    except Exception as exc:
                        status, probabilities, reason = "invalid", None, f"inference failed: {exc}"
            self._publish(SpeakerCountAnnotation(
                block.session_id, block.stream_epoch, start_sample, end_sample,
                count, probabilities, self.config.model_id,
                None if self._model is None else self._model.model_hash,
                status, reason, input_rms_dbfs, input_gain_db,
            ), generation)
