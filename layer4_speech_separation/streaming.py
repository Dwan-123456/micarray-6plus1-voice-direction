from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .contracts import L4_MODEL_SAMPLE_RATE, Layer4CandidatePair
from .interfaces import Layer4SeparationBackend
from .resampling import Layer4Resampler


L4_STREAM_BATCH_SAMPLES_48K = 10 * 48_000
L4_STREAM_OVERLAP_SAMPLES_48K = 48_000
_L3_HOP_SAMPLES = 960


def _readonly_audio(value: NDArray[np.float32], name: str) -> NDArray[np.float32]:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or not len(array)
        or array.dtype != np.float32
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be finite non-empty C-contiguous float32 mono audio")
    result = np.frombuffer(array.tobytes(), dtype=np.float32)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class Layer4StreamInputChunk:
    """One new, authoritative and overlap-free portion of an L3 track."""

    session_id: str
    stream_epoch: int
    track_id: int
    speaker_count: Literal[1, 2]
    start_sample_48k: int
    theta_deg: float
    waveform_48k: NDArray[np.float32]
    is_final: bool = False

    def __post_init__(self) -> None:
        if not self.session_id or self.stream_epoch < 0 or self.track_id <= 0:
            raise ValueError("L4 stream input identity is invalid")
        if type(self.speaker_count) is not int or self.speaker_count not in {1, 2}:
            raise ValueError("L4 stream input speaker_count must be one or two")
        if self.start_sample_48k < 0 or self.start_sample_48k % _L3_HOP_SAMPLES:
            raise ValueError("L4 stream input must start on the authoritative 20 ms grid")
        if not np.isfinite(self.theta_deg) or not 0.0 <= self.theta_deg < 360.0:
            raise ValueError("L4 stream input angle must be finite and in [0,360)")
        if type(self.is_final) is not bool:
            raise ValueError("L4 stream input final flag must be bool")
        waveform = _readonly_audio(self.waveform_48k, "L4 stream input")
        if len(waveform) % _L3_HOP_SAMPLES:
            raise ValueError("L4 stream input must contain complete 20 ms hops")
        object.__setattr__(self, "waveform_48k", waveform)

    @property
    def end_sample_48k(self) -> int:
        return self.start_sample_48k + len(self.waveform_48k)


@dataclass(frozen=True, slots=True)
class Layer4StreamOutputChunk:
    """An immutable L4 interval that will not be revised by a later push."""

    session_id: str
    stream_epoch: int
    track_id: int
    speaker_count: Literal[1, 2]
    branch_id: Literal[0, 1]
    commit_id: int
    start_sample_48k: int
    end_sample_48k: int
    theta_deg: float
    sample_rate: Literal[16000]
    waveform_16k: NDArray[np.float32]
    path: Literal["single_speaker_bypass", "two_speaker_separation"]
    request_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id or self.stream_epoch < 0 or self.track_id <= 0:
            raise ValueError("L4 stream output identity is invalid")
        if self.speaker_count not in {1, 2} or self.branch_id not in {0, 1}:
            raise ValueError("L4 stream output speaker or branch identity is invalid")
        if self.commit_id < 0 or self.sample_rate != L4_MODEL_SAMPLE_RATE:
            raise ValueError("L4 stream output commit metadata is invalid")
        if (
            self.start_sample_48k < 0
            or self.end_sample_48k <= self.start_sample_48k
            or self.start_sample_48k % _L3_HOP_SAMPLES
            or self.end_sample_48k % _L3_HOP_SAMPLES
        ):
            raise ValueError("L4 stream output timeline is invalid")
        if not np.isfinite(self.theta_deg) or not 0.0 <= self.theta_deg < 360.0:
            raise ValueError("L4 stream output angle must be finite and in [0,360)")
        waveform = _readonly_audio(self.waveform_16k, "L4 stream output")
        if len(waveform) * 3 != self.end_sample_48k - self.start_sample_48k:
            raise ValueError("L4 stream output must preserve the authoritative timeline")
        if len(waveform) % 320:
            raise ValueError("L4 stream output must contain complete 20 ms hops")
        if self.speaker_count == 1:
            if self.branch_id != 0 or self.path != "single_speaker_bypass" or self.request_id is not None:
                raise ValueError("single-speaker L4 stream output must be branch-zero bypass audio")
        elif self.path != "two_speaker_separation" or not self.request_id:
            raise ValueError("two-speaker L4 stream output requires separation provenance")
        object.__setattr__(self, "waveform_16k", waveform)


class Layer4StreamSession:
    """Ten-second L4 micro-batch adapter with one-second permutation repair.

    The current MossFormer2 model is a non-streaming full-window model.  This
    adapter gives it a bounded state boundary: the first regular batch commits
    all but its final overlap, and every later batch is inferred together with
    that input overlap.  The old and new output overlaps determine the stable
    anonymous-source permutation and are crossfaded before commitment.
    """

    def __init__(
        self,
        *,
        speaker_count: Literal[1, 2],
        backend: Layer4SeparationBackend | None = None,
        resampler: Layer4Resampler | None = None,
        batch_samples_48k: int = L4_STREAM_BATCH_SAMPLES_48K,
        overlap_samples_48k: int = L4_STREAM_OVERLAP_SAMPLES_48K,
    ) -> None:
        if type(speaker_count) is not int or speaker_count not in {1, 2}:
            raise ValueError("L4 stream session speaker_count must be one or two")
        if speaker_count == 2 and backend is None:
            raise ValueError("two-speaker L4 stream sessions require a separation backend")
        if backend is not None and (
            int(getattr(backend, "sample_rate", 0)) != L4_MODEL_SAMPLE_RATE
            or int(getattr(backend, "source_count", 0)) != 2
        ):
            raise ValueError("L4 stream backend must produce two 16 kHz sources")
        if (
            type(batch_samples_48k) is not int
            or type(overlap_samples_48k) is not int
            or batch_samples_48k <= overlap_samples_48k
            or overlap_samples_48k <= 0
            or batch_samples_48k % _L3_HOP_SAMPLES
            or overlap_samples_48k % _L3_HOP_SAMPLES
        ):
            raise ValueError("L4 stream batch and overlap must be ordered complete 20 ms intervals")

        self.speaker_count = speaker_count
        self.backend = backend
        self.resampler = resampler or Layer4Resampler()
        self.batch_samples_48k = batch_samples_48k
        self.overlap_samples_48k = overlap_samples_48k
        self.batch_samples_16k = batch_samples_48k // 3
        self.overlap_samples_16k = overlap_samples_48k // 3

        self._identity: tuple[str, int, int] | None = None
        self._next_input_sample_48k: int | None = None
        self._next_output_sample_48k: int | None = None
        self._theta_deg = 0.0
        self._pending_start_sample_48k: int | None = None
        self._pending_48k = np.empty(0, dtype=np.float32)
        self._previous_input_tail_48k: np.ndarray | None = None
        self._previous_output_tails_16k: tuple[np.ndarray, np.ndarray] | None = None
        self._previous_request_id: str | None = None
        self._processed_regular_batch = False
        self._request_sequence = 0
        self._commit_sequence = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending_samples_48k(self) -> int:
        return len(self._pending_48k)

    @property
    def model_request_count(self) -> int:
        """Number of two-speaker backend windows inferred by this session."""

        return self._request_sequence

    @staticmethod
    def _similarity(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= np.finfo(np.float32).eps:
            return 0.0
        return float(np.dot(left, right) / denominator)

    def _request_id(self) -> str:
        assert self._identity is not None
        session_id, stream_epoch, track_id = self._identity
        value = (
            f"{session_id}:epoch{stream_epoch}:track{track_id}:"
            f"l4-stream-window{self._request_sequence:06d}"
        )
        self._request_sequence += 1
        return value

    def _resample(self, waveform_48k: np.ndarray) -> np.ndarray:
        output = np.asarray(
            self.resampler.to_16k(np.ascontiguousarray(waveform_48k, dtype=np.float32))
        )
        expected = len(waveform_48k) // 3
        if (
            output.shape != (expected,)
            or output.dtype != np.float32
            or not output.flags.c_contiguous
            or not np.isfinite(output).all()
        ):
            raise RuntimeError("L4 stream resampler did not preserve the exact 48/16 kHz timeline")
        return np.ascontiguousarray(output, dtype=np.float32)

    def _separate(self, waveform_48k: np.ndarray) -> Layer4CandidatePair:
        assert self.backend is not None
        request_id = self._request_id()
        audio_16k = self._resample(waveform_48k)
        result = self.backend.separate(request_id, audio_16k)
        if result.request_id != request_id or result.sample_rate != L4_MODEL_SAMPLE_RATE:
            raise RuntimeError("L4 stream backend returned incompatible request provenance")
        if any(len(source) != len(audio_16k) for source in result.sources):
            raise RuntimeError("L4 stream backend changed the source timeline")
        return result

    def _aligned_sources(
        self,
        candidates: Layer4CandidatePair,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._previous_output_tails_16k is not None
        current = tuple(
            np.ascontiguousarray(source, dtype=np.float32) for source in candidates.sources
        )
        head = tuple(source[: self.overlap_samples_16k] for source in current)
        previous = self._previous_output_tails_16k
        straight = self._similarity(previous[0], head[0]) + self._similarity(previous[1], head[1])
        swapped = self._similarity(previous[0], head[1]) + self._similarity(previous[1], head[0])
        return current if straight >= swapped else (current[1], current[0])

    def _crossfade(
        self,
        current: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._previous_output_tails_16k is not None
        ramp = np.linspace(
            0.0, 1.0, self.overlap_samples_16k, endpoint=False, dtype=np.float32,
        )
        inverse = np.float32(1.0) - ramp
        return tuple(
            np.ascontiguousarray(previous * inverse + source[: self.overlap_samples_16k] * ramp)
            for previous, source in zip(self._previous_output_tails_16k, current, strict=True)
        )  # type: ignore[return-value]

    def _emit(
        self,
        waveforms_16k: tuple[np.ndarray, ...],
        *,
        request_id: str | None,
    ) -> tuple[Layer4StreamOutputChunk, ...]:
        assert self._identity is not None and self._next_output_sample_48k is not None
        if not waveforms_16k or any(len(value) != len(waveforms_16k[0]) for value in waveforms_16k):
            raise RuntimeError("L4 stream commit requires equal non-empty branch audio")
        start = self._next_output_sample_48k
        end = start + len(waveforms_16k[0]) * 3
        session_id, stream_epoch, track_id = self._identity
        commit_id = self._commit_sequence
        self._commit_sequence += 1
        self._next_output_sample_48k = end
        path = "single_speaker_bypass" if self.speaker_count == 1 else "two_speaker_separation"
        return tuple(
            Layer4StreamOutputChunk(
                session_id=session_id,
                stream_epoch=stream_epoch,
                track_id=track_id,
                speaker_count=self.speaker_count,
                branch_id=branch_id,  # type: ignore[arg-type]
                commit_id=commit_id,
                start_sample_48k=start,
                end_sample_48k=end,
                theta_deg=self._theta_deg,
                sample_rate=L4_MODEL_SAMPLE_RATE,
                waveform_16k=np.ascontiguousarray(waveform, dtype=np.float32),
                path=path,
                request_id=request_id,
            )
            for branch_id, waveform in enumerate(waveforms_16k)
        )

    def _process_regular_batch(
        self,
        block_48k: np.ndarray,
    ) -> tuple[Layer4StreamOutputChunk, ...]:
        if self.speaker_count == 1:
            output = self._resample(block_48k)
            return self._emit((output,), request_id=None)

        if not self._processed_regular_batch:
            candidates = self._separate(block_48k)
            sources = tuple(
                np.ascontiguousarray(source, dtype=np.float32) for source in candidates.sources
            )
            prefix = tuple(source[: -self.overlap_samples_16k] for source in sources)
            self._previous_input_tail_48k = np.ascontiguousarray(
                block_48k[-self.overlap_samples_48k :], dtype=np.float32,
            )
            self._previous_output_tails_16k = tuple(
                np.ascontiguousarray(source[-self.overlap_samples_16k :], dtype=np.float32)
                for source in sources
            )  # type: ignore[assignment]
            self._previous_request_id = candidates.request_id
            self._processed_regular_batch = True
            return self._emit(prefix, request_id=candidates.request_id)

        assert self._previous_input_tail_48k is not None
        window = np.ascontiguousarray(
            np.concatenate((self._previous_input_tail_48k, block_48k)), dtype=np.float32,
        )
        candidates = self._separate(window)
        current = self._aligned_sources(candidates)
        blended = self._crossfade(current)
        committed = tuple(
            np.ascontiguousarray(
                np.concatenate((blend, source[self.overlap_samples_16k : -self.overlap_samples_16k])),
                dtype=np.float32,
            )
            for blend, source in zip(blended, current, strict=True)
        )
        self._previous_input_tail_48k = np.ascontiguousarray(
            block_48k[-self.overlap_samples_48k :], dtype=np.float32,
        )
        self._previous_output_tails_16k = tuple(
            np.ascontiguousarray(source[-self.overlap_samples_16k :], dtype=np.float32)
            for source in current
        )  # type: ignore[assignment]
        self._previous_request_id = candidates.request_id
        return self._emit(committed, request_id=candidates.request_id)

    def _accept_identity(self, item: Layer4StreamInputChunk) -> None:
        identity = (item.session_id, item.stream_epoch, item.track_id)
        if self._identity is None:
            self._identity = identity
            self._next_input_sample_48k = item.start_sample_48k
            self._next_output_sample_48k = item.start_sample_48k
            self._pending_start_sample_48k = item.start_sample_48k
        elif identity != self._identity:
            raise ValueError("one L4 stream session cannot mix track identities")
        if item.speaker_count != self.speaker_count:
            raise ValueError("L4 stream speaker_count cannot change inside a session")
        if item.start_sample_48k != self._next_input_sample_48k:
            raise ValueError("L4 stream input chunks must be contiguous and overlap-free")

    def push(self, item: Layer4StreamInputChunk) -> tuple[Layer4StreamOutputChunk, ...]:
        if self._closed:
            raise RuntimeError("L4 stream session is already closed")
        self._accept_identity(item)
        self._theta_deg = item.theta_deg
        self._pending_48k = np.ascontiguousarray(
            np.concatenate((self._pending_48k, item.waveform_48k)), dtype=np.float32,
        )
        self._next_input_sample_48k = item.end_sample_48k

        outputs: list[Layer4StreamOutputChunk] = []
        while len(self._pending_48k) >= self.batch_samples_48k:
            block = np.ascontiguousarray(
                self._pending_48k[: self.batch_samples_48k], dtype=np.float32,
            )
            self._pending_48k = np.ascontiguousarray(
                self._pending_48k[self.batch_samples_48k :], dtype=np.float32,
            )
            assert self._pending_start_sample_48k is not None
            self._pending_start_sample_48k += self.batch_samples_48k
            outputs.extend(self._process_regular_batch(block))

        if item.is_final:
            outputs.extend(self._finish())
            self._closed = True
        return tuple(outputs)

    def _finish(self) -> tuple[Layer4StreamOutputChunk, ...]:
        if self._identity is None:
            raise RuntimeError("cannot flush an L4 stream session before its first input")
        remainder = self._pending_48k
        self._pending_48k = np.empty(0, dtype=np.float32)

        if self.speaker_count == 1:
            if not len(remainder):
                return ()
            return self._emit((self._resample(remainder),), request_id=None)

        if not self._processed_regular_batch:
            if not len(remainder):
                return ()
            candidates = self._separate(remainder)
            return self._emit(
                tuple(np.ascontiguousarray(value, dtype=np.float32) for value in candidates.sources),
                request_id=candidates.request_id,
            )

        assert self._previous_output_tails_16k is not None
        if not len(remainder):
            return self._emit(
                self._previous_output_tails_16k,
                request_id=self._previous_request_id,
            )

        assert self._previous_input_tail_48k is not None
        window = np.ascontiguousarray(
            np.concatenate((self._previous_input_tail_48k, remainder)), dtype=np.float32,
        )
        candidates = self._separate(window)
        current = self._aligned_sources(candidates)
        blended = self._crossfade(current)
        committed = tuple(
            np.ascontiguousarray(
                np.concatenate((blend, source[self.overlap_samples_16k :])), dtype=np.float32,
            )
            for blend, source in zip(blended, current, strict=True)
        )
        return self._emit(committed, request_id=candidates.request_id)

    def flush(self) -> tuple[Layer4StreamOutputChunk, ...]:
        if self._closed:
            raise RuntimeError("L4 stream session is already closed")
        outputs = self._finish()
        self._closed = True
        return outputs
