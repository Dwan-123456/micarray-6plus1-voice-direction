from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from layer4_speech_separation import (
    L4_STREAM_BATCH_SAMPLES_48K,
    Layer4CandidatePair,
    Layer4StreamInputChunk,
    Layer4StreamOutputChunk,
    Layer4StreamSession,
)
from layer4_speech_separation.models import _OfficialModelBackend


class _DecimatingResampler:
    def __init__(self) -> None:
        self.input_lengths: list[int] = []

    def to_16k(self, waveform_48k: np.ndarray) -> np.ndarray:
        self.input_lengths.append(len(waveform_48k))
        return np.ascontiguousarray(waveform_48k[::3], dtype=np.float32)


class _PairBackend:
    sample_rate = 16_000
    source_count = 2
    model_id = "pair-spy"
    model_revision = "test"

    def __init__(
        self,
        factory: Callable[[int, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.factory = factory or (
            lambda _index, audio: (audio.copy(), -audio.copy())
        )
        self.inputs: list[np.ndarray] = []

    def separate(self, request_id: str, waveform_16k: np.ndarray) -> Layer4CandidatePair:
        audio = np.ascontiguousarray(waveform_16k, dtype=np.float32)
        self.inputs.append(audio.copy())
        sources = self.factory(len(self.inputs) - 1, audio)
        return Layer4CandidatePair(
            request_id,
            self.model_id,
            self.model_revision,
            16_000,
            tuple(np.ascontiguousarray(item, dtype=np.float32) for item in sources),
        )


def _input(
    waveform: np.ndarray,
    *,
    start: int,
    speaker_count: int = 2,
    final: bool = False,
) -> Layer4StreamInputChunk:
    return Layer4StreamInputChunk(
        session_id="session",
        stream_epoch=0,
        track_id=7,
        speaker_count=speaker_count,  # type: ignore[arg-type]
        start_sample_48k=start,
        theta_deg=45.0,
        waveform_48k=np.ascontiguousarray(waveform, dtype=np.float32),
        is_final=final,
    )


def _branch(
    outputs: tuple[Layer4StreamOutputChunk, ...], branch_id: int,
) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate([
        item.waveform_16k for item in outputs if item.branch_id == branch_id
    ]), dtype=np.float32)


def test_default_ten_second_stream_uses_ten_then_eleven_second_model_windows() -> None:
    samples = 2 * L4_STREAM_BATCH_SAMPLES_48K
    time = np.arange(samples, dtype=np.float32) / np.float32(48_000)
    source = np.ascontiguousarray(np.sin(2 * np.pi * 317.0 * time), dtype=np.float32)
    resampler = _DecimatingResampler()

    def alternating(index: int, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pair = (audio.copy(), -audio.copy())
        return pair if index % 2 == 0 else pair[::-1]

    backend = _PairBackend(alternating)
    stream = Layer4StreamSession(
        speaker_count=2,
        backend=backend,
        resampler=resampler,  # type: ignore[arg-type]
    )

    first = stream.push(_input(source[: L4_STREAM_BATCH_SAMPLES_48K], start=0))
    second = stream.push(_input(
        source[L4_STREAM_BATCH_SAMPLES_48K :],
        start=L4_STREAM_BATCH_SAMPLES_48K,
    ))
    tail = stream.flush()
    outputs = (*first, *second, *tail)

    assert [len(item) for item in backend.inputs] == [160_000, 176_000]
    assert resampler.input_lengths == [480_000, 528_000]
    assert [(item.start_sample_48k, item.end_sample_48k) for item in outputs if item.branch_id == 0] == [
        (0, 432_000),
        (432_000, 912_000),
        (912_000, 960_000),
    ]
    assert [item.commit_id for item in outputs if item.branch_id == 0] == [0, 1, 2]
    expected = np.ascontiguousarray(source[::3], dtype=np.float32)
    np.testing.assert_allclose(_branch(outputs, 0), expected, atol=2e-7)
    np.testing.assert_allclose(_branch(outputs, 1), -expected, atol=2e-7)
    assert all(not item.waveform_16k.flags.writeable for item in outputs)


def test_current_official_backend_does_not_expand_ten_seconds_to_its_thirty_second_chunk() -> None:
    backend = _OfficialModelBackend.__new__(_OfficialModelBackend)
    backend.chunk_samples = 30 * 16_000
    backend.overlap_samples = 16_000
    backend.model_id = "mf2-spy"
    backend.model_revision = "test"
    seen: list[int] = []

    def forward(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seen.append(len(audio))
        return audio.copy(), -audio.copy()

    backend._forward = forward
    audio = np.zeros(10 * 16_000, dtype=np.float32)

    result = backend.separate("ten-seconds", audio)

    assert seen == [10 * 16_000]
    assert tuple(len(item) for item in result.sources) == (10 * 16_000, 10 * 16_000)


def test_two_speaker_overlap_is_crossfaded_with_cached_weights(monkeypatch) -> None:
    batch = 10 * 960
    overlap = 2 * 960

    def levels(index: int, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        level = np.float32(1.0 if index == 0 else 3.0)
        positive = np.full(len(audio), level, dtype=np.float32)
        return positive, -positive

    backend = _PairBackend(levels)
    stream = Layer4StreamSession(
        speaker_count=2,
        backend=backend,
        resampler=_DecimatingResampler(),  # type: ignore[arg-type]
        batch_samples_48k=batch,
        overlap_samples_48k=overlap,
    )
    overlap_16k = overlap // 3
    ramp = np.linspace(0.0, 1.0, overlap_16k, endpoint=False, dtype=np.float32)

    def unexpected_linspace(*_args, **_kwargs):
        raise AssertionError("crossfade weights must be cached at session construction")

    monkeypatch.setattr(np, "linspace", unexpected_linspace)
    stream.push(_input(np.zeros(batch, np.float32), start=0))
    outputs = stream.push(_input(np.zeros(batch, np.float32), start=batch))
    positive = next(item.waveform_16k for item in outputs if item.branch_id == 0)

    np.testing.assert_allclose(positive[:overlap_16k], 1.0 + 2.0 * ramp, atol=1e-6)
    np.testing.assert_array_equal(positive[overlap_16k:], np.float32(3.0))


def test_exact_four_second_batch_skips_empty_pending_concatenate(monkeypatch) -> None:
    batch = 4 * 48_000
    overlap = 48_000
    source = np.ascontiguousarray(
        np.linspace(-0.25, 0.25, batch, dtype=np.float32),
    )
    item = _input(source, start=0, speaker_count=1)
    resampler = _DecimatingResampler()
    stream = Layer4StreamSession(
        speaker_count=1,
        resampler=resampler,  # type: ignore[arg-type]
        batch_samples_48k=batch,
        overlap_samples_48k=overlap,
    )
    original_concatenate = np.concatenate
    calls = 0

    def counted_concatenate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_concatenate(*args, **kwargs)

    monkeypatch.setattr(np, "concatenate", counted_concatenate)

    outputs = stream.push(item)

    assert calls == 0
    assert stream.pending_samples_48k == 0
    assert resampler.input_lengths == [batch]
    assert len(outputs) == 1
    np.testing.assert_array_equal(outputs[0].waveform_16k, source[::3])


def test_single_speaker_bypass_never_calls_the_separator_and_flushes_remainder() -> None:
    batch = 10 * 960
    remainder = 3 * 960
    backend = _PairBackend()
    resampler = _DecimatingResampler()
    source = np.ascontiguousarray(
        np.linspace(-0.5, 0.5, batch + remainder, dtype=np.float32),
    )
    stream = Layer4StreamSession(
        speaker_count=1,
        backend=backend,
        resampler=resampler,  # type: ignore[arg-type]
        batch_samples_48k=batch,
        overlap_samples_48k=2 * 960,
    )

    assert stream.push(_input(source[: 4 * 960], start=0, speaker_count=1)) == ()
    regular = stream.push(_input(
        source[4 * 960 : batch], start=4 * 960, speaker_count=1,
    ))
    assert stream.push(_input(source[batch:], start=batch, speaker_count=1)) == ()
    final = stream.flush()

    assert backend.inputs == []
    assert resampler.input_lengths == [batch, remainder]
    assert len(regular) == len(final) == 1
    assert regular[0].path == final[0].path == "single_speaker_bypass"
    assert regular[0].request_id is final[0].request_id is None
    assert (regular[0].start_sample_48k, regular[0].end_sample_48k) == (0, batch)
    assert (final[0].start_sample_48k, final[0].end_sample_48k) == (
        batch, batch + remainder,
    )
    np.testing.assert_array_equal(
        np.concatenate((regular[0].waveform_16k, final[0].waveform_16k)),
        source[::3],
    )


def test_short_final_two_speaker_stream_is_inferred_once_without_a_regular_batch() -> None:
    backend = _PairBackend()
    source = np.ascontiguousarray(np.linspace(-0.2, 0.2, 3 * 960, dtype=np.float32))
    stream = Layer4StreamSession(
        speaker_count=2,
        backend=backend,
        resampler=_DecimatingResampler(),  # type: ignore[arg-type]
        batch_samples_48k=10 * 960,
        overlap_samples_48k=2 * 960,
    )

    outputs = stream.push(_input(source, start=0, final=True))

    assert len(backend.inputs) == 1 and len(backend.inputs[0]) == len(source) // 3
    assert len(outputs) == 2
    assert {(item.start_sample_48k, item.end_sample_48k) for item in outputs} == {
        (0, len(source)),
    }
    assert stream.closed
    with pytest.raises(RuntimeError, match="already closed"):
        stream.flush()
    with pytest.raises(RuntimeError, match="already closed"):
        stream.push(_input(source, start=len(source)))


def test_two_speaker_flush_reprocesses_overlap_with_a_partial_tail() -> None:
    batch = 10 * 960
    overlap = 2 * 960
    remainder = 3 * 960
    source = np.ascontiguousarray(
        np.sin(2 * np.pi * 0.017 * np.arange(batch + remainder)), dtype=np.float32,
    )
    backend = _PairBackend()
    stream = Layer4StreamSession(
        speaker_count=2,
        backend=backend,
        resampler=_DecimatingResampler(),  # type: ignore[arg-type]
        batch_samples_48k=batch,
        overlap_samples_48k=overlap,
    )

    regular = stream.push(_input(source[:batch], start=0))
    assert stream.push(_input(source[batch:], start=batch)) == ()
    final = stream.flush()
    outputs = (*regular, *final)

    assert [len(item) for item in backend.inputs] == [batch // 3, (overlap + remainder) // 3]
    assert [(item.start_sample_48k, item.end_sample_48k) for item in outputs if item.branch_id == 0] == [
        (0, batch - overlap),
        (batch - overlap, batch + remainder),
    ]
    np.testing.assert_allclose(_branch(outputs, 0), source[::3], atol=2e-7)
    np.testing.assert_allclose(_branch(outputs, 1), -source[::3], atol=2e-7)


def test_stream_rejects_timeline_gaps_and_speaker_count_changes() -> None:
    stream = Layer4StreamSession(
        speaker_count=1,
        resampler=_DecimatingResampler(),  # type: ignore[arg-type]
        batch_samples_48k=10 * 960,
        overlap_samples_48k=2 * 960,
    )
    first = np.zeros(2 * 960, dtype=np.float32)
    assert stream.push(_input(first, start=0, speaker_count=1)) == ()

    with pytest.raises(ValueError, match="contiguous"):
        stream.push(_input(first, start=3 * 960, speaker_count=1))
    with pytest.raises(ValueError, match="cannot change"):
        stream.push(_input(first, start=2 * 960, speaker_count=2))
