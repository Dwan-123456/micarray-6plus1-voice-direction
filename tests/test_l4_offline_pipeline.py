from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from layer4_speech_separation import (
    DirectionCountSpeakerClassifier,
    Layer4CandidatePair,
    Layer4LongAudioInput,
)
from layer4_speech_separation.offline import OfflineLayer4Pipeline
from layer4_speech_separation.models import _OfficialModelBackend
from gui.dev_test_ui.offline_l4_store import OfflineLayer4UiStore


def _source(counts: tuple[int, ...]) -> Layer4LongAudioInput:
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    waveform = np.ascontiguousarray(np.sin(2 * np.pi * 2_700 * time), dtype=np.float32)
    return Layer4LongAudioInput(
        "asset", hashlib.sha256(waveform.tobytes()).hexdigest(), "session", 0, 9, 120.0,
        0, 48_000, waveform,
        tuple((960 * (index + 1), count) for index, count in enumerate(counts)),
    )


class _Backend:
    model_id = "separator"
    model_revision = "1"
    sample_rate = 16_000
    source_count = 2

    def __init__(self) -> None:
        self.calls = 0

    def separate(self, request_id, waveform_16k):
        self.calls += 1
        other = np.ascontiguousarray(
            np.sin(2 * np.pi * 3_500 * np.arange(len(waveform_16k)) / 16_000),
            dtype=np.float32,
        )
        return Layer4CandidatePair(
            request_id, self.model_id, self.model_revision, 16_000,
            (other, np.ascontiguousarray(waveform_16k)),
        )


class _L5:
    def __init__(self):
        self.calls = 0

    def process_long_audio_20ms(self, item):
        self.calls += 1
        count = len(item.waveform) // 960
        probabilities = np.linspace(0.1, 0.9, count, dtype=np.float32)
        return SimpleNamespace(
            model_id="l5",
            threshold=0.7,
            probabilities_20ms=probabilities,
            is_voice_20ms=tuple(bool(value >= 0.7) for value in probabilities),
            summary_probability=0.8,
            summary_is_voice=True,
            metadata={"frame_shift_ms": 20},
        )


def test_direction_count_classifier_uses_maximum_recorded_l2_output_count() -> None:
    decision = DirectionCountSpeakerClassifier().classify(_source((1, 1, 2, 1)))
    assert decision.speaker_count == 2
    assert decision.confidence == 1.0
    assert decision.metadata["maximum_l2_direction_count"] == 2


def test_direction_count_classifier_caps_transient_extra_directions_at_two() -> None:
    decision = DirectionCountSpeakerClassifier().classify(_source((1, 3, 2, 3)))
    assert decision.speaker_count == 2
    assert decision.metadata["maximum_l2_direction_count"] == 3
    assert decision.metadata["effective_speaker_count"] == 2
    assert decision.metadata["aggregation"] == "min(2, maximum)"


def test_one_speaker_is_resampled_and_bypasses_separator_before_l5() -> None:
    backend = _Backend()
    pipeline = OfflineLayer4Pipeline(
        speaker_counter=DirectionCountSpeakerClassifier(),
        backends={"mossformer2_ss_16k": backend},
        layer5=_L5(),
        default_backend="mossformer2_ss_16k",
    )
    result = pipeline.process(_source((1, 1, 1)), request_id="one")
    assert result.path == "single_speaker_bypass"
    assert result.selected is None
    assert backend.calls == 0
    assert result.l5_is_voice


def test_two_speakers_are_separated_matched_and_keep_parent_identity() -> None:
    backend = _Backend()
    pipeline = OfflineLayer4Pipeline(
        speaker_counter=DirectionCountSpeakerClassifier(),
        backends={"mossformer2_ss_16k": backend},
        layer5=_L5(),
        default_backend="mossformer2_ss_16k",
    )
    result = pipeline.process(_source((1, 2, 2)), request_id="two")
    assert result.path == "two_speaker_separation"
    assert result.selected is not None
    assert result.selected.selected_source_index == 1
    assert result.selected.track_id == 9 and result.selected.theta_deg == 120.0
    assert backend.calls == 1


def test_l4_and_l5_can_only_run_as_two_explicit_ui_send_steps() -> None:
    backend = _Backend()
    layer5 = _L5()
    pipeline = OfflineLayer4Pipeline(
        speaker_counter=DirectionCountSpeakerClassifier(),
        backends={"mossformer2_ss_16k": backend},
        layer5=layer5,
        default_backend="mossformer2_ss_16k",
    )
    processed = pipeline.process_l4_sealed((_source((1, 2)),))
    assert len(processed) == 1
    assert processed[0].source.track_id == 9
    assert layer5.calls == 0
    results = pipeline.process_l5_sealed(processed)
    assert layer5.calls == 1
    assert results[0].l5_is_voice is True

    store = OfflineLayer4UiStore()
    try:
        store.set_processed(processed)
        assert store.audio_path(9).is_file()
        assert all(item is None for item in store.snapshots()[0].voice_annotations_20ms)
        store.apply_l5(results)
        annotations = store.snapshots()[0].voice_annotations_20ms
        assert annotations and all(item is not None for item in annotations)
        probabilities = tuple(item.probability for item in annotations if item is not None)
        assert probabilities[0] == pytest.approx(0.1)
        assert probabilities[-1] == pytest.approx(0.9)
        assert any(not item.is_voice for item in annotations if item is not None)
        assert any(item.is_voice for item in annotations if item is not None)
    finally:
        store.close()


def test_long_audio_adapter_repairs_swapped_chunk_outputs_before_crossfade() -> None:
    backend = _OfficialModelBackend.__new__(_OfficialModelBackend)
    backend.chunk_samples = 100
    backend.overlap_samples = 20
    calls = 0

    def forward(audio):
        nonlocal calls
        calls += 1
        pair = (audio.copy(), -audio.copy())
        return pair if calls % 2 else pair[::-1]

    backend._forward = forward
    audio = np.ascontiguousarray(
        np.sin(2 * np.pi * 0.037 * np.arange(260)), dtype=np.float32,
    )
    left, right = backend._chunked(audio)
    assert calls > 1
    assert np.allclose(left, audio, atol=1e-6)
    assert np.allclose(right, -audio, atol=1e-6)


def test_mossformer_adapter_restores_quiet_input_rms_and_pads_short_audio() -> None:
    backend = _OfficialModelBackend.__new__(_OfficialModelBackend)
    backend.device = torch.device("cpu")
    backend.minimum_input_samples = 16_000
    backend.restore_input_rms = True

    class FixedAmplitudeModel:
        def __call__(self, value):
            assert value.shape == (1, 16_000)
            return torch.stack((torch.ones_like(value), -torch.ones_like(value)))

    backend.model = FixedAmplitudeModel()
    audio = np.ascontiguousarray(
        0.003 * np.sin(2 * np.pi * 300 * np.arange(8_000) / 16_000),
        dtype=np.float32,
    )
    left, right = backend._forward(audio)
    expected_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert len(left) == len(right) == len(audio)
    assert np.sqrt(np.mean(left.astype(np.float64) ** 2)) == pytest.approx(expected_rms)
    assert np.sqrt(np.mean(right.astype(np.float64) ** 2)) == pytest.approx(expected_rms)


def test_l4_output_is_attenuated_before_pcm16_clipping() -> None:
    pipeline = OfflineLayer4Pipeline(
        speaker_counter=DirectionCountSpeakerClassifier(),
        backends={"mossformer2_ss_16k": _Backend()},
        layer5=_L5(),
        default_backend="mossformer2_ss_16k",
    )
    processed = pipeline.process_l4(_source((1, 1)), request_id="peak-safe")
    assert np.max(np.abs(processed.waveform_48k)) <= 32767.0 / 32768.0
    assert float(processed.metadata["pcm16_peak_safety_gain"]) < 1.0
