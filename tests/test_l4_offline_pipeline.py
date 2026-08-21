from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

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

    def process(self, inputs):
        self.calls += 1
        item = inputs[0]
        return SimpleNamespace(detections=(SimpleNamespace(
            probability=0.8, is_voice=True, model_id="l5", track_id=item.track_id,
        ),))


def test_direction_count_classifier_uses_maximum_recorded_l2_output_count() -> None:
    decision = DirectionCountSpeakerClassifier().classify(_source((1, 1, 2, 1)))
    assert decision.speaker_count == 2
    assert decision.confidence == 1.0
    assert decision.metadata["maximum_l2_direction_count"] == 2


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
        assert annotations and all(item is not None and item.is_voice for item in annotations)
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
