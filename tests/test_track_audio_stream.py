from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample_poly

from gui.dev_test_ui.audio_id_tracker import AudioIdTracker
from layer5_voice_classifier import (
    InputGainCompensationSettings,
    Layer5AudioSegment,
    Layer5Result,
    ModelPrediction,
    NvidiaMarbleNetPlugin,
    VoiceDetection,
)
from track_audio_stream import TrackAudioStreamHub, TrackAudioWindow, TrackVoiceAnnotation
from track_audio_stream.service import _CROSSFADE_SAMPLES


def _window(decision: int, track_id: int = 7, *, level: float = 1.0e-3):
    absolute = np.arange(decision - 3_840, decision, dtype=np.float64)
    waveform = np.ascontiguousarray(
        level * np.sin(2.0 * np.pi * 500.0 * absolute / 48_000.0), np.float32
    )
    return TrackAudioWindow(
        "session", 0, decision // 960, decision, track_id, 30.0,
        waveform, (0.9,) * 4, "optimized",
    )


def _identity(decision: int):
    return ("session", 0, decision // 960, decision)


def _active(track_id: int = 7):
    return SimpleNamespace(
        track_id=track_id,
        track_state="confirmed",
        theta_deg=30.0,
        normalized_score=0.8,
    )


def test_hub_appends_one_aligned_compensated_hop_per_id_and_grows_context():
    hub = TrackAudioStreamHub(InputGainCompensationSettings(), context_ms=160)
    first = hub.process((_window(7_680),), active_track_ids=(7,), identity=_identity(7_680))
    second = hub.process((_window(8_640),), active_track_ids=(7,), identity=_identity(8_640))

    assert len(first.emitted_hops) == 1
    assert first.emitted_hops[0].start_sample == 5_760
    assert first.emitted_hops[0].end_sample == 6_720
    assert len(second.continuous_audio[0].waveform) == 1_920
    assert second.continuous_audio[0].effective_end_sample == 7_680
    assert second.continuous_audio[0].gain_diagnostic.enabled is True


def test_hub_seals_complete_long_audio_with_aligned_l2_direction_counts() -> None:
    hub = TrackAudioStreamHub(InputGainCompensationSettings(enabled=False), context_ms=60)
    for index, decision in enumerate((7_680, 8_640, 9_600, 10_560)):
        hub.process(
            (_window(decision),), active_track_ids=(7,), identity=_identity(decision),
            l2_direction_count=1 if index < 2 else 2,
        )
    sealed = hub.seal()
    assert len(sealed) == 1
    assert len(sealed[0].waveform) == 4 * 960
    assert sealed[0].start_sample == 5_760
    assert sealed[0].l2_direction_counts == (
        (6_720, 1), (7_680, 1), (8_640, 2), (9_600, 2),
    )
    assert not sealed[0].waveform.flags.writeable


def test_adjacent_windows_crossfade_the_future_overlap_without_a_20ms_seam():
    def offset_window(decision: int, offset: float) -> TrackAudioWindow:
        absolute = np.arange(decision - 3_840, decision, dtype=np.float64)
        waveform = np.ascontiguousarray(
            0.1 * np.sin(2.0 * np.pi * 523.0 * absolute / 48_000.0) + offset,
            np.float32,
        )
        return TrackAudioWindow(
            "session", 0, decision // 960, decision, 7, 30.0,
            waveform, (0.1,) * 4, "optimized",
        )

    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=160,
    )
    first_window = offset_window(7_680, 0.0)
    second_window = offset_window(8_640, 0.04)
    first = hub.process(
        (first_window,), active_track_ids=(7,), identity=_identity(7_680),
    ).emitted_hops[0].waveform
    second = hub.process(
        (second_window,), active_track_ids=(7,), identity=_identity(8_640),
    ).emitted_hops[0].waveform

    previous_future = first_window.waveform[-960:]
    current = second_window.waveform[-1_920:-960]
    phase = np.linspace(0.0, np.pi / 2.0, _CROSSFADE_SAMPLES, dtype=np.float32)
    expected_prefix = (
        previous_future[:_CROSSFADE_SAMPLES] * np.cos(phase) ** 2
        + current[:_CROSSFADE_SAMPLES] * np.sin(phase) ** 2
    )
    np.testing.assert_allclose(second[:_CROSSFADE_SAMPLES], expected_prefix, atol=1e-7)
    np.testing.assert_array_equal(second[_CROSSFADE_SAMPLES:], current[_CROSSFADE_SAMPLES:])
    # The seam now follows two adjacent samples from the same older L3
    # estimate instead of jumping directly to the newer window's DC/weights.
    assert second[0] - first[-1] == pytest.approx(
        previous_future[0] - first[-1], abs=1e-7,
    )
    assert abs(second[0] - first[-1]) < abs(current[0] - first[-1])


def test_gain_switch_is_realtime_and_does_not_reset_the_track_context():
    hub = TrackAudioStreamHub(InputGainCompensationSettings(), context_ms=160)
    first = hub.process((_window(7_680),), active_track_ids=(7,), identity=_identity(7_680))
    assert hub.set_gain_compensation_enabled(False) is False
    second = hub.process((_window(8_640),), active_track_ids=(7,), identity=_identity(8_640))
    assert len(second.continuous_audio[0].waveform) == 1_920
    assert second.continuous_audio[0].gain_diagnostic.enabled is False
    np.testing.assert_array_equal(
        second.continuous_audio[0].waveform[:960], first.emitted_hops[0].waveform
    )
    assert second.emitted_hops[0].waveform is not None


def test_test_ui_caches_exact_same_compensated_samples_as_cnn(tmp_path):
    hub = TrackAudioStreamHub(InputGainCompensationSettings(), context_ms=160)
    batch = hub.process((_window(7_680),), active_track_ids=(7,), identity=_identity(7_680))
    tracker = AudioIdTracker("cache", project_root=tmp_path, downstream_window_samples=3_840)
    tracker.consume_stream_batch(batch, active_tracks=(_active(),))
    path = tracker.audio_cache_path(7)
    assert path is not None
    playback = np.asarray(np.memmap(path, dtype=np.float32, mode="r"))
    np.testing.assert_array_equal(playback, batch.continuous_audio[0].waveform)


def test_l5_annotation_is_attached_to_the_exact_cached_20ms_id_hop(tmp_path):
    hub = TrackAudioStreamHub(InputGainCompensationSettings(), context_ms=160)
    batch = hub.process((_window(7_680),), active_track_ids=(7,), identity=_identity(7_680))
    tracker = AudioIdTracker("cache", project_root=tmp_path, downstream_window_samples=3_840)
    tracker.consume_stream_batch(batch, active_tracks=(_active(),))
    hop = batch.emitted_hops[0]
    annotation = TrackVoiceAnnotation(
        "session", 0, 8, 7_680, 7, hop.start_sample, hop.end_sample,
        0.7, True, "nv", 0.7,
    )

    snapshots = tracker.apply_l5_annotations((annotation,))

    assert snapshots[0].voice_annotations_20ms == (annotation,)
    with pytest.raises(ValueError, match="no matching cached audio ID"):
        tracker.apply_l5_annotations((
            TrackVoiceAnnotation(
                "session", 0, 8, 7_680, 9, hop.start_sample, hop.end_sample,
                0.2, False, "nv", 0.7,
            ),
        ))


def test_runtime_l5_annotation_uses_latest_continuous_hop_and_strict_identity():
    from app.runtime import ApplicationRuntime

    waveform = np.zeros(1_920, np.float32)
    audio = Layer5AudioSegment(
        "session", 0, 8, 7_680, 30.0, 48_000, waveform,
        (0.5, 0.9), 7, 5_760, 7_680,
    )
    detection = VoiceDetection(
        "session", 0, 8, 7_680, 30.0, 0.7, True, "nv", 7,
    )
    prediction = ModelPrediction("nv", np.asarray([0.7], np.float32), 1.0, {})
    result = Layer5Result((detection,), (prediction,), "nv", 0.7)

    annotations = ApplicationRuntime._track_voice_annotations((audio,), result)

    assert (annotations[0].start_sample, annotations[0].end_sample) == (6_720, 7_680)
    assert annotations[0].is_voice is True
    wrong_id = Layer5Result(
        (VoiceDetection("session", 0, 8, 7_680, 30.0, 0.7, True, "nv", 8),),
        (prediction,), "nv", 0.7,
    )
    with pytest.raises(ValueError, match="identity/order/theta mismatch"):
        ApplicationRuntime._track_voice_annotations((audio,), wrong_id)


def test_hub_keeps_ids_separate_and_rejects_duplicate_or_stale_windows():
    hub = TrackAudioStreamHub(InputGainCompensationSettings(), context_ms=160)
    batch = hub.process(
        (_window(7_680, 2), _window(7_680, 3)),
        active_track_ids=(2, 3), identity=_identity(7_680),
    )
    assert tuple(item.track_id for item in batch.continuous_audio) == (2, 3)
    with pytest.raises(ValueError, match="unique"):
        hub.process(
            (_window(8_640, 2), _window(8_640, 2)),
            active_track_ids=(2,), identity=_identity(8_640),
        )
    with pytest.raises(ValueError, match="strictly ordered"):
        hub.process((_window(7_680, 2),), active_track_ids=(2,), identity=_identity(7_680))


def test_unrecoverable_gap_preserves_listening_time_but_resets_cnn_context():
    hub = TrackAudioStreamHub(InputGainCompensationSettings(), context_ms=160)
    hub.process((_window(7_680),), active_track_ids=(7,), identity=_identity(7_680))
    later = 7_680 + 10 * 960
    batch = hub.process((_window(later),), active_track_ids=(7,), identity=_identity(later))
    assert any(not item.observed for item in batch.emitted_hops)
    assert len(batch.continuous_audio[0].waveform) <= 4 * 960


def test_real_20ms_library_audio_drives_the_same_test_ui_cache_and_nvidia_input(tmp_path):
    project = Path(__file__).parents[1]
    model_dir = project / "models" / "nv_marblenet_baseline_v1"
    sample_rate, source = wavfile.read(model_dir / "source" / "smoke_speech.wav")
    source = source.astype(np.float32) / 32768.0
    nonzero = int(np.argmax(np.abs(source) > 1.0e-3))
    start = max(0, nonzero - nonzero % 320)
    hop_16k = source[start:start + 320]
    assert len(hop_16k) == 320
    hop_48k = np.ascontiguousarray(resample_poly(hop_16k, 3, 1), np.float32)

    waveform = np.zeros(3_840, np.float32)
    waveform[1_920:2_880] = hop_48k
    window = TrackAudioWindow(
        "session", 0, 8, 7_680, 7, 30.0, waveform,
        (0.1, 0.1, 0.9, 0.1), "optimized",
    )
    hub = TrackAudioStreamHub(InputGainCompensationSettings(), context_ms=3_200)
    batch = hub.process((window,), active_track_ids=(7,), identity=("session", 0, 8, 7_680))

    tracker = AudioIdTracker("cache", project_root=tmp_path, downstream_window_samples=3_840)
    tracker.consume_stream_batch(batch, active_tracks=(_active(),))
    playback = np.asarray(np.memmap(tracker.audio_cache_path(7), dtype=np.float32, mode="r"))
    cnn_audio = batch.continuous_audio[0].waveform
    np.testing.assert_array_equal(playback, cnn_audio)

    prediction = NvidiaMarbleNetPlugin("nv", model_dir, device="cpu").predict(
        cnn_audio[None, :]
    )
    assert prediction.probabilities.shape == (1,)
    assert np.isfinite(prediction.probabilities[0])
