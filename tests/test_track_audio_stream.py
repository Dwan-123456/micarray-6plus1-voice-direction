from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample_poly

from gui.dev_test_ui.audio_id_tracker import AudioIdTracker
from layer4_voice_classifier import InputGainCompensationSettings, NvidiaMarbleNetPlugin
from track_audio_stream import TrackAudioStreamHub, TrackAudioWindow


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
