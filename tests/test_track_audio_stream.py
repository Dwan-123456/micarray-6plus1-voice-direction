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
from track_audio_stream.service import _ArchivedHop, _CROSSFADE_SAMPLES


def _window(
    decision: int,
    track_id: int = 7,
    *,
    level: float = 1.0e-3,
    processing_mode: str = "optimized",
):
    absolute = np.arange(decision - 3_840, decision, dtype=np.float64)
    waveform = np.ascontiguousarray(
        level * np.sin(2.0 * np.pi * 500.0 * absolute / 48_000.0), np.float32
    )
    return TrackAudioWindow(
        "session", 0, decision // 960, decision, track_id, 30.0,
        waveform, (0.9,) * 4, processing_mode,
    )


def _identity(decision: int):
    return ("session", 0, decision // 960, decision)


def _active(track_id: int = 7, *, state: str = "confirmed"):
    return SimpleNamespace(
        track_id=track_id,
        track_state=state,
        theta_deg=30.0,
        normalized_score=0.8,
    )


def _observe_l2(
    hub: TrackAudioStreamHub,
    decisions: tuple[int, ...],
    *,
    track_id: int = 7,
    processing_mode: str = "optimized",
) -> None:
    for decision in decisions:
        hub.observe_l2(
            identity=_identity(decision),
            active_tracks=(_active(track_id),),
            processing_mode=processing_mode,
            l2_direction_count=1,
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


def test_tentative_l3_audio_is_hidden_then_backfilled_when_same_id_confirms(tmp_path) -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=160,
    )
    tracker = AudioIdTracker(
        "cache", project_root=tmp_path, downstream_window_samples=3_840,
        minimum_listening_track_seconds=2.0,
    )
    tentative = _active(state="tentative")
    confirmed = _active(state="confirmed")

    hub.observe_l2(
        identity=_identity(7_680), active_tracks=(tentative,),
        processing_mode="optimized", l2_direction_count=1,
    )
    first = hub.process(
        (_window(7_680),), active_track_ids=(7,), identity=_identity(7_680),
    )
    assert tracker.consume_stream_batch(first, active_tracks=(tentative,)) == ()

    hub.observe_l2(
        identity=_identity(8_640), active_tracks=(confirmed,),
        processing_mode="optimized", l2_direction_count=1,
    )
    second = hub.process(
        (_window(8_640),), active_track_ids=(7,), identity=_identity(8_640),
    )
    rows = tracker.consume_stream_batch(second, active_tracks=(confirmed,))

    assert len(rows) == 1
    assert rows[0].track_id == 7
    assert rows[0].audio_sample_count == 2 * 960
    playback = np.asarray(np.memmap(tracker.audio_cache_path(7), dtype=np.float32, mode="r"))
    np.testing.assert_array_equal(playback, second.continuous_audio[0].waveform)
    del playback

    # Confirmation makes the opening available for this live pass, but the
    # existing post-stitch two-second rule still removes the short final row
    # and its exact Hub identity before offline L4/L5 submission.
    assert tracker.finalize_capture() == ()
    assert hub.seal(allowed_track_keys=set()) == ()


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


def test_l2_authoritative_timeline_fixes_length_independent_of_bf_throughput() -> None:
    decisions = tuple(7_680 + index * 960 for index in range(8))
    fast = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=160,
    )
    slow = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=160,
    )
    _observe_l2(fast, decisions)
    _observe_l2(slow, decisions)

    for decision in decisions:
        fast.process(
            (_window(decision),), active_track_ids=(7,), identity=_identity(decision),
        )
    for decision in decisions[2:6:3]:
        slow.process(
            (_window(decision),), active_track_ids=(7,), identity=_identity(decision),
        )

    terminal = slow.finalize_missing_hops()
    fast_sealed = fast.seal()[0]
    slow_sealed = slow.seal()[0]

    assert terminal[-1].end_sample == decisions[-1] - 960
    assert len(fast_sealed.waveform) == len(slow_sealed.waveform) == len(decisions) * 960
    assert fast_sealed.start_sample == slow_sealed.start_sample == decisions[0] - 1_920
    assert fast_sealed.end_sample == slow_sealed.end_sample == decisions[-1] - 960
    assert np.count_nonzero(slow_sealed.waveform == 0.0) > 0


def test_test_ui_terminal_hops_extend_playback_to_l2_authoritative_end(tmp_path) -> None:
    decisions = tuple(7_680 + index * 960 for index in range(6))
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=160,
    )
    _observe_l2(hub, decisions)
    batch = hub.process(
        (_window(decisions[1], level=0.01),),
        active_track_ids=(7,),
        identity=_identity(decisions[1]),
    )
    tracker = AudioIdTracker(
        "cache", project_root=tmp_path, downstream_window_samples=3_840,
    )
    tracker.consume_stream_batch(batch, active_tracks=(_active(),))

    tracker.append_terminal_hops(hub.finalize_missing_hops())
    rows = tracker.finalize_capture()

    assert rows[0].audio_sample_count == len(decisions) * 960
    playback = np.asarray(np.memmap(tracker.audio_cache_path(7), dtype=np.float32, mode="r"))
    assert len(playback) == len(decisions) * 960
    np.testing.assert_array_equal(playback[-2 * 960:], 0.0)


def test_hub_seals_discontinuous_runs_as_one_unique_id_with_silent_gap() -> None:
    hub = TrackAudioStreamHub(InputGainCompensationSettings(enabled=False), context_ms=60)
    hub._archive[("session", 0, 7)] = [
        _ArchivedHop(0, 960, 30.0, 1, np.ones(960, np.float32)),
        _ArchivedHop(2_880, 3_840, 32.0, 2, np.full(960, 2.0, np.float32)),
    ]

    sealed = hub.seal()

    assert len(sealed) == 1
    assert sealed[0].track_id == 7
    assert sealed[0].start_sample == 0
    assert sealed[0].end_sample == 3_840
    np.testing.assert_array_equal(sealed[0].waveform[:960], 1.0)
    np.testing.assert_array_equal(sealed[0].waveform[960:2_880], 0.0)
    np.testing.assert_array_equal(sealed[0].waveform[2_880:], 2.0)
    assert sealed[0].l2_direction_counts == (
        (960, 1), (1_920, 0), (2_880, 0), (3_840, 2),
    )


def test_hub_purges_sub_two_second_track_before_offline_l4() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False),
        context_ms=60,
        minimum_output_seconds=2.0,
    )
    short_key = ("session", 0, 2)
    long_key = ("session", 0, 3)
    hub._archive[short_key] = [
        _ArchivedHop(
            index * 960, (index + 1) * 960, 20.0, 1,
            np.ones(960, np.float32),
        )
        for index in range(36)
    ]
    hub._archive[long_key] = [
        _ArchivedHop(
            index * 960, (index + 1) * 960, 30.0, 1,
            np.ones(960, np.float32),
        )
        for index in range(100)
    ]

    sealed = hub.seal()

    assert tuple(item.track_id for item in sealed) == (3,)
    assert short_key not in hub._archive
    assert long_key in hub._archive


def test_hub_never_seals_unconfirmed_tentative_id_even_over_two_seconds() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=160,
        minimum_output_seconds=2.0,
    )
    tentative = _active(state="tentative")
    for decision in range(7_680, 7_680 + 101 * 960, 960):
        hub.observe_l2(
            identity=_identity(decision), active_tracks=(tentative,),
            processing_mode="optimized", l2_direction_count=1,
        )
        hub.process(
            (_window(decision),), active_track_ids=(7,), identity=_identity(decision),
        )

    assert hub.seal() == ()
    assert ("session", 0, 7) not in hub._archive


def test_hub_purges_every_track_not_retained_by_the_test_ui() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=60,
    )
    visible_key = ("session", 0, 3)
    hidden_key = ("session", 0, 4)
    for key in (visible_key, hidden_key):
        hub._archive[key] = [
            _ArchivedHop(
                index * 960, (index + 1) * 960, 30.0, 1,
                np.ones(960, np.float32),
            )
            for index in range(100)
        ]

    sealed = hub.seal(allowed_track_keys={visible_key})

    assert tuple(item.track_id for item in sealed) == (3,)
    assert visible_key in hub._archive
    assert hidden_key not in hub._archive


def test_empty_test_ui_allowlist_purges_all_offline_l4_audio() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=60,
    )
    key = ("session", 0, 3)
    hub._archive[key] = [
        _ArchivedHop(0, 960, 30.0, 1, np.ones(960, np.float32)),
    ]

    assert hub.seal(allowed_track_keys=set()) == ()
    assert hub._archive == {}


def test_hub_mode_change_removes_the_hidden_old_mode_audio() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=60,
    )
    hub.process(
        (_window(7_680, level=0.01),),
        active_track_ids=(7,), identity=_identity(7_680),
    )
    hub.process(
        (_window(8_640, level=0.02, processing_mode="ds_baseline"),),
        active_track_ids=(7,), identity=_identity(8_640),
    )

    sealed = hub.seal(allowed_track_keys={("session", 0, 7)})

    assert len(sealed) == 1
    assert len(sealed[0].waveform) == 960


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
