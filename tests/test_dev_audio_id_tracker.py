from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from common.data_types import TrackedDirection
from gui.dev_test_ui.audio_id_tracker import AudioIdTracker
from gui.dev_test_ui.contracts import BeamformPreview


def _direction(
    track_id: int,
    decision: int,
    theta: float,
    *,
    measured: float | None = None,
    state: str = "confirmed",
    observed: bool = True,
    new: bool = False,
) -> TrackedDirection:
    measured = theta if measured is None and observed else measured
    last = decision if observed else decision - 960
    return TrackedDirection(
        "session", 0, decision // 960, decision, decision - 1_920, decision,
        track_id, 1, measured, theta, 1.0, 0.8, state, observed, new,
        7_680, last, decision - last, False,
    )


def _preview(track_id: int, decision: int, theta: float, *, backend: str = "ds_baseline") -> BeamformPreview:
    absolute = np.arange(decision - 7_680, decision, dtype=np.float64)
    waveform = np.ascontiguousarray(
        0.1 * np.sin(2.0 * np.pi * 500.0 * absolute / 48_000.0),
        dtype=np.float32,
    )
    return BeamformPreview(
        "session", 0, decision // 960, decision, theta, waveform, backend,
        track_id=track_id, track_state="confirmed",
    )


def _silent_preview(track_id: int, decision: int, theta: float) -> BeamformPreview:
    return BeamformPreview(
        "session", 0, decision // 960, decision, theta,
        np.zeros(7_680, dtype=np.float32), "ds_baseline",
        track_id=track_id, track_state="confirmed",
    )


def _low_level_preview(track_id: int, decision: int, theta: float) -> BeamformPreview:
    absolute = np.arange(decision - 7_680, decision, dtype=np.float64)
    peak = np.sqrt(2.0) * 10.0 ** (-55.0 / 20.0)
    waveform = np.ascontiguousarray(
        peak * np.sin(2.0 * np.pi * 500.0 * absolute / 48_000.0),
        dtype=np.float32,
    )
    return BeamformPreview(
        "session", 0, decision // 960, decision, theta, waveform, "ds_baseline",
        track_id=track_id, track_state="confirmed",
    )


def _window(decision: int):
    return SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=decision)


def test_exact_authoritative_id_is_the_only_join_key_and_cross_zero_never_splits(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    for decision, theta in ((7_680, 359.0), (8_640, 0.0), (9_600, 1.0)):
        item = _direction(7, decision, theta)
        tracker.update(_window(decision), (item,), (_preview(7, decision, theta),), active_tracks=(item,))
    tracker.update(_window(18_240), (), (), active_tracks=())

    rows = tuple(item for item in tracker.snapshots() if item.track_id != 0)
    assert len(rows) == 1
    assert rows[0].track_id == 7
    assert rows[0].state == "ended"
    assert rows[0].audio_sample_count == 3 * 960


def test_nearby_different_authoritative_ids_are_never_merged(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    decision = 7_680
    left, right = _direction(2, decision, 10.0), _direction(3, decision, 11.0)
    tracker.update(
        _window(decision), (left, right),
        (_preview(2, decision, 10.0), _preview(3, decision, 11.0)),
        active_tracks=(left, right),
    )
    assert {item.track_id for item in tracker.snapshots()} == {2, 3}


def test_id_rollover_is_not_repaired_by_angle(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    first = _direction(2, 7_680, 206.0)
    tracker.update(_window(7_680), (first,), (_preview(2, 7_680, 206.0),), active_tracks=(first,))
    second = _direction(3, 8_640, 206.5)
    tracker.update(_window(8_640), (second,), (_preview(3, 8_640, 206.5),), active_tracks=(second,))
    rows = {item.track_id: item for item in tracker.snapshots()}
    assert rows[2].state == "ended"
    assert rows[3].state == "active"


def test_coasting_keeps_row_until_l2_deletes_track(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    observed = _direction(4, 7_680, 45.0)
    tracker.update(_window(7_680), (observed,), (_preview(4, 7_680, 45.0),), active_tracks=(observed,))
    coast = _direction(4, 8_640, 46.0, measured=None, state="coasting", observed=False)
    tracker.update(_window(8_640), (), (), active_tracks=(coast,))
    assert tracker.snapshots()[0].state == "coasting"
    tracker.update(_window(9_600), (), (), active_tracks=())
    assert tracker.snapshots()[0].state == "ended"


def test_coasting_bf_preview_appends_real_audio_to_same_authoritative_id(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    first = _direction(5, 7_680, 45.0)
    tracker.update(
        _window(7_680), (first,), (_preview(5, 7_680, 45.0),),
        active_tracks=(first,),
    )
    coast = _direction(
        5, 8_640, 46.0, measured=None, state="coasting", observed=False,
    )
    tracker.update(
        _window(8_640), (coast,), (_preview(5, 8_640, 46.0),),
        active_tracks=(coast,),
    )
    tracker.update(_window(9_600), (), (), active_tracks=())

    cache = tracker.audio_cache_path(5)
    assert cache is not None
    audio = np.memmap(cache, dtype=np.float32, mode="r")
    assert len(audio) == 2 * 960
    assert np.sqrt(np.mean(np.square(audio[960:], dtype=np.float64))) > 0.01


def test_skipped_results_recover_absolute_timeline_without_speedup(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    first = _direction(9, 7_680, 30.0)
    later = _direction(9, 11_520, 32.0)
    tracker.update(_window(7_680), (first,), (_preview(9, 7_680, 30.0),), active_tracks=(first,))
    tracker.update(_window(11_520), (later,), (_preview(9, 11_520, 32.0),), active_tracks=(later,))
    tracker.update(_window(12_480), (), (), active_tracks=())
    row = tracker.snapshots()[0]
    assert row.audio_sample_count == 5 * 960


def test_unrecoverable_gap_preserves_duration_and_uses_newest_full_160ms(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    first_decision = 7_680
    later_decision = first_decision + 20 * 960
    first = _direction(1, first_decision, 30.0)
    later = _direction(1, later_decision, 30.0)
    tracker.update(_window(first_decision), (first,), (_preview(1, first_decision, 30.0),), active_tracks=(first,))
    tracker.update(_window(later_decision), (later,), (_preview(1, later_decision, 30.0),), active_tracks=(later,))
    tracker.update(_window(later_decision + 960), (), (), active_tracks=())
    assert tracker.snapshots()[0].audio_sample_count == 21 * 960


def test_ended_track_at_or_below_thirty_percent_sound_is_deleted(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    for index in range(10):
        decision = 7_680 + index * 960
        direction = _direction(12, decision, 80.0)
        preview = (
            _preview(12, decision, 80.0)
            if index < 3 else _silent_preview(12, decision, 80.0)
        )
        tracker.update(_window(decision), (direction,), (preview,), active_tracks=(direction,))
    tracker.update(_window(24_960), (), (), active_tracks=())
    assert tracker.snapshots() == ()
    assert not tuple((tmp_path / "cache").rglob("track_012/segment_*.f32"))


def test_ended_track_above_thirty_percent_sound_is_retained(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    for index in range(10):
        decision = 7_680 + index * 960
        direction = _direction(13, decision, 90.0)
        preview = (
            _preview(13, decision, 90.0)
            if index < 4 else _silent_preview(13, decision, 90.0)
        )
        tracker.update(_window(decision), (direction,), (preview,), active_tracks=(direction,))
    tracker.update(_window(24_960), (), (), active_tracks=())
    assert tracker.snapshots()[0].track_id == 13


def test_minus_fifty_five_dbfs_l3_audio_counts_as_sound(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    for index in range(10):
        decision = 7_680 + index * 960
        direction = _direction(16, decision, 95.0)
        preview = (
            _low_level_preview(16, decision, 95.0)
            if index < 4 else _silent_preview(16, decision, 95.0)
        )
        tracker.update(_window(decision), (direction,), (preview,), active_tracks=(direction,))
    tracker.update(_window(24_960), (), (), active_tracks=())

    assert tracker.snapshots()[0].track_id == 16
    assert tracker.audio_cache_path(16) is not None


def test_playable_track_survives_silent_tail_while_replay_queue_drains(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    for index in range(250):
        decision = 7_680 + index * 960
        direction = _direction(14, decision, 100.0)
        preview = (
            _preview(14, decision, 100.0)
            if index < 100 else _silent_preview(14, decision, 100.0)
        )
        tracker.update(_window(decision), (direction,), (preview,), active_tracks=(direction,))
    tracker.update(_window(7_680 + 250 * 960), (), (), active_tracks=())
    rows = tracker.snapshots()
    assert len(rows) == 1
    assert rows[0].track_id == 14
    assert tracker.audio_cache_path(14) is not None


def test_coasting_timeline_silence_does_not_delete_playable_observed_audio(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    first_decision = 7_680
    observed = _direction(15, first_decision, 120.0)
    tracker.update(
        _window(first_decision),
        (observed,),
        (_preview(15, first_decision, 120.0),),
        active_tracks=(observed,),
    )
    for index in range(1, 201):
        decision = first_decision + index * 960
        coast = _direction(
            15, decision, 120.0,
            measured=None, state="coasting", observed=False,
        )
        tracker.update(_window(decision), (), (), active_tracks=(coast,))
    tracker.update(_window(first_decision + 201 * 960), (), (), active_tracks=())

    rows = tracker.snapshots()
    assert len(rows) == 1
    assert rows[0].track_id == 15
    assert rows[0].audio_sample_count == 201 * 960
    cache = tracker.audio_cache_path(15)
    assert cache is not None
    assert cache.stat().st_size == 201 * 960 * np.dtype(np.float32).itemsize


def test_mode_change_seals_and_isolates_cache_partitions(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    first = _direction(1, 7_680, 30.0)
    tracker.update(_window(7_680), (first,), (_preview(1, 7_680, 30.0),), active_tracks=(first,))
    tracker.seal_mode("constant_beamwidth_baseline")
    second = _direction(1, 8_640, 30.0)
    tracker.update(
        _window(8_640), (second,),
        (_preview(1, 8_640, 30.0, backend="constant_beamwidth_baseline"),),
        active_tracks=(second,),
    )
    tracker.update(_window(9_600), (), (), active_tracks=())
    paths = tuple((tmp_path / "cache").rglob("track_001/segment_*.f32"))
    assert len(paths) == 2
    assert len({path.parent.parent.name for path in paths}) == 2


def test_center_reference_is_full_capture_and_deleted_only_on_close(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path, segment_seconds=0.02, retained_segments=1)
    for index in range(5):
        samples = np.zeros((960, 8), dtype=np.float32)
        samples[:, 6] = index + 1
        tracker.append_center_reference(SimpleNamespace(
            session_id="session", stream_epoch=0, end_sample=(index + 1) * 960, samples=samples,
        ))
    assert tracker.snapshots()[0].audio_sample_count == 5 * 960
    assert len(tuple((tmp_path / "cache/track_000").glob("segment_*.f32"))) == 5
    tracker.close(delete_files=True)
    assert not (tmp_path / "cache").exists()
