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
        15_360, last, decision - last, False,
    )


def _preview(track_id: int, decision: int, theta: float, *, backend: str = "ds_baseline") -> BeamformPreview:
    absolute = np.arange(decision - 15_360, decision, dtype=np.float64)
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
        np.zeros(15_360, dtype=np.float32), "ds_baseline",
        track_id=track_id, track_state="confirmed",
    )


def _window(decision: int):
    return SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=decision)


def test_exact_authoritative_id_is_the_only_join_key_and_cross_zero_never_splits(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    for decision, theta in ((15_360, 359.0), (16_320, 0.0), (17_280, 1.0)):
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
    decision = 15_360
    left, right = _direction(2, decision, 10.0), _direction(3, decision, 11.0)
    tracker.update(
        _window(decision), (left, right),
        (_preview(2, decision, 10.0), _preview(3, decision, 11.0)),
        active_tracks=(left, right),
    )
    assert {item.track_id for item in tracker.snapshots()} == {2, 3}


def test_id_rollover_is_not_repaired_by_angle(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    first = _direction(2, 15_360, 206.0)
    tracker.update(_window(15_360), (first,), (_preview(2, 15_360, 206.0),), active_tracks=(first,))
    second = _direction(3, 16_320, 206.5)
    tracker.update(_window(16_320), (second,), (_preview(3, 16_320, 206.5),), active_tracks=(second,))
    rows = {item.track_id: item for item in tracker.snapshots()}
    assert rows[2].state == "ended"
    assert rows[3].state == "active"


def test_coasting_keeps_row_until_l2_deletes_track(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    observed = _direction(4, 15_360, 45.0)
    tracker.update(_window(15_360), (observed,), (_preview(4, 15_360, 45.0),), active_tracks=(observed,))
    coast = _direction(4, 16_320, 46.0, measured=None, state="coasting", observed=False)
    tracker.update(_window(16_320), (), (), active_tracks=(coast,))
    assert tracker.snapshots()[0].state == "coasting"
    tracker.update(_window(17_280), (), (), active_tracks=())
    assert tracker.snapshots()[0].state == "ended"


def test_skipped_results_recover_absolute_timeline_without_speedup(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    first = _direction(9, 15_360, 30.0)
    later = _direction(9, 19_200, 32.0)
    tracker.update(_window(15_360), (first,), (_preview(9, 15_360, 30.0),), active_tracks=(first,))
    tracker.update(_window(19_200), (later,), (_preview(9, 19_200, 32.0),), active_tracks=(later,))
    tracker.update(_window(20_160), (), (), active_tracks=())
    row = tracker.snapshots()[0]
    assert row.audio_sample_count == 5 * 960


def test_unrecoverable_gap_preserves_duration_and_uses_newest_full_320ms(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    first_decision = 15_360
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
        decision = 15_360 + index * 960
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
        decision = 15_360 + index * 960
        direction = _direction(13, decision, 90.0)
        preview = (
            _preview(13, decision, 90.0)
            if index < 4 else _silent_preview(13, decision, 90.0)
        )
        tracker.update(_window(decision), (direction,), (preview,), active_tracks=(direction,))
    tracker.update(_window(24_960), (), (), active_tracks=())
    assert tracker.snapshots()[0].track_id == 13


def test_ended_track_with_three_seconds_continuous_silence_is_deleted(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    for index in range(250):
        decision = 15_360 + index * 960
        direction = _direction(14, decision, 100.0)
        preview = (
            _preview(14, decision, 100.0)
            if index < 100 else _silent_preview(14, decision, 100.0)
        )
        tracker.update(_window(decision), (direction,), (preview,), active_tracks=(direction,))
    tracker.update(_window(15_360 + 250 * 960), (), (), active_tracks=())
    assert tracker.snapshots() == ()


def test_mode_change_seals_and_isolates_cache_partitions(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    first = _direction(1, 15_360, 30.0)
    tracker.update(_window(15_360), (first,), (_preview(1, 15_360, 30.0),), active_tracks=(first,))
    tracker.seal_mode("constant_beamwidth_baseline")
    second = _direction(1, 16_320, 30.0)
    tracker.update(
        _window(16_320), (second,),
        (_preview(1, 16_320, 30.0, backend="constant_beamwidth_baseline"),),
        active_tracks=(second,),
    )
    tracker.update(_window(17_280), (), (), active_tracks=())
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
