from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gui.dev_test_ui.audio_id_tracker import AudioIdTracker
from gui.dev_test_ui.contracts import BeamformPreview


def _preview(*, backend: str, decision_sample: int, epoch: int = 0) -> BeamformPreview:
    return BeamformPreview(
        "session",
        epoch,
        decision_sample // 960,
        decision_sample,
        30.0,
        np.ones(15_360, np.float32),
        backend,
    )


def test_gate_recovery_epoch_keeps_cached_recordings_and_uses_new_ui_id(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    candidate = SimpleNamespace(theta_deg=30.0, normalized_score=0.8)

    for decision in (15_360, 16_320):
        tracker.update(
            SimpleNamespace(
                session_id="session", stream_epoch=0, decision_sample=decision,
            ),
            (candidate,),
            (_preview(backend="ds_baseline", decision_sample=decision),),
            track_ids=(1,),
            formal_flags=(True,),
        )

    old_path = tmp_path / "cache/track_001/segment_000000.f32"
    assert old_path.is_file()
    old_size = old_path.stat().st_size

    # Gate unavailable in the same epoch is only a no-candidate update.  It
    # must not remove either the row or its playable audio file.
    unavailable = tracker.update(
        SimpleNamespace(
            session_id="session", stream_epoch=0, decision_sample=17_280,
        ),
        (),
        (),
    )
    assert tuple(item.track_id for item in unavailable) == (1,)
    assert old_path.is_file()
    assert old_path.stat().st_size >= old_size

    # Gate/IMCRA recovery may begin a new stream epoch.  Preserve the prior
    # recording as an ended row and rebind its display identity to the current
    # epoch so the immutable DevUiFrame remains stream-consistent.
    center = np.zeros((960, 8), dtype=np.float32)
    tracker.append_center_reference(SimpleNamespace(
        session_id="session", stream_epoch=1, end_sample=960, samples=center,
    ))
    archived = tracker.snapshots()
    assert tuple(item.track_id for item in archived) == (0, 1)
    assert archived[1].state == "ended"
    assert all(item.stream_epoch == 1 for item in archived)
    assert old_path.is_file()

    # L2 private IDs restart at an epoch boundary.  A repeated formal ID gets
    # a new session-unique listening ID instead of overwriting the archive.
    tracker.update(
        SimpleNamespace(
            session_id="session", stream_epoch=1, decision_sample=15_360,
        ),
        (candidate,),
        (_preview(backend="ds_baseline", decision_sample=15_360, epoch=1),),
        track_ids=(1,),
        formal_flags=(True,),
    )
    current = tracker.snapshots()
    assert tuple(item.track_id for item in current) == (0, 1, 2)
    assert current[1].state == "ended"
    assert current[2].state == "active"
    assert old_path.is_file()
    tracker.close()


def test_switching_between_optimized_and_ds_resets_the_listening_cache(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    candidate = SimpleNamespace(theta_deg=30.0, normalized_score=0.8)
    first_window = SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=15_360)
    second_window = SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=16_320)

    first = tracker.update(
        first_window,
        (candidate,),
        (_preview(backend="imcra_spatial_separation", decision_sample=15_360),),
    )
    second = tracker.update(
        second_window,
        (candidate,),
        (_preview(backend="ds_baseline", decision_sample=16_320),),
    )
    third_window = SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=17_280)
    third = tracker.update(
        third_window,
        (candidate,),
        (_preview(backend="ds_baseline", decision_sample=17_280),),
    )

    assert len(first) == len(second) == len(third) == 1
    # Continuous overlap stitching intentionally waits for the next preview.
    assert first[0].audio_sample_count == second[0].audio_sample_count == 0
    assert third[0].audio_sample_count == 960
    assert second[0].track_id == 1
    tracker.close()


def test_overlap_stitch_uses_next_preview_for_a_continuous_boundary(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    candidate = SimpleNamespace(theta_deg=30.0, normalized_score=0.8)

    def preview(decision_sample: int, scale: float, offset: float) -> BeamformPreview:
        absolute = np.arange(decision_sample - 15_360, decision_sample, dtype=np.float64)
        waveform = np.ascontiguousarray(
            scale * np.sin(2 * np.pi * 1_000 * absolute / 48_000) + offset,
            dtype=np.float32,
        )
        return BeamformPreview(
            "session", 0, decision_sample // 960, decision_sample,
            30.0, waveform, "ds_baseline",
        )

    for decision, scale, offset in (
        (15_360, 1.0, 0.10),
        (16_320, 0.6, -0.15),
        (17_280, 1.4, 0.20),
    ):
        tracker.update(
            SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=decision),
            (candidate,),
            (preview(decision, scale, offset),),
        )

    cached = np.fromfile(tmp_path / "cache/track_001/segment_000000.f32", np.float32)
    assert cached.shape == (1_920,)
    seam = abs(float(cached[960] - cached[959]))
    local_differences = np.abs(np.diff(cached[940:980]))
    assert seam <= float(np.percentile(local_differences, 95)) * 1.05
    tracker.close()


def test_skipped_l3_results_preserve_absolute_duration_and_formal_id(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)

    def item(decision_sample: int, theta: float):
        absolute = np.arange(decision_sample - 15_360, decision_sample, dtype=np.float64)
        waveform = np.ascontiguousarray(
            0.1 * np.sin(2 * np.pi * 500 * absolute / 48_000), dtype=np.float32,
        )
        candidate = SimpleNamespace(theta_deg=theta, normalized_score=0.8)
        preview = BeamformPreview(
            "session", 0, decision_sample // 960, decision_sample,
            theta, waveform, "ds_baseline",
        )
        window = SimpleNamespace(
            session_id="session", stream_epoch=0, decision_sample=decision_sample,
        )
        return window, candidate, preview

    first = item(15_360, 10.0)
    second = item(19_200, 170.0)  # Four hops later and outside the old 30-degree gate.
    tracker.update(first[0], (first[1],), (first[2],), track_ids=(7,))
    snapshots = tracker.update(second[0], (second[1],), (second[2],), track_ids=(7,))

    cached = np.fromfile(tmp_path / "cache/track_007/segment_000000.f32", np.float32)
    assert cached.shape == (3_840,)
    assert np.count_nonzero(cached) > 3_000
    assert len(snapshots) == 1
    assert snapshots[0].track_id == 7
    assert snapshots[0].audio_sample_count == 3_840
    tracker.close()


def test_unrecoverable_gap_uses_full_320ms_then_silence_without_time_compression(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    candidate = SimpleNamespace(theta_deg=30.0, normalized_score=0.8)
    first_decision = 15_360
    second_decision = first_decision + 20 * 960
    tracker.update(
        SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=first_decision),
        (candidate,), (_preview(backend="ds_baseline", decision_sample=first_decision),),
        track_ids=(1,),
    )
    snapshots = tracker.update(
        SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=second_decision),
        (candidate,), (_preview(backend="ds_baseline", decision_sample=second_decision),),
        track_ids=(1,),
    )

    cached = np.fromfile(tmp_path / "cache/track_001/segment_000000.f32", np.float32)
    assert cached.shape == (20 * 960,)
    assert np.count_nonzero(cached[960:3_840]) == 0
    assert np.all(cached[-15_120:] == 1.0)
    assert snapshots[0].audio_sample_count == 20 * 960
    tracker.close()


def test_center_microphone_reference_is_first_snapshot_and_uses_logical_channel_six(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    samples = np.zeros((960, 8), dtype=np.float32)
    samples[:, 6] = np.linspace(-0.25, 0.25, 960, dtype=np.float32)
    block = SimpleNamespace(
        session_id="session", stream_epoch=0, end_sample=960, samples=samples,
    )

    tracker.append_center_reference(block)
    snapshots = tracker.snapshots()
    cache_path = tracker.audio_cache_path(0)

    assert snapshots[0].track_id == 0
    assert snapshots[0].audio_sample_count == 960
    assert cache_path is not None
    assert np.array_equal(np.fromfile(cache_path, np.float32), samples[:, 6])
    tracker.close()
    assert not (tmp_path / "cache").exists()


def test_center_reference_keeps_every_segment_until_ui_close(tmp_path):
    tracker = AudioIdTracker(
        "cache", project_root=tmp_path,
        segment_seconds=0.02, retained_segments=3,
    )
    for index in range(5):
        samples = np.zeros((960, 8), dtype=np.float32)
        samples[:, 6] = index + 1
        tracker.append_center_reference(SimpleNamespace(
            session_id="session", stream_epoch=0,
            end_sample=(index + 1) * 960, samples=samples,
        ))

    snapshots = tracker.snapshots()
    cache_path = tracker.audio_cache_path(0)
    assert len(list((tmp_path / "cache/track_000").glob("segment_*.f32"))) == 5
    assert snapshots[0].audio_sample_count == 5 * 960
    assert cache_path is not None
    assert np.fromfile(cache_path, np.float32).shape == (5 * 960,)
    tracker.close()
    assert not (tmp_path / "cache").exists()


def test_l2_id_rollover_continues_one_unambiguous_listening_cache(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    first_candidate = SimpleNamespace(theta_deg=206.0, normalized_score=0.4)
    second_candidate = SimpleNamespace(theta_deg=208.5, normalized_score=0.7)
    first_decision, second_decision = 15_360, 16_320

    tracker.update(
        SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=first_decision),
        (first_candidate,),
        (BeamformPreview(
            "session", 0, 16, first_decision, 206.0,
            np.ones(15_360, np.float32), "ds_baseline",
        ),),
        track_ids=(2,),
    )
    snapshots = tracker.update(
        SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=second_decision),
        (second_candidate,),
        (BeamformPreview(
            "session", 0, 17, second_decision, 208.5,
            np.ones(15_360, np.float32), "ds_baseline",
        ),),
        track_ids=(3,),
    )

    assert tuple(item.track_id for item in snapshots) == (2,)
    assert snapshots[0].audio_sample_count == 960
    assert tracker._formal_aliases[3] == 2
    assert not (tmp_path / "cache/track_003").exists()
    tracker.close()


def test_l2_id_rollover_can_rejoin_after_four_seconds(tmp_path):
    tracker = AudioIdTracker(
        "cache",
        project_root=tmp_path,
        wait_seconds=3.0,
        formal_id_rollover_wait_seconds=5.0,
    )
    first_decision = 15_360
    second_decision = first_decision + 4 * 48_000

    def preview(track_id: int, decision: int, theta: float) -> BeamformPreview:
        return BeamformPreview(
            "session", 0, track_id, decision, theta,
            np.ones(15_360, np.float32), "ds_baseline",
        )

    tracker.update(
        SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=first_decision),
        (SimpleNamespace(theta_deg=206.0, normalized_score=0.4),),
        (preview(1, first_decision, 206.0),),
        track_ids=(2,),
    )
    snapshots = tracker.update(
        SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=second_decision),
        (SimpleNamespace(theta_deg=208.0, normalized_score=0.7),),
        (preview(2, second_decision, 208.0),),
        track_ids=(3,),
    )

    assert tuple(item.track_id for item in snapshots) == (2,)
    assert tracker._formal_aliases[3] == 2
    tracker.close()


def test_simultaneous_l2_ids_are_never_merged_even_when_angles_are_close(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    first_decision, second_decision = 15_360, 16_320
    tracker.update(
        SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=first_decision),
        (SimpleNamespace(theta_deg=206.0, normalized_score=0.4),),
        (BeamformPreview(
            "session", 0, 16, first_decision, 206.0,
            np.ones(15_360, np.float32), "ds_baseline",
        ),),
        track_ids=(2,),
    )
    snapshots = tracker.update(
        SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=second_decision),
        (
            SimpleNamespace(theta_deg=206.5, normalized_score=0.5),
            SimpleNamespace(theta_deg=208.5, normalized_score=0.7),
        ),
        (
            BeamformPreview(
                "session", 0, 17, second_decision, 206.5,
                np.ones(15_360, np.float32), "ds_baseline",
            ),
            BeamformPreview(
                "session", 0, 17, second_decision, 208.5,
                np.ones(15_360, np.float32), "ds_baseline",
            ),
        ),
        track_ids=(2, 3),
    )

    assert tuple(item.track_id for item in snapshots) == (2, 3)
    assert tracker._formal_aliases[2] == 2
    assert tracker._formal_aliases[3] == 3
    tracker.close()


def test_ended_listening_tracks_are_not_pruned_while_ui_session_is_open(tmp_path):
    tracker = AudioIdTracker(
        "cache", project_root=tmp_path, wait_seconds=0.02, max_ended_tracks=1,
    )
    decision = 15_360
    candidates = (
        SimpleNamespace(theta_deg=0.0, normalized_score=0.5),
        SimpleNamespace(theta_deg=180.0, normalized_score=0.6),
    )
    previews = tuple(
        BeamformPreview(
            "session", 0, 16, decision, theta,
            np.ones(15_360, np.float32), "ds_baseline",
        )
        for theta in (0.0, 180.0)
    )
    tracker.update(
        SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=decision),
        candidates, previews, track_ids=(1, 2),
    )
    snapshots = tracker.update(
        SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=decision + 1_920),
        (), (),
    )

    assert tuple(item.track_id for item in snapshots) == (1, 2)
    assert all(item.state == "ended" for item in snapshots)
    assert (tmp_path / "cache/track_001/segment_000000.f32").is_file()
    assert (tmp_path / "cache/track_002/segment_000000.f32").is_file()
    tracker.close()


def test_provisional_l2_id_starts_cache_when_kalman_ready(tmp_path):
    tracker = AudioIdTracker("cache", project_root=tmp_path)
    candidate = SimpleNamespace(theta_deg=45.0, normalized_score=0.8)

    def preview(decision_sample: int) -> BeamformPreview:
        return BeamformPreview(
            "session", 0, decision_sample // 960, decision_sample, 45.0,
            np.ones(15_360, np.float32), "ds_baseline",
        )

    first = 15_360
    provisional = tracker.update(
        SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=first),
        (candidate,), (preview(first),), track_ids=(4,), formal_flags=(False,),
        kalman_ready_flags=(False,),
    )
    first_ready = tracker.update(
        SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=first + 960),
        (candidate,), (preview(first + 960),), track_ids=(4,), formal_flags=(False,),
        kalman_ready_flags=(True,),
    )
    continued = tracker.update(
        SimpleNamespace(session_id="session", stream_epoch=0, decision_sample=first + 1_920),
        (candidate,), (preview(first + 1_920),), track_ids=(4,), formal_flags=(True,),
        kalman_ready_flags=(True,),
    )

    assert provisional == ()
    assert tuple(item.track_id for item in first_ready) == (4,)
    assert first_ready[0].audio_sample_count == 0
    assert continued[0].audio_sample_count == 960
    assert (tmp_path / "cache/track_004/segment_000000.f32").stat().st_size == 960 * 4
    tracker.close()
