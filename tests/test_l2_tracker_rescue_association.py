from __future__ import annotations

from common.data_types import CandidateDirection
from layer2_source_detection.global_tracker import GlobalDirectionTracker, GlobalTrackerConfig


def _candidate(sample: int, theta: float, rank: int = 1) -> CandidateDirection:
    return CandidateDirection(
        "rescue",
        0,
        sample // 960,
        sample,
        sample - 1_920,
        sample,
        theta,
        1.0,
        0.9,
    )


def _update(
    tracker: GlobalDirectionTracker,
    sample: int,
    angles: tuple[float, ...],
):
    return tracker.update(
        "rescue",
        0,
        sample,
        tuple(_candidate(sample, theta, rank) for rank, theta in enumerate(angles, 1)),
        window_id=sample // 960,
        doa_start_sample=sample - 1_920,
        doa_end_sample=sample,
    )


def _tracker() -> GlobalDirectionTracker:
    return GlobalDirectionTracker(
        GlobalTrackerConfig(
            association_gate_deg=50.0,
            association_chi2=0.01,
            minimum_association_probability=0.99,
            confirmation_observations=3,
            confirmation_window_samples=9_600,
            tentative_ttl_samples=24_000,
            coasting_ttl_samples=96_000,
        )
    )


def test_candidate_within_50_degrees_keeps_id_and_publishes_filtered_angle() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (10.0,))
    _update(tracker, 16_320, (10.0,))
    _update(tracker, 17_280, (10.0,))

    observed, active = _update(tracker, 18_240, (50.0,))

    assert len(observed) == 1
    assert observed[0].track_id == first[0].track_id
    assert observed[0].measured_theta_deg == 50.0
    assert 10.0 < observed[0].theta_deg < 50.0
    assert {item.track_id for item in active} == {first[0].track_id}


def test_second_nearby_candidate_is_suppressed_instead_of_becoming_an_id() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (10.0,))

    _, active = _update(tracker, 16_320, (12.0, 45.0))

    assert {item.track_id for item in active} == {first[0].track_id}


def test_candidate_beyond_50_degrees_can_create_a_new_id() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (10.0,))

    observed, active = _update(tracker, 16_320, (61.0,))

    assert len(observed) == 1
    assert observed[0].track_id != first[0].track_id
    assert {item.track_id for item in active} == {first[0].track_id, observed[0].track_id}


def test_rescue_association_uses_circular_distance_across_zero() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (359.0,))

    observed, active = _update(tracker, 16_320, (20.0,))

    assert observed[0].track_id == first[0].track_id
    assert {item.track_id for item in active} == {first[0].track_id}


def test_last_observed_angle_recovers_id_when_prediction_has_drifted() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (10.0,))
    track = tracker._tracks[first[0].track_id]
    for model in track.models:
        model.mean[0] = 100.0
        model.mean[1] = 0.0

    observed, active = _update(tracker, 16_320, (20.0,))

    assert observed[0].track_id == first[0].track_id
    assert {item.track_id for item in active} == {first[0].track_id}


def test_alternating_nearby_tracks_merge_into_older_confirmed_id() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (10.0, 100.0))
    _update(tracker, 16_320, (10.0, 100.0))
    confirmed, _ = _update(tracker, 17_280, (10.0, 100.0))
    older_id = first[0].track_id
    duplicate_id = next(item.track_id for item in confirmed if item.track_id != older_id)

    _update(tracker, 27_840, ())
    for model in tracker._tracks[older_id].models:
        model.mean[:] = (20.0, 0.0)
    for model in tracker._tracks[duplicate_id].models:
        model.mean[:] = (30.0, 0.0)

    _update(tracker, 28_800, (20.0,))
    _update(tracker, 29_760, (30.0,))
    observed, active = _update(tracker, 30_720, (20.0,))

    assert {item.track_id for item in active} == {older_id}
    assert observed[0].track_id == older_id
    assert tracker.last_diagnostics.merged_track_ids == ((duplicate_id, older_id),)


def test_simultaneously_observed_nearby_tracks_are_not_merged() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (10.0, 100.0))
    _update(tracker, 16_320, (10.0, 100.0))
    confirmed, _ = _update(tracker, 17_280, (10.0, 100.0))
    track_ids = {item.track_id for item in confirmed}

    _update(tracker, 27_840, ())
    for theta, track_id in zip((20.0, 30.0), sorted(track_ids), strict=True):
        for model in tracker._tracks[track_id].models:
            model.mean[:] = (theta, 0.0)

    _update(tracker, 28_800, (20.0, 30.0))
    _update(tracker, 29_760, (20.0, 30.0))
    _, active = _update(tracker, 30_720, (20.0, 30.0))

    assert {item.track_id for item in active} == track_ids
    assert tracker.last_diagnostics.merged_track_ids == ()
