from __future__ import annotations

import numpy as np

from common.data_types import CandidateDirection
from layer2_source_detection.circular_kalman_v2 import (
    CircularKalmanFilterV2,
    CircularKalmanV2Config,
)
from layer2_source_detection.direction_id_tracking_v2 import (
    DirectionIdTrackerV2,
    DirectionIdTrackingV2Config,
)


def _candidate(sample: int, theta: float, norm: float = 0.8) -> CandidateDirection:
    return CandidateDirection(
        "s", 0, sample // 960, sample, sample - 1920, sample,
        theta % 360.0, 1.0, norm,
    )


def test_v2_first_hit_is_temp_and_second_hit_enables_kalman() -> None:
    tracker = DirectionIdTrackerV2()
    tracker.update("s", 0, 1920, (_candidate(1920, 50),))
    assert tracker.last_assignments == (1,)
    assert tracker.kalman_ready_track_ids == ()
    tracker.update("s", 0, 2880, (_candidate(2880, 52),))
    assert tracker.last_assignments == (1,)
    assert tracker.kalman_ready_track_ids == (1,)


def test_v2_internal_four_tracks_public_three_with_circular_45_degree_nms() -> None:
    tracker = DirectionIdTrackerV2()
    candidates = tuple(_candidate(1920, theta, norm) for theta, norm in (
        (359, .9), (2, .8), (110, .7), (210, .6)
    ))
    accepted = tracker.update("s", 0, 1920, candidates)
    assert tracker.active_track_count == 4
    public, ids, flags = tracker.select_public(
        accepted, tracker.last_assignments, (False,) * 4, 45.0
    )
    assert len(public) == len(ids) == len(flags) == 3
    assert public[0].theta_deg == 359
    assert public[1].theta_deg == 110
    assert public[2].theta_deg == 210


def test_v2_nonvoice_fan_cannot_become_formal_but_voice_can() -> None:
    config = DirectionIdTrackingV2Config(
        confirmation_age_samples=3 * 960,
        formal_lease_samples=3 * 960,
        provisional_hold_samples=20 * 960,
    )
    tracker = DirectionIdTrackerV2(config)
    for i in range(6):
        sample = 1920 + i * 720
        candidate = _candidate(sample, 0)
        tracker.update("s", 0, sample, (candidate,))
        tracker.record_published(sample, (candidate,), tracker.last_assignments)
        tracker.apply_classification_feedback("s", 0, sample, 0, 0.05, False)
    assert tracker.confirmed_track_ids == ()
    public, _, _ = tracker.select_public(
        (candidate,), tracker.last_assignments, (False,), 45.0
    )
    assert public == (candidate,)

    tracker = DirectionIdTrackerV2(config)
    for i in range(6):
        sample = 1920 + i * 720
        candidate = _candidate(sample, 110)
        tracker.update("s", 0, sample, (candidate,))
        tracker.record_published(sample, (candidate,), tracker.last_assignments)
        tracker.apply_classification_feedback("s", 0, sample, 110, 0.95, True)
    assert tracker.confirmed_track_ids == (1,)


def test_v2_nonvoice_never_hides_angle_and_voice_clears_negative_evidence() -> None:
    tracker = DirectionIdTrackerV2()
    samples = (1920, 2880, 3840, 4800, 5760, 6720)
    for index, sample in enumerate(samples):
        candidate = _candidate(sample, 40)
        accepted = tracker.update("s", 0, sample, (candidate,))
        tracker.record_published(sample, accepted, tracker.last_assignments)
        is_voice = index == 2
        tracker.apply_classification_feedback(
            "s", 0, sample, 40, 0.95 if is_voice else 0.05, is_voice
        )
    public, _, _ = tracker.select_public(
        (candidate,), tracker.last_assignments, (False,), 45.0
    )
    assert public == (candidate,)


def test_v2_kalman_damps_velocity_and_caps_it() -> None:
    kalman = CircularKalmanFilterV2(CircularKalmanV2Config(
        velocity_half_life_seconds=0.5, max_velocity_dps=60.0,
    ))
    c1 = _candidate(1920, 50)
    c2 = _candidate(2880, 55)
    kalman.update("s", 0, 1920, (c1,), (1,), 1.0, 1.0, (1,))
    kalman.update("s", 0, 2880, (c2,), (1,), 1.0, 0.02, (1,))
    first = kalman.forecast_angles("s", 0, 2880 + 24_000, (1,))[1]
    second = kalman.forecast_angles("s", 0, 2880 + 48_000, (1,))[1]
    assert abs(((second - first + 180) % 360) - 180) < 30.0
    assert np.isfinite((first, second)).all()
