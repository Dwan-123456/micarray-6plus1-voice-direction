from __future__ import annotations

from common.data_types import CandidateDirection
from layer2_source_detection.circular_kalman import CircularKalmanConfig, CircularKalmanFilter
from layer2_source_detection.direction_id_tracking import (
    DirectionIdTracker,
    DirectionIdTrackingConfig,
)


def candidate(theta: float, sample: int = 15_360) -> CandidateDirection:
    return CandidateDirection("s", 0, sample // 960, sample, sample - 1_920, sample, theta, 1.0, .9)


def mark_current_candidate_as_voice(
    tracker: DirectionIdTracker,
    session_id: str,
    sample: int,
    observed: tuple[CandidateDirection, ...],
) -> None:
    tracker.record_published(sample, observed, tracker.last_assignments)
    assert tracker.apply_voice_feedback(session_id, 0, sample, observed[0].theta_deg) is not None


def test_circular_kalman_crosses_zero_without_long_way_jump():
    kalman = CircularKalmanFilter()
    first = kalman.update("s", 0, 15_360, (candidate(359.0),), (1,), 1.0, 1.0)
    second = kalman.update("s", 0, 16_320, (candidate(1.0, 16_320),), (1,), 1.0, 1.0)
    assert first[0].theta_deg == 359.0
    assert second[0].theta_deg > 359.0 or second[0].theta_deg < 2.0


def test_private_id_tracker_does_not_modify_public_candidate():
    tracker = DirectionIdTracker()
    raw = (candidate(10.0), candidate(120.0))
    assert tracker.update("s", 0, 15_360, raw) == raw
    assert tracker.last_assignments == (1, 2)
    moved = (candidate(12.0, 16_320), candidate(118.0, 16_320))
    assert tracker.update("s", 0, 16_320, moved) == moved
    assert tracker.last_assignments == (1, 2)


def test_reacquired_measurement_matches_current_kalman_forecast_within_30_degrees():
    tracker = DirectionIdTracker()
    tracker.update("forecast", 0, 15_360, (candidate(100.0),))
    tracker.update("forecast", 0, 16_320, ())

    returned = (candidate(128.0, 17_280),)
    tracker.update(
        "forecast", 0, 17_280, returned,
        predicted_angles={1: 125.0},
    )

    assert tracker.last_assignments == (1,)


def test_measurement_chooses_the_nearest_predicted_id():
    tracker = DirectionIdTracker()
    tracker.update(
        "nearest", 0, 15_360,
        (candidate(100.0), candidate(160.0)),
    )

    tracker.update(
        "nearest", 0, 16_320,
        (candidate(134.0, 16_320),),
        predicted_angles={1: 125.0, 2: 145.0},
    )

    assert tracker.last_assignments == (1,)


def test_kalman_forecast_is_read_only_and_update_still_owns_prediction_step():
    kalman = CircularKalmanFilter()
    kalman.update("forecast", 0, 15_360, (candidate(100.0),), (1,), 1.0, 1.0)
    forecast = kalman.forecast_angles("forecast", 0, 16_320, (1,))
    assert forecast == kalman.forecast_angles("forecast", 0, 16_320, (1,))
    result = kalman.update(
        "forecast", 0, 16_320, (candidate(105.0, 16_320),), (1,), 1.0, 1.0,
    )
    assert result[0].theta_deg > 100.0


def test_kalman_state_follows_id_when_candidate_rank_swaps():
    tracker = DirectionIdTracker()
    kalman = CircularKalmanFilter()
    first = (candidate(10.0), candidate(100.0))
    tracker.update("s", 0, 15_360, first)
    kalman.update("s", 0, 15_360, first, tracker.last_assignments, 1.0, 1.0)

    swapped = (candidate(102.0, 16_320), candidate(12.0, 16_320))
    tracker.update("s", 0, 16_320, swapped)
    assert tracker.last_assignments == (2, 1)
    filtered = kalman.update("s", 0, 16_320, swapped, tracker.last_assignments, 1.0, 1.0)
    assert abs(filtered[0].theta_deg - 101.0) < 5.0
    assert abs(filtered[1].theta_deg - 11.0) < 5.0


def test_mature_id_predicts_for_bounded_hold_but_immature_id_does_not():
    tracking_config = DirectionIdTrackingConfig(
        max_missed_windows=2,
        confirmation_min_age_windows=3,
        confirmation_min_matches=3,
        prediction_hold_windows=2,
    )
    tracker = DirectionIdTracker(tracking_config)
    kalman = CircularKalmanFilter(CircularKalmanConfig(max_missed_windows=2))
    for index in range(3):
        sample = 15_360 + index * 960
        observed = (candidate(40.0 + index, sample),)
        tracker.update("s", 0, sample, observed)
        if index == 0:
            mark_current_candidate_as_voice(tracker, "s", sample, observed)
        kalman.update("s", 0, sample, observed, tracker.last_assignments, 1.0, 1.0)

    for missing_index in (1, 2):
        sample = 15_360 + (2 + missing_index) * 960
        tracker.update("s", 0, sample, ())
        kalman.update("s", 0, sample, (), (), 1.0, 1.0)
        assert tracker.prediction_track_ids == (1,)
        assert len(kalman.predicted_angles((1,))) == 1

    sample = 15_360 + 5 * 960
    tracker.update("s", 0, sample, ())
    kalman.update("s", 0, sample, (), (), 1.0, 1.0)
    assert tracker.prediction_track_ids == (1,)

    sample = 15_360 + 6 * 960
    tracker.update("s", 0, sample, ())
    kalman.update("s", 0, sample, (), (), 1.0, 1.0)
    assert tracker.prediction_track_ids == ()

    immature = DirectionIdTracker(tracking_config)
    for index in range(2):
        sample = 15_360 + index * 960
        immature.update("s", 0, sample, (candidate(80.0, sample),))
    immature.update("s", 0, 17_280, ())
    assert immature.prediction_track_ids == ()


def test_candidate_after_two_second_no_match_boundary_cannot_revive_expired_id():
    config = DirectionIdTrackingConfig(
        max_missed_windows=2,
        confirmation_min_age_windows=2,
        confirmation_min_matches=2,
        prediction_hold_windows=2,
    )
    tracker = DirectionIdTracker(config)
    for index in range(3):
        sample = 15_360 + index * 960
        observed = (candidate(30.0, sample),)
        tracker.update("expiry", 0, sample, observed)
        if index == 0:
            mark_current_candidate_as_voice(tracker, "expiry", sample, observed)
    assert tracker.confirmed_track_ids == (1,)
    for index in (3, 4):
        tracker.update("expiry", 0, 15_360 + index * 960, ())
    late_sample = 15_360 + 5 * 960
    tracker.update("expiry", 0, late_sample, (candidate(31.0, late_sample),))
    assert tracker.last_assignments == (2,)
    assert tracker.confirmed_track_ids == ()


def test_default_id_matures_after_three_seconds_with_five_sparse_matches_and_voice():
    tracker = DirectionIdTracker()
    match_indices = {0, 37, 75, 112, 150}
    for index in range(151):
        sample = 15_360 + index * 960
        if index in match_indices:
            theta = 20.0 + index / 37.5
            tracker.update("moving", 0, sample, (candidate(theta, sample),))
            if index == 0:
                mark_current_candidate_as_voice(
                    tracker, "moving", sample, (candidate(theta, sample),)
                )
            assert tracker.last_assignments == (1,)
        else:
            tracker.update("moving", 0, sample, ())
    tracker.update("moving", 0, 15_360 + 151 * 960, ())
    assert tracker.prediction_track_ids == (1,)


def test_provisional_id_with_fewer_than_five_matches_is_not_confirmed():
    tracker = DirectionIdTracker()
    match_indices = {0, 40, 80, 120}
    for index in range(151):
        sample = 15_360 + index * 960
        observed = (candidate(40.0, sample),) if index in match_indices else ()
        tracker.update("sparse", 0, sample, observed)
    assert tracker.active_track_count == 0
    assert tracker.prediction_track_ids == ()


def test_provisional_id_with_five_matches_but_no_l4_voice_is_not_confirmed():
    tracker = DirectionIdTracker()
    match_indices = {0, 37, 75, 112, 150}
    for index in range(151):
        sample = 15_360 + index * 960
        observed = (candidate(60.0, sample),) if index in match_indices else ()
        tracker.update("nonvoice", 0, sample, observed)
    assert tracker.confirmed_track_ids == ()
    assert tracker.prediction_track_ids == ()


def test_reacquired_measurement_uses_double_confidence_for_one_update():
    class ConfidenceSpyKalman(CircularKalmanFilter):
        def __init__(self):
            super().__init__()
            self.confidences: list[float] = []

        def _correct(self, state, theta_deg, measurement_noise_scale, measurement_confidence):
            self.confidences.append(measurement_confidence)
            return super()._correct(
                state, theta_deg, measurement_noise_scale, measurement_confidence
            )

    kalman = ConfidenceSpyKalman()
    kalman.update("reacquire", 0, 15_360, (candidate(20.0),), (1,), 1.0, 1.0)
    kalman.update("reacquire", 0, 16_320, (candidate(22.0, 16_320),), (1,), 1.0, 1.0)
    kalman.update("reacquire", 0, 17_280, (), (), 1.0, 1.0)
    kalman.update("reacquire", 0, 18_240, (candidate(35.0, 18_240),), (1,), 1.0, 1.0)
    kalman.update("reacquire", 0, 19_200, (candidate(36.0, 19_200),), (1,), 1.0, 1.0)
    assert kalman.confidences == [1.0, 2.0, 1.0]


def test_q_and_r_scales_change_kalman_response_without_resetting_identity():
    def response(q_scale: float, r_scale: float) -> float:
        kalman = CircularKalmanFilter(CircularKalmanConfig(max_missed_windows=50))
        kalman.update("s", 0, 15_360, (candidate(0.0),), (1,), q_scale, r_scale)
        for index in range(1, 21):
            kalman.update("s", 0, 15_360 + index * 960, (), (), q_scale, r_scale)
        measured = candidate(30.0, 15_360 + 21 * 960)
        return kalman.update(
            "s", 0, measured.decision_sample, (measured,), (1,), q_scale, r_scale
        )[0].theta_deg

    assert response(10.0, 1.0) > response(0.02, 1.0)
    assert response(1.0, 0.02) > response(1.0, 10.0)


def test_angle_matches_change_position_but_do_not_extend_formal_lifetime():
    config = DirectionIdTrackingConfig(
        max_missed_windows=10,
        confirmation_min_age_windows=2,
        confirmation_min_matches=2,
        prediction_hold_windows=2,
    )
    tracker = DirectionIdTracker(config)
    base = 15_360
    for index in range(5):
        sample = base + index * 960
        observed = (candidate(20.0 + index, sample),)
        tracker.update("lease", 0, sample, observed)
        if index == 0:
            mark_current_candidate_as_voice(tracker, "lease", sample, observed)
    assert tracker.confirmed_track_ids == (1,)

    expired_sample = base + 5 * 960
    tracker.update("lease", 0, expired_sample, (candidate(25.0, expired_sample),))
    assert tracker.last_assignments == (2,)
    assert tracker.confirmed_track_ids == ()


def test_l4_voice_angle_matches_historical_id_and_extends_lease():
    config = DirectionIdTrackingConfig(
        max_missed_windows=10,
        confirmation_min_age_windows=2,
        confirmation_min_matches=2,
        prediction_hold_windows=2,
    )
    tracker = DirectionIdTracker(config)
    base = 15_360
    for index in range(3):
        sample = base + index * 960
        item = (candidate(359.0, sample),)
        tracker.update("voice", 0, sample, item)
        tracker.record_published(sample, item, tracker.last_assignments)
        if index == 0:
            assert tracker.apply_voice_feedback("voice", 0, sample, 2.0) == 1

    for index in (3, 4):
        sample = base + index * 960
        item = (candidate(5.0, sample),)
        tracker.update("voice", 0, sample, item)
        tracker.record_published(sample, item, tracker.last_assignments)
    assert tracker.apply_voice_feedback("voice", 0, base + 4 * 960, 5.0) == 1
    tracker.update("voice", 0, base + 6 * 960, ())
    assert tracker.confirmed_track_ids == (1,)
    tracker.update("voice", 0, base + 7 * 960, ())
    assert tracker.confirmed_track_ids == ()


def test_forced_gate_association_cannot_create_or_promote_an_id():
    config = DirectionIdTrackingConfig(
        max_missed_windows=10,
        confirmation_min_age_windows=2,
        confirmation_min_matches=2,
        prediction_hold_windows=2,
    )
    tracker = DirectionIdTracker(config)
    base = 15_360
    for index in range(3):
        sample = base + index * 960
        observed = (candidate(40.0, sample),)
        tracker.update("forced", 0, sample, observed)
        if index == 0:
            mark_current_candidate_as_voice(tracker, "forced", sample, observed)
    assert tracker.confirmed_track_ids == (1,)
    sample = base + 3 * 960
    accepted = tracker.update(
        "forced", 0, sample,
        (candidate(42.0, sample), candidate(140.0, sample)),
        existing_formal_only=True,
    )
    assert tuple(item.theta_deg for item in accepted) == (42.0,)
    assert tracker.last_assignments == (1,)
    assert tracker.active_track_ids == (1,)
