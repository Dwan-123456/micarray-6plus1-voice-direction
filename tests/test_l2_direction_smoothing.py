from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from common.data_types import CandidateDirection
from layer2_source_detection import (
    DirectionSmoother,
    DirectionSmoothingConfig,
    DirectionSmoothingError,
    circular_delta_deg,
)


def candidate(
    theta: float,
    index: int,
    *,
    session_id: str = "session",
    epoch: int = 0,
    raw: float = 1.0,
    normalized: float = 0.8,
) -> CandidateDirection:
    decision = 15_360 + 960 * index
    return CandidateDirection(
        session_id,
        epoch,
        index,
        decision,
        decision - 1_920,
        decision,
        theta,
        raw,
        normalized,
    )


def update(smoother: DirectionSmoother, items: tuple[CandidateDirection, ...]):
    reference = items[0]
    return smoother.update(
        reference.session_id,
        reference.stream_epoch,
        reference.decision_sample,
        items,
    )


def test_wraparound_is_smoothed_on_the_circle_and_public_fields_are_preserved():
    smoother = DirectionSmoother()
    outputs = []
    for index, theta in enumerate((358.0, 359.0, 1.0, 2.0)):
        raw = candidate(theta, index, raw=2.0 + index, normalized=0.9 - index * 0.1)
        result = update(smoother, (raw,))[0]
        outputs.append(result.theta_deg)
        assert replace(result, theta_deg=raw.theta_deg) == raw
    assert outputs[0] == 358.0
    assert all(abs(circular_delta_deg(theta, 0.0)) < 5.0 for theta in outputs)


def test_stationary_jitter_variance_is_reduced_after_warmup():
    measurements = np.asarray((90, 94, 87, 93, 88, 92, 89, 91, 87, 93, 90, 92), dtype=float)
    smoother = DirectionSmoother()
    filtered = np.asarray([
        update(smoother, (candidate(theta, index),))[0].theta_deg
        for index, theta in enumerate(measurements)
    ])
    assert np.var(filtered[3:]) < np.var(measurements[3:])


def test_two_source_assignment_is_global_deterministic_and_keeps_candidate_rank():
    smoother = DirectionSmoother()
    update(smoother, (candidate(10.0, 0, raw=4.0), candidate(100.0, 0, raw=3.0)))
    raw = (candidate(102.0, 1, raw=9.0), candidate(12.0, 1, raw=8.0))
    result = update(smoother, raw)
    assert len(result) == 2
    assert abs(circular_delta_deg(result[0].theta_deg, 100.0)) < 5.0
    assert abs(circular_delta_deg(result[1].theta_deg, 10.0)) < 5.0
    assert [item.raw_score for item in result] == [9.0, 8.0]


def test_empty_windows_never_publish_predictions_and_tracks_expire():
    smoother = DirectionSmoother(replace(DirectionSmoothingConfig(), max_missed_windows=2))
    first = candidate(30.0, 0)
    assert update(smoother, (first,))[0].theta_deg == 30.0
    for index in (1, 2, 3):
        decision = 15_360 + 960 * index
        assert smoother.update("session", 0, decision, ()) == ()
    assert smoother.active_track_count == 0
    reacquired = update(smoother, (candidate(41.0, 4),))[0]
    assert reacquired.theta_deg == 41.0


def test_epoch_change_resets_state_and_first_angle_is_raw():
    smoother = DirectionSmoother()
    update(smoother, (candidate(20.0, 0),))
    update(smoother, (candidate(25.0, 1),))
    reset_candidate = candidate(200.0, 0, epoch=1)
    assert update(smoother, (reset_candidate,))[0].theta_deg == 200.0


def test_skipped_samples_use_real_elapsed_time_and_nonmonotonic_input_fails():
    smoother = DirectionSmoother()
    update(smoother, (candidate(45.0, 0),))
    skipped = candidate(50.0, 4)
    assert update(smoother, (skipped,))[0].decision_sample == skipped.decision_sample
    with pytest.raises(DirectionSmoothingError, match="monotonically"):
        update(smoother, (skipped,))


def test_disabled_smoother_returns_the_exact_raw_contract():
    smoother = DirectionSmoother(replace(DirectionSmoothingConfig(), enabled=False))
    raw = (candidate(12.0, 0), candidate(120.0, 0))
    assert update(smoother, raw) == raw
    assert smoother.active_track_count == 0


def test_identity_mismatch_and_candidate_limit_are_rejected():
    smoother = DirectionSmoother()
    with pytest.raises(ValueError, match="another session"):
        smoother.update("other", 0, candidate(1.0, 0).decision_sample, (candidate(1.0, 0),))
    with pytest.raises(ValueError, match="at most 2"):
        update(smoother, (candidate(1.0, 0), candidate(2.0, 0), candidate(3.0, 0)))


def test_moving_source_follows_without_wrap_or_unbounded_lag():
    smoother = DirectionSmoother()
    measurements = tuple(float(value % 360) for value in range(340, 391, 5))
    filtered = tuple(
        update(smoother, (candidate(theta, index),))[0].theta_deg
        for index, theta in enumerate(measurements)
    )
    assert all(np.isfinite(filtered))
    assert max(
        abs(circular_delta_deg(actual, measured))
        for actual, measured in zip(filtered, measurements, strict=True)
    ) < 15.0


def test_fixed_sequence_is_bitwise_deterministic_and_has_no_public_id_field():
    sequence = ((10.0, 130.0), (12.0, 128.0), (15.0, 125.0), (19.0, 121.0))

    def run_once():
        smoother = DirectionSmoother()
        return tuple(
            tuple(item.theta_deg for item in update(
                smoother,
                tuple(candidate(theta, index, raw=5.0 - rank) for rank, theta in enumerate(batch)),
            ))
            for index, batch in enumerate(sequence)
        )

    assert run_once() == run_once()
    public_fields = {item.name for item in fields(CandidateDirection)}
    assert "track_id" not in public_fields and "source_id" not in public_fields


def test_exact_posterior_collision_falls_back_to_the_lower_rank_raw_angle():
    raw = (candidate(10.0, 0), candidate(30.0, 0))
    assert DirectionSmoother._avoid_rank_collisions([20.0, 20.0], raw) == [20.0, 30.0]
