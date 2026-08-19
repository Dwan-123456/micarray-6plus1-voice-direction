from __future__ import annotations

import pytest

from data_management import DecisionRecord


def _candidate(theta: float = 30.0) -> dict[str, object]:
    return {"theta_deg": theta, "raw_score": 1.0, "normalized_score": 0.8}


def _detection(theta: float = 30.0, *, voice: bool = True) -> dict[str, object]:
    return {
        "theta_deg": theta,
        "probability": 0.9,
        "is_voice": voice,
        "model_id": "primary",
    }


def test_successful_record_requires_one_same_order_detection_per_candidate() -> None:
    with pytest.raises(ValueError, match="每个候选"):
        DecisionRecord(
            "session", 0, 0, 15_360, (13_440, 15_360), (0, 15_360), "ok",
            candidates=(_candidate(),),
        )
    with pytest.raises(ValueError, match="同序"):
        DecisionRecord(
            "session", 0, 0, 15_360, (13_440, 15_360), (0, 15_360), "ok",
            candidates=(_candidate(30.0),),
            detections=(_detection(31.0),),
            voice_direction_count=1,
        )


def test_failed_stage_cannot_be_misreported_as_degraded() -> None:
    with pytest.raises(ValueError, match="必须为error"):
        DecisionRecord(
            "session", 0, 0, 15_360, (13_440, 15_360), (0, 15_360), "degraded",
            stage_statuses={"l2": "completed", "l3": "failed", "l4": "skipped"},
        )


def test_voice_count_is_derived_from_formal_detections() -> None:
    with pytest.raises(ValueError, match="voice_direction_count"):
        DecisionRecord(
            "session", 0, 0, 15_360, (13_440, 15_360), (0, 15_360), "ok",
            candidates=(_candidate(),),
            detections=(_detection(voice=False),),
            voice_direction_count=1,
        )

