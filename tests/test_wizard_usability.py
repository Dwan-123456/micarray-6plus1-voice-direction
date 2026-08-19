from __future__ import annotations

from data_management.wizard import WizardInput, validate_wizard


def wizard_input(
    *, source_count: int = 1, theta: tuple[float, ...] = (0.0,), distance: tuple[float, ...] = (1.0,)
) -> WizardInput:
    return WizardInput(
        "dataset-a",
        "room-a",
        "quiet",
        "pose-a",
        source_count,
        "granted",
        ("research",),
        theta,
        distance,
        recording_name="诊室测试录音",
    )


def test_wizard_allows_unspecified_source_count():
    assert validate_wizard(wizard_input(source_count=0, theta=(), distance=())) == []


def test_wizard_does_not_require_a_preset_recording_time():
    assert validate_wizard(wizard_input()) == []


def test_wizard_requires_one_angle_and_distance_per_source():
    errors = validate_wizard(wizard_input(source_count=2))
    assert "真实角度的数量必须与声源数量一致" in errors
    assert "声源距离的数量必须与声源数量一致" in errors
