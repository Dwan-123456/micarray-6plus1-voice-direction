from __future__ import annotations

from dataclasses import replace

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
        source_categories=tuple("human_voice" for _ in range(source_count)),
        recording_name="诊室测试录音",
        source_movements=tuple("stationary" for _ in range(source_count)),
        noise_source="air_conditioner",
    )


def test_wizard_allows_unspecified_source_count():
    assert validate_wizard(wizard_input(source_count=0, theta=(), distance=())) == []


def test_wizard_requires_one_angle_and_distance_per_source():
    errors = validate_wizard(wizard_input(source_count=2))
    assert "真实角度的数量必须与声源数量一致" in errors
    assert "声源距离的数量必须与声源数量一致" in errors


def test_wizard_requires_one_movement_per_source_when_movements_are_supplied():
    data = wizard_input(source_count=2, theta=(), distance=())
    errors = validate_wizard(replace(data, source_movements=("stationary",)))
    assert "声源移动方式的数量必须与声源数量一致" in errors
