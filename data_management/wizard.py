from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WizardInput:
    dataset_id: str
    room_id: str
    environment_id: str
    array_pose_id: str
    source_count: int
    consent_status: str
    allowed_uses: tuple[str, ...]
    theta_degrees: tuple[float, ...] = ()
    distance_m: tuple[float, ...] = ()
    source_categories: tuple[str, ...] = ("human_voice",)
    speaker_ids: tuple[str, ...] = ()
    language_tags: tuple[str, ...] = ()
    license_id: str = "internal"
    expires_at_utc: str | None = None
    notes: str = ""
    recording_name: str = ""
    source_movements: tuple[str, ...] = ()
    noise_source: str = ""


def validate_wizard(data: WizardInput) -> list[str]:
    errors = []
    name = data.recording_name.strip()
    if not name:
        errors.append("请填写音频名称")
    elif len(name) > 100 or any(ord(char) < 32 for char in name):
        errors.append("音频名称必须在100个字符以内且不能包含控制字符")
    if not all((data.dataset_id, data.room_id, data.environment_id, data.array_pose_id)):
        errors.append("数据集、房间、环境和阵列姿态均必填")
    if data.consent_status not in {"granted", "not_applicable"}:
        errors.append("参与者同意未获授权")
    if not data.allowed_uses:
        errors.append("至少选择一种允许用途")
    if data.source_count < 0:
        errors.append("声源数量不能小于0")
    if data.source_movements and len(data.source_movements) != data.source_count:
        errors.append("声源移动方式的数量必须与声源数量一致")
    if data.source_movements and len(data.source_categories) != data.source_count:
        errors.append("声源类型的数量必须与声源数量一致")
    if any(not value.strip() for value in data.source_categories):
        errors.append("声源类型不能为空")
    if any(not value.strip() for value in data.source_movements):
        errors.append("声源移动方式不能为空")
    if len(data.noise_source) > 200 or any(ord(char) < 32 for char in data.noise_source):
        errors.append("噪音来源必须在200个字符以内且不能包含控制字符")
    if data.theta_degrees and len(data.theta_degrees) != data.source_count:
        errors.append("真实角度的数量必须与声源数量一致")
    if data.distance_m and len(data.distance_m) != data.source_count:
        errors.append("声源距离的数量必须与声源数量一致")
    if any(not 0 <= x < 360 for x in data.theta_degrees):
        errors.append("真实角度必须位于[0,360)")
    if any(x <= 0 for x in data.distance_m):
        errors.append("距离必须为正数")
    return errors
