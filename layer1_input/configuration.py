from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from common.config import ProjectConfig, calibration_config_hash
from common.data_types import CalibrationAssetIdentity, CalibrationMetadata


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int
    device_channels: int
    physical_channel_map: tuple[int, ...]
    logical_channel_map: tuple[int, ...]
    block_size: int
    device_name: str
    host_api: str
    handoff_blocks: int

    @classmethod
    def from_project(cls, config: ProjectConfig) -> "AudioConfig":
        device = config.device
        return cls(
            device.sample_rate,
            device.device_channels,
            device.physical_channel_map,
            device.logical_channel_map,
            device.block_size_samples,
            device.device_name,
            device.host_api,
            config.runtime.capture_handoff_blocks,
        )


@dataclass(slots=True)
class CdcConfig:
    enabled: bool
    port: str
    baudrate: int
    required: bool

    @classmethod
    def from_project(cls, config: ProjectConfig) -> "CdcConfig":
        device = config.device
        return cls(device.serial_enabled, device.serial_port, device.serial_baud, device.serial_required)

    def validate(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.required, bool):
            raise ValueError("CDC enabled/required 必须是布尔值")
        if not self.port.strip():
            raise ValueError("CDC port 不能为空")
        if isinstance(self.baudrate, bool) or self.baudrate <= 0:
            raise ValueError("CDC baudrate 必须是正整数")


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    gains: tuple[float, ...]
    polarity: tuple[int, ...]
    delay_samples: tuple[int, ...]
    version: str = "gain_polarity_integer_delay_v1"
    correction_model: str = "gain_polarity_integer_delay_v1"
    status: str = "unverified"
    report_hash: str | None = None
    fractional_delay_asset: CalibrationAssetIdentity | None = None
    frequency_response_asset: CalibrationAssetIdentity | None = None
    calibration_hash: str = ""

    def __post_init__(self) -> None:
        if not self.calibration_hash:
            payload = {
                "correction_model": self.correction_model,
                "delay_samples": self.delay_samples,
                "fractional_delay_asset": self.fractional_delay_asset,
                "frequency_response_asset": self.frequency_response_asset,
                "gains": self.gains,
                "polarity": self.polarity,
                "version": self.version,
            }
            canonical = json.dumps(
                payload,
                default=lambda value: {
                    "sha256": value.sha256,
                    "uri": value.uri,
                    "version": value.version,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            object.__setattr__(self, "calibration_hash", hashlib.sha256(canonical.encode()).hexdigest())

    @staticmethod
    def _asset(value: object | None) -> CalibrationAssetIdentity | None:
        if value is None:
            return None
        return CalibrationAssetIdentity(value.uri, value.version, value.sha256)

    @classmethod
    def from_project(cls, config: ProjectConfig) -> "CalibrationConfig":
        value = config.calibration
        return cls(
            value.gains,
            value.polarity,
            value.delay_samples,
            value.version,
            value.correction_model,
            config.hardware.hardware_calibration_status,
            config.hardware.hardware_calibration_report_hash,
            cls._asset(value.fractional_delay_asset),
            cls._asset(value.frequency_response_asset),
            calibration_config_hash(value),
        )

    def validate(self, channels: int = 7) -> None:
        if not all(len(values) == channels for values in (self.gains, self.polarity, self.delay_samples)):
            raise ValueError("校准参数数量必须与物理通道数一致")
        if any(value not in (-1, 1) for value in self.polarity):
            raise ValueError("polarity 只能是 -1 或 1")
        if any(value < 0 for value in self.delay_samples):
            raise ValueError("delay_samples 只能是非负整数")
        if not self.version or self.correction_model != "gain_polarity_integer_delay_v1":
            raise ValueError("当前L1只支持版本化gain/polarity/integer-delay校准")
        if self.status not in {"verified", "unverified"}:
            raise ValueError("校准状态必须为verified或unverified")
        if len(self.calibration_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.calibration_hash
        ):
            raise ValueError("calibration_hash必须为小写SHA-256")
        if self.fractional_delay_asset is not None or self.frequency_response_asset is not None:
            raise ValueError("亚采样/频率相关校准资产接口已预留，但当前L1尚未实现应用")

    @property
    def metadata(self) -> CalibrationMetadata:
        return CalibrationMetadata(
            self.status,
            self.version,
            self.calibration_hash,
            self.correction_model,
            self.report_hash,
            self.fractional_delay_asset,
            self.frequency_response_asset,
        )
