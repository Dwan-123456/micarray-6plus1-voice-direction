from __future__ import annotations

from dataclasses import dataclass

from common.config import ProjectConfig


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


@dataclass(slots=True)
class CalibrationConfig:
    gains: tuple[float, ...]
    polarity: tuple[int, ...]
    delay_samples: tuple[int, ...]

    @classmethod
    def from_project(cls, config: ProjectConfig) -> "CalibrationConfig":
        value = config.calibration
        return cls(value.gains, value.polarity, value.delay_samples)

    def validate(self, channels: int = 7) -> None:
        if not all(len(values) == channels for values in (self.gains, self.polarity, self.delay_samples)):
            raise ValueError("校准参数数量必须与物理通道数一致")
        if any(value not in (-1, 1) for value in self.polarity):
            raise ValueError("polarity 只能是 -1 或 1")
        if any(value < 0 for value in self.delay_samples):
            raise ValueError("delay_samples 只能是非负整数")
