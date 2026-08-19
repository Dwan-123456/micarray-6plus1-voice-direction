from dataclasses import dataclass
from pathlib import Path

from common.config import load_config


@dataclass(frozen=True)
class Settings:
    device_name: str
    host_api: str
    sample_rate: int
    channels: int
    block_size: int
    serial_port: str
    serial_baud: int
    serial_required: bool
    endpoint_id: str = "{0.0.1.00000000}.{bcf8fa2f-9e38-4adb-ad22-159888f98e3b}"


_ROOT = Path(__file__).resolve().parents[1]
_config = load_config(_ROOT / "config/config.yaml")
settings = Settings(
    _config.device.device_name, _config.device.host_api, _config.device.sample_rate,
    _config.device.device_channels, _config.device.block_size_samples,
    _config.device.serial_port, _config.device.serial_baud, _config.device.serial_required,
)
