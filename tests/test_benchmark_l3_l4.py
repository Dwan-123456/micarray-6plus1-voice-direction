from __future__ import annotations

from common.config import load_config
from scripts.benchmark_l3_l4 import PROJECT_ROOT, _resolve_device


def test_benchmark_auto_device_follows_the_single_runtime_config() -> None:
    config = load_config(PROJECT_ROOT / "config/config.yaml", environ={})

    assert _resolve_device("auto", config) == config.runtime.preferred_device.casefold()
