from __future__ import annotations

from common.config import load_config
import pytest

from scripts.benchmark_l3_l5 import PROJECT_ROOT, _candidates, _l3_windows, _resolve_device


def test_benchmark_auto_device_follows_the_l3_runtime_config() -> None:
    config = load_config(PROJECT_ROOT / "config/config.yaml", environ={})

    assert _resolve_device("auto", config, layer="l3") == config.runtime.l3_device
    assert _resolve_device("auto", config, layer="l5") == config.runtime.l5_device


def test_benchmark_builds_a_real_three_candidate_batch() -> None:
    window = _l3_windows(1, seed=20260819)[0]

    candidates = _candidates(window, 3)

    assert tuple(item.theta_deg for item in candidates) == (20.0, 120.0, 240.0)


@pytest.mark.parametrize("count", (0, 4))
def test_benchmark_rejects_candidate_counts_outside_runtime_contract(count: int) -> None:
    window = _l3_windows(1, seed=20260820)[0]

    with pytest.raises(ValueError, match="one, two, or three"):
        _candidates(window, count)
