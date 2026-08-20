from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from common.config import RuntimeConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parallel_runtime_limits_are_loaded_from_the_single_config():
    runtime = load_config(PROJECT_ROOT / "config" / "config.yaml").runtime

    assert runtime.stage_queue_windows == 2_000
    assert (runtime.l2_queue_windows, runtime.l3_queue_windows, runtime.l4_queue_windows) == (
        2_000,
        2_000,
        2_000,
    )
    assert runtime.completion_queue_windows == 8
    assert runtime.max_inflight_windows == 6_003
    assert runtime.compute_cache_max_bytes == 64 * 1024 * 1024
    assert runtime.overflow_policy == "drop_oldest"
    assert runtime.graceful_shutdown_timeout_seconds == 10.0


def test_two_thousand_window_defaults_are_covered_by_joiner_capacity():
    runtime = load_config(PROJECT_ROOT / "config" / "config.yaml").runtime

    assert RuntimeConfig.model_fields["stage_queue_windows"].default == 2_000
    assert RuntimeConfig.model_fields["l2_queue_windows"].default is None
    assert RuntimeConfig.model_fields["l3_queue_windows"].default is None
    assert RuntimeConfig.model_fields["l4_queue_windows"].default is None
    assert RuntimeConfig.model_fields["max_inflight_windows"].default is None
    assert runtime.max_inflight_windows == (
        runtime.l2_queue_windows
        + runtime.l3_queue_windows
        + runtime.l4_queue_windows
        + 3
    )

    with pytest.raises(ValidationError, match="must cover all staged queues"):
        RuntimeConfig.model_validate(
            {**runtime.model_dump(), "max_inflight_windows": 6_002}
        )


def test_shared_stage_queue_variable_resizes_all_stages_and_joiner_capacity():
    runtime = RuntimeConfig.model_validate({
        **load_config(PROJECT_ROOT / "config" / "config.yaml").runtime.model_dump(),
        "stage_queue_windows": 250,
        "l2_queue_windows": None,
        "l3_queue_windows": None,
        "l4_queue_windows": None,
        "max_inflight_windows": None,
    })
    assert (
        runtime.l2_queue_windows,
        runtime.l3_queue_windows,
        runtime.l4_queue_windows,
        runtime.max_inflight_windows,
    ) == (250, 250, 250, 753)
