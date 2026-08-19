from __future__ import annotations

from pathlib import Path

from common.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parallel_runtime_limits_are_loaded_from_the_single_config():
    runtime = load_config(PROJECT_ROOT / "config" / "config.yaml").runtime

    assert (runtime.l2_queue_windows, runtime.l3_queue_windows, runtime.l4_queue_windows) == (4, 3, 3)
    assert runtime.completion_queue_windows == 8
    assert runtime.max_inflight_windows == 16
    assert runtime.compute_cache_max_bytes == 64 * 1024 * 1024
    assert runtime.overflow_policy == "drop_oldest"
    assert runtime.graceful_shutdown_timeout_seconds == 10.0
