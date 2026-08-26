from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from common.config import RuntimeConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parallel_runtime_limits_are_loaded_from_the_single_config():
    config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    runtime = config.runtime

    assert runtime.stage_queue_windows == 100
    assert (runtime.l2_queue_windows, runtime.l3_queue_windows, runtime.l5_queue_windows) == (
        100,
        100,
        100,
    )
    assert runtime.completion_queue_windows == 8
    assert runtime.max_inflight_windows == 303
    assert runtime.compute_cache_max_bytes == 64 * 1024 * 1024
    assert runtime.layer456_resident_memory_budget_bytes == 128 * 1024 * 1024
    assert runtime.layer456_spool_min_free_bytes == 5 * 1024**3
    assert config.layer6.embedding_cache_max_segments == 600
    assert runtime.overflow_policy == "drop_oldest"
    assert runtime.graceful_shutdown_timeout_seconds == 10.0
    assert config.layer4.streaming.model_dump() == {
        "enabled": True,
        "chunk_seconds": 4,
        "overlap_seconds": 1,
        "queue_chunks": 2,
    }


def test_hundred_window_defaults_are_covered_by_joiner_capacity():
    runtime = load_config(PROJECT_ROOT / "config" / "config.yaml").runtime

    assert RuntimeConfig.model_fields["stage_queue_windows"].default == 100
    assert RuntimeConfig.model_fields["l2_queue_windows"].default is None
    assert RuntimeConfig.model_fields["l3_queue_windows"].default is None
    assert RuntimeConfig.model_fields["l5_queue_windows"].default is None
    assert RuntimeConfig.model_fields["max_inflight_windows"].default is None
    assert runtime.max_inflight_windows == (
        runtime.l2_queue_windows
        + runtime.l3_queue_windows
        + runtime.l5_queue_windows
        + 3
    )

    with pytest.raises(ValidationError, match="must cover all staged queues"):
        RuntimeConfig.model_validate(
            {**runtime.model_dump(), "max_inflight_windows": 302}
        )


def test_shared_stage_queue_variable_resizes_all_stages_and_joiner_capacity():
    runtime = RuntimeConfig.model_validate({
        **load_config(PROJECT_ROOT / "config" / "config.yaml").runtime.model_dump(),
        "stage_queue_windows": 250,
        "l2_queue_windows": None,
        "l3_queue_windows": None,
        "l5_queue_windows": None,
        "max_inflight_windows": None,
    })
    assert (
        runtime.l2_queue_windows,
        runtime.l3_queue_windows,
        runtime.l5_queue_windows,
        runtime.max_inflight_windows,
    ) == (250, 250, 250, 753)


@pytest.mark.parametrize(
    "updates",
    (
        {"chunk_seconds": 2},
        {"chunk_seconds": 16},
        {"chunk_seconds": 10, "overlap_seconds": 10},
        {"queue_chunks": 0},
        {"queue_chunks": 65},
    ),
)
def test_layer4_streaming_limits_are_strict_and_overlap_advances(updates):
    streaming = load_config(
        PROJECT_ROOT / "config" / "config.yaml"
    ).layer4.streaming

    with pytest.raises(ValidationError):
        streaming.__class__.model_validate({**streaming.model_dump(), **updates})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("layer456_resident_memory_budget_bytes", 1),
        ("layer456_spool_min_free_bytes", -1),
    ),
)
def test_long_session_runtime_guardrails_are_strict(field, value):
    runtime = load_config(PROJECT_ROOT / "config" / "config.yaml").runtime

    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate({**runtime.model_dump(), field: value})


def test_layer6_voiceprint_cache_limit_is_strict():
    layer6 = load_config(PROJECT_ROOT / "config" / "config.yaml").layer6

    with pytest.raises(ValidationError):
        layer6.__class__.model_validate({
            **layer6.model_dump(),
            "embedding_cache_max_segments": 0,
        })
