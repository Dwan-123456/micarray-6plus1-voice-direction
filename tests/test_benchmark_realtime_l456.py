from __future__ import annotations

import argparse
from pathlib import Path
import queue
from types import SimpleNamespace

import pytest

from scripts import benchmark_realtime_l456 as benchmark


@pytest.mark.parametrize("seconds", (3, 5, 8, 10, 15))
def test_chunk_seconds_accepts_every_representative_integer(seconds: int) -> None:
    assert benchmark._chunk_seconds(str(seconds)) == seconds


@pytest.mark.parametrize("seconds", (2, 16))
def test_chunk_seconds_rejects_values_outside_three_to_fifteen(seconds: int) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="3..15"):
        benchmark._chunk_seconds(str(seconds))


def test_benchmark_cli_default_matches_the_runtime_four_second_cadence() -> None:
    args = benchmark._parser().parse_args(["recording_manifest.json"])

    assert args.chunk_seconds == 4


def test_benchmark_config_freezes_the_ephemeral_ds_mf2_profile(tmp_path: Path) -> None:
    config = benchmark._benchmark_config(
        benchmark.DEFAULT_CONFIG,
        chunk_seconds=5,
        ephemeral_data_root=tmp_path / "ephemeral",
    )

    assert config.paths.data_root == str((tmp_path / "ephemeral").resolve())
    assert config.layer1_pre_denoise.enabled is False
    assert config.layer1_speaker_count.enabled is False
    assert config.layer2.scanner_backend == "frequency_normalized_music"
    assert config.layer2.dpd_rank1_enabled is False
    assert config.layer2.noise_whitening_enabled is True
    assert config.layer4.default_backend == "mossformer2_ss_16k"
    assert config.layer4.streaming.chunk_seconds == 5
    assert config.layer5.input_gain_compensation.enabled is False
    assert config.recording.runtime.mode == "off"


def test_process_rss_probe_returns_a_positive_value() -> None:
    assert benchmark._current_rss_bytes() > 0


def test_observed_metrics_tracks_latest_only_previews_and_sampled_queue_high_water() -> None:
    mailbox: queue.Queue[object] = queue.Queue()
    mailbox.put(SimpleNamespace(
        revision=2,
        is_final=False,
        valid_through_sample_48k=350_400,
        stage_durations_seconds=(("l4", 1.0), ("l5", 0.2), ("l6", 0.1)),
    ))
    runtime = SimpleNamespace(
        latest_realtime_postprocessing=mailbox,
        processing_status={
            "queue_depths": {"l2": 3, "l3": 2},
            "layer456_stream": {"queued_blocks": 2, "latest_revision": 2},
        },
    )
    metrics = benchmark._ObservedMetrics()

    metrics.observe(runtime, elapsed_seconds=10.4, source_seconds=10.2)
    runtime.processing_status["queue_depths"] = {"l2": 1, "l3": 4}
    runtime.processing_status["layer456_stream"]["queued_blocks"] = 1
    metrics.observe(runtime, elapsed_seconds=10.5, source_seconds=10.3)

    assert metrics.queue_high_water == {"l2": 3, "l3": 4, "layer456": 2}
    assert metrics.observed_revisions == [2]
    assert metrics.first_preview_wall_seconds == 10.4
    assert metrics.first_preview_source_seconds == 10.2
    assert benchmark._stage_durations(metrics.latest_snapshot) == {
        "model_load": 0.0,
        "l4": 1.0,
        "dnsmos": 0.0,
        "l5": 0.2,
        "l6": 0.1,
        "snapshot": 0.0,
    }
    assert metrics.latest_preview_lag_seconds == pytest.approx(2.9)
    assert metrics.maximum_preview_lag_seconds == pytest.approx(2.9)


def test_preview_coverage_requires_every_sealed_track_at_its_exact_end() -> None:
    def source(track_id: int, end_sample: int) -> SimpleNamespace:
        return SimpleNamespace(
            session_id="session",
            stream_epoch=0,
            track_id=track_id,
            end_sample=end_sample,
        )

    sealed = (source(1, 480_000), source(2, 960_000))
    complete = SimpleNamespace(l4_processed=(
        SimpleNamespace(source=source(1, 480_000)),
        # Two separated branches must collapse to one exact track identity.
        SimpleNamespace(source=source(2, 960_000)),
        SimpleNamespace(source=source(2, 960_000)),
    ))
    missing_track = SimpleNamespace(l4_processed=(
        SimpleNamespace(source=source(1, 480_000)),
    ))
    truncated = SimpleNamespace(l4_processed=(
        SimpleNamespace(source=source(1, 480_000)),
        SimpleNamespace(source=source(2, 900_000)),
    ))

    assert benchmark._preview_covers_sealed_sources(complete, sealed)
    assert not benchmark._preview_covers_sealed_sources(missing_track, sealed)
    assert not benchmark._preview_covers_sealed_sources(truncated, sealed)
