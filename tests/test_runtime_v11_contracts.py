from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.processing_contracts import (
    JoinedWindowResult, L2StageResult, L3StageResult, L5StageResult,
    ProcessingConfigSnapshot, WindowKey, WindowWorkItem,
)
from common.config import load_config
from common.data_types import CandidateDirection, DecisionWindow, TrackedDirection
from common.geometry import physical_6plus1_geometry
from layer2_source_detection import DirectionScanConfig, RollingNormMusicScanner
from layer2_source_detection.global_tracker import GlobalDirectionTracker, GlobalTrackerConfig


def _window(window_id: int, sample: int, *, epoch: int = 0) -> DecisionWindow:
    samples = np.random.default_rng(window_id + 100 * epoch).normal(
        0.0, 0.01, (7_680, 8),
    ).astype(np.float32)
    return DecisionWindow(
        "runtime-v11", epoch, window_id, sample, sample - 1_920, sample,
        sample - 7_680, sample, 48_000,
        samples, (window_id,),
    )


def _candidate(window: DecisionWindow, theta: float) -> CandidateDirection:
    return CandidateDirection(
        window.session_id, window.stream_epoch, window.window_id, window.decision_sample,
        window.doa_start_sample, window.doa_end_sample, theta, 1.0, 0.9,
    )


def _direction(window: DecisionWindow, track_id: int, theta: float) -> TrackedDirection:
    return TrackedDirection(
        window.session_id, window.stream_epoch, window.window_id, window.decision_sample,
        window.doa_start_sample, window.doa_end_sample, track_id, 1, theta, theta,
        1.0, 0.9, "confirmed", True, False,
        window.decision_sample - 960, window.decision_sample, 0, False,
    )


def test_processing_config_has_music_mdl_lifecycle_and_no_removed_switches() -> None:
    config = load_config("config/config.yaml", environ={})
    values = config.layer2.model_dump()
    assert config.layer2.scanner_backend == "frequency_normalized_music"
    assert values["music"]["context_ms"] in {160, 240, 320}
    assert (values["n_fft"], values["win_length"], values["hop_length"]) == (1024, 960, 480)
    assert values["mdl_max_age_ms"] <= 100
    assert values["dpd_rank1_enabled"] is False
    assert values["noise_whitening_enabled"] is False
    assert values["direction_id_tracking"]["coasting_ttl_ms"] > 0
    assert "iterative_peak_search_enabled" not in values
    assert "enabled" not in values["direction_id_tracking"]


def test_tracker_uses_sample_lifecycle_and_preserves_session_counter_across_epoch() -> None:
    tracker = GlobalDirectionTracker(GlobalTrackerConfig(
        association_gate_deg=20.0, max_velocity_dps=60.0,
        confirmation_observations=2, confirmation_window_samples=9_600,
        coasting_ttl_samples=1_920, miss_cost=30.0, birth_cost=30.0,
    ))
    first = _window(0, 7_680)
    directions, _ = tracker.update(
        first.session_id, 0, first.decision_sample, (_candidate(first, 359.0),),
        window_id=0, doa_start_sample=first.doa_start_sample,
    )
    first_id = directions[0].track_id
    second = _window(1, 8_640)
    directions, _ = tracker.update(
        second.session_id, 0, second.decision_sample, (_candidate(second, 1.0),),
        window_id=1, doa_start_sample=second.doa_start_sample,
    )
    assert directions[0].track_id == first_id
    assert directions[0].track_state == "confirmed"

    # A sample jump beyond TTL expires the old track; an epoch change clears
    # motion state but must not rewind the session-scoped ID counter.
    jumped = _window(2, 12_480)
    directions, _ = tracker.update(
        jumped.session_id, 0, jumped.decision_sample, (_candidate(jumped, 1.0),),
        window_id=2, doa_start_sample=jumped.doa_start_sample,
    )
    assert directions[0].track_id > first_id
    next_epoch = _window(3, 7_680, epoch=1)
    directions, _ = tracker.update(
        next_epoch.session_id, 1, next_epoch.decision_sample, (_candidate(next_epoch, 1.0),),
        window_id=3, doa_start_sample=next_epoch.doa_start_sample,
    )
    assert directions[0].track_id > first_id + 1


def test_music_rolls_continuously_and_rebuilds_on_sample_gap() -> None:
    config = load_config("config/config.yaml", environ={})
    scan = DirectionScanConfig.from_project(config)
    scanner = RollingNormMusicScanner()
    geometry = physical_6plus1_geometry()
    first = _window(0, 7_680)
    scanner.scan_detailed(first, geometry, scan, 4)
    assert scanner.last_state_diagnostic.state == "rebuilt"
    assert scanner.last_state_diagnostic.steering_cache_rebuilt
    second = _window(1, 8_640)
    scanner.scan_detailed(second, geometry, scan, 4)
    assert scanner.last_state_diagnostic.state == "advanced"
    assert scanner.last_state_diagnostic.added_frames == 2
    jumped = _window(3, 10_560)
    scanner.scan_detailed(jumped, geometry, scan, 4)
    assert scanner.last_state_diagnostic.state == "rebuilt"
    assert scanner.last_state_diagnostic.reason == "sample_discontinuity"
    assert scanner.last_state_diagnostic.gap_samples == 960


def test_joined_result_rejects_cross_layer_id_order_and_angle_mismatch() -> None:
    window = _window(0, 7_680)
    key = WindowKey.from_window(window)
    work = WindowWorkItem(
        key, window, ProcessingConfigSnapshot(0, "hash", "geometry", "raw", {}), 1,
    )
    first = _direction(window, 1, 10.0)
    second = _direction(window, 2, 80.0)
    l2 = L2StageResult.completed(key, SimpleNamespace(directions=(first, second)), finished_monotonic_ns=2)
    l3 = L3StageResult.completed(key, SimpleNamespace(enhanced_audio=(
        SimpleNamespace(**{name: getattr(first, name) for name in (
            "session_id", "stream_epoch", "window_id", "decision_sample", "track_id", "theta_deg"
        )}),
        SimpleNamespace(**{name: getattr(second, name) for name in (
            "session_id", "stream_epoch", "window_id", "decision_sample", "track_id", "theta_deg"
        )}),
    )), finished_monotonic_ns=3)
    l5 = L5StageResult.completed(key, SimpleNamespace(detections=(
        SimpleNamespace(
            session_id=second.session_id, stream_epoch=second.stream_epoch,
            window_id=second.window_id, decision_sample=second.decision_sample,
            track_id=second.track_id, theta_deg=second.theta_deg,
        ),
        SimpleNamespace(
            session_id=first.session_id, stream_epoch=first.stream_epoch,
            window_id=first.window_id, decision_sample=first.decision_sample,
            track_id=first.track_id, theta_deg=first.theta_deg,
        ),
    )), finished_monotonic_ns=4)
    with pytest.raises(ValueError, match="identical ordered track IDs and angles"):
        JoinedWindowResult(work, l2, l3, l5, "", 5)
