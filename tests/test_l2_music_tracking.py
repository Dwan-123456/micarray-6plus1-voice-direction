from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from common.config import load_config
from common.data_types import CandidateDirection, DecisionWindow
from common.geometry import MIC_POSITIONS_M, physical_6plus1_geometry
from layer2_source_detection import DirectionScanConfig, Layer2Pipeline, RollingNormMusicScanner
from layer2_source_detection.global_tracker import GlobalDirectionTracker, GlobalTrackerConfig
from layer2_source_detection.pipeline import _select_l3_directions
from layer2_source_detection.probability_gate import SourceProbability20ms, SourceProbabilityState


CONFIG = Path(__file__).parents[1] / "config" / "config.yaml"


def _audio(directions: tuple[float, ...], *, seed: int = 1, samples: int = 16_320) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frequencies = np.fft.rfftfreq(samples, 1.0 / 48_000.0)
    band = (frequencies >= 2_000.0) & (frequencies <= 4_000.0)
    output = np.zeros((samples, 8), dtype=np.float64)
    for theta in directions:
        source = np.zeros(frequencies.size, dtype=np.complex128)
        source[band] = rng.normal(size=band.sum()) + 1j * rng.normal(size=band.sum())
        unit = np.array((np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))))
        delay = -(MIC_POSITIONS_M @ unit) / 343.0
        for mic in range(7):
            spectrum = source * np.exp(-2j * np.pi * frequencies * delay[mic])
            output[:, mic] += np.fft.irfft(spectrum, n=samples)
    peak = np.max(np.abs(output[:, :7]))
    if peak:
        output[:, :7] /= peak
    return np.asarray(output, dtype=np.float32)


def _window(audio: np.ndarray, index: int = 0, *, session: str = "music", epoch: int = 0) -> DecisionWindow:
    start = index * 960
    decision = 15_360 + start
    samples = np.ascontiguousarray(audio[start:start + 15_360])
    return DecisionWindow(
        session, epoch, index, decision, decision - 1_920, decision,
        decision - 15_360, decision, 48_000, samples, (index,),
    )


def _candidate(sample: int, theta: float, rank: int = 1) -> CandidateDirection:
    return CandidateDirection("track", 0, sample // 960, sample, sample - 1_920, sample, theta, 1.0, 0.9)


def _tracker() -> GlobalDirectionTracker:
    return GlobalDirectionTracker(GlobalTrackerConfig(
        association_gate_deg=45.0, max_velocity_dps=60.0,
        confirmation_observations=2, confirmation_window_samples=9_600,
        coasting_ttl_samples=4_800, miss_cost=1.0, birth_cost=1.0,
    ))


def _update(tracker: GlobalDirectionTracker, sample: int, angles: tuple[float, ...], **kwargs):
    candidates = tuple(_candidate(sample, angle, rank) for rank, angle in enumerate(angles, 1))
    return tracker.update(
        "track", 0, sample, candidates, window_id=sample // 960,
        doa_start_sample=sample - 1_920, doa_end_sample=sample, **kwargs,
    )


def test_music_configuration_and_hardware_mix_contract() -> None:
    config = load_config(CONFIG, environ={})
    scan = DirectionScanConfig.from_project(config)
    assert config.layer2.scanner_backend == "frequency_normalized_music"
    assert (scan.n_fft, scan.win_length, scan.hop_length) == (1024, 960, 480)
    assert scan.context_ms in {160, 240, 320}
    assert (scan.frequency_min_hz, scan.frequency_max_hz) == (2_000.0, 4_000.0)
    assert scan.max_candidates == 3 and scan.min_peak_distance_deg == 45.0


@pytest.mark.parametrize("source_count", range(4))
def test_mdl_model_order_covers_zero_through_three_sources(source_count: int) -> None:
    eigenvalues = np.r_[
        np.ones(7 - source_count),
        np.arange(1, source_count + 1, dtype=np.float64) * 20.0 + 20.0,
    ]
    order, consistency = RollingNormMusicScanner._mdl_order(
        np.tile(eigenvalues, (100, 1)), snapshots=31
    )
    assert (order, consistency) == (source_count, 1.0)


@pytest.mark.parametrize("context_ms, expected_frames", ((160, 15), (240, 23), (320, 31)))
def test_music_history_candidates_preserve_direction(context_ms: int, expected_frames: int) -> None:
    scan = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        context_ms=context_ms,
    )
    response, _, diagnostics = RollingNormMusicScanner().scan_detailed(
        _window(_audio((73.0,), seed=31)), physical_6plus1_geometry(), scan,
    )
    assert diagnostics.state.added_frames == expected_frames
    assert abs(((int(np.argmax(response.raw_scores)) - 73 + 180) % 360) - 180) <= 2


@pytest.mark.parametrize("theta", (0.0, 1.0, 73.0, 180.0, 359.0))
def test_music_single_source_direction_and_no_mirror(theta: float) -> None:
    scan = DirectionScanConfig.from_project(load_config(CONFIG, environ={}))
    response, candidates, diagnostics = RollingNormMusicScanner().scan_detailed(
        _window(_audio((theta,), seed=7)), physical_6plus1_geometry(), scan,
    )
    error = abs(((float(np.argmax(response.raw_scores)) - theta + 180.0) % 360.0) - 180.0)
    assert error <= 2.0
    assert diagnostics.model_order.estimated_sources == 1
    assert candidates and abs(((candidates[0].theta_deg - theta + 180.0) % 360.0) - 180.0) <= 2.0


def test_music_hardware_mix_is_excluded_and_incremental_update_is_two_frames() -> None:
    config = DirectionScanConfig.from_project(load_config(CONFIG, environ={}))
    audio = _audio((45.0,), seed=11)
    scanner = RollingNormMusicScanner()
    first, _, first_diag = scanner.scan_detailed(_window(audio, 0), physical_6plus1_geometry(), config)
    changed = audio.copy()
    changed[:, 7] = np.linspace(-100.0, 100.0, changed.shape[0], dtype=np.float32)
    second, _, second_diag = scanner.scan_detailed(_window(changed, 1), physical_6plus1_geometry(), config)
    fresh, _, _ = RollingNormMusicScanner().scan_detailed(_window(changed, 1), physical_6plus1_geometry(), config)
    assert first_diag.state.added_frames in {15, 23, 31}
    assert (second_diag.state.added_frames, second_diag.state.removed_frames) == (2, 2)
    np.testing.assert_allclose(second.raw_scores, fresh.raw_scores, rtol=1e-5, atol=1e-6)


def test_music_two_sources_are_45_degree_nms_separated() -> None:
    config = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        direction_threshold=0.15, min_cross_frequency_consistency=0.0,
    )
    response, candidates, diagnostics = RollingNormMusicScanner().scan_detailed(
        _window(_audio((40.0, 170.0), seed=13)), physical_6plus1_geometry(), config,
    )
    assert 0 <= diagnostics.model_order.estimated_sources <= 3
    assert len(candidates) <= 3
    assert all(
        abs(((left.theta_deg - right.theta_deg + 180.0) % 360.0) - 180.0) >= 45.0
        for index, left in enumerate(candidates) for right in candidates[index + 1:]
    )
    assert response.raw_scores.shape == (360,)


def test_music_rolling_p95_is_below_hard_20ms_budget() -> None:
    scan = DirectionScanConfig.from_project(load_config(CONFIG, environ={}))
    audio = _audio((80.0,), seed=17, samples=15_360 + 30 * 960)
    scanner = RollingNormMusicScanner()
    times = []
    for index in range(30):
        started = perf_counter()
        scanner.scan_detailed(_window(audio, index), physical_6plus1_geometry(), scan)
        times.append((perf_counter() - started) * 1000.0)
    assert np.percentile(times[2:], 95) <= 15.0
    assert max(times[2:]) < 20.0


def test_global_assignment_crosses_zero_and_survives_rank_swap() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (358.0, 120.0))
    ids = {round(item.theta_deg): item.track_id for item in first}
    second, _ = _update(tracker, 16_320, (121.0, 359.0))
    third, _ = _update(tracker, 17_280, (1.0, 122.0))
    assert {item.track_id for item in second} == set(ids.values())
    near_zero = min(third, key=lambda item: abs(((item.theta_deg + 180.0) % 360.0) - 180.0))
    assert near_zero.track_id == ids[358]
    assert near_zero.track_state == "confirmed"


def test_birth_coast_reacquire_ttl_and_session_scoped_monotonic_ids() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (30.0,))
    first_id = first[0].track_id
    confirmed, _ = _update(tracker, 16_320, (30.5,))
    assert confirmed[0].track_id == first_id
    _, active = _update(tracker, 17_280, ())
    assert active[0].track_id == first_id and active[0].track_state == "coasting"
    recovered, _ = _update(tracker, 18_240, (31.0,))
    assert recovered[0].track_id == first_id
    _update(tracker, 24_000, ())
    replacement, _ = _update(tracker, 24_960, (31.0,))
    assert replacement[0].track_id > first_id
    epoch_track = tracker.update(
        "track", 1, 25_920, (_candidate(25_920, 31.0),),
        window_id=27, doa_start_sample=24_000, doa_end_sample=25_920,
    )[0][0]
    assert epoch_track.track_id > replacement[0].track_id


def test_kalman_toggle_changes_only_angle_not_id_or_lifecycle() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (50.0,), kalman_enabled=False)
    second, _ = _update(tracker, 16_320, (55.0,), kalman_enabled=True, q_scale=0.2, r_scale=2.0)
    third, _ = _update(tracker, 17_280, (54.0,), kalman_enabled=False)
    assert first[0].track_id == second[0].track_id == third[0].track_id
    assert second[0].kalman_applied and not third[0].kalman_applied
    assert third[0].track_state == "confirmed"


def test_confirmed_coasting_id_is_selected_as_an_l3_bf_target() -> None:
    tracker = _tracker()
    _update(tracker, 15_360, (20.0,), kalman_enabled=True)
    confirmed, _ = _update(tracker, 16_320, (22.0,), kalman_enabled=True)
    observed, active = _update(tracker, 17_280, (), kalman_enabled=True)

    selected = _select_l3_directions(observed, active)

    assert len(selected) == 1
    assert selected[0].track_id == confirmed[0].track_id
    assert selected[0].track_state == "coasting"
    assert not selected[0].is_observed


def test_tentative_missing_id_is_not_selected_as_an_l3_bf_target() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (20.0,), kalman_enabled=True)
    observed, active = _update(tracker, 16_320, (), kalman_enabled=True)

    assert first[0].track_state == "tentative"
    assert active[0].track_state == "tentative"
    assert _select_l3_directions(observed, active) == ()


def test_pipeline_gate_closed_advances_track_to_coasting_without_music_observation() -> None:
    config = load_config(CONFIG, environ={})
    pipeline = Layer2Pipeline.from_project(config)
    window = _window(_audio((30.0,), seed=23))
    probabilities = tuple(SourceProbability20ms(
        window.session_id, window.stream_epoch, start, start + 960, 0.1,
        SourceProbabilityState.READY, "ready",
    ) for start in (window.doa_start_sample, window.doa_start_sample + 960))
    result = pipeline.process(
        window, probabilities, physical_6plus1_geometry(), DirectionScanConfig.from_project(config),
        gate_threshold=0.6, gate_config_revision=0,
    )
    assert result.spatial_response is None and result.directions == ()
