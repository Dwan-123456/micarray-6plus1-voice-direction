from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from common.config import load_config
from common.data_types import CandidateDirection, DecisionWindow, ImcraHopSnapshot
from common.geometry import MIC_POSITIONS_M, physical_6plus1_geometry
from layer2_source_detection import DirectionScanConfig, Layer2Pipeline, RollingNormMusicScanner
from layer2_source_detection.global_tracker import GlobalDirectionTracker, GlobalTrackerConfig
from layer2_source_detection.pipeline import _select_l3_directions
from layer2_source_detection.probability_gate import (
    ProbabilityGateState,
    SourceProbability20ms,
    SourceProbabilityState,
)


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


def _imcra_hops(
    index: int = 0,
    *,
    noise_by_mic: np.ndarray | None = None,
    spp: float = 0.8,
    prior_snr: float = 3.0,
) -> tuple[ImcraHopSnapshot, ...]:
    frequencies = np.fft.rfftfreq(2048, 1.0 / 48_000.0).astype(np.float32)
    frequencies = frequencies[frequencies <= 8_000.0]
    shape = (7, frequencies.size)
    noise_by_mic = (
        np.ones(7, dtype=np.float32)
        if noise_by_mic is None
        else np.asarray(noise_by_mic, dtype=np.float32)
    )
    noise = np.broadcast_to(noise_by_mic[:, None], shape).copy()
    ones = np.ones(shape, dtype=np.float32)
    spp_values = np.full(shape, spp, dtype=np.float32)
    start = index * 960
    return tuple(
        ImcraHopSnapshot(
            "music", 0, start + hop * 960, start + (hop + 1) * 960, (index * 16 + hop,),
            "cohen_imcra_2003_l1_v2", "ready", frequencies,
            noise, ones * 2.0, ones * 1.5, ones * 0.5, ones * 0.4,
            spp_values, 1.0 - spp_values, ones * (prior_snr + 1.0), ones * prior_snr,
            np.ones((7, 4), dtype=np.float32),
            10.0 * np.log10(np.maximum(noise_by_mic, 1.0e-12)),
            np.full(7, spp, dtype=np.float32), spp,
        )
        for hop in range(16)
    )


def _window(
    audio: np.ndarray,
    index: int = 0,
    *,
    session: str = "music",
    epoch: int = 0,
    imcra_hops: tuple[ImcraHopSnapshot, ...] = (),
) -> DecisionWindow:
    start = index * 960
    decision = 15_360 + start
    samples = np.ascontiguousarray(audio[start:start + 15_360])
    return DecisionWindow(
        session, epoch, index, decision, decision - 1_920, decision,
        decision - 15_360, decision, 48_000, samples, (index,),
        imcra_hops,
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
    assert not scan.dpd_rank1_enabled
    assert not scan.noise_whitening_enabled


@pytest.mark.parametrize("source_count", range(7))
def test_mdl_model_order_covers_zero_through_six_sources(source_count: int) -> None:
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
    assert 0 <= diagnostics.model_order.estimated_sources <= 6
    assert len(candidates) <= 3
    assert all(
        abs(((left.theta_deg - right.theta_deg + 180.0) % 360.0) - 180.0) >= 45.0
        for index, left in enumerate(candidates) for right in candidates[index + 1:]
    )
    assert response.raw_scores.shape == (360,)


@pytest.mark.parametrize("manual_limit", (1, 2, 3))
def test_manual_music_order_limit_caps_diagnostic_mdl_without_hiding_it(
    monkeypatch, manual_limit: int,
) -> None:
    monkeypatch.setattr(
        RollingNormMusicScanner,
        "_mdl_order",
        staticmethod(lambda _eigenvalues, _snapshots: (5, 1.0)),
    )
    config = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        effective_order_limit=manual_limit,
    )
    _, _, diagnostics = RollingNormMusicScanner().scan_detailed(
        _window(_audio((73.0,), seed=47)), physical_6plus1_geometry(), config,
    )
    assert diagnostics.model_order.estimated_sources == 5
    assert diagnostics.effective_model_order == manual_limit
    assert diagnostics.mdl_saturated
    assert not diagnostics.births_allowed
    assert diagnostics.stop_reason == "mdl_saturated"


def test_optional_dpd_rank1_uses_real_frequency_support_and_manual_ceiling() -> None:
    config = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        dpd_rank1_enabled=True,
        effective_order_limit=1,
        direction_threshold=0.15,
    )
    response, candidates, diagnostics = RollingNormMusicScanner().scan_detailed(
        _window(_audio((73.0,), seed=53)), physical_6plus1_geometry(), config,
    )
    assert diagnostics.dpd_rank1_enabled
    assert diagnostics.effective_model_order == 1
    assert diagnostics.selected_frequency_bins >= config.min_valid_frequency_bins
    assert len(candidates) == 1
    assert abs(((candidates[0].theta_deg - 73.0 + 180.0) % 360.0) - 180.0) <= 2.0
    assert int(np.argmax(response.raw_scores)) in range(71, 76)
    assert diagnostics.evidence[0].supporting_frequency_bins < diagnostics.valid_frequency_bins + 1
    assert diagnostics.evidence[0].frequency_support_ratio >= config.dpd_min_frequency_support_ratio


def test_optional_dpd_rank1_separates_two_sources_across_zero_boundary() -> None:
    config = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        dpd_rank1_enabled=True,
        direction_threshold=0.15,
        effective_order_limit=2,
    )
    _, candidates, diagnostics = RollingNormMusicScanner().scan_detailed(
        _window(_audio((359.0, 90.0), seed=81), imcra_hops=_imcra_hops()),
        physical_6plus1_geometry(), config,
    )
    assert len(candidates) == 2
    angles = tuple(candidate.theta_deg for candidate in candidates)
    assert min(abs(((angle - 359.0 + 180.0) % 360.0) - 180.0) for angle in angles) <= 3.0
    assert min(abs(((angle - 90.0 + 180.0) % 360.0) - 180.0) for angle in angles) <= 3.0
    assert all(
        item.frequency_support_ratio >= config.dpd_min_frequency_support_ratio
        for item in diagnostics.evidence
    )


def test_optional_imcra_noise_psd_whitening_is_independent_and_safe() -> None:
    base = DirectionScanConfig.from_project(load_config(CONFIG, environ={}))
    config = replace(base, noise_whitening_enabled=True)
    hops = _imcra_hops(noise_by_mic=np.asarray((1, 2, 3, 4, 5, 6, 7), np.float32))
    response, candidates, diagnostics = RollingNormMusicScanner().scan_detailed(
        _window(_audio((35.0,), seed=59), imcra_hops=hops),
        physical_6plus1_geometry(), config,
    )
    assert diagnostics.noise_whitening_enabled
    assert diagnostics.whitening_status == "imcra_psd"
    assert diagnostics.imcra_noise_hops == 16
    assert np.isfinite(response.raw_scores).all()
    assert candidates


def test_diagonal_imcra_whitening_matches_generic_cholesky_reference() -> None:
    base = DirectionScanConfig.from_project(load_config(CONFIG, environ={}))
    config = replace(base, noise_whitening_enabled=True)
    scanner = RollingNormMusicScanner()
    geometry = physical_6plus1_geometry()
    window = _window(
        _audio((35.0,), seed=159),
        imcra_hops=_imcra_hops(
            noise_by_mic=np.asarray((1, 2, 3, 4, 5, 6, 7), np.float32)
        ),
    )
    steering, _ = scanner._steering_tensor(geometry, config, 0)
    frequencies = scanner._target_frequencies(config)
    rng = np.random.default_rng(160)
    matrix = rng.normal(size=(frequencies.size, 7, 7)) + 1j * rng.normal(
        size=(frequencies.size, 7, 7)
    )
    covariance = np.einsum("fij,fkj->fik", matrix, matrix.conj())

    actual_covariance, actual_steering, status = scanner._whiten(
        covariance, steering, window, config
    )
    _, _, noise_psd = scanner._imcra_metrics(window, config)
    assert noise_psd is not None
    diagonal = np.maximum(noise_psd.T, config.eigenvalue_floor)
    trace = np.mean(diagonal, axis=1)
    effective = (
        (1.0 - config.noise_covariance_shrinkage) * diagonal
        + config.noise_covariance_shrinkage * trace[:, None]
        + config.diagonal_loading
        * np.maximum(trace, config.eigenvalue_floor)[:, None]
    )
    noise = np.zeros_like(covariance)
    indices = np.arange(7)
    noise[:, indices, indices] = effective
    factor = np.linalg.cholesky(noise)
    left = np.linalg.solve(factor, covariance)
    expected_covariance = np.linalg.solve(
        factor, left.conj().transpose(0, 2, 1)
    ).conj().transpose(0, 2, 1)
    expected_covariance = 0.5 * (
        expected_covariance + expected_covariance.conj().transpose(0, 2, 1)
    )
    expected_steering = np.linalg.solve(
        factor, steering.transpose(0, 2, 1)
    ).transpose(0, 2, 1)
    expected_steering /= np.maximum(
        np.linalg.norm(expected_steering, axis=2, keepdims=True), 1.0e-12
    )

    assert status == "imcra_psd"
    np.testing.assert_allclose(actual_covariance, expected_covariance, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual_steering, expected_steering, rtol=1e-11, atol=1e-11)


def test_optional_whitening_without_ready_imcra_falls_back_without_failure() -> None:
    config = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        noise_whitening_enabled=True,
    )
    response, candidates, diagnostics = RollingNormMusicScanner().scan_detailed(
        _window(_audio((120.0,), seed=61)), physical_6plus1_geometry(), config,
    )
    assert diagnostics.whitening_status == "unavailable"
    assert diagnostics.covariance_quality == "degraded"
    assert diagnostics.fallback_reason == "imcra_noise_psd_unavailable"
    assert np.isfinite(response.raw_scores).all()
    assert candidates


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


def test_optional_dpd_and_imcra_whitening_fit_20ms_hard_budget() -> None:
    scan = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        dpd_rank1_enabled=True,
        noise_whitening_enabled=True,
        direction_threshold=0.15,
    )
    audio = _audio((80.0,), seed=71, samples=15_360 + 30 * 960)
    scanner = RollingNormMusicScanner()
    times = []
    for index in range(30):
        started = perf_counter()
        scanner.scan_detailed(
            _window(audio, index, imcra_hops=_imcra_hops(index)),
            physical_6plus1_geometry(), scan,
        )
        times.append((perf_counter() - started) * 1_000.0)
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


def test_tentative_confirmation_retries_in_a_rolling_sample_window() -> None:
    tracker = GlobalDirectionTracker(GlobalTrackerConfig(
        association_gate_deg=45.0,
        max_velocity_dps=60.0,
        confirmation_observations=2,
        confirmation_window_samples=9_600,
        coasting_ttl_samples=20_000,
        miss_cost=1.0,
        birth_cost=1.0,
    ))
    first, _ = _update(tracker, 15_360, (30.0,))
    outside_original_window, _ = _update(tracker, 25_920, (30.0,))
    confirmed, active = _update(tracker, 26_880, (30.0,))

    assert first[0].track_state == "tentative"
    assert outside_original_window[0].track_id == first[0].track_id
    assert outside_original_window[0].track_state == "tentative"
    assert confirmed[0].track_id == first[0].track_id
    assert confirmed[0].first_seen_sample == first[0].first_seen_sample
    assert confirmed[0].track_state == "confirmed"
    assert _select_l3_directions(confirmed, active) == confirmed


def test_tracker_blocks_birth_for_saturated_mdl_window() -> None:
    tracker = _tracker()
    directions, active = tracker.update(
        "track", 0, 15_360, (_candidate(15_360, 30.0),),
        window_id=16, doa_start_sample=13_440, doa_end_sample=15_360,
        allow_births=False,
    )
    assert directions == active == ()


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


def test_kalman_off_coasting_holds_last_observed_angle_without_prediction() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (10.0,), kalman_enabled=False)
    second, _ = _update(tracker, 16_320, (20.0,), kalman_enabled=False)
    _, active = _update(tracker, 18_240, (), kalman_enabled=False)
    assert first[0].track_id == second[0].track_id == active[0].track_id
    assert active[0].track_state == "coasting"
    assert active[0].theta_deg == pytest.approx(20.0)
    assert not active[0].kalman_applied


def test_kalman_off_zero_order_hold_is_circular_and_switch_does_not_change_id() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (359.0,), kalman_enabled=True)
    second, _ = _update(tracker, 16_320, (1.0,), kalman_enabled=True)
    _, predicted = _update(tracker, 17_280, (), kalman_enabled=True)
    _, held = _update(tracker, 18_240, (), kalman_enabled=False)
    assert first[0].track_id == second[0].track_id == predicted[0].track_id == held[0].track_id
    assert held[0].theta_deg == pytest.approx(1.0)
    assert not held[0].kalman_applied


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


def test_live_id_forces_closed_probability_gate_open_for_three_second_ttl() -> None:
    config = load_config(CONFIG, environ={})
    pipeline = Layer2Pipeline.from_project(config)
    audio = _audio((30.0,), seed=29, samples=15_360 + 960)

    def probabilities(window: DecisionWindow, value: float) -> tuple[SourceProbability20ms, ...]:
        return tuple(SourceProbability20ms(
            window.session_id, window.stream_epoch, start, start + 960, value,
            SourceProbabilityState.READY, "ready",
        ) for start in (window.doa_start_sample, window.doa_start_sample + 960))

    first_window = _window(audio, 0)
    first = pipeline.process(
        first_window, probabilities(first_window, 1.0), physical_6plus1_geometry(),
        DirectionScanConfig.from_project(config), gate_threshold=0.6,
        gate_config_revision=0,
    )
    assert first.active_tracks
    assert first.active_tracks[0].track_state == "tentative"
    assert first.directions == ()

    second_window = _window(audio, 1)
    forced = pipeline.process(
        second_window, probabilities(second_window, 0.0), physical_6plus1_geometry(),
        DirectionScanConfig.from_project(config), gate_threshold=0.6,
        gate_config_revision=0,
    )
    assert forced.gate_decision.state is ProbabilityGateState.OPEN
    assert forced.gate_decision.probability_40ms == 0.0
    assert forced.gate_decision.reason == "active_id_force_open"
    assert forced.spatial_response is not None

    expired_decision = second_window.decision_sample + 3 * 48_000 + 960
    expired_window = DecisionWindow(
        second_window.session_id, second_window.stream_epoch, 999, expired_decision,
        expired_decision - 1_920, expired_decision,
        expired_decision - 15_360, expired_decision, 48_000,
        np.zeros((15_360, 8), dtype=np.float32), (999,),
    )
    restored = pipeline.process(
        expired_window, probabilities(expired_window, 0.0), physical_6plus1_geometry(),
        DirectionScanConfig.from_project(config), gate_threshold=0.6,
        gate_config_revision=0,
    )
    assert restored.gate_decision.state is ProbabilityGateState.CLOSED
    assert restored.gate_decision.reason == "probability_below_threshold"
    assert restored.spatial_response is None and restored.active_tracks == ()


def test_three_seconds_without_voice_marks_nonexclusive_noise_track() -> None:
    tracker = GlobalDirectionTracker(GlobalTrackerConfig(
        association_gate_deg=45.0, max_velocity_dps=60.0,
        confirmation_observations=2, confirmation_window_samples=9_600,
        coasting_ttl_samples=200_000, miss_cost=1.0, birth_cost=1.0,
    ))
    first_sample = 15_360
    first, _ = _update(tracker, first_sample, (50.0,))
    old_id = first[0].track_id
    for sample in range(first_sample + 960, first_sample + 3 * 48_000, 960):
        observed, _ = _update(tracker, sample, (50.0,))
        assert observed[0].track_id == old_id

    boundary = first_sample + 3 * 48_000
    observed, active = _update(tracker, boundary, (52.0,))
    assert observed[0].track_id == old_id
    noise = next(item for item in active if item.track_id == old_id)
    assert noise.is_noise_interference
    assert abs(noise.theta_deg - 52.0) < 1e-6

    for offset in range(5):
        assert tracker.apply_voice_feedback(
            "track", 0, boundary + offset * 960, old_id, 0.95, True
        )
        if offset == 1:
            assert tracker.apply_voice_feedback(
                "track", 0, boundary + offset * 960 + 1, old_id, 0.05, False
            )
    assert not tracker._tracks[old_id].noise_interference


def test_normal_track_moving_near_noise_track_is_not_merged_into_it() -> None:
    tracker = GlobalDirectionTracker(GlobalTrackerConfig(
        association_gate_deg=45.0, max_velocity_dps=60.0,
        confirmation_observations=2, confirmation_window_samples=9_600,
        coasting_ttl_samples=200_000, miss_cost=1.0, birth_cost=1.0,
    ))
    first_sample = 15_360
    noise, _ = _update(tracker, first_sample, (50.0,))
    noise_id = noise[0].track_id
    tracker._tracks[noise_id].noise_interference = True

    moving, _ = _update(tracker, first_sample + 960, (120.0,))
    moving_id = moving[0].track_id
    assert moving_id != noise_id
    moved, active = _update(tracker, first_sample + 1_920, (85.0,))
    assert moved[0].track_id == moving_id
    assert any(item.track_id == noise_id and item.is_noise_interference for item in active)
    for offset in range(5):
        assert tracker.apply_voice_feedback(
            "track", 0, first_sample + 1_920 + offset, noise_id, 0.95, True
        )
    assert tracker._tracks[noise_id].noise_interference


def test_internal_tracker_never_exceeds_four_active_ids() -> None:
    tracker = GlobalDirectionTracker(GlobalTrackerConfig(
        association_gate_deg=45.0, max_velocity_dps=60.0,
        confirmation_observations=2, confirmation_window_samples=9_600,
        coasting_ttl_samples=200_000, miss_cost=1.0, birth_cost=1.0,
    ))
    first, _ = _update(tracker, 15_360, (0.0, 120.0, 240.0))
    second, active = _update(tracker, 16_320, (60.0, 180.0, 300.0))
    assert len(first) == len(second) == 3
    assert len(active) == tracker.active_track_count == 4
    assert len(set(tracker.active_track_ids)) == 4
