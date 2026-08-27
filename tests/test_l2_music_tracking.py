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
from layer2_source_detection.music import _DpdVoteCluster
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
    frequencies = frequencies[frequencies <= 10_000.0]
    shape = (7, frequencies.size)
    noise_by_mic = (
        np.ones(7, dtype=np.float32)
        if noise_by_mic is None
        else np.asarray(noise_by_mic, dtype=np.float32)
    )
    noise = np.broadcast_to(noise_by_mic[:, None], shape).copy()
    noise_covariance = np.zeros((frequencies.size, 7, 7), dtype=np.complex64)
    diagonal_indices = np.arange(7)
    noise_covariance[:, diagonal_indices, diagonal_indices] = noise.T
    cross_power = np.float32(0.2 * np.sqrt(noise_by_mic[0] * noise_by_mic[1]))
    noise_covariance[:, 0, 1] = 1j * cross_power
    noise_covariance[:, 1, 0] = -1j * cross_power
    ones = np.ones(shape, dtype=np.float32)
    spp_values = np.full(shape, spp, dtype=np.float32)
    start = index * 960
    return tuple(
        ImcraHopSnapshot(
            "music", 0, start + hop * 960, start + (hop + 1) * 960, (index * 8 + hop,),
            "cohen_imcra_2003_l1_v8", "ready", frequencies,
            noise, ones * 2.0, ones * 1.5, ones * 0.5, ones * 0.4,
            spp_values, 1.0 - spp_values, ones * (prior_snr + 1.0), ones * prior_snr,
            np.ones((7, 4), dtype=np.float32),
            10.0 * np.log10(np.maximum(noise_by_mic, 1.0e-12)),
            np.full(7, spp, dtype=np.float32), spp, noise_covariance,
        )
        for hop in range(8)
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
    decision = 7_680 + start
    samples = np.ascontiguousarray(audio[start:start + 7_680])
    return DecisionWindow(
        session, epoch, index, decision, decision - 1_920, decision,
        decision - 7_680, decision, 48_000, samples, (index,),
        imcra_hops,
    )


def _candidate(sample: int, theta: float, rank: int = 1) -> CandidateDirection:
    return CandidateDirection("track", 0, sample // 960, sample, sample - 1_920, sample, theta, 1.0, 0.9)


def _tracker() -> GlobalDirectionTracker:
    return GlobalDirectionTracker(GlobalTrackerConfig(
        association_gate_deg=45.0, max_velocity_dps=60.0,
        confirmation_observations=3, confirmation_window_samples=9_600,
        tentative_ttl_samples=24_000, coasting_ttl_samples=96_000,
    ))


def _update(tracker: GlobalDirectionTracker, sample: int, angles: tuple[float, ...], **kwargs):
    candidates = tuple(_candidate(sample, angle, rank) for rank, angle in enumerate(angles, 1))
    return tracker.update(
        "track", 0, sample, candidates, window_id=sample // 960,
        doa_start_sample=sample - 1_920, doa_end_sample=sample, **kwargs,
    )


def _circular_error_deg(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def test_music_configuration_and_hardware_mix_contract() -> None:
    config = load_config(CONFIG, environ={})
    scan = DirectionScanConfig.from_project(config)
    assert config.layer2.scanner_backend == "frequency_normalized_music"
    assert (scan.n_fft, scan.win_length, scan.hop_length) == (1024, 960, 480)
    assert scan.context_ms in {160, 200, 240, 320}
    assert (scan.frequency_min_hz, scan.frequency_max_hz) == (2_000.0, 4_000.0)
    assert scan.max_candidates == 3 and scan.min_peak_distance_deg == 50.0
    assert not scan.dpd_rank1_enabled
    assert not scan.noise_whitening_enabled


def test_music_fixed_geometry_frequency_weights_match_2_to_4khz_contract() -> None:
    frequencies = np.asarray(
        (2_000, 2_299, 2_300, 2_500, 2_700, 3_000, 3_600, 3_700, 3_800, 3_900, 4_000),
        dtype=np.float64,
    )

    weights = RollingNormMusicScanner._geometry_frequency_weights(frequencies)

    np.testing.assert_allclose(
        weights,
        (0.35, 0.35, 0.55, 0.75, 0.90, 1.00, 1.00, 0.875, 0.75, 0.60, 0.45),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_music_cross_frequency_fusion_uses_fixed_geometry_weights() -> None:
    frequencies = np.asarray((2_100.0, 3_300.0), dtype=np.float64)
    per_frequency = np.zeros((2, 360), dtype=np.float64)
    per_frequency[0, 20] = 1.0
    per_frequency[1, 140] = 1.0

    fused = RollingNormMusicScanner._geometry_weighted_mean(
        per_frequency, frequencies,
    )

    assert fused[140] > fused[20]
    np.testing.assert_allclose(fused[[20, 140]], (0.35 / 1.35, 1.0 / 1.35))


@pytest.mark.parametrize("context_ms, expected_frames", ((160, 15), (200, 19), (240, 23), (320, 31)))
def test_music_history_candidates_preserve_direction(context_ms: int, expected_frames: int) -> None:
    scan = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        context_ms=context_ms,
    )
    audio = _audio((73.0,), seed=31)
    scanner = RollingNormMusicScanner()
    response = None
    for index in range(1 + (context_ms - 160) // 20):
        response, _, _ = scanner.scan_detailed(
            _window(audio, index), physical_6plus1_geometry(), scan,
        )
    assert response is not None
    assert len(scanner._frame_covariances) == expected_frames
    assert abs(((int(np.argmax(response.raw_scores)) - 73 + 180) % 360) - 180) <= 2


@pytest.mark.parametrize("theta", (0.0, 1.0, 73.0, 180.0, 359.0))
def test_music_single_source_direction_and_no_mirror(theta: float) -> None:
    scan = DirectionScanConfig.from_project(load_config(CONFIG, environ={}))
    response, candidates, diagnostics = RollingNormMusicScanner().scan_detailed(
        _window(_audio((theta,), seed=7)), physical_6plus1_geometry(), scan,
    )
    error = abs(((float(np.argmax(response.raw_scores)) - theta + 180.0) % 360.0) - 180.0)
    assert error <= 2.0
    assert diagnostics.model_order.estimated_sources == scan.effective_order_limit
    assert diagnostics.model_order.mdl_age_samples == 0
    assert candidates and abs(((candidates[0].theta_deg - theta + 180.0) % 360.0) - 180.0) <= 2.0


def test_music_hardware_mix_is_excluded_and_incremental_update_is_two_frames() -> None:
    config = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        context_ms=160,
    )
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


def test_music_two_sources_are_50_degree_nms_separated() -> None:
    config = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        direction_threshold=0.15,
    )
    response, candidates, diagnostics = RollingNormMusicScanner().scan_detailed(
        _window(_audio((40.0, 170.0), seed=13)), physical_6plus1_geometry(), config,
    )
    assert diagnostics.model_order.estimated_sources == config.effective_order_limit
    assert len(candidates) <= 3
    assert all(
        abs(((left.theta_deg - right.theta_deg + 180.0) % 360.0) - 180.0) >= 50.0
        for index, left in enumerate(candidates) for right in candidates[index + 1:]
    )
    assert response.raw_scores.shape == (360,)


@pytest.mark.parametrize("manual_limit", (1, 2, 3))
def test_manual_music_order_directly_controls_subspace_and_peak_search(manual_limit: int) -> None:
    config = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        effective_order_limit=manual_limit,
    )
    _, _, diagnostics = RollingNormMusicScanner().scan_detailed(
        _window(_audio((73.0,), seed=47)), physical_6plus1_geometry(), config,
    )
    assert diagnostics.model_order.estimated_sources == manual_limit
    assert diagnostics.effective_model_order == manual_limit
    assert diagnostics.births_allowed
    assert diagnostics.stop_reason == "manual_order_greedy_peak_search"


def test_manual_music_order_two_finds_two_peaks() -> None:
    config = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        effective_order_limit=2,
        direction_threshold=0.15,
    )
    _, candidates, diagnostics = RollingNormMusicScanner().scan_detailed(
        _window(_audio((30.0, 210.0), seed=49)), physical_6plus1_geometry(), config,
    )
    assert diagnostics.model_order.estimated_sources == 2
    assert diagnostics.effective_model_order == 2
    assert len(candidates) == 2
    assert all(
        abs(((left.theta_deg - right.theta_deg + 180.0) % 360.0) - 180.0) >= 50.0
        for index, left in enumerate(candidates) for right in candidates[index + 1:]
    )


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
    assert diagnostics.evidence[0].supporting_frequency_bins >= config.dpd_min_cluster_frequency_bins
    assert diagnostics.evidence[0].supporting_frequency_subbands >= config.dpd_min_cluster_subbands
    assert diagnostics.evidence[0].circular_concentration >= config.dpd_min_circular_concentration


def test_optional_dpd_rank1_separates_two_sources_across_zero_boundary() -> None:
    config = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        dpd_rank1_enabled=True,
        noise_whitening_enabled=False,
        direction_threshold=0.15,
        effective_order_limit=2,
    )
    _, candidates, diagnostics = RollingNormMusicScanner().scan_detailed(
        _window(_audio((359.0, 90.0), seed=81), imcra_hops=_imcra_hops()),
        physical_6plus1_geometry(), config,
    )
    assert len(candidates) == 2
    angles = tuple(candidate.theta_deg for candidate in candidates)
    assert min(abs(((angle - 359.0 + 180.0) % 360.0) - 180.0) for angle in angles) <= 4.0
    assert min(abs(((angle - 90.0 + 180.0) % 360.0) - 180.0) for angle in angles) <= 3.0
    assert all(
        item.frequency_support_ratio >= config.dpd_min_frequency_support_ratio
        for item in diagnostics.evidence
    )


def test_dpd_vote_clustering_wraps_zero_and_rejects_narrowband_cluster() -> None:
    config = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        dpd_rank1_enabled=True,
        effective_order_limit=3,
    )
    scanner = RollingNormMusicScanner()
    peak_angles = np.asarray((358, 359, 0, 1, 2, 0), dtype=int)
    per_frequency = np.zeros((len(peak_angles), 360), dtype=np.float64)
    per_frequency[np.arange(len(peak_angles)), peak_angles] = 1.0
    selected = np.ones(len(peak_angles), dtype=bool)
    weights = np.ones(len(peak_angles), dtype=np.float64)
    plane_fit = np.full(len(peak_angles), 0.9, dtype=np.float64)
    grid = np.arange(360, dtype=np.float64)
    distance = np.abs((grid + 180.0) % 360.0 - 180.0)
    vote = np.maximum(0.0, 1.0 - distance / (config.dpd_angle_tolerance_deg + 1.0))

    narrowband = np.linspace(2_050.0, 2_200.0, len(peak_angles))
    assert scanner._dpd_vote_clusters(
        vote, per_frequency, selected, weights, plane_fit, narrowband, config
    ) == ()

    broadband = np.asarray((2_050.0, 2_200.0, 2_650.0, 2_850.0, 3_250.0, 3_750.0))
    clusters = scanner._dpd_vote_clusters(
        vote, per_frequency, selected, weights, plane_fit, broadband, config
    )
    assert len(clusters) == 1
    assert clusters[0].angle_index == 0
    assert clusters[0].supporting_frequency_bins == len(peak_angles)
    assert clusters[0].supporting_frequency_subbands >= config.dpd_min_cluster_subbands
    assert clusters[0].circular_concentration >= config.dpd_min_circular_concentration


def test_dpd_strong_nearby_peaks_fuse_with_unique_frequency_weight() -> None:
    config = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        dpd_rank1_enabled=True,
        direction_threshold=0.20,
    )
    scanner = RollingNormMusicScanner()
    normalized = np.zeros(360, dtype=np.float64)
    normalized[[20, 25, 30]] = (0.80, 0.50, 0.90)
    peak_angles = np.asarray((20, 20, 20, 30, 30, 30), dtype=int)
    per_frequency = np.zeros((peak_angles.size, 360), dtype=np.float64)
    per_frequency[np.arange(peak_angles.size), peak_angles] = 1.0
    selected = np.ones(peak_angles.size, dtype=bool)
    weights = np.ones(peak_angles.size, dtype=np.float64)
    plane_fit = np.full(peak_angles.size, 0.9, dtype=np.float64)
    frequencies = np.asarray((2_050, 2_250, 2_650, 2_850, 3_250, 3_750), dtype=np.float64)
    clusters = (
        _DpdVoteCluster(20, 4, 4 / 6, 0.9, 2, 1.0, 4.0, (0, 1, 2, 3)),
        _DpdVoteCluster(30, 4, 4 / 6, 0.9, 3, 1.0, 4.0, (2, 3, 4, 5)),
    )

    fused = scanner._merge_dpd_peak_clusters(
        clusters, normalized, per_frequency, selected, weights, plane_fit,
        frequencies, config,
    )

    assert len(fused) == 1
    assert fused[0].angle_index == 25
    assert fused[0].supporting_frequency_bins == 6
    assert fused[0].cluster_weight == pytest.approx(6.0)
    assert fused[0].supporting_frequency_indices == tuple(range(6))


def test_dpd_peak_fusion_requires_strictly_above_point_seven_and_avoids_chaining() -> None:
    config = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        dpd_rank1_enabled=True,
        direction_threshold=0.20,
        dpd_min_frequency_support_ratio=0.10,
        dpd_min_cluster_frequency_bins=2,
        dpd_min_circular_concentration=0.90,
    )
    scanner = RollingNormMusicScanner()
    selected = np.ones(6, dtype=bool)
    weights = np.ones(6, dtype=np.float64)
    plane_fit = np.full(6, 0.9, dtype=np.float64)
    frequencies = np.asarray((2_050, 2_250, 2_650, 2_850, 3_250, 3_750), dtype=np.float64)
    peak_angles = np.asarray((10, 10, 50, 50, 90, 90), dtype=int)
    per_frequency = np.zeros((6, 360), dtype=np.float64)
    per_frequency[np.arange(6), peak_angles] = 1.0
    clusters = (
        _DpdVoteCluster(10, 2, 1 / 3, 0.9, 2, 1.0, 2.0, (0, 1)),
        _DpdVoteCluster(50, 2, 1 / 3, 0.9, 2, 1.0, 2.0, (2, 3)),
        _DpdVoteCluster(90, 2, 1 / 3, 0.9, 2, 1.0, 2.0, (4, 5)),
    )
    normalized = np.full(360, 0.50, dtype=np.float64)
    normalized[[10, 50, 90]] = 0.90

    fused = scanner._merge_dpd_peak_clusters(
        clusters, normalized, per_frequency, selected, weights, plane_fit,
        frequencies, config,
    )
    assert len(fused) == 2
    assert tuple(item.angle_index for item in fused) == (30, 90)

    normalized[10] = 0.70
    not_fused = scanner._merge_dpd_peak_clusters(
        clusters[:2], normalized, per_frequency, selected, weights, plane_fit,
        frequencies, config,
    )
    assert len(not_fused) == 2
    assert {item.angle_index for item in not_fused} == {10, 50}


def test_dpd_peak_fusion_wraps_zero_degrees() -> None:
    config = replace(
        DirectionScanConfig.from_project(load_config(CONFIG, environ={})),
        dpd_rank1_enabled=True,
        direction_threshold=0.20,
        dpd_min_frequency_support_ratio=0.10,
        dpd_min_cluster_frequency_bins=2,
    )
    scanner = RollingNormMusicScanner()
    normalized = np.full(360, 0.50, dtype=np.float64)
    normalized[[350, 10]] = 0.90
    peak_angles = np.asarray((350, 350, 350, 10, 10, 10), dtype=int)
    per_frequency = np.zeros((6, 360), dtype=np.float64)
    per_frequency[np.arange(6), peak_angles] = 1.0
    clusters = (
        _DpdVoteCluster(350, 3, 0.5, 0.9, 2, 1.0, 3.0, (0, 1, 2)),
        _DpdVoteCluster(10, 3, 0.5, 0.9, 2, 1.0, 3.0, (3, 4, 5)),
    )
    fused = scanner._merge_dpd_peak_clusters(
        clusters, normalized, per_frequency, np.ones(6, dtype=bool),
        np.ones(6), np.full(6, 0.9),
        np.asarray((2_050, 2_250, 2_650, 2_850, 3_250, 3_750), dtype=np.float64),
        config,
    )
    assert len(fused) == 1
    assert fused[0].angle_index == 0


def test_optional_imcra_spatial_covariance_whitening_is_independent_and_safe() -> None:
    base = DirectionScanConfig.from_project(load_config(CONFIG, environ={}))
    config = replace(base, noise_whitening_enabled=True)
    hops = _imcra_hops(noise_by_mic=np.asarray((1, 2, 3, 4, 5, 6, 7), np.float32))
    response, candidates, diagnostics = RollingNormMusicScanner().scan_detailed(
        _window(_audio((35.0,), seed=59), imcra_hops=hops),
        physical_6plus1_geometry(), config,
    )
    assert diagnostics.noise_whitening_enabled
    assert diagnostics.whitening_status == "imcra_spatial_covariance"
    assert diagnostics.imcra_noise_hops == 8
    assert np.isfinite(response.raw_scores).all()
    assert candidates


def test_imcra_complex_cross_spectrum_survives_l2_frequency_interpolation() -> None:
    config = DirectionScanConfig.from_project(load_config(CONFIG, environ={}))
    scanner = RollingNormMusicScanner()
    window = _window(_audio((35.0,), seed=158), imcra_hops=_imcra_hops())

    _, _, covariance = scanner._imcra_metrics(window, config)

    assert covariance is not None
    assert np.max(np.abs(np.imag(covariance[:, 0, 1]))) > 0.0
    np.testing.assert_allclose(
        covariance,
        covariance.conj().transpose(0, 2, 1),
        rtol=2e-5,
        atol=1e-7,
    )


def test_imcra_spatial_covariance_whitening_matches_cholesky_reference() -> None:
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
    _, _, noise_covariance = scanner._imcra_metrics(window, config)
    assert noise_covariance is not None
    trace = np.maximum(
        np.real(np.trace(noise_covariance, axis1=1, axis2=2)) / 7.0,
        config.eigenvalue_floor,
    )
    identity = np.eye(7, dtype=np.complex128)[None, :, :]
    effective = (
        (1.0 - config.noise_covariance_shrinkage) * noise_covariance
        + config.noise_covariance_shrinkage * trace[:, None, None] * identity
        + config.diagonal_loading * trace[:, None, None] * identity
    )
    factor = np.linalg.cholesky(effective)
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

    assert status == "imcra_spatial_covariance"
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
    assert diagnostics.fallback_reason == "imcra_noise_covariance_unavailable"
    assert np.isfinite(response.raw_scores).all()
    assert candidates


def test_music_rolling_p95_is_below_hard_20ms_budget() -> None:
    scan = DirectionScanConfig.from_project(load_config(CONFIG, environ={}))
    audio = _audio((80.0,), seed=17, samples=7_680 + 30 * 960)
    scanner = RollingNormMusicScanner()
    times = []
    for index in range(30):
        started = perf_counter()
        scanner.scan_detailed(_window(audio, index), physical_6plus1_geometry(), scan)
        times.append((perf_counter() - started) * 1000.0)
    assert np.percentile(times[2:], 95) <= 15.0
    assert max(times[2:]) < 20.0


def test_circular_imm_jpda_crosses_zero_and_survives_rank_swap() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (358.0, 120.0))
    ids = {round(item.measured_theta_deg): item.track_id for item in first}
    second, _ = _update(tracker, 16_320, (121.0, 359.0))
    third, _ = _update(tracker, 17_280, (1.0, 122.0))
    assert {item.track_id for item in second} == set(ids.values())
    assert min(third, key=lambda item: _circular_error_deg(item.theta_deg, 0)).track_id == ids[358]
    assert all(item.track_state == "confirmed" for item in third)


def test_imm_jpda_confirmation_coasting_reacquisition_and_ttl() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (30.0,))
    _update(tracker, 16_320, (30.5,))
    confirmed, _ = _update(tracker, 17_280, (31.0,))
    track_id = first[0].track_id
    assert confirmed[0].track_id == track_id and confirmed[0].track_state == "confirmed"
    _, coasting = _update(tracker, 18_240, ())
    assert coasting[0].track_id == track_id and coasting[0].kalman_applied
    recovered, _ = _update(tracker, 19_200, (32.0,))
    assert recovered[0].track_id == track_id
    _, expired = _update(tracker, 19_200 + 96_001, ())
    assert expired == ()


def test_confirmed_track_survives_probability_floor_until_absolute_sample_ttl() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (30.0,))
    _update(tracker, 16_320, (30.5,))
    confirmed, _ = _update(tracker, 17_280, (31.0,))
    track_id = first[0].track_id
    assert confirmed[0].track_id == track_id

    # A confirmed track remains alive throughout coasting even after its
    # diagnostic existence probability falls below the tentative-track floor.
    _, coasting = _update(tracker, 17_280 + 95_040, ())
    assert coasting[0].track_id == track_id
    assert coasting[0].track_state == "coasting"
    assert tracker._tracks[track_id].existence_probability < 0.05

    _, expired = _update(tracker, 17_280 + 96_000, ())
    assert expired == ()


def test_confirmed_track_reacquires_original_id_after_long_coasting() -> None:
    tracker = _tracker()
    first, _ = _update(tracker, 15_360, (30.0,))
    _update(tracker, 16_320, (30.5,))
    _update(tracker, 17_280, (31.0,))
    track_id = first[0].track_id

    _, coasting = _update(tracker, 17_280 + 72_000, ())
    before = tracker._tracks[track_id].existence_probability
    assert coasting[0].track_id == track_id
    recovered, _ = _update(tracker, 17_280 + 72_960, (32.0,))
    assert recovered[0].track_id == track_id
    assert recovered[0].track_state == "confirmed"
    assert tracker._tracks[track_id].existence_probability > before


def test_jpda_is_one_to_one_and_exposes_probabilistic_diagnostics() -> None:
    tracker = _tracker()
    _update(tracker, 15_360, (20.0, 200.0))
    observed, _ = _update(tracker, 16_320, (202.0, 22.0))
    assert len({item.track_id for item in observed}) == 2
    assert tracker.last_diagnostics.backend == "circular_imm_jpda_v1"
    assert tracker.last_diagnostics.joint_hypotheses > 0
    assert len(tracker.last_diagnostics.association_probabilities) == 2


def test_internal_unwrapped_state_is_periodically_rebased() -> None:
    tracker = _tracker()
    sample = 15_360
    for angle in tuple(range(0, 360, 10)) * 3:
        _update(tracker, sample, (float(angle),))
        sample += 960
    for track in tracker._tracks.values():
        assert all(abs(model.mean[0]) <= 180.0 for model in track.models)


def test_tracker_respects_explicit_birth_suppression() -> None:
    tracker = _tracker()
    observed, active = tracker.update(
        "track", 0, 15_360, (_candidate(15_360, 30.0),),
        window_id=16, doa_start_sample=13_440, allow_births=False,
    )
    assert observed == active == ()

    first, _ = _update(tracker, 16_320, (30.0,))
    observed, active = tracker.update(
        "track", 0, 17_280, (_candidate(17_280, 31.0),),
        window_id=18, doa_start_sample=15_360, allow_births=False,
    )
    assert observed[0].track_id == first[0].track_id
    assert active[0].track_id == first[0].track_id


def test_tentative_observation_then_confirmed_and_coasting_are_published() -> None:
    tracker = _tracker()
    first, active = _update(tracker, 15_360, (20.0,))
    assert first[0].track_state == "tentative"
    assert _select_l3_directions(first, active) == first
    _update(tracker, 16_320, (20.5,))
    confirmed, active = _update(tracker, 17_280, (21.0,))
    assert _select_l3_directions(confirmed, active) == confirmed
    observed, active = _update(tracker, 18_240, ())
    selected = _select_l3_directions(observed, active)
    assert len(selected) == 1 and selected[0].track_state == "coasting"


def test_internal_tracker_never_exceeds_four_tracks() -> None:
    tracker = _tracker()
    sample = 15_360
    for angles in ((0.0, 60.0, 120.0), (180.0, 240.0, 300.0), (30.0, 90.0, 150.0)):
        _update(tracker, sample, angles)
        sample += 960
        assert tracker.active_track_count <= 4


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


def test_pipeline_requires_one_continuously_open_covariance_context_before_music_angles() -> None:
    config = load_config(CONFIG, environ={})
    pipeline = Layer2Pipeline.from_project(config)
    audio = _audio((30.0,), seed=37, samples=7_680 + 15 * 960)
    scan_config = DirectionScanConfig.from_project(config)

    def probabilities(window: DecisionWindow, value: float) -> tuple[SourceProbability20ms, ...]:
        return tuple(SourceProbability20ms(
            window.session_id, window.stream_epoch, start, start + 960, value,
            SourceProbabilityState.READY, "ready",
        ) for start in (window.doa_start_sample, window.doa_start_sample + 960))

    for index in range(4):
        window = _window(audio, index)
        closed = pipeline.process(
            window, probabilities(window, 0.0), physical_6plus1_geometry(), scan_config,
            gate_threshold=0.6, gate_config_revision=0,
        )
        assert closed.gate_decision.state is ProbabilityGateState.CLOSED
        assert closed.spatial_response is None
        assert closed.search_diagnostics is None
        assert closed.music_state is not None

    first_open_window = _window(audio, 4)
    first_open = pipeline.process(
        first_open_window, probabilities(first_open_window, 1.0),
        physical_6plus1_geometry(), scan_config,
        gate_threshold=0.6, gate_config_revision=0,
    )
    assert first_open.spatial_response is not None
    assert first_open.search_diagnostics is not None
    assert first_open.search_diagnostics.model_order.snapshot_count == 19
    assert first_open.search_diagnostics.active_frame_count == 1
    assert first_open.search_diagnostics.birth_required_active_frames == 10
    assert not first_open.search_diagnostics.births_allowed
    assert first_open.candidates == ()
    assert first_open.music_state is not None
    assert first_open.music_state.state == "advanced"
    assert first_open.music_state.added_frames == 2
    assert first_open.active_tracks == ()

    result = first_open
    for index in range(5, 14):
        window = _window(audio, index)
        result = pipeline.process(
            window, probabilities(window, 1.0), physical_6plus1_geometry(), scan_config,
            gate_threshold=0.6, gate_config_revision=0,
        )
    assert result.search_diagnostics is not None
    assert result.search_diagnostics.active_frame_count == 10
    assert result.search_diagnostics.births_allowed
    assert result.active_tracks
    assert result.active_tracks[0].track_state == "tentative"

    closed_window = _window(audio, 14)
    closed = pipeline.process(
        closed_window, probabilities(closed_window, 0.0), physical_6plus1_geometry(), scan_config,
        gate_threshold=0.6, gate_config_revision=0,
    )
    assert closed.gate_decision.state is ProbabilityGateState.CLOSED
    assert closed.spatial_response is None
    assert closed.search_diagnostics is None
    assert all(not item.is_observed for item in closed.directions)

    reopened_window = _window(audio, 15)
    reopened = pipeline.process(
        reopened_window, probabilities(reopened_window, 1.0),
        physical_6plus1_geometry(), scan_config,
        gate_threshold=0.6, gate_config_revision=0,
    )
    assert reopened.search_diagnostics is not None
    assert reopened.search_diagnostics.active_frame_count == 1
    assert reopened.search_diagnostics.birth_required_active_frames == 10
    assert all(not item.is_observed for item in reopened.directions)


def test_pipeline_tracking_off_publishes_only_raw_music_peaks_and_reenable_resets_ids() -> None:
    config = load_config(CONFIG, environ={})
    pipeline = Layer2Pipeline.from_project(config)
    audio = _audio((30.0,), seed=31, samples=7_680 + 10 * 960)

    def probabilities(window: DecisionWindow) -> tuple[SourceProbability20ms, ...]:
        return tuple(SourceProbability20ms(
            window.session_id, window.stream_epoch, start, start + 960, 1.0,
            SourceProbabilityState.READY, "ready",
        ) for start in (window.doa_start_sample, window.doa_start_sample + 960))

    raw = None
    for index in range(10):
        raw_window = _window(audio, index)
        raw = pipeline.process(
            raw_window, probabilities(raw_window), physical_6plus1_geometry(),
            DirectionScanConfig.from_project(config), gate_threshold=0.6,
            gate_config_revision=0, direction_id_tracking_enabled=False,
        )
    assert raw is not None
    assert raw.direction_id_tracking_enabled is False
    assert raw.spatial_response is not None
    assert raw.candidates
    assert raw.directions == raw.active_tracks == ()
    assert raw.candidate_track_ids == (None,) * len(raw.candidates)
    assert raw.candidate_track_is_formal == (False,) * len(raw.candidates)

    tracked_window = _window(audio, 10)
    tracked = pipeline.process(
        tracked_window, probabilities(tracked_window), physical_6plus1_geometry(),
        DirectionScanConfig.from_project(config), gate_threshold=0.6,
        gate_config_revision=0, direction_id_tracking_enabled=True,
    )
    assert tracked.direction_id_tracking_enabled is True
    assert tracked.active_tracks
    assert tracked.active_tracks[0].track_id == 1
    assert tracked.active_tracks[0].track_state == "tentative"
