from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from common import MIC_POSITIONS_M, DecisionWindow, circular_distance_deg, physical_6plus1_geometry
from common.config import load_config
from layer2_source_detection import (
    CircularKalmanConfig,
    CircularKalmanFilter,
    DirectionIdTracker,
    DirectionIdTrackingConfig,
    DirectionScanConfig,
    DirectionScanError,
    DirectionSmoothingError,
    Layer2ExecutionState,
    Layer2Pipeline,
    Layer2PipelineResult,
    ProbabilityGate,
    ProbabilityGateState,
    SourceProbability20ms,
    SourceProbabilityState,
    SrpPhatScanner,
    robust_z_sigmoid,
    select_candidate_indices,
)
from gui.dev_test_ui.srp_panel import SrpPanelSnapshot


PROJECT_CONFIG = Path(__file__).parents[1] / "config" / "config.yaml"


@pytest.fixture(scope="module")
def scan_config() -> DirectionScanConfig:
    return DirectionScanConfig.from_project(load_config(PROJECT_CONFIG, environ={}))


def decision_window(samples: np.ndarray, window_id: int = 7) -> DecisionWindow:
    context = np.zeros((15_360, 8), dtype=np.float32)
    context[-len(samples):, :7] = samples
    return DecisionWindow(
        "test-session", 2, window_id, 15_360, 13_440, 15_360,
        0, 15_360, 48_000, context, (4, 5),
    )


def plane_wave(theta_deg: float, *, seed: int = 0, noise_snr_db: float | None = None) -> np.ndarray:
    sample_rate, length = 48_000, 1_920
    time = np.arange(length, dtype=np.float64) / sample_rate
    direction = np.array((np.cos(np.deg2rad(theta_deg)), np.sin(np.deg2rad(theta_deg))))
    tau = -(MIC_POSITIONS_M @ direction) / 343.0
    rng = np.random.default_rng(seed)
    frequencies = np.arange(600.0, 3_901.0, 137.0)
    phases = rng.uniform(0.0, 2.0 * np.pi, len(frequencies))
    amplitudes = rng.uniform(0.5, 1.0, len(frequencies))
    signal = np.sum(
        amplitudes[:, None, None]
        * np.cos(2.0 * np.pi * frequencies[:, None, None] * (time[None, :, None] - tau[None, None, :])
                 + phases[:, None, None]),
        axis=0,
    )
    signal /= np.max(np.abs(signal))
    if noise_snr_db is not None:
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10.0 ** (noise_snr_db / 10.0))
        signal += rng.normal(scale=np.sqrt(noise_power), size=signal.shape)
    return np.asarray(signal, dtype=np.float32)


def test_physical_geometry_is_exact_and_read_only() -> None:
    expected = np.array([
        (.04, 0.0), (.02, .034641016), (-.02, .034641016), (-.04, 0.0),
        (-.02, -.034641016), (.02, -.034641016), (0.0, 0.0),
    ], dtype=np.float64)
    geometry = physical_6plus1_geometry()
    assert np.max(np.abs(geometry.positions_m - expected)) < 1e-9
    assert not geometry.positions_m.flags.writeable
    with pytest.raises(ValueError):
        geometry.positions_m[0, 0] = 1.0


def test_config_is_taken_from_the_unique_project_config(scan_config: DirectionScanConfig) -> None:
    assert scan_config.n_fft == 2_048
    assert scan_config.gcc_interpolation == 16
    assert (scan_config.frequency_min_hz, scan_config.frequency_max_hz) == (2_000.0, 4_000.0)
    assert scan_config.min_peak_distance_deg == 45.0
    assert scan_config.max_candidates == 2


def test_every_degree_has_no_mirror_reverse_or_fixed_rotation(scan_config: DirectionScanConfig) -> None:
    scanner, geometry = SrpPhatScanner(), physical_6plus1_geometry()
    errors = []
    for theta in range(360):
        raw = scanner.raw_spatial_response(plane_wave(theta, seed=11), geometry, scan_config)
        errors.append(circular_distance_deg(int(np.argmax(raw)), theta))
    assert max(errors) <= 1.0


def test_spatial_response_contract_and_input_immutability(scan_config: DirectionScanConfig) -> None:
    window = decision_window(plane_wave(73))
    before = window.samples.copy()
    response, candidates = SrpPhatScanner().scan(window, physical_6plus1_geometry(), scan_config)
    assert response.theta_degrees.tolist() == list(range(360))
    for values in (response.theta_degrees, response.raw_scores, response.normalized_scores):
        assert values.dtype == np.float32 and values.flags.c_contiguous and not values.flags.writeable
    assert np.array_equal(window.samples, before)
    assert len(candidates) <= 2
    assert all(candidate.window_id == window.window_id for candidate in candidates)


def test_srp_phat_uses_only_fft_bins_inside_2000_4000_hz(
    scan_config: DirectionScanConfig,
) -> None:
    scanner = SrpPhatScanner()
    _spectrum, phat, frequencies, band = scanner._spectrum_and_phat(
        plane_wave(73, seed=17), scan_config
    )
    assert np.all((frequencies[band] >= 2_000.0) & (frequencies[band] <= 4_000.0))
    assert frequencies[band][0] == pytest.approx(2_015.625)
    assert frequencies[band][-1] == pytest.approx(3_984.375)
    assert np.count_nonzero(phat[:, ~band]) == 0
    assert np.count_nonzero(phat[:, band]) > 0


def test_srp_static_frontend_plan_is_reused_without_reusing_window_artifacts(
    scan_config: DirectionScanConfig,
) -> None:
    scanner = SrpPhatScanner()
    samples = plane_wave(73, seed=170)
    spectrum_a, phat_a, frequencies_a, band_a = scanner._spectrum_and_phat(samples, scan_config)
    spectrum_b, phat_b, frequencies_b, band_b = scanner._spectrum_and_phat(samples, scan_config)

    assert scanner._periodic_hann(1_920) is scanner._periodic_hann(1_920)
    assert frequencies_a is frequencies_b and band_a is band_b
    assert not frequencies_a.flags.writeable and not band_a.flags.writeable
    assert spectrum_a is not spectrum_b and phat_a is not phat_b
    np.testing.assert_array_equal(spectrum_a, spectrum_b)
    np.testing.assert_array_equal(phat_a, phat_b)


def test_unified_frontend_is_bit_exact_to_the_previous_gcc_formula(
    scan_config: DirectionScanConfig,
) -> None:
    scanner = SrpPhatScanner()
    samples = plane_wave(211, seed=171)
    signal = np.asarray(samples, dtype=np.float64)
    signal = signal - np.mean(signal, axis=0, keepdims=True)
    spectrum = np.fft.rfft(
        signal * np.asarray(np.hanning(1_921)[:-1], dtype=np.float64)[:, None],
        n=scan_config.n_fft,
        axis=0,
    )
    frequencies = np.fft.rfftfreq(scan_config.n_fft, d=1.0 / 48_000.0)
    band = (
        (frequencies >= scan_config.frequency_min_hz)
        & (frequencies <= scan_config.frequency_max_hz)
    )
    cross = np.asarray(
        (
            spectrum[:, scanner._pair_left]
            * np.conj(spectrum[:, scanner._pair_right])
        ).T,
        dtype=np.complex128,
    )
    cross /= np.maximum(np.abs(cross), scan_config.phat_epsilon)
    cross[:, ~band] = 0.0
    expected = np.fft.irfft(
        cross,
        n=scan_config.n_fft * scan_config.gcc_interpolation,
        axis=1,
    )

    np.testing.assert_array_equal(scanner._gcc_phat(samples, scan_config), expected)


@pytest.mark.parametrize("iterative", [False, True])
def test_each_scan_builds_only_one_ephemeral_frontend(
    scan_config: DirectionScanConfig, iterative: bool,
) -> None:
    class CountingScanner(SrpPhatScanner):
        def __init__(self):
            super().__init__()
            self.frontend_calls = 0

        def _frontend(self, *args, **kwargs):
            self.frontend_calls += 1
            return super()._frontend(*args, **kwargs)

    scanner = CountingScanner()
    scanner.scan_detailed(
        decision_window(plane_wave(73, seed=172)),
        physical_6plus1_geometry(),
        replace(scan_config, iterative_peak_search_enabled=iterative),
    )
    assert scanner.frontend_calls == 1


def test_srp_steering_cache_includes_sample_rate_and_has_a_hard_limit() -> None:
    scanner = SrpPhatScanner()
    geometry = physical_6plus1_geometry()
    first = scanner._steering_lookup(geometry, 48_000, 32_768, 16)
    second = scanner._steering_lookup(geometry, 24_000, 32_768, 16)
    assert len(scanner._steering_cache) == 2
    assert not np.array_equal(first[0], second[0])

    for sample_rate in range(30_000, 30_000 + scanner._STEERING_CACHE_LIMIT + 2):
        scanner._steering_lookup(geometry, sample_rate, 32_768, 16)
    assert len(scanner._steering_cache) == scanner._STEERING_CACHE_LIMIT
    assert all(key[2] != 48_000 for key in scanner._steering_cache)


def test_robust_z_sigmoid_matches_specification(scan_config: DirectionScanConfig) -> None:
    raw = np.linspace(-2.0, 3.0, 360, dtype=np.float32)
    normalized = robust_z_sigmoid(raw, scan_config)
    median = np.median(raw)
    scale = max(1.4826 * np.median(np.abs(raw - median)), 1e-6)
    expected = 1.0 / (1.0 + np.exp(-np.clip((raw - median) / scale - 2.0, -80.0, 80.0)))
    assert np.allclose(normalized, expected.astype(np.float32))


def test_circular_peak_boundary_nms_ties_and_top_two(scan_config: DirectionScanConfig) -> None:
    config = replace(scan_config, direction_threshold=.35, peak_prominence=.05)
    scores = np.full(360, .1, dtype=np.float32)
    for index, value in ((359, .99), (2, .95), (20, .9), (40, .9), (60, .89), (80, .88),
                         (100, .87), (120, .86), (140, .85), (160, .84), (180, .83), (200, .82), (220, .81)):
        scores[index] = value
    selected = select_candidate_indices(scores, config)
    assert selected[0] == 359
    assert 2 not in selected
    assert len(selected) == 2
    assert selected == (359, 60)
    assert circular_distance_deg(359.0, 2.0) == 3.0


def test_circular_nms_keeps_peak_exactly_at_configured_45_degree_boundary(
    scan_config: DirectionScanConfig,
) -> None:
    scores = np.full(360, .1, dtype=np.float32)
    scores[350] = .99
    scores[35] = .95
    scores[34] = .90
    assert select_candidate_indices(scores, scan_config) == (350, 35)


def test_flat_response_has_no_candidates(scan_config: DirectionScanConfig) -> None:
    assert select_candidate_indices(np.full(360, .5, np.float32), scan_config) == ()


def test_nonfinite_input_is_rejected_before_formal_output(scan_config: DirectionScanConfig) -> None:
    samples = np.zeros((1_920, 7), np.float32)
    samples[0, 0] = np.nan
    with pytest.raises(ValueError):
        decision_window(samples)
    with pytest.raises(DirectionScanError):
        SrpPhatScanner().raw_spatial_response(samples, physical_6plus1_geometry(), scan_config)


@pytest.mark.parametrize(
    ("field", "value"),
    (("phat_epsilon", np.inf), ("phat_epsilon", np.nan),
     ("iterative_phase_power", np.inf), ("iterative_phase_power", np.nan)),
)
def test_nonfinite_scan_configuration_is_rejected(
    scan_config: DirectionScanConfig, field: str, value: float
) -> None:
    with pytest.raises(ValueError, match="finite"):
        replace(scan_config, **{field: value})


def test_twenty_db_noise_p95_is_within_five_degrees(scan_config: DirectionScanConfig) -> None:
    scanner, geometry = SrpPhatScanner(), physical_6plus1_geometry()
    errors = []
    for theta in range(0, 360, 10):
        raw = scanner.raw_spatial_response(plane_wave(theta, seed=theta + 99, noise_snr_db=20.0), geometry, scan_config)
        errors.append(circular_distance_deg(int(np.argmax(raw)), theta))
    assert np.percentile(errors, 95) <= 5.0


def test_ui_snapshot_requires_one_window_and_rank_order(scan_config: DirectionScanConfig) -> None:
    response, candidates = SrpPhatScanner().scan(
        decision_window(plane_wave(45)), physical_6plus1_geometry(), scan_config
    )
    snapshot = SrpPanelSnapshot(response, candidates, 1.0)
    assert snapshot.response.window_id == 7
    if candidates:
        mismatched = replace(candidates[0], window_id=8)
        with pytest.raises(ValueError, match="同一window"):
            SrpPanelSnapshot(response, (mismatched,), 1.0)


def test_iterative_off_does_not_enter_iterative_branch_and_is_exact(scan_config: DirectionScanConfig) -> None:
    class GuardedScanner(SrpPhatScanner):
        def _iterative_scan(self, *args, **kwargs):
            raise AssertionError("OFF must not call iterative search")

    window = decision_window(plane_wave(73, seed=19))
    scanner = GuardedScanner()
    classic_response, classic_candidates = scanner._single_pass(
        window, physical_6plus1_geometry(), scan_config
    )
    response, candidates, diagnostics = scanner.scan_detailed(
        window, physical_6plus1_geometry(), scan_config, config_revision=4
    )
    assert np.array_equal(response.raw_scores, classic_response.raw_scores)
    assert np.array_equal(response.normalized_scores, classic_response.normalized_scores)
    assert candidates == classic_candidates
    assert diagnostics.mode == "single_pass" and diagnostics.config_revision == 4


def test_iterative_mode_keeps_blue_base_response_and_is_bounded(scan_config: DirectionScanConfig) -> None:
    samples = plane_wave(30, seed=130) + .70 * plane_wave(150, seed=650)
    samples = np.asarray(samples / np.max(np.abs(samples)), dtype=np.float32)
    window = decision_window(samples)
    scanner, geometry = SrpPhatScanner(), physical_6plus1_geometry()
    base_response, _ = scanner.scan(window, geometry, scan_config)
    iterative_config = replace(scan_config, iterative_peak_search_enabled=True)
    response, candidates, diagnostics = scanner.scan_detailed(window, geometry, iterative_config, 7)
    assert np.array_equal(response.raw_scores, base_response.raw_scores)
    assert np.array_equal(response.normalized_scores, base_response.normalized_scores)
    assert 1 <= len(candidates) <= 2
    assert diagnostics.mode == "iterative_rank1_projection_v1"
    assert diagnostics.config_revision == 7
    assert 1 <= diagnostics.iterations_used <= 2
    assert len(diagnostics.evidence) == len(candidates)
    assert min(circular_distance_deg(item.theta_deg, 30.0) for item in candidates) <= 3.0
    assert min(circular_distance_deg(item.theta_deg, 150.0) for item in candidates) <= 3.0
    assert diagnostics.iterations_used == 2 and diagnostics.stop_reason == "candidate_limit_reached"
    assert diagnostics.candidate_limit == 2 and diagnostics.candidate_limit_applied


def test_iterative_fallback_only_handles_declared_numerical_failures(scan_config: DirectionScanConfig) -> None:
    iterative_config = replace(scan_config, iterative_peak_search_enabled=True)
    window = decision_window(plane_wave(73, seed=81))
    geometry = physical_6plus1_geometry()

    class NumericalFailure(SrpPhatScanner):
        def _iterative_scan(self, *args, **kwargs):
            raise DirectionScanError("synthetic numerical failure")

    response, candidates, diagnostics = NumericalFailure().scan_detailed(
        window, geometry, iterative_config
    )
    assert response is not None and candidates
    assert diagnostics.stop_reason == "legacy_fallback"
    assert "synthetic numerical failure" in diagnostics.fallback_reason

    class ProgrammingFailure(SrpPhatScanner):
        def _iterative_scan(self, *args, **kwargs):
            raise RuntimeError("synthetic programming failure")

    with pytest.raises(RuntimeError, match="synthetic programming failure"):
        ProgrammingFailure().scan_detailed(window, geometry, iterative_config)


def test_probability_gate_then_srp_preserves_existing_scan(scan_config: DirectionScanConfig) -> None:
    window = decision_window(plane_wave(73, seed=44))
    geometry = physical_6plus1_geometry()
    scanner = SrpPhatScanner()
    expected_response, expected_candidates, expected_diagnostics = scanner.scan_detailed(
        window, geometry, scan_config, config_revision=9
    )

    probabilities = (
        SourceProbability20ms(
            window.session_id, window.stream_epoch, window.doa_start_sample,
            window.doa_start_sample + 960, 0.8, SourceProbabilityState.READY, "ready",
        ),
        SourceProbability20ms(
            window.session_id, window.stream_epoch, window.doa_start_sample + 960,
            window.doa_end_sample, 0.9, SourceProbabilityState.READY, "ready",
        ),
    )
    result = Layer2Pipeline(ProbabilityGate(), scanner).process(
        window,
        probabilities,
        geometry,
        scan_config,
        gate_threshold=0.60,
        gate_config_revision=4,
        scan_config_revision=9,
    )

    assert result.state is Layer2ExecutionState.PROCESSED
    assert result.gate_decision.state is ProbabilityGateState.OPEN
    assert result.gate_decision.probability_40ms == pytest.approx(0.85)
    assert result.gate_decision.allow_srp is True
    assert np.array_equal(result.spatial_response.raw_scores, expected_response.raw_scores)
    assert np.array_equal(result.spatial_response.normalized_scores, expected_response.normalized_scores)
    assert result.candidates == expected_candidates
    assert result.search_diagnostics == expected_diagnostics


def test_layer2_result_and_ui_reject_more_than_two_formal_candidates(
    scan_config: DirectionScanConfig,
) -> None:
    window = decision_window(plane_wave(73, seed=91))
    geometry = physical_6plus1_geometry()
    scanner = SrpPhatScanner()
    probabilities = (
        SourceProbability20ms(
            window.session_id, window.stream_epoch, window.doa_start_sample,
            window.doa_start_sample + 960, 0.8, SourceProbabilityState.READY, "ready",
        ),
        SourceProbability20ms(
            window.session_id, window.stream_epoch, window.doa_start_sample + 960,
            window.doa_end_sample, 0.9, SourceProbabilityState.READY, "ready",
        ),
    )
    result = Layer2Pipeline(ProbabilityGate(), scanner).process(
        window, probabilities, geometry, scan_config,
        gate_threshold=0.60, gate_config_revision=0,
    )
    base = result.candidates[0]
    invalid_candidates = (
        replace(base, theta_deg=10.0),
        replace(base, theta_deg=120.0),
        replace(base, theta_deg=240.0),
    )
    with pytest.raises(ValueError, match="more than 2"):
        Layer2PipelineResult(
            result.state, result.gate_decision, result.spatial_response,
            invalid_candidates, result.search_diagnostics,
        )
    with pytest.raises(ValueError, match="最多显示2"):
        SrpPanelSnapshot(result.spatial_response, invalid_candidates, 1.0)
    circularly_too_close = (
        replace(base, theta_deg=359.0),
        replace(base, theta_deg=2.0),
    )
    with pytest.raises(ValueError, match="45 circular degrees"):
        Layer2PipelineResult(
            result.state, result.gate_decision, result.spatial_response,
            circularly_too_close, result.search_diagnostics,
        )


def test_layer2_smoother_failure_resets_and_falls_back_to_raw_candidates(
    scan_config: DirectionScanConfig,
) -> None:
    window = decision_window(plane_wave(73, seed=144))
    geometry = physical_6plus1_geometry()
    scanner = SrpPhatScanner()
    _response, raw_candidates, _diagnostics = scanner.scan_detailed(
        window, geometry, scan_config
    )

    class FailingSmoother:
        def __init__(self):
            self.reset_count = 0

        def update(self, *args, **kwargs):
            raise DirectionSmoothingError("synthetic smoother failure")

        def reset(self):
            self.reset_count += 1

    smoother = FailingSmoother()
    probabilities = (
        SourceProbability20ms(
            window.session_id, window.stream_epoch, window.doa_start_sample,
            window.doa_start_sample + 960, 0.8, SourceProbabilityState.READY, "ready",
        ),
        SourceProbability20ms(
            window.session_id, window.stream_epoch, window.doa_start_sample + 960,
            window.doa_end_sample, 0.9, SourceProbabilityState.READY, "ready",
        ),
    )
    pipeline = Layer2Pipeline(ProbabilityGate(), scanner, smoother)
    result = pipeline.process(
        window, probabilities, geometry, scan_config,
        gate_threshold=0.6, gate_config_revision=0,
        direction_kalman_enabled=True,
        direction_id_tracking_enabled=True,
    )

    assert result.candidates == raw_candidates
    assert pipeline.last_kalman_error == "synthetic smoother failure"
    assert smoother.reset_count == 2


def test_pipeline_rejects_kalman_without_id_tracking(scan_config: DirectionScanConfig) -> None:
    window = decision_window(plane_wave(73, seed=146))
    probabilities = (
        SourceProbability20ms(
            window.session_id, window.stream_epoch, window.doa_start_sample,
            window.doa_start_sample + 960, 0.8, SourceProbabilityState.READY, "ready",
        ),
        SourceProbability20ms(
            window.session_id, window.stream_epoch, window.doa_start_sample + 960,
            window.doa_end_sample, 0.9, SourceProbabilityState.READY, "ready",
        ),
    )
    pipeline = Layer2Pipeline(ProbabilityGate(), SrpPhatScanner())
    with pytest.raises(ValueError, match="requires private ID tracking"):
        pipeline.process(
            window, probabilities, physical_6plus1_geometry(), scan_config,
            gate_threshold=0.6, gate_config_revision=0,
            direction_kalman_enabled=True,
            direction_id_tracking_enabled=False,
        )


def test_mature_track_is_published_when_current_srp_has_no_mergeable_peak(
    scan_config: DirectionScanConfig,
) -> None:
    pipeline = Layer2Pipeline(
        ProbabilityGate(),
        SrpPhatScanner(),
        CircularKalmanFilter(CircularKalmanConfig(max_missed_windows=2)),
        DirectionIdTracker(DirectionIdTrackingConfig(
            max_missed_windows=2,
            confirmation_min_age_windows=3,
            confirmation_min_matches=3,
            prediction_hold_windows=2,
        )),
    )
    geometry = physical_6plus1_geometry()

    def window_at(index: int, *, with_signal: bool = True) -> DecisionWindow:
        decision = 15_360 + index * 960
        context = np.zeros((15_360, 8), dtype=np.float32)
        if with_signal:
            context[-1_920:, :7] = plane_wave(75.0, seed=700 + index)
        return DecisionWindow(
            "prediction-integration", 0, index, decision,
            decision - 1_920, decision, decision - 15_360, decision,
            48_000, context, (index,),
        )

    def probabilities(window: DecisionWindow, value: float):
        return (
            SourceProbability20ms(
                window.session_id, window.stream_epoch, window.doa_start_sample,
                window.doa_start_sample + 960, value, SourceProbabilityState.READY, "ready",
            ),
            SourceProbability20ms(
                window.session_id, window.stream_epoch, window.doa_start_sample + 960,
                window.doa_end_sample, value, SourceProbabilityState.READY, "ready",
            ),
        )

    for index in range(3):
        window = window_at(index)
        result = pipeline.process(
            window, probabilities(window, .9), geometry, scan_config,
            gate_threshold=.6, gate_config_revision=0,
            direction_id_tracking_enabled=True, direction_kalman_enabled=True,
        )
        assert result.state is Layer2ExecutionState.PROCESSED
        if index == 0:
            assert pipeline.submit_voice_feedback(
                window.session_id, window.stream_epoch,
                window.decision_sample, result.candidates[0].theta_deg,
            )

    missing = window_at(3, with_signal=False)
    predicted = pipeline.process(
        missing, probabilities(missing, .9), geometry, scan_config,
        gate_threshold=.6, gate_config_revision=0,
        direction_id_tracking_enabled=True, direction_kalman_enabled=True,
    )
    assert predicted.state is Layer2ExecutionState.PROCESSED
    assert predicted.spatial_response is not None
    assert len(predicted.candidates) == 1
    assert predicted.candidates[0].window_id == missing.window_id
    score_index = round(predicted.candidates[0].theta_deg) % 360
    assert predicted.candidates[0].raw_score == pytest.approx(
        float(predicted.spatial_response.raw_scores[score_index])
    )
    assert predicted.candidates[0].normalized_score == pytest.approx(
        float(predicted.spatial_response.normalized_scores[score_index])
    )

    forced_first = window_at(4, with_signal=False)
    forced = pipeline.process(
        forced_first, probabilities(forced_first, .1), geometry, scan_config,
        gate_threshold=.6, gate_config_revision=0,
        direction_id_tracking_enabled=True, direction_kalman_enabled=True,
    )
    assert forced.state is Layer2ExecutionState.PROCESSED
    assert forced.gate_decision.state is ProbabilityGateState.OPEN
    assert forced.gate_decision.probability_40ms == pytest.approx(.1)
    assert forced.gate_decision.reason == "confirmed_id_gate_hold"

    still_forced = window_at(5, with_signal=False)
    still_open = pipeline.process(
        still_forced, probabilities(still_forced, .1), geometry, scan_config,
        gate_threshold=.6, gate_config_revision=0,
        direction_id_tracking_enabled=True, direction_kalman_enabled=True,
    )
    assert still_open.state is Layer2ExecutionState.PROCESSED

    gate_closed = window_at(6, with_signal=False)
    blocked = pipeline.process(
        gate_closed, probabilities(gate_closed, .1), geometry, scan_config,
        gate_threshold=.6, gate_config_revision=0,
        direction_id_tracking_enabled=True, direction_kalman_enabled=True,
    )
    assert blocked.state is Layer2ExecutionState.BLOCKED
    assert blocked.spatial_response is None
    assert blocked.candidates == ()


def test_forced_gate_fan_matches_cannot_extend_formal_id_lifetime(
    scan_config: DirectionScanConfig,
) -> None:
    pipeline = Layer2Pipeline(
        ProbabilityGate(),
        SrpPhatScanner(),
        CircularKalmanFilter(CircularKalmanConfig(max_missed_windows=20)),
        DirectionIdTracker(DirectionIdTrackingConfig(
            max_missed_windows=20,
            confirmation_min_age_windows=2,
            confirmation_min_matches=2,
            prediction_hold_windows=2,
        )),
    )
    geometry = physical_6plus1_geometry()

    def run(index: int, probability: float):
        decision = 15_360 + index * 960
        context = np.zeros((15_360, 8), dtype=np.float32)
        context[-1_920:, :7] = plane_wave(350.0, seed=900 + index)
        window = DecisionWindow(
            "fan-lease", 0, index, decision,
            decision - 1_920, decision, decision - 15_360, decision,
            48_000, context, (index,),
        )
        slots = (
            SourceProbability20ms(
                window.session_id, 0, window.doa_start_sample,
                window.doa_start_sample + 960, probability,
                SourceProbabilityState.READY, "ready",
            ),
            SourceProbability20ms(
                window.session_id, 0, window.doa_start_sample + 960,
                window.doa_end_sample, probability,
                SourceProbabilityState.READY, "ready",
            ),
        )
        return pipeline.process(
            window, slots, geometry, scan_config,
            gate_threshold=.6, gate_config_revision=0,
            direction_id_tracking_enabled=True, direction_kalman_enabled=True,
        )

    for index in range(3):
        result = run(index, .9)
        assert result.state is Layer2ExecutionState.PROCESSED
        if index == 0:
            assert pipeline.submit_voice_feedback(
                result.candidates[0].session_id,
                result.candidates[0].stream_epoch,
                result.candidates[0].decision_sample,
                result.candidates[0].theta_deg,
            )
    assert pipeline.id_tracker.confirmed_track_ids == (1,)
    assert run(3, .1).gate_decision.reason == "confirmed_id_gate_hold"
    assert run(4, .1).gate_decision.reason == "confirmed_id_gate_hold"
    expired = run(5, .1)
    assert expired.state is Layer2ExecutionState.BLOCKED
    assert pipeline.id_tracker.confirmed_track_ids == ()


def test_queued_l4_voice_angle_feedback_extends_matching_formal_id(
    scan_config: DirectionScanConfig,
) -> None:
    pipeline = Layer2Pipeline(
        ProbabilityGate(), SrpPhatScanner(),
        CircularKalmanFilter(CircularKalmanConfig(max_missed_windows=20)),
        DirectionIdTracker(DirectionIdTrackingConfig(
            max_missed_windows=20,
            confirmation_min_age_windows=2,
            confirmation_min_matches=2,
            prediction_hold_windows=2,
        )),
    )
    geometry = physical_6plus1_geometry()

    def window_and_probabilities(index: int, probability: float):
        decision = 15_360 + index * 960
        context = np.zeros((15_360, 8), dtype=np.float32)
        context[-1_920:, :7] = plane_wave(80.0, seed=950 + index)
        window = DecisionWindow(
            "voice-lease", 0, index, decision,
            decision - 1_920, decision, decision - 15_360, decision,
            48_000, context, (index,),
        )
        slots = tuple(
            SourceProbability20ms(
                window.session_id, 0, start, start + 960, probability,
                SourceProbabilityState.READY, "ready",
            )
            for start in (window.doa_start_sample, window.doa_start_sample + 960)
        )
        return window, slots

    results = []
    for index in range(3):
        window, slots = window_and_probabilities(index, .9)
        results.append(pipeline.process(
            window, slots, geometry, scan_config,
            gate_threshold=.6, gate_config_revision=0,
            direction_id_tracking_enabled=True, direction_kalman_enabled=True,
        ))
        if index == 0:
            assert pipeline.submit_voice_feedback(
                window.session_id, window.stream_epoch,
                window.decision_sample, results[-1].candidates[0].theta_deg,
            )
    forced_window, forced_slots = window_and_probabilities(3, .1)
    forced = pipeline.process(
        forced_window, forced_slots, geometry, scan_config,
        gate_threshold=.6, gate_config_revision=0,
        direction_id_tracking_enabled=True, direction_kalman_enabled=True,
    )
    assert forced.candidates
    assert pipeline.submit_voice_feedback(
        forced_window.session_id, forced_window.stream_epoch,
        forced_window.decision_sample, forced.candidates[0].theta_deg,
    )

    for index in (4, 5):
        window, slots = window_and_probabilities(index, .1)
        result = pipeline.process(
            window, slots, geometry, scan_config,
            gate_threshold=.6, gate_config_revision=0,
            direction_id_tracking_enabled=True, direction_kalman_enabled=True,
        )
        assert result.state is Layer2ExecutionState.PROCESSED
    assert pipeline.voice_feedback_applied == 2
    window, slots = window_and_probabilities(6, .1)
    assert pipeline.process(
        window, slots, geometry, scan_config,
        gate_threshold=.6, gate_config_revision=0,
        direction_id_tracking_enabled=True, direction_kalman_enabled=True,
    ).state is Layer2ExecutionState.BLOCKED


def test_pipeline_rejects_smoothed_wraparound_pair_below_45_degrees(
    scan_config: DirectionScanConfig,
) -> None:
    samples = plane_wave(30, seed=130) + .70 * plane_wave(150, seed=650)
    window = decision_window(np.asarray(samples / np.max(np.abs(samples)), dtype=np.float32))
    geometry = physical_6plus1_geometry()
    iterative_config = replace(scan_config, iterative_peak_search_enabled=True)
    scanner = SrpPhatScanner()
    response, raw_candidates, diagnostics = scanner.scan_detailed(
        window, geometry, iterative_config
    )
    assert len(raw_candidates) == 2

    class FixedScanner:
        def scan_detailed(self, *args, **kwargs):
            return response, raw_candidates, diagnostics

    class CollapsingSmoother:
        def __init__(self):
            self.reset_count = 0

        def update(self, _session, _epoch, _sample, candidates, _track_ids, _q, _r):
            return (
                replace(candidates[0], theta_deg=359.0),
                replace(candidates[1], theta_deg=2.0),
            )

        def reset(self):
            self.reset_count += 1

    probabilities = (
        SourceProbability20ms(
            window.session_id, window.stream_epoch, window.doa_start_sample,
            window.doa_start_sample + 960, 0.8, SourceProbabilityState.READY, "ready",
        ),
        SourceProbability20ms(
            window.session_id, window.stream_epoch, window.doa_start_sample + 960,
            window.doa_end_sample, 0.9, SourceProbabilityState.READY, "ready",
        ),
    )
    smoother = CollapsingSmoother()
    pipeline = Layer2Pipeline(ProbabilityGate(), FixedScanner(), smoother)
    result = pipeline.process(
        window, probabilities, geometry, iterative_config,
        gate_threshold=0.6, gate_config_revision=0,
        direction_kalman_enabled=True,
        direction_id_tracking_enabled=True,
    )

    assert result.candidates == raw_candidates
    assert pipeline.last_kalman_error == (
        "the two smoothed source points violate the 45-degree circular separation"
    )
    assert smoother.reset_count == 2


def test_gate_blocked_window_advances_smoother_with_an_empty_measurement(
    scan_config: DirectionScanConfig,
) -> None:
    window = decision_window(plane_wave(73, seed=145))

    class CountingSmoother:
        def __init__(self):
            self.calls = []

        def update(self, session_id, stream_epoch, decision_sample, candidates, _track_ids, _q, _r):
            self.calls.append((session_id, stream_epoch, decision_sample, tuple(candidates)))
            return tuple(candidates)

        def reset(self):
            pass

    smoother = CountingSmoother()
    probabilities = (
        SourceProbability20ms(
            window.session_id, window.stream_epoch, window.doa_start_sample,
            window.doa_start_sample + 960, 0.1, SourceProbabilityState.READY, "ready",
        ),
        SourceProbability20ms(
            window.session_id, window.stream_epoch, window.doa_start_sample + 960,
            window.doa_end_sample, 0.2, SourceProbabilityState.READY, "ready",
        ),
    )
    result = Layer2Pipeline(ProbabilityGate(), SrpPhatScanner(), smoother).process(
        window, probabilities, physical_6plus1_geometry(), scan_config,
        gate_threshold=0.6, gate_config_revision=0,
        direction_kalman_enabled=True,
        direction_id_tracking_enabled=True,
    )

    assert result.state is Layer2ExecutionState.BLOCKED
    assert smoother.calls == [(window.session_id, window.stream_epoch, window.decision_sample, ())]


def test_real_pipeline_replaces_only_the_second_window_candidate_angle(
    scan_config: DirectionScanConfig,
) -> None:
    geometry = physical_6plus1_geometry()
    pipeline = Layer2Pipeline(ProbabilityGate(), SrpPhatScanner())

    def shifted_window(theta: float, index: int) -> DecisionWindow:
        decision = 15_360 + index * 960
        context = np.zeros((15_360, 8), dtype=np.float32)
        context[-1_920:, :7] = plane_wave(theta, seed=300 + index)
        return DecisionWindow(
            "smooth-integration", 0, index, decision, decision - 1_920, decision,
            decision - 15_360, decision, 48_000, context, (index,),
        )

    def open_probabilities(window: DecisionWindow):
        return (
            SourceProbability20ms(
                window.session_id, window.stream_epoch, window.doa_start_sample,
                window.doa_start_sample + 960, 0.9, SourceProbabilityState.READY, "ready",
            ),
            SourceProbability20ms(
                window.session_id, window.stream_epoch, window.doa_start_sample + 960,
                window.doa_end_sample, 0.9, SourceProbabilityState.READY, "ready",
            ),
        )

    first_window = shifted_window(70.0, 0)
    second_window = shifted_window(80.0, 1)
    first_raw = SrpPhatScanner().scan(first_window, geometry, scan_config)[1]
    second_raw = SrpPhatScanner().scan(second_window, geometry, scan_config)[1]
    first = pipeline.process(
        first_window, open_probabilities(first_window), geometry, scan_config,
        gate_threshold=0.6, gate_config_revision=0,
        direction_kalman_enabled=True,
        direction_id_tracking_enabled=True,
    )
    second = pipeline.process(
        second_window, open_probabilities(second_window), geometry, scan_config,
        gate_threshold=0.6, gate_config_revision=0,
        direction_kalman_enabled=True,
        direction_id_tracking_enabled=True,
    )

    assert first.candidates == first_raw
    assert len(second.candidates) == len(second_raw) == 1
    assert replace(second.candidates[0], theta_deg=second_raw[0].theta_deg) == second_raw[0]
    assert second.candidates[0].theta_deg != second_raw[0].theta_deg
