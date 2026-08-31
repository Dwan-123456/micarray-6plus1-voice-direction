from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from common.config import load_config
from common.data_types import CandidateDirection, DecisionWindow
from common.geometry import MIC_POSITIONS_M, physical_6plus1_geometry
from layer2_source_detection import (
    DirectionScanConfig,
    Layer2ExecutionState,
    Layer2Pipeline,
    RollingNormMusicScanner,
)
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


def _window(
    audio: np.ndarray,
    index: int = 0,
    *,
    session: str = "music",
    epoch: int = 0,
) -> DecisionWindow:
    start = index * 960
    decision = 7_680 + start
    samples = np.ascontiguousarray(audio[start:start + 7_680])
    return DecisionWindow(
        session, epoch, index, decision, decision - 1_920, decision,
        decision - 7_680, decision, 48_000, samples, (index,),
        (),
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


def test_confirmation_window_expands_for_adaptive_observation_period() -> None:
    tracker = GlobalDirectionTracker(
        GlobalTrackerConfig(
            confirmation_observations=10,
            confirmation_window_samples=24_000,
            tentative_ttl_samples=24_000,
            coasting_ttl_samples=96_000,
        )
    )
    observed = ()
    for index in range(10):
        sample = 15_360 + index * 9_600
        observed, _ = _update(
            tracker,
            sample,
            (30.0,),
            observation_period_samples=9_600,
        )

    assert observed[0].track_state == "confirmed"
    assert observed[0].track_id == 1


def _ready_probabilities(
    window: DecisionWindow,
    value: float = 1.0,
) -> tuple[SourceProbability20ms, ...]:
    return tuple(
        SourceProbability20ms(
            window.session_id,
            window.stream_epoch,
            start,
            start + 960,
            value,
            SourceProbabilityState.READY,
            "ready",
        )
        for start in (window.doa_start_sample, window.doa_start_sample + 960)
    )


@pytest.mark.parametrize("signal_order", (1, 2, 3))
def test_music_signal_complement_matches_direct_noise_projection(signal_order: int) -> None:
    rng = np.random.default_rng(820_2026 + signal_order)
    frequency_bins = 9
    eigenvectors = np.stack([
        np.linalg.qr(
            rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7)),
        )[0]
        for _ in range(frequency_bins)
    ])
    steering = rng.normal(size=(frequency_bins, 360, 7)) + 1j * rng.normal(
        size=(frequency_bins, 360, 7),
    )
    noise = eigenvectors[:, :, : 7 - signal_order]
    direct_projection = np.einsum(
        "fcn,fac->fan", noise.conj(), steering, optimize=True,
    )
    direct = np.sum(np.abs(direct_projection) ** 2, axis=2)
    complement = RollingNormMusicScanner._noise_projection_denominator(
        eigenvectors,
        steering,
        signal_order,
        np.sum(np.abs(steering) ** 2, axis=2),
    )
    np.testing.assert_allclose(complement, direct, rtol=2e-13, atol=2e-13)


def test_music_configuration_and_hardware_mix_contract() -> None:
    config = load_config(CONFIG, environ={})
    scan = DirectionScanConfig.from_project(config)
    assert config.layer2.scanner_backend == "frequency_normalized_music"
    assert config.layer2.probability_gate.threshold == 0.80
    assert (scan.n_fft, scan.win_length, scan.hop_length) == (1024, 960, 480)
    assert scan.context_ms in {160, 200, 240, 320}
    assert (scan.frequency_min_hz, scan.frequency_max_hz) == (2_000.0, 4_000.0)
    assert scan.max_candidates == 3 and scan.min_peak_distance_deg == 50.0
    assert "dpd_rank1_enabled" not in type(config.layer2).model_fields
    assert "noise_whitening_enabled" not in type(config.layer2).model_fields


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


@pytest.mark.parametrize(
    ("effective_order", "skip_reason"),
    ((0, "source_count_zero"), (None, "source_count_warming")),
)
def test_process_prepared_skips_music_for_zero_or_warming_source_count(
    effective_order: int | None,
    skip_reason: str,
) -> None:
    class ForbiddenScanner:
        def __init__(self) -> None:
            self.scan_calls = 0
            self.covariance_calls = 0
            self.reset_calls = 0

        def scan_detailed(self, *args, **kwargs):
            self.scan_calls += 1
            raise AssertionError("order 0/None must not scan MUSIC")

        def observe_covariance(self, *args, **kwargs):
            self.covariance_calls += 1
            raise AssertionError("order 0/None must not maintain MUSIC covariance")

        def reset(self) -> None:
            self.reset_calls += 1

    config = load_config(CONFIG, environ={})
    scanner = ForbiddenScanner()
    pipeline = Layer2Pipeline.from_project(config, scanner=scanner)
    window = _window(_audio((30.0,), seed=61))
    decision, active_frame_count = pipeline.evaluate_gate(
        window,
        _ready_probabilities(window),
        gate_threshold=0.6,
        gate_config_revision=4,
    )

    result = pipeline.process_prepared(
        window,
        decision,
        active_frame_count,
        physical_6plus1_geometry(),
        DirectionScanConfig.from_project(config),
        music_effective_order=effective_order,
        music_skip_reason=skip_reason,
        scan_config_revision=7,
        direction_id_tracking_enabled=False,
    )

    assert decision.state is ProbabilityGateState.OPEN
    assert result.state is Layer2ExecutionState.MUSIC_SKIPPED
    assert result.music_effective_order == effective_order
    assert result.music_skip_reason == skip_reason
    assert result.spatial_response is None
    assert result.search_diagnostics is None
    assert result.model_order is None
    assert result.music_state is None
    assert result.candidates == ()
    assert scanner.scan_calls == scanner.covariance_calls == 0
    assert scanner.reset_calls == 1


def test_positive_music_order_rewarms_after_gate_open_count_zero_windows() -> None:
    config = load_config(CONFIG, environ={})
    pipeline = Layer2Pipeline.from_project(config)
    scan_config = DirectionScanConfig.from_project(config)
    audio = _audio((30.0,), seed=63, samples=20_000)

    for index in range(10):
        window = _window(audio, index)
        decision, active_frame_count = pipeline.evaluate_gate(
            window,
            _ready_probabilities(window),
            gate_threshold=0.6,
            gate_config_revision=4,
        )
        skipped = pipeline.process_prepared(
            window,
            decision,
            active_frame_count,
            physical_6plus1_geometry(),
            scan_config,
            music_effective_order=0,
            music_skip_reason="source_count_zero",
        )
        assert skipped.state is Layer2ExecutionState.MUSIC_SKIPPED

    window = _window(audio, 10)
    decision, active_frame_count = pipeline.evaluate_gate(
        window,
        _ready_probabilities(window),
        gate_threshold=0.6,
        gate_config_revision=4,
    )
    result = pipeline.process_prepared(
        window,
        decision,
        active_frame_count,
        physical_6plus1_geometry(),
        scan_config,
        music_effective_order=1,
    )

    assert active_frame_count == 1
    assert result.search_diagnostics is not None
    assert result.search_diagnostics.active_frame_count == 1
    assert not result.search_diagnostics.births_allowed


@pytest.mark.parametrize("effective_order", (1, 2))
def test_process_prepared_applies_same_window_source_count_as_music_order(
    effective_order: int,
) -> None:
    class RecordingScanner:
        def __init__(self) -> None:
            self.delegate = RollingNormMusicScanner()
            self.orders: list[int] = []

        @property
        def last_state_diagnostic(self):
            return self.delegate.last_state_diagnostic

        def scan_detailed(self, window, geometry, scan_config, config_revision=0):
            self.orders.append(scan_config.effective_order_limit)
            return self.delegate.scan_detailed(
                window,
                geometry,
                scan_config,
                config_revision,
            )

        def reset(self) -> None:
            self.delegate.reset()

    config = load_config(CONFIG, environ={})
    scanner = RecordingScanner()
    pipeline = Layer2Pipeline.from_project(config, scanner=scanner)
    window = _window(_audio((30.0, 210.0), seed=67))
    decision, active_frame_count = pipeline.evaluate_gate(
        window,
        _ready_probabilities(window),
        gate_threshold=0.6,
        gate_config_revision=5,
    )

    result = pipeline.process_prepared(
        window,
        decision,
        active_frame_count,
        physical_6plus1_geometry(),
        DirectionScanConfig.from_project(config),
        music_effective_order=effective_order,
        scan_config_revision=8,
        direction_id_tracking_enabled=False,
    )

    assert result.state is Layer2ExecutionState.PROCESSED
    assert scanner.orders == [effective_order]
    assert result.music_effective_order == effective_order
    assert result.music_skip_reason is None
    assert result.search_diagnostics is not None
    assert result.search_diagnostics.config_revision == 8
    assert result.search_diagnostics.model_order.estimated_sources == effective_order
    assert result.search_diagnostics.effective_model_order == effective_order
    assert result.spatial_response is not None
    assert result.spatial_response.model_order is not None
    assert result.spatial_response.model_order.estimated_sources == effective_order


def test_process_prepared_rejects_gate_from_a_different_window_before_music() -> None:
    class ForbiddenScanner:
        def scan_detailed(self, *args, **kwargs):
            raise AssertionError("a mismatched Gate must be rejected before MUSIC")

    config = load_config(CONFIG, environ={})
    pipeline = Layer2Pipeline.from_project(config, scanner=ForbiddenScanner())
    audio = _audio((30.0,), seed=71)
    gate_window = _window(audio, 0)
    current_window = _window(audio, 1)
    decision, active_frame_count = pipeline.evaluate_gate(
        gate_window,
        _ready_probabilities(gate_window),
        gate_threshold=0.6,
        gate_config_revision=6,
    )

    with pytest.raises(ValueError, match="prepared L2 Gate does not match the current window"):
        pipeline.process_prepared(
            current_window,
            decision,
            active_frame_count,
            physical_6plus1_geometry(),
            DirectionScanConfig.from_project(config),
            music_effective_order=1,
            scan_config_revision=9,
            direction_id_tracking_enabled=False,
        )


def test_pipeline_requires_one_continuously_open_covariance_context_before_music_angles() -> None:
    config = load_config(CONFIG, environ={})
    pipeline = Layer2Pipeline.from_project(config)
    audio = _audio((30.0,), seed=37, samples=7_680 + 15 * 960)
    scan_config = DirectionScanConfig.from_project(config)
    initial_window_frames = 1 + (7_680 - scan_config.win_length) // scan_config.hop_length

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
        assert closed.music_state is None
        assert len(pipeline.scanner._frame_covariances) == 0

    first_open_window = _window(audio, 4)
    first_open = pipeline.process(
        first_open_window, probabilities(first_open_window, 1.0),
        physical_6plus1_geometry(), scan_config,
        gate_threshold=0.6, gate_config_revision=0,
    )
    assert first_open.spatial_response is not None
    assert first_open.search_diagnostics is not None
    assert first_open.search_diagnostics.model_order.snapshot_count == initial_window_frames
    assert first_open.search_diagnostics.active_frame_count == 1
    assert first_open.search_diagnostics.birth_required_active_frames == 10
    assert not first_open.search_diagnostics.births_allowed
    assert first_open.candidates == ()
    assert first_open.music_state is not None
    assert first_open.music_state.state == "rebuilt"
    assert first_open.music_state.added_frames == initial_window_frames
    assert first_open.music_state.removed_frames == 0
    assert first_open.music_state.reason == "new_stream"
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
    assert closed.music_state is None
    assert len(pipeline.scanner._frame_covariances) == 0
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
    assert reopened.music_state is not None
    assert reopened.music_state.state == "rebuilt"
    assert reopened.music_state.added_frames == initial_window_frames
    assert all(not item.is_observed for item in reopened.directions)


def test_pipeline_tracking_off_publishes_only_raw_music_peaks_and_reenable_resets_ids() -> None:
    config = load_config(CONFIG, environ={})
    pipeline = Layer2Pipeline.from_project(config)
    audio = _audio((30.0,), seed=31, samples=7_680 + 12 * 960)

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


def test_gate_activity_resets_when_intermediate_windows_are_missing() -> None:
    config = load_config(CONFIG, environ={})
    pipeline = Layer2Pipeline.from_project(config)
    audio = _audio((30.0,), seed=41, samples=7_680 + 10 * 960)
    scan_config = DirectionScanConfig.from_project(config)

    def probabilities(window: DecisionWindow) -> tuple[SourceProbability20ms, ...]:
        return tuple(SourceProbability20ms(
            window.session_id, window.stream_epoch, start, start + 960, 1.0,
            SourceProbabilityState.READY, "ready",
        ) for start in (window.doa_start_sample, window.doa_start_sample + 960))

    counts = []
    for index in (0, 2, 4, 6, 8):
        window = _window(audio, index)
        result = pipeline.process(
            window, probabilities(window), physical_6plus1_geometry(), scan_config,
            gate_threshold=0.6, gate_config_revision=0,
        )
        assert result.search_diagnostics is not None
        counts.append(result.search_diagnostics.active_frame_count)
    assert counts == [1, 1, 1, 1, 1]


def test_music_model_order_property_is_initialized_and_updated() -> None:
    config = load_config(CONFIG, environ={})
    scanner = RollingNormMusicScanner()
    assert scanner.model_order is None
    audio = _audio((30.0,), seed=43, samples=7_680)
    scanner.scan(
        _window(audio, 0),
        physical_6plus1_geometry(),
        DirectionScanConfig.from_project(config),
    )
    assert scanner.model_order is not None
    scanner.reset()
    assert scanner.model_order is None


@pytest.mark.parametrize("stride", (2, 3, 5, 10))
def test_sparse_full_scan_keeps_complete_rolling_covariance(stride: int) -> None:
    config = load_config(CONFIG, environ={})
    scan = DirectionScanConfig.from_project(config)
    scanner = RollingNormMusicScanner()
    audio = _audio((30.0,), seed=47, samples=7_680 + 12 * 960)
    last_diagnostics = None

    for index in range(11):
        window = _window(audio, index)
        if index % stride == 0:
            _, _, last_diagnostics = scanner.scan_detailed(
                window, physical_6plus1_geometry(), scan,
            )
        else:
            scanner.observe_covariance(window, scan)

    assert last_diagnostics is not None
    assert last_diagnostics.model_order.snapshot_count == 19
