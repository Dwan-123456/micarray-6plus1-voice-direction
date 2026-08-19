from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np

from common.angle import THETA_DEGREES, circular_distance_deg
from common.data_types import CandidateDirection, DecisionWindow, SpatialResponse
from common.geometry import MicGeometry

from .candidates import rank_candidate_indices, robust_z_sigmoid, select_candidate_indices
from .configuration import DirectionScanConfig
from .interface import DirectionScanError
from .iterative import CandidateSearchDiagnostics, CandidateSearchEvidence


@dataclass(frozen=True, slots=True)
class _PairPlan:
    pairs: tuple[tuple[int, int], ...]
    left: np.ndarray
    right: np.ndarray


def _build_pair_plan() -> _PairPlan:
    pairs = tuple(combinations(range(7), 2))
    left = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    right = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    left.setflags(write=False)
    right.setflags(write=False)
    return _PairPlan(pairs, left, right)


_PAIR_PLAN = _build_pair_plan()


@lru_cache(maxsize=4)
def _periodic_hann(length: int) -> np.ndarray:
    result = np.asarray(np.hanning(length + 1)[:-1], dtype=np.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class _FrontendPlan:
    window: np.ndarray
    frequencies: np.ndarray
    band: np.ndarray


@lru_cache(maxsize=16)
def _frontend_plan(
    frame_samples: int,
    sample_rate: int,
    n_fft: int,
    window_name: str,
    frequency_min_hz: float,
    frequency_max_hz: float,
) -> _FrontendPlan:
    if window_name != "hann_periodic":
        raise DirectionScanError(f"unsupported SRP window: {window_name}")
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    band = (frequencies >= frequency_min_hz) & (frequencies <= frequency_max_hz)
    frequencies.setflags(write=False)
    band.setflags(write=False)
    return _FrontendPlan(_periodic_hann(frame_samples), frequencies, band)


@dataclass(frozen=True, slots=True)
class _FrontendArtifacts:
    spectrum: np.ndarray
    phat: np.ndarray
    frequencies: np.ndarray
    band: np.ndarray


class SrpPhatScanner:
    """Specification-locked single-pass scanner plus optional two-pass de-emphasis."""

    _pairs = _PAIR_PLAN.pairs
    _pair_left = _PAIR_PLAN.left
    _pair_right = _PAIR_PLAN.right
    _STEERING_CACHE_LIMIT = 16

    def __init__(self) -> None:
        self._steering_cache: OrderedDict[
            tuple[bytes, float, int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = OrderedDict()

    @staticmethod
    def _periodic_hann(length: int) -> np.ndarray:
        return _periodic_hann(length)

    def _frontend(self, samples: np.ndarray, config: DirectionScanConfig) -> _FrontendArtifacts:
        plan = _frontend_plan(
            1_920,
            48_000,
            config.n_fft,
            config.window,
            config.frequency_min_hz,
            config.frequency_max_hz,
        )
        signal = np.asarray(samples, dtype=np.float64)
        if signal.shape != (1_920, 7) or not np.isfinite(signal).all():
            raise DirectionScanError("DOA input must be finite [1920,7]")
        if config.remove_channel_mean:
            signal = signal - np.mean(signal, axis=0, keepdims=True)
        spectrum = np.fft.rfft(signal * plan.window[:, None], n=config.n_fft, axis=0)
        cross = np.asarray(
            (spectrum[:, self._pair_left] * np.conj(spectrum[:, self._pair_right])).T,
            dtype=np.complex128,
        )
        phat = cross / np.maximum(np.abs(cross), config.phat_epsilon)
        phat[:, ~plan.band] = 0.0
        return _FrontendArtifacts(spectrum, phat, plan.frequencies, plan.band)

    def _spectrum_and_phat(
        self, samples: np.ndarray, config: DirectionScanConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        artifacts = self._frontend(samples, config)
        return artifacts.spectrum, artifacts.phat, artifacts.frequencies, artifacts.band

    def _gcc_phat(self, samples: np.ndarray, config: DirectionScanConfig) -> np.ndarray:
        artifacts = self._frontend(samples, config)
        gcc = np.fft.irfft(
            artifacts.phat, n=config.n_fft * config.gcc_interpolation, axis=1
        )
        if not np.isfinite(gcc).all():
            raise DirectionScanError("GCC-PHAT produced non-finite values")
        return gcc

    def _steering_lookup(
        self, geometry: MicGeometry, sample_rate: int, gcc_length: int, interpolation: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (
            geometry.positions_m.tobytes(),
            geometry.speed_of_sound_mps,
            sample_rate,
            gcc_length,
            interpolation,
        )
        cached = self._steering_cache.get(key)
        if cached is not None:
            self._steering_cache.move_to_end(key)
            return cached
        radians = np.deg2rad(THETA_DEGREES.astype(np.float64))
        directions = np.column_stack((np.cos(radians), np.sin(radians)))
        arrival_seconds = -(geometry.positions_m @ directions.T) / geometry.speed_of_sound_mps
        pair_delays = np.asarray(
            [arrival_seconds[left] - arrival_seconds[right] for left, right in self._pairs], dtype=np.float64
        )
        positions = pair_delays * sample_rate * interpolation
        floors = np.floor(positions).astype(np.int64)
        fractions = positions - floors
        lookup = (floors % gcc_length, (floors + 1) % gcc_length, fractions)
        self._steering_cache[key] = lookup
        self._steering_cache.move_to_end(key)
        while len(self._steering_cache) > self._STEERING_CACHE_LIMIT:
            self._steering_cache.popitem(last=False)
        return lookup

    def _raw_from_gcc(
        self, gcc: np.ndarray, geometry: MicGeometry, config: DirectionScanConfig, sample_rate: int
    ) -> np.ndarray:
        left, right, fraction = self._steering_lookup(
            geometry, sample_rate, gcc.shape[1], config.gcc_interpolation
        )
        pair_axis = np.arange(len(self._pairs), dtype=np.int64)[:, None]
        interpolated = gcc[pair_axis, left] * (1.0 - fraction) + gcc[pair_axis, right] * fraction
        raw = np.asarray(np.mean(interpolated, axis=0), dtype=np.float32)
        if raw.shape != (360,) or not np.isfinite(raw).all():
            raise DirectionScanError("SRP-PHAT response must be finite float32[360]")
        return raw

    def raw_spatial_response(
        self, samples: np.ndarray, geometry: MicGeometry, config: DirectionScanConfig, sample_rate: int = 48_000
    ) -> np.ndarray:
        return self._raw_from_gcc(self._gcc_phat(samples, config), geometry, config, sample_rate)

    @staticmethod
    def _response(window: DecisionWindow, raw: np.ndarray, normalized: np.ndarray) -> SpatialResponse:
        return SpatialResponse(
            window.session_id, window.stream_epoch, window.window_id, window.decision_sample,
            window.doa_start_sample, window.doa_end_sample, THETA_DEGREES, raw, normalized,
        )

    @staticmethod
    def _candidates(
        window: DecisionWindow, indices: tuple[int, ...] | list[int], raw: np.ndarray, normalized: np.ndarray
    ) -> tuple[CandidateDirection, ...]:
        return tuple(
            CandidateDirection(
                window.session_id, window.stream_epoch, window.window_id, window.decision_sample,
                window.doa_start_sample, window.doa_end_sample, float(index),
                float(raw[index]), float(normalized[index]),
            )
            for index in indices
        )

    def _single_pass(
        self, window: DecisionWindow, geometry: MicGeometry, config: DirectionScanConfig
    ) -> tuple[SpatialResponse, tuple[CandidateDirection, ...]]:
        response, candidates, _ = self._single_pass_detailed(window, geometry, config)
        return response, candidates

    def _single_pass_detailed(
        self, window: DecisionWindow, geometry: MicGeometry, config: DirectionScanConfig
    ) -> tuple[SpatialResponse, tuple[CandidateDirection, ...], int]:
        raw = self.raw_spatial_response(window.samples[-1_920:, :7], geometry, config, window.sample_rate)
        normalized = robust_z_sigmoid(raw, config)
        eligible = rank_candidate_indices(normalized, config)
        selected = eligible[: config.max_candidates]
        return (
            self._response(window, raw, normalized),
            self._candidates(window, selected, raw, normalized),
            len(eligible),
        )

    @staticmethod
    def _fixed_scale_sigmoid(raw: np.ndarray, scale: float, config: DirectionScanConfig) -> np.ndarray:
        z_score = (np.asarray(raw, dtype=np.float64) - float(np.median(raw))) / scale
        logits = np.clip(config.normalization_alpha * (z_score - config.normalization_beta), -80.0, 80.0)
        return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float32)

    def _support_and_suppression(
        self, phat: np.ndarray, weights: np.ndarray, frequencies: np.ndarray, geometry: MicGeometry,
        theta_deg: float, config: DirectionScanConfig,
    ) -> tuple[np.ndarray, int, int]:
        direction = np.asarray((np.cos(np.deg2rad(theta_deg)), np.sin(np.deg2rad(theta_deg))))
        arrival = -(geometry.positions_m @ direction) / geometry.speed_of_sound_mps
        delays = np.asarray([arrival[left] - arrival[right] for left, right in self._pairs])
        steering = np.exp(2j * np.pi * delays[:, None] * frequencies[None, :])
        agreement = np.clip(np.real(phat * steering), 0.0, 1.0) ** config.iterative_phase_power
        active = weights > 0.0
        supported = (agreement >= config.iterative_pair_phase_threshold) & active
        active_per_pair = np.maximum(1, np.count_nonzero(active, axis=1))
        pair_support = int(np.count_nonzero(np.count_nonzero(supported, axis=1) / active_per_pair >= 0.25))
        frequency_counts = np.count_nonzero(supported, axis=0)
        frequency_support = int(np.count_nonzero(frequency_counts >= config.iterative_min_pair_support))
        consensus = np.clip(
            (frequency_counts - config.iterative_min_pair_support + 1)
            / max(1, len(self._pairs) - config.iterative_min_pair_support + 1), 0.0, 1.0
        )
        return agreement * consensus[None, :], pair_support, frequency_support

    def _iterative_scan(
        self, window: DecisionWindow, geometry: MicGeometry, config: DirectionScanConfig, revision: int
    ) -> tuple[SpatialResponse, tuple[CandidateDirection, ...], CandidateSearchDiagnostics]:
        artifacts = self._frontend(window.samples[-1_920:, :7], config)
        spectrum, phat, frequencies, band = (
            artifacts.spectrum,
            artifacts.phat,
            artifacts.frequencies,
            artifacts.band,
        )
        weights = np.broadcast_to(band, phat.shape).astype(np.float64).copy()
        raw0 = self._raw_from_gcc(
            np.fft.irfft(phat, n=config.n_fft * config.gcc_interpolation, axis=1),
            geometry, config, window.sample_rate,
        )
        norm0 = robust_z_sigmoid(raw0, config)
        response = self._response(window, raw0, norm0)
        initial_eligible = rank_candidate_indices(norm0, config)
        initial = initial_eligible[: config.max_candidates]
        if not initial:
            return response, (), CandidateSearchDiagnostics(
                "iterative_rank1_projection_v1", "iterative_rank1_projection_v1", revision,
                1, "no_initial_candidate", 1.0, eligible_peak_count=0,
            )

        first = int(initial[0])
        selected = [first]
        suppression, pair1, freq1 = self._support_and_suppression(
            phat, weights, frequencies, geometry, float(first), config
        )
        evidence = [CandidateSearchEvidence(float(first), 0, float(raw0[first]), float(norm0[first]), pair1, freq1)]
        weights *= 1.0 - config.iterative_suppression_strength * suppression
        remaining = float(np.mean(weights[:, band]))
        reason, iterations = "search_complete", 1

        if remaining < config.iterative_min_remaining_weight_ratio:
            reason = "insufficient_remaining_weight"
        elif pair1 < config.iterative_min_pair_support or freq1 < config.iterative_min_frequency_support:
            reason = "insufficient_initial_support"
        else:
            direction = np.asarray((np.cos(np.deg2rad(first)), np.sin(np.deg2rad(first))))
            arrival = -(geometry.positions_m @ direction) / geometry.speed_of_sound_mps
            steering = np.exp(-2j * np.pi * frequencies[:, None] * arrival[None, :])
            source = np.sum(np.conj(steering) * spectrum, axis=1) / spectrum.shape[1]
            coherence = np.abs(np.sum(np.conj(steering) * spectrum, axis=1)) / np.maximum(
                np.sqrt(spectrum.shape[1]) * np.linalg.norm(spectrum, axis=1), config.phat_epsilon
            )
            residual_spectrum = spectrum - (
                config.iterative_suppression_strength * coherence**config.iterative_phase_power
            )[:, None] * steering * source[:, None]
            residual_cross = (
                residual_spectrum[:, self._pair_left]
                * np.conj(residual_spectrum[:, self._pair_right])
            ).T
            original_cross = (
                spectrum[:, self._pair_left] * np.conj(spectrum[:, self._pair_right])
            ).T
            residual_validity = np.clip(
                np.abs(residual_cross) / np.maximum(np.abs(original_cross), config.phat_epsilon), 0.0, 1.0
            ) ** 0.25
            residual_phat = residual_cross / np.maximum(np.abs(residual_cross), config.phat_epsilon)
            residual_phat *= residual_validity
            residual_phat[:, ~band] = 0.0
            residual_raw = self._raw_from_gcc(
                np.fft.irfft(residual_phat, n=config.n_fft * config.gcc_interpolation, axis=1),
                geometry, config, window.sample_rate,
            )
            scale0 = max(1.4826 * float(np.median(np.abs(raw0 - np.median(raw0)))), 1e-6)
            residual_norm = self._fixed_scale_sigmoid(residual_raw, scale0, config)
            first_height = max(float(raw0[first] - np.median(raw0)), config.phat_epsilon)
            second = next((
                int(index) for index in select_candidate_indices(residual_norm, config)
                if circular_distance_deg(int(index), first) >= config.min_peak_distance_deg
                and float(residual_raw[index] - np.median(residual_raw))
                >= config.iterative_min_residual_peak_ratio * first_height
            ), None)
            if second is None:
                reason = "no_residual_candidate"
            else:
                _, pair2, freq2 = self._support_and_suppression(
                    residual_phat, weights * residual_validity, frequencies, geometry, float(second), config
                )
                if pair2 < config.iterative_min_pair_support or freq2 < config.iterative_min_frequency_support:
                    reason = "insufficient_residual_support"
                else:
                    selected.append(second)
                    iterations = 2
                    reason = "candidate_limit_reached"
                    evidence.append(CandidateSearchEvidence(
                        float(second), 1, float(residual_raw[second]), float(residual_norm[second]), pair2, freq2
                    ))

        selected.sort(key=lambda index: (-float(norm0[index]), index))
        diagnostics = CandidateSearchDiagnostics(
            "iterative_rank1_projection_v1", "iterative_rank1_projection_v1", revision,
            iterations, reason, remaining, evidence=tuple(evidence),
            eligible_peak_count=len(selected), candidate_limit_applied=len(selected) == config.max_candidates,
        )
        return response, self._candidates(window, selected, raw0, norm0), diagnostics

    def scan_detailed(
        self, window: DecisionWindow, geometry: MicGeometry, config: DirectionScanConfig, config_revision: int = 0
    ) -> tuple[SpatialResponse, tuple[CandidateDirection, ...], CandidateSearchDiagnostics]:
        if window.sample_rate != 48_000:
            raise DirectionScanError("Layer 2 only accepts 48 kHz DecisionWindow")
        if not config.iterative_peak_search_enabled:
            response, candidates, eligible_count = self._single_pass_detailed(window, geometry, config)
            return response, candidates, CandidateSearchDiagnostics(
                "single_pass", "srp_phat_single_pass_v1", config_revision, 1,
                "candidate_limit_reached" if eligible_count > config.max_candidates else "single_pass",
                1.0, eligible_peak_count=eligible_count,
                candidate_limit_applied=eligible_count > config.max_candidates,
            )
        try:
            return self._iterative_scan(window, geometry, config, config_revision)
        except (DirectionScanError, FloatingPointError, np.linalg.LinAlgError) as exc:
            response, candidates, eligible_count = self._single_pass_detailed(window, geometry, config)
            return response, candidates, CandidateSearchDiagnostics(
                "iterative_rank1_projection_v1", "iterative_rank1_projection_v1", config_revision,
                1, "legacy_fallback", 1.0, fallback_reason=f"{type(exc).__name__}: {exc}",
                eligible_peak_count=eligible_count,
                candidate_limit_applied=eligible_count > config.max_candidates,
            )

    def scan(
        self, window: DecisionWindow, geometry: MicGeometry, config: DirectionScanConfig
    ) -> tuple[SpatialResponse, tuple[CandidateDirection, ...]]:
        response, candidates, _ = self.scan_detailed(window, geometry, config)
        return response, candidates
