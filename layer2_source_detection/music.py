"""Rolling broadband frequency-normalized MUSIC for the L2 runtime.

This is a project-specific implementation of Schmidt's MUSIC formulation and
the per-frequency normalization described by Pyroomacoustics NormMUSIC (MIT).
The signal-subspace order is supplied explicitly by the Test UI. No Israel Cohen MUSIC source is
claimed or copied; his material is used only for the separately documented
noise-estimation background.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.signal import find_peaks

from common.data_types import CandidateDirection, DecisionWindow, ModelOrderEstimate, SpatialResponse
from common.geometry import MicGeometry

from .configuration import DirectionScanConfig
from .interface import DirectionScanError


@dataclass(frozen=True, slots=True)
class MusicPeakEvidence:
    theta_deg: float
    search_iteration: int
    residual_raw_score: float
    fixed_reference_normalized_score: float
    supporting_pairs: int
    supporting_frequency_bins: int
    frequency_support_ratio: float = 0.0
    mean_plane_wave_fit: float = 0.0
    supporting_frequency_subbands: int = 0
    circular_concentration: float = 0.0
    cluster_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class _DpdVoteCluster:
    angle_index: int
    supporting_frequency_bins: int
    frequency_support_ratio: float
    mean_plane_wave_fit: float
    supporting_frequency_subbands: int
    circular_concentration: float
    cluster_weight: float
    supporting_frequency_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class MusicStateDiagnostic:
    state: str
    previous_decision_sample: int | None
    decision_sample: int
    gap_samples: int
    reused_frames: int
    added_frames: int
    removed_frames: int
    steering_cache_rebuilt: bool
    reason: str


@dataclass(frozen=True, slots=True)
class MusicDiagnostics:
    mode: str
    algorithm_version: str
    config_revision: int
    model_order: ModelOrderEstimate
    state: MusicStateDiagnostic
    valid_frequency_bins: int
    covariance_quality: str
    iterations_used: int = 1
    stop_reason: str = "manual_order_greedy_peak_search"
    remaining_weight_ratio: float = 1.0
    fallback_reason: str | None = None
    evidence: tuple[MusicPeakEvidence, ...] = ()
    eligible_peak_count: int = 0
    candidate_limit: int = 3
    candidate_limit_applied: bool = False
    effective_model_order: int | None = None
    births_allowed: bool = True
    active_frame_count: int = 0
    birth_required_active_frames: int = 0
    dpd_rank1_enabled: bool = False
    selected_frequency_bins: int = 0
    noise_whitening_enabled: bool = False
    whitening_status: str = "disabled"
    imcra_noise_hops: int = 0
    covariance_update_ms: float = 0.0
    eigensolve_ms: float = 0.0
    spectrum_ms: float = 0.0
    total_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.mode != "frequency_normalized_music" or self.config_revision < 0:
            raise ValueError("invalid L2 DOA diagnostics identity")
        if self.valid_frequency_bins < 0 or self.covariance_quality not in {
            "ready", "degraded", "failed",
        }:
            raise ValueError("invalid MUSIC covariance diagnostics")
        effective_model_order = (
            min(self.model_order.estimated_sources, 3)
            if self.effective_model_order is None else self.effective_model_order
        )
        if not 0 <= effective_model_order <= 3:
            raise ValueError("effective MUSIC order must be 0..3")
        if type(self.births_allowed) is not bool:
            raise TypeError("MUSIC birth flag must be bool")
        if type(self.dpd_rank1_enabled) is not bool or type(self.noise_whitening_enabled) is not bool:
            raise TypeError("MUSIC DPD/whitening flags must be bool")
        if min(
            self.selected_frequency_bins,
            self.imcra_noise_hops,
            self.active_frame_count,
            self.birth_required_active_frames,
        ) < 0:
            raise ValueError("MUSIC DPD/noise counts must be non-negative")
        if self.whitening_status not in {
            "disabled", "imcra_spatial_covariance", "unavailable",
        }:
            raise ValueError("invalid MUSIC whitening status")
        timings = (self.covariance_update_ms, self.eigensolve_ms, self.spectrum_ms, self.total_ms)
        if not all(np.isfinite(value) and value >= 0.0 for value in timings):
            raise ValueError("invalid MUSIC timing diagnostics")
        object.__setattr__(self, "effective_model_order", effective_model_order)
        object.__setattr__(self, "evidence", tuple(self.evidence))


class RollingNormMusicScanner:
    """Incremental frequency-normalized MUSIC owned by the single L2 worker."""

    algorithm_version = "frequency_normalized_music_manual_order_v9"

    def __init__(self) -> None:
        self._stream_key: tuple[str, int] | None = None
        self._last_sample: int | None = None
        self._frame_covariances: deque[np.ndarray] = deque()
        self._covariance_sum: np.ndarray | None = None
        self._frequency_indices: np.ndarray | None = None
        self._frequencies_hz: np.ndarray | None = None
        self._frequency_key: tuple[object, ...] | None = None
        self._geometry_weights: np.ndarray | None = None
        self._analysis_window_key: int | None = None
        self._analysis_window: np.ndarray | None = None
        self._steering: np.ndarray | None = None
        self._steering_energy: np.ndarray | None = None
        self._steering_key: tuple[object, ...] | None = None
        self._identity_7 = np.eye(7, dtype=np.complex128)[None, :, :]
        self._theta_degrees = np.arange(360, dtype=np.float32)
        self._last_covariance_update_ms = 0.0
        self.last_state_diagnostic: MusicStateDiagnostic | None = None

    def reset(self) -> None:
        self._stream_key = None
        self._last_sample = None
        self._frame_covariances.clear()
        self._covariance_sum = None
        self._last_covariance_update_ms = 0.0
        self.last_state_diagnostic = None

    @staticmethod
    def _greedy_circular_peaks(
        normalized: np.ndarray,
        config: DirectionScanConfig,
        limit: int,
    ) -> tuple[list[int], int, bool]:
        """Pick up to ``limit`` local maxima using the UI threshold and circular NMS."""
        tiled = np.tile(normalized, 3)
        peaks, _ = find_peaks(tiled, prominence=config.peak_prominence)
        eligible = sorted(
            (
                int(index - 360)
                for index in peaks
                if 360 <= index < 720
                and normalized[int(index - 360)] >= config.direction_threshold
            ),
            key=lambda index: (-float(normalized[index]), index),
        )
        chosen: list[int] = []
        for _ in range(limit):
            next_peak = next(
                (
                    index for index in eligible
                    if index not in chosen
                    and all(
                        abs(((index - old + 180) % 360) - 180)
                        >= config.min_peak_distance_deg
                        for old in chosen
                    )
                ),
                None,
            )
            if next_peak is None:
                break
            chosen.append(next_peak)
        unselected_separated_peak = any(
            index not in chosen
            and all(
                abs(((index - old + 180) % 360) - 180)
                >= config.min_peak_distance_deg
                for old in chosen
            )
            for index in eligible
        )
        return chosen, len(eligible), bool(len(chosen) == limit and unselected_separated_peak)

    @staticmethod
    def _periodic_hann(length: int) -> np.ndarray:
        return np.hanning(length + 1)[:-1].astype(np.float64)

    def _prepare_frequency_axis(self, config: DirectionScanConfig) -> None:
        key = (
            config.n_fft,
            config.frequency_min_hz,
            config.frequency_max_hz,
            config.min_valid_frequency_bins,
        )
        if key == self._frequency_key:
            return
        frequencies = np.fft.rfftfreq(config.n_fft, 1.0 / 48_000.0)
        indices = np.flatnonzero(
            (frequencies >= config.frequency_min_hz)
            & (frequencies <= config.frequency_max_hz)
        )
        if indices.size < config.min_valid_frequency_bins:
            raise DirectionScanError("MUSIC has too few configured frequency bins")
        self._frequency_indices = indices
        self._frequencies_hz = frequencies[indices]
        self._geometry_weights = self._geometry_frequency_weights(self._frequencies_hz)
        self._frequency_key = key
        self._steering = None
        self._steering_energy = None
        self._steering_key = None

    def _frame_covariance(self, frame: np.ndarray, config: DirectionScanConfig) -> np.ndarray:
        self._prepare_frequency_axis(config)
        assert self._frequency_indices is not None
        if self._analysis_window_key != config.win_length:
            self._analysis_window = self._periodic_hann(config.win_length)
            self._analysis_window_key = config.win_length
        assert self._analysis_window is not None
        physical = np.asarray(frame[:, :7], dtype=np.float64)
        physical = physical - physical.mean(axis=0, keepdims=True)
        spectrum = np.fft.rfft(
            physical * self._analysis_window[:, None],
            n=config.n_fft,
            axis=0,
        )[self._frequency_indices]
        return spectrum[:, :, None] * spectrum[:, None, :].conj()

    def _target_frequencies(self, config: DirectionScanConfig) -> np.ndarray:
        self._prepare_frequency_axis(config)
        assert self._frequencies_hz is not None
        return self._frequencies_hz

    @staticmethod
    def _geometry_frequency_weights(frequencies_hz: np.ndarray) -> np.ndarray:
        """Return the fixed 2--4 kHz weights for this 4 cm array geometry."""
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        weights = np.zeros_like(frequencies)
        weights[(frequencies >= 2_000.0) & (frequencies < 2_300.0)] = 0.35
        weights[(frequencies >= 2_300.0) & (frequencies < 2_500.0)] = 0.55
        weights[(frequencies >= 2_500.0) & (frequencies < 2_700.0)] = 0.75
        weights[(frequencies >= 2_700.0) & (frequencies < 3_000.0)] = 0.90
        weights[(frequencies >= 3_000.0) & (frequencies < 3_600.0)] = 1.00
        falling_36_38 = (frequencies >= 3_600.0) & (frequencies < 3_800.0)
        weights[falling_36_38] = (
            1.00 - 0.25 * (frequencies[falling_36_38] - 3_600.0) / 200.0
        )
        falling_38_40 = (frequencies >= 3_800.0) & (frequencies <= 4_000.0)
        weights[falling_38_40] = (
            0.75 - 0.30 * (frequencies[falling_38_40] - 3_800.0) / 200.0
        )
        return weights

    @classmethod
    def _geometry_weighted_mean(
        cls,
        per_frequency: np.ndarray,
        frequencies_hz: np.ndarray,
    ) -> np.ndarray:
        """Fuse independently normalized MUSIC spectra using fixed weights."""
        spectra = np.asarray(per_frequency, dtype=np.float64)
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        if spectra.ndim != 2 or spectra.shape[0] != frequencies.size:
            raise ValueError("MUSIC frequency spectra and frequency axis do not match")
        weights = cls._geometry_frequency_weights(frequencies)
        total_weight = float(np.sum(weights))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise DirectionScanError("MUSIC fixed frequency weights have no support")
        return np.sum(spectra * weights[:, None], axis=0) / total_weight

    @staticmethod
    def _interpolate_imcra(
        source_frequencies: np.ndarray, values: np.ndarray, target_frequencies: np.ndarray
    ) -> np.ndarray:
        values = np.asarray(
            values,
            dtype=np.complex128 if np.iscomplexobj(values) else np.float64,
        )
        upper = np.searchsorted(source_frequencies, target_frequencies, side="left")
        upper = np.clip(upper, 1, len(source_frequencies) - 1)
        lower = upper - 1
        lower_frequency = source_frequencies[lower]
        fraction = (
            (target_frequencies - lower_frequency)
            / np.maximum(source_frequencies[upper] - lower_frequency, 1.0e-12)
        )
        return (
            values[..., lower] * (1.0 - fraction)
            + values[..., upper] * fraction
        )

    def _imcra_metrics(
        self, window: DecisionWindow, config: DirectionScanConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        frequencies = self._target_frequencies(config)
        ready = tuple(hop for hop in window.imcra_hops if hop.state == "ready")
        if not ready:
            ones = np.ones(frequencies.size, dtype=np.float64)
            return ones, ones, None
        source_frequencies = ready[0].frequencies_hz
        spp = np.mean(self._interpolate_imcra(
            source_frequencies,
            np.stack([hop.spp for hop in ready]),
            frequencies,
        ), axis=(0, 1))
        prior_snr = np.mean(self._interpolate_imcra(
            source_frequencies,
            np.stack([hop.prior_snr for hop in ready]),
            frequencies,
        ), axis=(0, 1))
        if any(hop.noise_covariance is None for hop in ready):
            noise_covariance = None
        else:
            stacked_covariance = np.stack([hop.noise_covariance for hop in ready])
            covariance_frequency_last = np.moveaxis(stacked_covariance, 1, -1)
            interpolated_covariance = self._interpolate_imcra(
                source_frequencies,
                covariance_frequency_last,
                frequencies,
            )
            noise_covariance = np.moveaxis(
                np.mean(interpolated_covariance, axis=0), -1, 0,
            )
            noise_covariance = 0.5 * (
                noise_covariance + noise_covariance.conj().transpose(0, 2, 1)
            )
        return np.clip(spp, 0.0, 1.0), np.maximum(prior_snr, 0.0), noise_covariance

    def _whiten(
        self,
        covariance: np.ndarray,
        steering: np.ndarray,
        window: DecisionWindow,
        config: DirectionScanConfig,
        imcra_metrics: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if not config.noise_whitening_enabled:
            return covariance, steering, "disabled"
        _, _, noise_covariance = (
            self._imcra_metrics(window, config)
            if imcra_metrics is None
            else imcra_metrics
        )
        if noise_covariance is None:
            return covariance, steering, "unavailable"
        status = "imcra_spatial_covariance"
        trace = np.real(np.trace(noise_covariance, axis1=1, axis2=2)) / 7.0
        trace = np.maximum(trace, config.eigenvalue_floor)
        effective_covariance = (
            (1.0 - config.noise_covariance_shrinkage) * noise_covariance
            + config.noise_covariance_shrinkage * trace[:, None, None] * self._identity_7
            + config.diagonal_loading * trace[:, None, None] * self._identity_7
        )
        effective_covariance = 0.5 * (
            effective_covariance + effective_covariance.conj().transpose(0, 2, 1)
        )
        if not np.isfinite(effective_covariance).all():
            return covariance, steering, "unavailable"
        try:
            factor = np.linalg.cholesky(effective_covariance)
            left = np.linalg.solve(factor, covariance)
            whitened_covariance = np.linalg.solve(
                factor, left.conj().transpose(0, 2, 1),
            ).conj().transpose(0, 2, 1)
            whitened_steering = np.linalg.solve(
                factor, steering.transpose(0, 2, 1),
            ).transpose(0, 2, 1)
        except np.linalg.LinAlgError:
            return covariance, steering, "unavailable"
        whitened_covariance = 0.5 * (
            whitened_covariance + whitened_covariance.conj().transpose(0, 2, 1)
        )
        norm = np.linalg.norm(whitened_steering, axis=2, keepdims=True)
        whitened_steering /= np.maximum(norm, 1.0e-12)
        return whitened_covariance, whitened_steering, status

    @staticmethod
    def _circular_distance_grid(angles: np.ndarray, center: float) -> np.ndarray:
        return np.abs((angles - center + 180.0) % 360.0 - 180.0)

    def _dpd_rank1_spectrum(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        steering: np.ndarray,
        window: DecisionWindow,
        config: DirectionScanConfig,
        frequency_mask: np.ndarray,
        geometry_weights: np.ndarray,
        steering_energy: np.ndarray | None,
        imcra_metrics: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        principal = eigenvectors[:, :, -1]
        principal_projection = np.abs(
            np.einsum("fac,fc->fa", steering.conj(), principal, optimize=True)
        ) ** 2
        total_energy = (
            np.sum(np.abs(steering) ** 2, axis=2)
            if steering_energy is None
            else steering_energy
        )
        denominator = np.maximum(total_energy - principal_projection, 1.0e-12)
        per_frequency = 1.0 / np.maximum(denominator, 1.0e-12)
        per_frequency /= np.maximum(per_frequency.max(axis=1, keepdims=True), 1.0e-12)

        plane_fit_by_angle = principal_projection / np.maximum(total_energy, 1.0e-12)
        plane_fit = np.max(plane_fit_by_angle, axis=1)
        peak_angles = np.argmax(per_frequency, axis=1).astype(np.float64)
        eigen_ratio = eigenvalues[:, -1] / np.maximum(eigenvalues[:, -2], 1.0e-12)
        spp, prior_snr, _ = (
            self._imcra_metrics(window, config)
            if imcra_metrics is None
            else imcra_metrics
        )
        spp = spp[frequency_mask]
        prior_snr = prior_snr[frequency_mask]
        eig_weight = np.clip(
            (eigen_ratio - config.dpd_min_eigenvalue_ratio)
            / np.maximum(eigen_ratio, 1.0e-12),
            0.0,
            1.0,
        )
        snr_weight = prior_snr / (1.0 + prior_snr)
        weights = (
            eig_weight
            * plane_fit
            * np.maximum(spp, 0.05)
            * np.maximum(snr_weight, 0.05)
            * geometry_weights
        )
        selected = (
            (eigen_ratio >= config.dpd_min_eigenvalue_ratio)
            & (plane_fit >= config.dpd_min_plane_wave_fit)
            & np.isfinite(weights)
            & (weights > 0.0)
        )
        grid = np.arange(360, dtype=np.float64)
        vote = np.zeros(360, dtype=np.float64)
        for angle, weight in zip(peak_angles[selected], weights[selected], strict=True):
            distance = self._circular_distance_grid(grid, float(angle))
            kernel = np.maximum(0.0, 1.0 - distance / (config.dpd_angle_tolerance_deg + 1.0))
            vote += float(weight) * kernel
        total_weight = float(np.sum(weights[selected]))
        if total_weight > 0.0:
            vote /= total_weight
        return vote, per_frequency, selected, weights, plane_fit

    def _dpd_vote_clusters(
        self,
        normalized_vote: np.ndarray,
        per_frequency: np.ndarray,
        selected: np.ndarray,
        weights: np.ndarray,
        plane_fit: np.ndarray,
        frequencies_hz: np.ndarray,
        config: DirectionScanConfig,
    ) -> tuple[_DpdVoteCluster, ...]:
        """Cluster reliable per-frequency rank-1 direction votes on a circle."""

        if not selected.any() or not np.any(normalized_vote > 0.0):
            return ()
        peak_angles = np.argmax(per_frequency, axis=1).astype(np.float64)
        tiled = np.tile(normalized_vote, 3)
        peaks, _ = find_peaks(tiled, prominence=config.peak_prominence)
        seeds = sorted(
            {int(index - 360) for index in peaks if 360 <= index < 720},
            key=lambda index: (-float(normalized_vote[index]), index),
        )
        total_weight = float(np.sum(weights[selected]))
        if total_weight <= 0.0:
            return ()
        subband_edges = np.linspace(
            config.frequency_min_hz,
            config.frequency_max_hz,
            config.dpd_frequency_subbands + 1,
        )
        frequency_subbands = np.clip(
            np.searchsorted(subband_edges, frequencies_hz, side="right") - 1,
            0,
            config.dpd_frequency_subbands - 1,
        )
        accepted: list[_DpdVoteCluster] = []
        for seed in seeds:
            supporters = selected & (
                self._circular_distance_grid(peak_angles, float(seed))
                <= config.dpd_angle_tolerance_deg
            )
            if not supporters.any():
                continue
            supporter_weights = weights[supporters]
            supporter_angles = np.deg2rad(peak_angles[supporters])
            vector = np.sum(supporter_weights * np.exp(1j * supporter_angles))
            weight_sum = float(np.sum(supporter_weights))
            if weight_sum <= 0.0:
                continue
            center = float(np.rad2deg(np.angle(vector)) % 360.0)
            # Refine membership once around the circular weighted mean so a
            # 359/0-degree cluster is treated exactly like every other one.
            supporters = selected & (
                self._circular_distance_grid(peak_angles, center)
                <= config.dpd_angle_tolerance_deg
            )
            supporter_weights = weights[supporters]
            supporter_angles = np.deg2rad(peak_angles[supporters])
            weight_sum = float(np.sum(supporter_weights))
            if weight_sum <= 0.0:
                continue
            vector = np.sum(supporter_weights * np.exp(1j * supporter_angles))
            center = float(np.rad2deg(np.angle(vector)) % 360.0)
            angle_index = int(np.rint(center)) % 360
            support_count = int(supporters.sum())
            support_ratio = weight_sum / total_weight
            concentration = float(abs(vector) / weight_sum)
            subband_count = int(np.unique(frequency_subbands[supporters]).size)
            mean_fit = float(
                np.average(plane_fit[supporters], weights=supporter_weights)
            )
            if (
                normalized_vote[angle_index] < config.direction_threshold
                or support_count < config.dpd_min_cluster_frequency_bins
                or support_ratio < config.dpd_min_frequency_support_ratio
                or subband_count < config.dpd_min_cluster_subbands
                or concentration < config.dpd_min_circular_concentration
            ):
                continue
            accepted.append(_DpdVoteCluster(
                angle_index,
                support_count,
                support_ratio,
                mean_fit,
                subband_count,
                concentration,
                weight_sum,
                tuple(int(index) for index in np.flatnonzero(supporters)),
            ))
        return tuple(accepted)

    def _merge_dpd_peak_clusters(
        self,
        clusters: tuple[_DpdVoteCluster, ...],
        normalized_vote: np.ndarray,
        per_frequency: np.ndarray,
        selected: np.ndarray,
        weights: np.ndarray,
        plane_fit: np.ndarray,
        frequencies_hz: np.ndarray,
        config: DirectionScanConfig,
    ) -> tuple[_DpdVoteCluster, ...]:
        """Fuse strong nearby DPD peaks using each unique frequency vote once."""

        if not clusters:
            return ()
        peak_angles = np.argmax(per_frequency, axis=1).astype(np.float64)
        total_weight = float(np.sum(weights[selected]))
        if total_weight <= 0.0:
            return ()
        subband_edges = np.linspace(
            config.frequency_min_hz,
            config.frequency_max_hz,
            config.dpd_frequency_subbands + 1,
        )
        frequency_subbands = np.clip(
            np.searchsorted(subband_edges, frequencies_hz, side="right") - 1,
            0,
            config.dpd_frequency_subbands - 1,
        )
        ranked = sorted(
            clusters,
            key=lambda item: (-float(item.cluster_weight), item.angle_index),
        )
        consumed: set[int] = set()
        fused: list[_DpdVoteCluster] = []
        for anchor_index, anchor in enumerate(ranked):
            if anchor_index in consumed:
                continue
            group = [anchor]
            consumed.add(anchor_index)
            anchor_is_strong = (
                normalized_vote[anchor.angle_index]
                > config.dpd_peak_fusion_min_normalized_score
            )
            if anchor_is_strong:
                for candidate_index, candidate in enumerate(ranked):
                    if candidate_index in consumed:
                        continue
                    if (
                        normalized_vote[candidate.angle_index]
                        <= config.dpd_peak_fusion_min_normalized_score
                    ):
                        continue
                    if all(
                        self._circular_distance_grid(
                            np.asarray([float(candidate.angle_index)]),
                            float(member.angle_index),
                        )[0] <= config.dpd_peak_fusion_distance_deg
                        for member in group
                    ):
                        group.append(candidate)
                        consumed.add(candidate_index)
            if len(group) == 1:
                fused.append(anchor)
                continue

            supporter_indices = np.asarray(sorted({
                index
                for member in group
                for index in member.supporting_frequency_indices
            }), dtype=np.int64)
            if supporter_indices.size == 0:
                fused.append(anchor)
                continue
            supporter_weights = weights[supporter_indices]
            weight_sum = float(np.sum(supporter_weights))
            if weight_sum <= 0.0:
                continue
            supporter_angles = np.deg2rad(peak_angles[supporter_indices])
            vector = np.sum(supporter_weights * np.exp(1j * supporter_angles))
            center = float(np.rad2deg(np.angle(vector)) % 360.0)
            angle_index = int(np.rint(center)) % 360
            support_count = int(supporter_indices.size)
            support_ratio = weight_sum / total_weight
            concentration = float(abs(vector) / weight_sum)
            subband_count = int(np.unique(frequency_subbands[supporter_indices]).size)
            mean_fit = float(np.average(
                plane_fit[supporter_indices], weights=supporter_weights,
            ))
            if (
                normalized_vote[angle_index] < config.direction_threshold
                or support_count < config.dpd_min_cluster_frequency_bins
                or support_ratio < config.dpd_min_frequency_support_ratio
                or subband_count < config.dpd_min_cluster_subbands
                or concentration < config.dpd_min_circular_concentration
            ):
                continue
            fused.append(_DpdVoteCluster(
                angle_index,
                support_count,
                support_ratio,
                mean_fit,
                subband_count,
                concentration,
                weight_sum,
                tuple(int(index) for index in supporter_indices),
            ))
        return tuple(sorted(
            fused,
            key=lambda item: (-float(item.cluster_weight), item.angle_index),
        ))

    @staticmethod
    def _nms_dpd_clusters(
        clusters: tuple[_DpdVoteCluster, ...],
        minimum_distance_deg: float,
    ) -> tuple[_DpdVoteCluster, ...]:
        selected: list[_DpdVoteCluster] = []
        for cluster in clusters:
            if any(
                abs(((cluster.angle_index - old.angle_index + 180) % 360) - 180)
                < minimum_distance_deg
                for old in selected
            ):
                continue
            selected.append(cluster)
        return tuple(selected)

    def _rebuild(self, window: DecisionWindow, config: DirectionScanConfig, reason: str) -> None:
        history_samples = config.context_ms * 48
        audio = window.samples[-history_samples:, :]
        frames = []
        for start in range(0, len(audio) - config.win_length + 1, config.hop_length):
            frames.append(self._frame_covariance(audio[start : start + config.win_length], config))
        if not frames:
            raise DirectionScanError("MUSIC history does not contain one complete STFT frame")
        self._frame_covariances = deque(frames)
        self._covariance_sum = np.sum(np.stack(frames), axis=0)
        previous = self._last_sample
        gap = 0 if previous is None else window.decision_sample - previous - 960
        self.last_state_diagnostic = MusicStateDiagnostic(
            "rebuilt", previous, window.decision_sample, max(0, gap), 0,
            len(frames), 0, False, reason,
        )

    def _advance(self, window: DecisionWindow, config: DirectionScanConfig) -> None:
        # Two 50%-overlapped frames become available per 20 ms decision hop.
        new_audio = window.samples[-1_440:, :]
        added = [
            self._frame_covariance(new_audio[start : start + config.win_length], config)
            for start in (0, config.hop_length)
        ]
        assert self._covariance_sum is not None
        old_count = len(self._frame_covariances)
        for covariance in added:
            self._frame_covariances.append(covariance)
            self._covariance_sum += covariance
        max_frames = 1 + (config.context_ms * 48 - config.win_length) // config.hop_length
        removed = 0
        while len(self._frame_covariances) > max_frames:
            self._covariance_sum -= self._frame_covariances.popleft()
            removed += 1
        self.last_state_diagnostic = MusicStateDiagnostic(
            "advanced", self._last_sample, window.decision_sample, 0,
            old_count, len(added), removed, False, "sample_continuous",
        )

    def observe_covariance(
        self, window: DecisionWindow, config: DirectionScanConfig
    ) -> MusicStateDiagnostic:
        """Maintain rolling MUSIC spatial covariance without eigensolve or peak search."""

        stream_key = (window.session_id, window.stream_epoch)
        if self._stream_key == stream_key and self._last_sample == window.decision_sample:
            assert self.last_state_diagnostic is not None
            return self.last_state_diagnostic
        started = perf_counter()
        continuous = self._stream_key == stream_key and self._last_sample is not None and (
            window.decision_sample == self._last_sample + 960
        )
        if not continuous:
            reason = "new_stream" if self._stream_key != stream_key else "sample_discontinuity"
            self._rebuild(window, config, reason)
        else:
            self._advance(window, config)
        self._stream_key = stream_key
        self._last_sample = window.decision_sample
        self._last_covariance_update_ms = (perf_counter() - started) * 1_000.0
        assert self.last_state_diagnostic is not None
        return self.last_state_diagnostic

    def _steering_tensor(
        self, geometry: MicGeometry, config: DirectionScanConfig, revision: int
    ) -> tuple[np.ndarray, bool]:
        self._prepare_frequency_axis(config)
        assert self._frequencies_hz is not None
        key = (
            geometry.version, geometry.speed_of_sound_mps,
            tuple(np.asarray(geometry.positions_m).ravel()),
            config.frequency_min_hz, config.frequency_max_hz, config.n_fft, revision,
        )
        rebuilt = key != self._steering_key
        if rebuilt:
            theta = np.deg2rad(np.arange(360, dtype=np.float64))
            direction = np.stack((np.cos(theta), np.sin(theta)), axis=1)
            # Project convention: a wave arriving from theta has microphone
            # delay -(position dot unit_direction)/c.
            delays = -(direction @ np.asarray(geometry.positions_m).T) / geometry.speed_of_sound_mps
            self._steering = np.exp(
                -2j * np.pi * self._frequencies_hz[:, None, None] * delays[None, :, :]
            )
            self._steering_energy = np.sum(np.abs(self._steering) ** 2, axis=2)
            self._steering_key = key
        assert self._steering is not None
        return self._steering, rebuilt

    @staticmethod
    def _noise_projection_denominator(
        eigenvectors: np.ndarray,
        steering: np.ndarray,
        signal_order: int,
        steering_energy: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute noise-subspace energy through its smaller orthogonal complement."""

        signal = eigenvectors[:, :, -signal_order:]
        projection = np.einsum(
            "fcn,fac->fan", signal.conj(), steering, optimize=True,
        )
        signal_energy = np.sum(np.abs(projection) ** 2, axis=2)
        total_energy = (
            np.sum(np.abs(steering) ** 2, axis=2)
            if steering_energy is None
            else steering_energy
        )
        return np.maximum(total_energy - signal_energy, 1.0e-12)

    def scan_detailed(
        self,
        window: DecisionWindow,
        geometry: MicGeometry,
        config: DirectionScanConfig,
        config_revision: int = 0,
    ) -> tuple[SpatialResponse, tuple[CandidateDirection, ...], MusicDiagnostics]:
        started = perf_counter()
        self.observe_covariance(window, config)
        covariance_updated = perf_counter()
        assert self._covariance_sum is not None
        snapshots = len(self._frame_covariances)
        covariance = self._covariance_sum / max(snapshots, 1)
        trace = np.real(np.trace(covariance, axis1=1, axis2=2)) / 7.0
        covariance = (
            (1.0 - config.covariance_shrinkage) * covariance
            + config.covariance_shrinkage * trace[:, None, None] * self._identity_7
            + config.diagonal_loading
            * np.maximum(trace, config.eigenvalue_floor)[:, None, None]
            * self._identity_7
        )
        steering, rebuilt = self._steering_tensor(geometry, config, config_revision)
        imcra_metrics = (
            self._imcra_metrics(window, config)
            if config.noise_whitening_enabled or config.dpd_rank1_enabled
            else None
        )
        covariance, steering, whitening_status = self._whiten(
            covariance, steering, window, config, imcra_metrics
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigensolved = perf_counter()
        valid = np.isfinite(eigenvalues).all(axis=1) & (eigenvalues[:, -1] > config.eigenvalue_floor)
        valid_count = int(valid.sum())
        if valid_count < config.min_valid_frequency_bins:
            raise DirectionScanError("MUSIC covariance quality left too few valid frequency bins")
        all_valid = valid_count == valid.size
        target_frequencies = self._target_frequencies(config)
        assert self._geometry_weights is not None
        if all_valid:
            valid_frequencies = target_frequencies
            geometry_weights = self._geometry_weights
            steering_energy = (
                self._steering_energy
                if whitening_status == "disabled" else None
            )
        else:
            eigenvalues = eigenvalues[valid]
            eigenvectors = eigenvectors[valid]
            steering = steering[valid]
            valid_frequencies = target_frequencies[valid]
            geometry_weights = self._geometry_weights[valid]
            steering_energy = (
                self._steering_energy[valid]
                if whitening_status == "disabled" and self._steering_energy is not None
                else None
            )
        manual_order = config.effective_order_limit
        # Keep the established ModelOrderEstimate DTO for downstream recording
        # compatibility. Its order is now the explicit operator selection and
        # the legacy MDL age is always zero; no model-order estimator runs.
        model_order = ModelOrderEstimate(
            manual_order, valid_count, snapshots, 1.0, 0, "ready",
        )
        if config.dpd_rank1_enabled:
            raw, per_frequency, selected, weights, plane_fit = self._dpd_rank1_spectrum(
                eigenvalues, eigenvectors, steering, window, config, valid,
                geometry_weights, steering_energy, imcra_metrics,
            )
            limit = config.effective_order_limit
            fallback_reason = (
                "imcra_noise_covariance_unavailable"
                if config.noise_whitening_enabled and whitening_status == "unavailable"
                else None
            )
        else:
            # The Test UI order selector directly supplies the signal-subspace
            # order and the maximum number of peaks to search.
            effective_order = manual_order
            limit = effective_order
            denominator = self._noise_projection_denominator(
                eigenvectors, steering, effective_order, steering_energy,
            )
            per_frequency = 1.0 / np.maximum(denominator, 1.0e-12)
            per_frequency /= np.maximum(per_frequency.max(axis=1, keepdims=True), 1.0e-12)
            total_weight = float(np.sum(geometry_weights))
            if not np.isfinite(total_weight) or total_weight <= 0.0:
                raise DirectionScanError("MUSIC fixed frequency weights have no support")
            raw = np.sum(per_frequency * geometry_weights[:, None], axis=0) / total_weight
            stop_reason = "manual_order_greedy_peak_search"
            fallback_reason = (
                "imcra_noise_covariance_unavailable"
                if config.noise_whitening_enabled and whitening_status == "unavailable"
                else None
            )
            births_allowed = True
        normalized = np.asarray(
            (raw - raw.min()) / max(float(raw.max() - raw.min()), 1.0e-12), dtype=np.float32
        )
        raw32 = np.asarray(raw, dtype=np.float32)
        response = SpatialResponse(
            window.session_id, window.stream_epoch, window.window_id, window.decision_sample,
            window.doa_start_sample, window.doa_end_sample,
            self._theta_degrees, raw32, normalized,
            model_order,
            valid_count,
            "ready" if model_order.status == "ready" else "degraded",
            self.algorithm_version,
        )
        support_by_index: dict[int, _DpdVoteCluster] = {}
        if config.dpd_rank1_enabled:
            qualified_clusters = self._dpd_vote_clusters(
                normalized,
                per_frequency,
                selected,
                weights,
                plane_fit,
                valid_frequencies,
                config,
            )
            fused_clusters = self._merge_dpd_peak_clusters(
                qualified_clusters,
                normalized,
                per_frequency,
                selected,
                weights,
                plane_fit,
                valid_frequencies,
                config,
            )
            separated_clusters = self._nms_dpd_clusters(
                fused_clusters, config.min_peak_distance_deg,
            )
            selected_clusters = separated_clusters[:limit]
            chosen = [item.angle_index for item in selected_clusters]
            support_by_index = {
                item.angle_index: item for item in selected_clusters
            }
            effective_order = len(chosen)
            births_allowed = bool(chosen)
            if not selected.any():
                stop_reason = "dpd_no_reliable_bins"
            elif not qualified_clusters:
                stop_reason = "dpd_no_qualified_clusters"
            elif not fused_clusters:
                stop_reason = "dpd_no_qualified_fused_clusters"
            else:
                stop_reason = "dpd_circular_clusters_with_peak_fusion"
            eligible_peak_count = len(separated_clusters)
            candidate_limit_applied = len(separated_clusters) > limit
        else:
            chosen, eligible_peak_count, candidate_limit_applied = self._greedy_circular_peaks(
                normalized, config, limit,
            )
            births_allowed = bool(chosen)
        candidates = tuple(
            CandidateDirection(
                window.session_id, window.stream_epoch, window.window_id, window.decision_sample,
                window.doa_start_sample, window.doa_end_sample, float(index),
                float(raw32[index]), float(normalized[index]),
            )
            for index in chosen
        )
        spectrum_built = perf_counter()
        state = self.last_state_diagnostic
        if state is not None and rebuilt:
            self.last_state_diagnostic = MusicStateDiagnostic(
                state.state, state.previous_decision_sample, state.decision_sample,
                state.gap_samples, state.reused_frames, state.added_frames,
                state.removed_frames, True, state.reason,
            )
        evidence = tuple(
            MusicPeakEvidence(
                item.theta_deg,
                0,
                item.raw_score,
                item.normalized_score,
                7,
                (
                    support_by_index[int(item.theta_deg)].supporting_frequency_bins
                    if config.dpd_rank1_enabled else valid_count
                ),
                (
                    support_by_index[int(item.theta_deg)].frequency_support_ratio
                    if config.dpd_rank1_enabled else 1.0
                ),
                (
                    support_by_index[int(item.theta_deg)].mean_plane_wave_fit
                    if config.dpd_rank1_enabled else 0.0
                ),
                (
                    support_by_index[int(item.theta_deg)].supporting_frequency_subbands
                    if config.dpd_rank1_enabled else 0
                ),
                (
                    support_by_index[int(item.theta_deg)].circular_concentration
                    if config.dpd_rank1_enabled else 0.0
                ),
                (
                    support_by_index[int(item.theta_deg)].cluster_weight
                    if config.dpd_rank1_enabled else 0.0
                ),
            )
            for item in candidates
        )
        assert self.last_state_diagnostic is not None
        ready_imcra_hops = sum(hop.state == "ready" for hop in window.imcra_hops)
        covariance_quality = (
            "degraded"
            if (config.dpd_rank1_enabled and not births_allowed)
            or (config.noise_whitening_enabled and whitening_status == "unavailable")
            else "ready"
        )
        diagnostics = MusicDiagnostics(
            "frequency_normalized_music", self.algorithm_version, config_revision,
            model_order, self.last_state_diagnostic, valid_count,
            covariance_quality,
            stop_reason=stop_reason,
            fallback_reason=fallback_reason,
            evidence=evidence,
            eligible_peak_count=eligible_peak_count, candidate_limit=limit or 3,
            candidate_limit_applied=candidate_limit_applied,
            effective_model_order=effective_order,
            births_allowed=births_allowed,
            dpd_rank1_enabled=config.dpd_rank1_enabled,
            selected_frequency_bins=int(selected.sum()) if config.dpd_rank1_enabled else int(valid.sum()),
            noise_whitening_enabled=config.noise_whitening_enabled,
            whitening_status=whitening_status,
            imcra_noise_hops=ready_imcra_hops if config.noise_whitening_enabled else 0,
            covariance_update_ms=self._last_covariance_update_ms,
            eigensolve_ms=(eigensolved - covariance_updated) * 1_000.0,
            spectrum_ms=(spectrum_built - eigensolved) * 1_000.0,
            total_ms=(perf_counter() - started) * 1_000.0,
        )
        return response, candidates, diagnostics

    def scan(self, window: DecisionWindow, geometry: MicGeometry, config: DirectionScanConfig):
        response, candidates, _ = self.scan_detailed(window, geometry, config)
        return response, candidates

    @property
    def model_order(self) -> ModelOrderEstimate | None:
        return self._last_model_order
