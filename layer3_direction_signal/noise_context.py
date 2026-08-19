from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from common.data_types import DecisionWindow, ImcraHopSnapshot

from .configuration import SpatialSeparationConfig, StftSettings
from .interface import Layer3Error


_HOP_SAMPLES = 960
_CONTEXT_HOPS = 16
_STFT_FRAMES_PER_HOP = 2


@dataclass(frozen=True, slots=True)
class BeamformerNoiseContext:
    session_id: str
    stream_epoch: int
    context_start_sample: int
    context_end_sample: int
    algorithm_version: str
    frequencies_hz: np.ndarray
    noise_psd: np.ndarray
    posterior_noise_probability: np.ndarray
    prior_snr: np.ndarray
    posterior_snr: np.ndarray
    noise_level_db: np.ndarray

    def __post_init__(self) -> None:
        if not self.session_id or self.stream_epoch < 0 or not self.algorithm_version:
            raise ValueError("BF噪声上下文身份或算法版本无效")
        if self.context_end_sample - self.context_start_sample != 15_360:
            raise ValueError("BF噪声上下文必须覆盖320 ms")
        frequencies = np.asarray(self.frequencies_hz)
        expected_spectral = (16, 7, len(frequencies))
        arrays = {
            "noise_psd": (self.noise_psd, expected_spectral),
            "posterior_noise_probability": (self.posterior_noise_probability, expected_spectral),
            "prior_snr": (self.prior_snr, expected_spectral),
            "posterior_snr": (self.posterior_snr, expected_spectral),
            "noise_level_db": (self.noise_level_db, (16, 7)),
        }
        if frequencies.ndim != 1 or len(frequencies) < 2 or not np.all(np.diff(frequencies) > 0):
            raise ValueError("BF噪声上下文频率轴无效")
        if not np.isfinite(frequencies).all():
            raise ValueError("BF噪声上下文频率轴必须finite")
        object.__setattr__(
            self, "frequencies_hz",
            np.frombuffer(np.ascontiguousarray(frequencies, dtype=np.float32).tobytes(), dtype=np.float32),
        )
        for name, (value, shape) in arrays.items():
            raw = np.asarray(value)
            if raw.shape != shape or not np.isfinite(raw).all():
                raise ValueError(f"BF噪声上下文{name}形状或数值无效")
            if name != "noise_level_db" and np.any(raw < 0):
                raise ValueError(f"BF噪声上下文{name}不能为负")
            immutable = np.frombuffer(
                np.ascontiguousarray(raw, dtype=np.float32).tobytes(), dtype=np.float32,
            ).reshape(shape)
            object.__setattr__(self, name, immutable)
        if np.any(self.posterior_noise_probability > 1):
            raise ValueError("BF后验噪声概率必须位于[0,1]")

    @classmethod
    def from_window(cls, window: DecisionWindow) -> "BeamformerNoiseContext":
        hops, algorithm_version, _frequencies = _validated_window_hops(window)
        return cls(
            window.session_id,
            window.stream_epoch,
            window.context_start_sample,
            window.context_end_sample,
            algorithm_version,
            hops[0].frequencies_hz,
            np.stack([item.noise_psd for item in hops]),
            np.stack([1.0 - item.spp for item in hops]).astype(np.float32),
            np.stack([item.prior_snr for item in hops]),
            np.stack([item.posterior_snr for item in hops]),
            np.stack([item.noise_level_db for item in hops]),
        )


def _validated_window_hops(
    window: DecisionWindow,
) -> tuple[tuple[ImcraHopSnapshot, ...], str, np.ndarray]:
    hops = tuple(window.imcra_hops)
    expected_ranges = tuple(
        (start, start + 960)
        for start in range(window.context_start_sample, window.context_end_sample, 960)
    )
    actual_ranges = tuple((item.start_sample, item.end_sample) for item in hops)
    if len(hops) != 16 or actual_ranges != expected_ranges:
        raise Layer3Error("IMCRA噪声上下文必须包含与320 ms窗口连续对齐的16个hop")
    if any(item.state != "ready" for item in hops):
        raise Layer3Error("IMCRA噪声上下文尚未ready")
    versions = {item.algorithm_version for item in hops}
    if len(versions) != 1:
        raise Layer3Error("IMCRA噪声上下文算法版本不一致")
    frequency_axes = {item.frequencies_hz.tobytes() for item in hops}
    if len(frequency_axes) != 1:
        raise Layer3Error("IMCRA噪声上下文频率轴不一致")
    return hops, versions.pop(), hops[0].frequencies_hz


@dataclass(frozen=True, slots=True)
class NoiseStatistics:
    covariance_fcc: torch.Tensor
    noise_confidence_f: torch.Tensor
    frequency_gain_f: torch.Tensor


@dataclass(frozen=True, slots=True)
class NoiseCacheSnapshot:
    max_temporal_hops: int
    temporal_hops: int
    rolled: bool
    persistent_tensor_bytes: int


@dataclass(frozen=True, slots=True)
class _InterpolatedNoiseContext:
    noise_psd_hmf: torch.Tensor
    noise_probability_hmf: torch.Tensor
    prior_snr_hmf: torch.Tensor
    posterior_snr_hmf: torch.Tensor
    noise_level_db: torch.Tensor


@dataclass(frozen=True, slots=True)
class _InterpolationPlan:
    left_indices: torch.Tensor
    right_indices: torch.Tensor
    right_ratio: torch.Tensor


def _build_interpolation_plan(
    source_frequencies: torch.Tensor,
    target_frequencies: torch.Tensor,
) -> _InterpolationPlan:
    right = torch.searchsorted(source_frequencies, target_frequencies).clamp(
        1, len(source_frequencies) - 1,
    )
    left = right - 1
    low_frequency = source_frequencies[left]
    high_frequency = source_frequencies[right]
    ratio = ((target_frequencies - low_frequency) / (high_frequency - low_frequency)).clamp(0.0, 1.0)
    return _InterpolationPlan(left, right, ratio)


def _interpolate_last_axis(
    values: torch.Tensor,
    plan: _InterpolationPlan,
) -> torch.Tensor:
    return (
        values[..., plan.left_indices] * (1.0 - plan.right_ratio)
        + values[..., plan.right_indices] * plan.right_ratio
    )


def _full_interpolated_context(
    context: BeamformerNoiseContext,
    frequencies_hz: torch.Tensor,
    plan: _InterpolationPlan | None = None,
) -> _InterpolatedNoiseContext:
    device = frequencies_hz.device
    source_frequencies = torch.as_tensor(context.frequencies_hz.copy(), device=device)
    plan = plan or _build_interpolation_plan(source_frequencies, frequencies_hz)
    return _InterpolatedNoiseContext(
        _interpolate_last_axis(
            torch.as_tensor(context.noise_psd.copy(), device=device), plan,
        ),
        _interpolate_last_axis(
            torch.as_tensor(context.posterior_noise_probability.copy(), device=device),
            plan,
        ).clamp(0.0, 1.0),
        _interpolate_last_axis(
            torch.as_tensor(context.prior_snr.copy(), device=device), plan,
        ).clamp_min(0.0),
        _interpolate_last_axis(
            torch.as_tensor(context.posterior_snr.copy(), device=device), plan,
        ).clamp_min(0.0),
        torch.as_tensor(context.noise_level_db.copy(), device=device),
    )


def _append_interpolated_hops(
    previous: _InterpolatedNoiseContext,
    context: BeamformerNoiseContext,
    frequencies_hz: torch.Tensor,
    plan: _InterpolationPlan,
    hop_gap: int,
) -> _InterpolatedNoiseContext:
    device = frequencies_hz.device

    def append(values: torch.Tensor, raw: np.ndarray, *, probability: bool = False) -> torch.Tensor:
        newest = _interpolate_last_axis(
            torch.as_tensor(raw[-hop_gap:].copy(), device=device), plan,
        )
        if probability:
            newest = newest.clamp(0.0, 1.0)
        else:
            newest = newest.clamp_min(0.0)
        return torch.cat((values[hop_gap:], newest), dim=0)

    return _InterpolatedNoiseContext(
        append(previous.noise_psd_hmf, context.noise_psd),
        append(
            previous.noise_probability_hmf,
            context.posterior_noise_probability,
            probability=True,
        ),
        append(previous.prior_snr_hmf, context.prior_snr),
        append(previous.posterior_snr_hmf, context.posterior_snr),
        torch.cat((
            previous.noise_level_db[hop_gap:],
            torch.as_tensor(context.noise_level_db[-hop_gap:].copy(), device=device),
        ), dim=0),
    )


def _full_interpolated_hops(
    hops: tuple[ImcraHopSnapshot, ...],
    plan: _InterpolationPlan,
    *,
    device: torch.device,
) -> _InterpolatedNoiseContext:
    """Transfer and interpolate all 16 hops only for a cold/rebuilt window."""

    def stacked(name: str) -> torch.Tensor:
        return torch.as_tensor(
            np.stack([getattr(item, name) for item in hops]),
            dtype=torch.float32,
            device=device,
        )

    return _InterpolatedNoiseContext(
        _interpolate_last_axis(stacked("noise_psd"), plan).clamp_min(0.0),
        _interpolate_last_axis(1.0 - stacked("spp"), plan).clamp(0.0, 1.0),
        _interpolate_last_axis(stacked("prior_snr"), plan).clamp_min(0.0),
        _interpolate_last_axis(stacked("posterior_snr"), plan).clamp_min(0.0),
        torch.as_tensor(
            np.stack([item.noise_level_db for item in hops]),
            dtype=torch.float32,
            device=device,
        ),
    )


def _append_interpolated_snapshots(
    previous: _InterpolatedNoiseContext,
    hops: tuple[ImcraHopSnapshot, ...],
    plan: _InterpolationPlan,
    hop_gap: int,
    *,
    device: torch.device,
) -> _InterpolatedNoiseContext:
    """Advance a warm cache by transferring only the newly arrived hops."""

    def append(
        values: torch.Tensor,
        name: str,
        *,
        probability: bool = False,
    ) -> torch.Tensor:
        raw = np.stack([getattr(item, name) for item in hops[-hop_gap:]])
        if name == "spp":
            raw = 1.0 - raw
        newest = _interpolate_last_axis(
            torch.as_tensor(raw, dtype=torch.float32, device=device),
            plan,
        )
        newest = newest.clamp(0.0, 1.0) if probability else newest.clamp_min(0.0)
        return torch.cat((values[hop_gap:], newest), dim=0)

    return _InterpolatedNoiseContext(
        append(previous.noise_psd_hmf, "noise_psd"),
        append(previous.noise_probability_hmf, "spp", probability=True),
        append(previous.prior_snr_hmf, "prior_snr"),
        append(previous.posterior_snr_hmf, "posterior_snr"),
        torch.cat((
            previous.noise_level_db[hop_gap:],
            torch.as_tensor(
                np.stack([item.noise_level_db for item in hops[-hop_gap:]]),
                dtype=torch.float32,
                device=device,
            ),
        ), dim=0),
    )


def _frame_weights(
    interpolated: _InterpolatedNoiseContext,
    spectrum_fct: torch.Tensor,
    stft: StftSettings,
    hop_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    noise_probability_hf = interpolated.noise_probability_hmf.median(dim=1).values
    if hop_indices is None:
        frame_centres = torch.arange(spectrum_fct.shape[-1], device=spectrum_fct.device) * stft.hop_length
        hop_indices = torch.div(frame_centres, 960, rounding_mode="floor").clamp(0, 15)
    return noise_probability_hf[hop_indices].transpose(0, 1).contiguous()


def _covariance_numerator(
    spectrum_fct: torch.Tensor,
    frame_weights_ft: torch.Tensor,
) -> torch.Tensor:
    # One batched GEMM is materially cheaper than launching the equivalent
    # three-operand einsum for the four-frame rolling update on CUDA.
    return (spectrum_fct * frame_weights_ft[:, None, :]) @ spectrum_fct.mH


def _finalize_noise_statistics(
    interpolated: _InterpolatedNoiseContext,
    covariance_numerator_fcc: torch.Tensor,
    denominator_f: torch.Tensor,
    frequencies_hz: torch.Tensor,
    config: SpatialSeparationConfig,
) -> NoiseStatistics:
    covariance = covariance_numerator_fcc / denominator_f[:, None, None].clamp_min(1e-6)
    covariance = 0.5 * (covariance + covariance.mH)

    psd_fc = interpolated.noise_psd_hmf.mean(dim=0).transpose(0, 1).clamp_min(1e-12)
    level_c = torch.pow(10.0, interpolated.noise_level_db.mean(dim=0) / 10.0).clamp_min(1e-12)
    level_relative = level_c / level_c.mean()
    psd_relative = psd_fc / psd_fc.mean(dim=-1, keepdim=True)
    diagonal_relative = 0.8 * psd_relative + 0.2 * level_relative[None, :]
    covariance_scale = torch.diagonal(covariance, dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-8)
    diagonal_anchor = torch.diag_embed((diagonal_relative * covariance_scale[:, None]).to(torch.complex64))
    shrinkage = config.noise_covariance_shrinkage
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal_anchor
    covariance = 0.5 * (covariance + covariance.mH)

    confidence = (denominator_f / 33.0).clamp(0.0, 1.0)
    prior = interpolated.prior_snr_hmf.median(dim=0).values.median(dim=0).values
    posterior = interpolated.posterior_snr_hmf.median(dim=0).values.median(dim=0).values
    prior_gain = prior / (1.0 + prior)
    posterior_excess = torch.clamp(posterior - 1.0, min=0.0)
    posterior_gain = posterior_excess / (1.0 + posterior_excess)
    gain = torch.sqrt(prior_gain * posterior_gain).clamp(config.min_frequency_gain, 1.0)
    speech_band = (frequencies_hz >= config.frequency_min_hz) & (frequencies_hz <= config.frequency_max_hz)
    gain = torch.where(speech_band, gain, torch.zeros_like(gain))

    if not torch.isfinite(covariance).all() or not torch.isfinite(gain).all():
        raise Layer3Error("IMCRA noise statistics produced non-finite values")
    return NoiseStatistics(covariance.to(torch.complex64), confidence.to(torch.float32), gain.to(torch.float32))


class RollingNoiseStatisticsCache:
    """Exact rolling IMCRA interpolation and covariance state for one 320 ms window."""

    max_temporal_hops = 50

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._identity: tuple[str, int, int, int] | None = None
        self._context_key: tuple[str, bytes] | None = None
        self._config: SpatialSeparationConfig | None = None
        self._stft: StftSettings | None = None
        self._interpolated: _InterpolatedNoiseContext | None = None
        self._numerator: torch.Tensor | None = None
        self._denominator: torch.Tensor | None = None
        self._previous_spectrum: torch.Tensor | None = None
        self._previous_weights: torch.Tensor | None = None
        self._interpolation_plan_key: tuple[bytes, str, int, int] | None = None
        self._interpolation_plan: _InterpolationPlan | None = None
        self._current_indices: torch.Tensor | None = None
        self._previous_indices: torch.Tensor | None = None
        self._rolling_index_hop_gap: int | None = None
        self._frame_hop_indices_key: tuple[int, int, str, int] | None = None
        self._frame_hop_indices: torch.Tensor | None = None
        self._last_rolled = False

    def _plans(
        self,
        source_frequencies_np: np.ndarray,
        frequencies_hz: torch.Tensor,
        spectrum_fct: torch.Tensor,
        stft: StftSettings,
    ) -> tuple[_InterpolationPlan, torch.Tensor]:
        device = frequencies_hz.device
        device_key = (device.type, -1 if device.index is None else int(device.index))
        interpolation_key = (
            source_frequencies_np.tobytes(),
            device_key[0],
            device_key[1],
            frequencies_hz.data_ptr(),
        )
        if self._interpolation_plan_key != interpolation_key or self._interpolation_plan is None:
            source = torch.as_tensor(source_frequencies_np.copy(), device=device)
            self._interpolation_plan_key = interpolation_key
            self._interpolation_plan = _build_interpolation_plan(source, frequencies_hz)
        frame_key = (
            spectrum_fct.shape[-1],
            stft.hop_length,
            device_key[0],
            device_key[1],
        )
        if self._frame_hop_indices_key != frame_key or self._frame_hop_indices is None:
            centres = torch.arange(spectrum_fct.shape[-1], device=device) * stft.hop_length
            self._frame_hop_indices_key = frame_key
            self._frame_hop_indices = torch.div(
                centres,
                _HOP_SAMPLES,
                rounding_mode="floor",
            ).clamp(0, _CONTEXT_HOPS - 1)
        return self._interpolation_plan, self._frame_hop_indices

    def _rolling_indices(
        self,
        hop_gap: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._rolling_index_hop_gap != hop_gap
            or self._current_indices is None
            or self._current_indices.device != device
        ):
            # These complements match RollingStftCache's aligned interior:
            # remove the previous left edge/right reflected frame, then add
            # the current left reflected frame/right edge and all new frames.
            previous_expired = (*range(0, _STFT_FRAMES_PER_HOP * hop_gap + 1), 32)
            current_added = (
                0,
                *range(32 - _STFT_FRAMES_PER_HOP * hop_gap, 33),
            )
            self._current_indices = torch.tensor(
                current_added,
                dtype=torch.long,
                device=device,
            )
            self._previous_indices = torch.tensor(
                previous_expired,
                dtype=torch.long,
                device=device,
            )
            self._rolling_index_hop_gap = hop_gap
        assert self._previous_indices is not None
        return self._current_indices, self._previous_indices

    def estimate(
        self,
        context: BeamformerNoiseContext,
        spectrum_fct: torch.Tensor,
        frequencies_hz: torch.Tensor,
        config: SpatialSeparationConfig,
        stft: StftSettings,
        *,
        allow_rolling: bool,
    ) -> NoiseStatistics:
        identity = (
            context.session_id,
            context.stream_epoch,
            context.context_start_sample,
            context.context_end_sample,
        )
        context_key = (context.algorithm_version, context.frequencies_hz.tobytes())
        interpolation_plan, frame_hop_indices = self._plans(
            context.frequencies_hz, frequencies_hz, spectrum_fct, stft,
        )
        previous = self._identity
        sample_delta = 0 if previous is None else identity[2] - previous[2]
        hop_gap = sample_delta // _HOP_SAMPLES if sample_delta > 0 else 0
        can_roll = (
            allow_rolling
            and previous is not None
            and identity[:2] == previous[:2]
            and identity[3] - previous[3] == sample_delta
            and sample_delta % _HOP_SAMPLES == 0
            and 1 <= hop_gap < _CONTEXT_HOPS
            and context_key == self._context_key
            and config == self._config
            and stft == self._stft
            and self._interpolated is not None
            and self._numerator is not None
            and self._denominator is not None
            and self._previous_spectrum is not None
            and self._previous_weights is not None
        )
        interpolated = (
            _append_interpolated_hops(
                self._interpolated,
                context,
                frequencies_hz,
                interpolation_plan,
                hop_gap,
            )
            if can_roll
            else _full_interpolated_context(context, frequencies_hz, interpolation_plan)
        )
        weights = _frame_weights(interpolated, spectrum_fct, stft, frame_hop_indices)
        if can_roll:
            current_indices, previous_indices = self._rolling_indices(
                hop_gap,
                device=spectrum_fct.device,
            )
            current_edge_spectrum = spectrum_fct.index_select(-1, current_indices)
            current_edge_weights = weights.index_select(-1, current_indices)
            previous_edge_spectrum = self._previous_spectrum.index_select(-1, previous_indices)
            previous_edge_weights = self._previous_weights.index_select(-1, previous_indices)
            numerator = (
                self._numerator
                - _covariance_numerator(previous_edge_spectrum, previous_edge_weights)
                + _covariance_numerator(current_edge_spectrum, current_edge_weights)
            )
            denominator = (
                self._denominator
                - previous_edge_weights.sum(dim=-1)
                + current_edge_weights.sum(dim=-1)
            )
        else:
            numerator = _covariance_numerator(spectrum_fct, weights)
            denominator = weights.sum(dim=-1)
        result = _finalize_noise_statistics(interpolated, numerator, denominator, frequencies_hz, config)

        self._identity = identity
        self._context_key = context_key
        self._config = config
        self._stft = stft
        self._interpolated = interpolated
        self._numerator = numerator
        self._denominator = denominator
        self._previous_spectrum = spectrum_fct.clone()
        self._previous_weights = weights.clone()
        self._last_rolled = can_roll
        return result

    def estimate_window(
        self,
        window: DecisionWindow,
        spectrum_fct: torch.Tensor,
        frequencies_hz: torch.Tensor,
        config: SpatialSeparationConfig,
        stft: StftSettings,
        *,
        allow_rolling: bool,
    ) -> tuple[NoiseStatistics, str]:
        """Estimate directly from a DecisionWindow with an overlapping-window update.

        Unlike ``BeamformerNoiseContext.from_window`` this path does not stack
        and copy all 16 CPU hop arrays on every advance. A cold window transfers
        all hops; an overlapping window transfers only its newly arrived hops.
        """
        hops, algorithm_version, source_frequencies = _validated_window_hops(window)
        identity = (
            window.session_id,
            window.stream_epoch,
            window.context_start_sample,
            window.context_end_sample,
        )
        context_key = (algorithm_version, source_frequencies.tobytes())
        interpolation_plan, frame_hop_indices = self._plans(
            source_frequencies, frequencies_hz, spectrum_fct, stft,
        )
        previous = self._identity
        sample_delta = 0 if previous is None else identity[2] - previous[2]
        hop_gap = sample_delta // _HOP_SAMPLES if sample_delta > 0 else 0
        can_roll = (
            allow_rolling
            and previous is not None
            and identity[:2] == previous[:2]
            and identity[3] - previous[3] == sample_delta
            and sample_delta % _HOP_SAMPLES == 0
            and 1 <= hop_gap < _CONTEXT_HOPS
            and context_key == self._context_key
            and config == self._config
            and stft == self._stft
            and self._interpolated is not None
            and self._numerator is not None
            and self._denominator is not None
            and self._previous_spectrum is not None
            and self._previous_weights is not None
        )
        interpolated = (
            _append_interpolated_snapshots(
                self._interpolated,
                hops,
                interpolation_plan,
                hop_gap,
                device=frequencies_hz.device,
            )
            if can_roll
            else _full_interpolated_hops(
                hops,
                interpolation_plan,
                device=frequencies_hz.device,
            )
        )
        weights = _frame_weights(interpolated, spectrum_fct, stft, frame_hop_indices)
        if can_roll:
            current_indices, previous_indices = self._rolling_indices(
                hop_gap,
                device=spectrum_fct.device,
            )
            current_edge_spectrum = spectrum_fct.index_select(-1, current_indices)
            current_edge_weights = weights.index_select(-1, current_indices)
            previous_edge_spectrum = self._previous_spectrum.index_select(-1, previous_indices)
            previous_edge_weights = self._previous_weights.index_select(-1, previous_indices)
            numerator = (
                self._numerator
                - _covariance_numerator(previous_edge_spectrum, previous_edge_weights)
                + _covariance_numerator(current_edge_spectrum, current_edge_weights)
            )
            denominator = (
                self._denominator
                - previous_edge_weights.sum(dim=-1)
                + current_edge_weights.sum(dim=-1)
            )
        else:
            numerator = _covariance_numerator(spectrum_fct, weights)
            denominator = weights.sum(dim=-1)
        result = _finalize_noise_statistics(
            interpolated, numerator, denominator, frequencies_hz, config,
        )

        self._identity = identity
        self._context_key = context_key
        self._config = config
        self._stft = stft
        self._interpolated = interpolated
        self._numerator = numerator
        self._denominator = denominator
        self._previous_spectrum = spectrum_fct.clone()
        self._previous_weights = weights.clone()
        self._last_rolled = can_roll
        return result, algorithm_version

    def snapshot(self) -> NoiseCacheSnapshot:
        tensors: list[torch.Tensor] = []
        if self._interpolated is not None:
            tensors.extend((
                self._interpolated.noise_psd_hmf,
                self._interpolated.noise_probability_hmf,
                self._interpolated.prior_snr_hmf,
                self._interpolated.posterior_snr_hmf,
                self._interpolated.noise_level_db,
            ))
        tensors.extend(
            item for item in (
                self._numerator, self._denominator,
                self._previous_spectrum, self._previous_weights,
                self._current_indices, self._previous_indices, self._frame_hop_indices,
            ) if item is not None
        )
        if self._interpolation_plan is not None:
            tensors.extend((
                self._interpolation_plan.left_indices,
                self._interpolation_plan.right_indices,
                self._interpolation_plan.right_ratio,
            ))
        return NoiseCacheSnapshot(
            self.max_temporal_hops,
            0 if self._identity is None else 16,
            self._last_rolled,
            sum(item.numel() * item.element_size() for item in tensors),
        )


def estimate_noise_statistics(
    context: BeamformerNoiseContext,
    spectrum_fct: torch.Tensor,
    frequencies_hz: torch.Tensor,
    config: SpatialSeparationConfig,
    stft: StftSettings,
) -> NoiseStatistics:
    """Build IMCRA-controlled 7x7 spatial noise covariance for every BF bin."""
    interpolated = _full_interpolated_context(context, frequencies_hz)
    weights = _frame_weights(interpolated, spectrum_fct, stft)
    return _finalize_noise_statistics(
        interpolated,
        _covariance_numerator(spectrum_fct, weights),
        weights.sum(dim=-1),
        frequencies_hz,
        config,
    )
