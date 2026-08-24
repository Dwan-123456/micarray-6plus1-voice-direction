from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from common.data_types import DecisionWindow, ImcraHopSnapshot

from .configuration import SpatialSeparationConfig, StftSettings
from .interface import Layer3Error


_HOP_SAMPLES = 960
@dataclass(frozen=True, slots=True)
class BeamformerNoiseContext:
    session_id: str
    stream_epoch: int
    context_start_sample: int
    context_end_sample: int
    algorithm_version: str
    frequencies_hz: np.ndarray
    noise_psd: np.ndarray
    noise_covariance: np.ndarray
    posterior_noise_probability: np.ndarray
    prior_snr: np.ndarray
    posterior_snr: np.ndarray
    noise_level_db: np.ndarray

    def __post_init__(self) -> None:
        if not self.session_id or self.stream_epoch < 0 or not self.algorithm_version:
            raise ValueError("BF噪声上下文身份或算法版本无效")
        context_hops = (self.context_end_sample - self.context_start_sample) // _HOP_SAMPLES
        if context_hops not in {2, 4, 8}:
            raise ValueError("BF噪声上下文必须覆盖40、80或160 ms")
        frequencies = np.asarray(self.frequencies_hz)
        expected_spectral = (context_hops, 7, len(frequencies))
        arrays = {
            "noise_psd": (self.noise_psd, expected_spectral),
            "posterior_noise_probability": (self.posterior_noise_probability, expected_spectral),
            "prior_snr": (self.prior_snr, expected_spectral),
            "posterior_snr": (self.posterior_snr, expected_spectral),
            "noise_level_db": (self.noise_level_db, (context_hops, 7)),
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
        covariance_shape = (context_hops, len(frequencies), 7, 7)
        covariance = np.asarray(self.noise_covariance)
        if covariance.shape != covariance_shape or not np.isfinite(covariance).all():
            raise ValueError("BF噪声上下文noise_covariance形状或数值无效")
        if not np.allclose(
            covariance,
            covariance.conj().transpose(0, 1, 3, 2),
            rtol=2.0e-5,
            atol=1.0e-7,
        ):
            raise ValueError("BF噪声上下文noise_covariance必须为Hermitian")
        immutable_covariance = np.frombuffer(
            np.ascontiguousarray(covariance, dtype=np.complex64).tobytes(),
            dtype=np.complex64,
        ).reshape(covariance_shape)
        object.__setattr__(self, "noise_covariance", immutable_covariance)
        if np.any(self.posterior_noise_probability > 1):
            raise ValueError("BF后验噪声概率必须位于[0,1]")

    @classmethod
    def from_window(
        cls, window: DecisionWindow, stft: StftSettings,
    ) -> "BeamformerNoiseContext":
        hops, algorithm_version, _frequencies = _validated_window_hops(
            window, stft.window_hops,
        )
        return cls(
            window.session_id,
            window.stream_epoch,
            window.context_end_sample - stft.window_samples,
            window.context_end_sample,
            algorithm_version,
            hops[0].frequencies_hz,
            np.stack([item.noise_psd for item in hops]),
            np.stack([item.noise_covariance for item in hops]),
            np.stack([1.0 - item.spp for item in hops]).astype(np.float32),
            np.stack([item.prior_snr for item in hops]),
            np.stack([item.posterior_snr for item in hops]),
            np.stack([item.noise_level_db for item in hops]),
        )


def _validated_window_hops(
    window: DecisionWindow, window_hops: int,
) -> tuple[tuple[ImcraHopSnapshot, ...], str, np.ndarray]:
    hops = tuple(window.imcra_hops)[-window_hops:]
    context_start = window.context_end_sample - window_hops * _HOP_SAMPLES
    expected_ranges = tuple(
        (start, start + _HOP_SAMPLES)
        for start in range(context_start, window.context_end_sample, _HOP_SAMPLES)
    )
    actual_ranges = tuple((item.start_sample, item.end_sample) for item in hops)
    if len(hops) != window_hops or actual_ranges != expected_ranges:
        raise Layer3Error(
            f"IMCRA噪声上下文必须包含连续对齐的{window_hops}个20 ms hop"
        )
    if any(item.state != "ready" for item in hops):
        raise Layer3Error("IMCRA噪声上下文尚未ready")
    if any(item.noise_covariance is None for item in hops):
        raise Layer3Error("IMCRA空间噪声协方差不可用")
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
    noise_covariance_hfcc: torch.Tensor
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


def _interpolate_covariance(
    values: torch.Tensor,
    plan: _InterpolationPlan,
) -> torch.Tensor:
    frequency_last = values.movedim(-3, -1)
    return _interpolate_last_axis(frequency_last, plan).movedim(-1, -3)


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
        _interpolate_covariance(
            torch.as_tensor(context.noise_covariance.copy(), device=device), plan,
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
        torch.cat((
            previous.noise_covariance_hfcc[hop_gap:],
            _interpolate_covariance(
                torch.as_tensor(
                    context.noise_covariance[-hop_gap:].copy(), device=device,
                ),
                plan,
            ),
        ), dim=0),
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
    """Transfer and interpolate all 8 hops only for a cold/rebuilt window."""

    def stacked(name: str) -> torch.Tensor:
        return torch.as_tensor(
            np.stack([getattr(item, name) for item in hops]),
            dtype=torch.float32,
            device=device,
        )

    return _InterpolatedNoiseContext(
        _interpolate_last_axis(stacked("noise_psd"), plan).clamp_min(0.0),
        _interpolate_covariance(
            torch.as_tensor(
                np.stack([item.noise_covariance for item in hops]),
                dtype=torch.complex64,
                device=device,
            ),
            plan,
        ),
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
        torch.cat((
            previous.noise_covariance_hfcc[hop_gap:],
            _interpolate_covariance(
                torch.as_tensor(
                    np.stack([item.noise_covariance for item in hops[-hop_gap:]]),
                    dtype=torch.complex64,
                    device=device,
                ),
                plan,
            ),
        ), dim=0),
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


def _finalize_noise_statistics(
    interpolated: _InterpolatedNoiseContext,
    frequencies_hz: torch.Tensor,
    config: SpatialSeparationConfig,
    *,
    frequency_indices: torch.Tensor | None = None,
    validate_values: bool = True,
) -> NoiseStatistics:
    frequency_count = len(frequencies_hz)
    if frequency_indices is None:
        frequency_indices = torch.arange(
            frequency_count, dtype=torch.long, device=frequencies_hz.device,
        )
    covariance = interpolated.noise_covariance_hfcc.mean(dim=0).index_select(
        0, frequency_indices,
    )
    covariance = 0.5 * (covariance + covariance.mH)

    psd_fc = (
        interpolated.noise_psd_hmf.mean(dim=0)
        .transpose(0, 1)
        .index_select(0, frequency_indices)
        .clamp_min(1e-12)
    )
    diagonal_indices = torch.arange(7, device=covariance.device)
    covariance[:, diagonal_indices, diagonal_indices] = psd_fc.to(covariance.dtype)
    covariance = 0.5 * (covariance + covariance.mH)

    confidence_active = (
        interpolated.noise_probability_hmf.median(dim=1).values.mean(dim=0)
        .index_select(0, frequency_indices)
        .clamp(0.0, 1.0)
    )
    prior = interpolated.prior_snr_hmf.median(dim=0).values.median(dim=0).values
    posterior = interpolated.posterior_snr_hmf.median(dim=0).values.median(dim=0).values
    prior_gain = prior / (1.0 + prior)
    posterior_excess = torch.clamp(posterior - 1.0, min=0.0)
    posterior_gain = posterior_excess / (1.0 + posterior_excess)
    gain = torch.sqrt(prior_gain * posterior_gain).clamp(config.min_frequency_gain, 1.0)
    speech_band = (frequencies_hz >= config.frequency_min_hz) & (frequencies_hz <= config.frequency_max_hz)
    gain = torch.where(speech_band, gain, torch.zeros_like(gain))

    # Preserve the public full-frequency tensor contract while avoiding 7x7
    # covariance work for bins that every supported L3 target mode discards.
    if len(frequency_indices) != frequency_count:
        full_covariance = torch.zeros(
            (frequency_count, covariance.shape[-2], covariance.shape[-1]),
            dtype=covariance.dtype,
            device=covariance.device,
        )
        full_covariance[frequency_indices] = covariance
        covariance = full_covariance
        confidence = torch.zeros_like(frequencies_hz, dtype=torch.float32)
        confidence[frequency_indices] = confidence_active
    else:
        confidence = confidence_active

    if validate_values and not torch.stack((
        torch.isfinite(covariance).all(), torch.isfinite(gain).all(),
    )).all():
        raise Layer3Error("IMCRA noise statistics produced non-finite values")
    return NoiseStatistics(covariance.to(torch.complex64), confidence.to(torch.float32), gain.to(torch.float32))


class RollingNoiseStatisticsCache:
    """Exact rolling IMCRA interpolation/covariance for the configured direct window."""

    max_temporal_hops = 50

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._identity: tuple[str, int, int, int] | None = None
        self._context_key: tuple[str, bytes] | None = None
        self._config: SpatialSeparationConfig | None = None
        self._stft: StftSettings | None = None
        self._interpolated: _InterpolatedNoiseContext | None = None
        self._interpolation_plan_key: tuple[bytes, str, int, int] | None = None
        self._interpolation_plan: _InterpolationPlan | None = None
        self._last_rolled = False

    def _plans(
        self,
        source_frequencies_np: np.ndarray,
        frequencies_hz: torch.Tensor,
    ) -> _InterpolationPlan:
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
        return self._interpolation_plan

    @staticmethod
    def _active_frequency_indices(
        frequencies_hz: torch.Tensor,
        config: SpatialSeparationConfig,
    ) -> torch.Tensor:
        return torch.nonzero(
            (frequencies_hz >= config.frequency_min_hz)
            & (frequencies_hz <= config.frequency_max_hz),
            as_tuple=False,
        ).flatten()

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
        interpolation_plan = self._plans(context.frequencies_hz, frequencies_hz)
        frequency_indices = self._active_frequency_indices(frequencies_hz, config)
        previous = self._identity
        sample_delta = 0 if previous is None else identity[2] - previous[2]
        hop_gap = sample_delta // _HOP_SAMPLES if sample_delta > 0 else 0
        can_roll = (
            allow_rolling
            and previous is not None
            and identity[:2] == previous[:2]
            and identity[3] - previous[3] == sample_delta
            and sample_delta % _HOP_SAMPLES == 0
            and 1 <= hop_gap < stft.window_hops
            and context_key == self._context_key
            and config == self._config
            and stft == self._stft
            and self._interpolated is not None
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
        result = _finalize_noise_statistics(
            interpolated,
            frequencies_hz,
            config,
            frequency_indices=frequency_indices,
        )

        self._identity = identity
        self._context_key = context_key
        self._config = config
        self._stft = stft
        self._interpolated = interpolated
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
        validate_values: bool = True,
    ) -> tuple[NoiseStatistics, str]:
        """Estimate directly from a DecisionWindow with an overlapping-window update.

        Unlike ``BeamformerNoiseContext.from_window`` this path does not stack
        and copy all 8 CPU hop arrays on every 20 ms advance.  A cold window
        transfers all hops; an overlapping window transfers only its newly arrived hops.
        """
        hops, algorithm_version, source_frequencies = _validated_window_hops(
            window, stft.window_hops,
        )
        identity = (
            window.session_id,
            window.stream_epoch,
            window.context_end_sample - stft.window_samples,
            window.context_end_sample,
        )
        context_key = (algorithm_version, source_frequencies.tobytes())
        interpolation_plan = self._plans(source_frequencies, frequencies_hz)
        frequency_indices = self._active_frequency_indices(frequencies_hz, config)
        previous = self._identity
        sample_delta = 0 if previous is None else identity[2] - previous[2]
        hop_gap = sample_delta // _HOP_SAMPLES if sample_delta > 0 else 0
        can_roll = (
            allow_rolling
            and previous is not None
            and identity[:2] == previous[:2]
            and identity[3] - previous[3] == sample_delta
            and sample_delta % _HOP_SAMPLES == 0
            and 1 <= hop_gap < stft.window_hops
            and context_key == self._context_key
            and config == self._config
            and stft == self._stft
            and self._interpolated is not None
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
        result = _finalize_noise_statistics(
            interpolated,
            frequencies_hz,
            config,
            frequency_indices=frequency_indices,
            validate_values=validate_values,
        )

        self._identity = identity
        self._context_key = context_key
        self._config = config
        self._stft = stft
        self._interpolated = interpolated
        self._last_rolled = can_roll
        return result, algorithm_version

    def snapshot(self) -> NoiseCacheSnapshot:
        tensors: list[torch.Tensor] = []
        if self._interpolated is not None:
            tensors.extend((
                self._interpolated.noise_psd_hmf,
                self._interpolated.noise_covariance_hfcc,
                self._interpolated.noise_probability_hmf,
                self._interpolated.prior_snr_hmf,
                self._interpolated.posterior_snr_hmf,
                self._interpolated.noise_level_db,
            ))
        if self._interpolation_plan is not None:
            tensors.extend((
                self._interpolation_plan.left_indices,
                self._interpolation_plan.right_indices,
                self._interpolation_plan.right_ratio,
            ))
        return NoiseCacheSnapshot(
            self.max_temporal_hops,
            0 if self._identity is None or self._stft is None else self._stft.window_hops,
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
    """Read L1's persistent 7x7 spatial noise covariance for every BF bin."""
    del spectrum_fct, stft
    interpolated = _full_interpolated_context(context, frequencies_hz)
    return _finalize_noise_statistics(
        interpolated,
        frequencies_hz,
        config,
    )
