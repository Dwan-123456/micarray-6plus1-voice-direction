from __future__ import annotations

from dataclasses import dataclass

import torch

from .configuration import SpatialSeparationConfig, StftSettings
from .interface import (
    L3_MODE_CONSTANT_BEAMWIDTH,
    L3_MODE_DS_BASELINE,
    L3_MODE_OPTIMIZED,
    L3_PROCESSING_MODES,
    Layer3Error,
)
from .noise_context import NoiseStatistics


@dataclass(frozen=True, slots=True, eq=False)
class PreparedL3Context:
    """Candidate-independent, device-resident work for exactly one L3 window.

    The contained tensors are owned by L3 and are treated as read-only.  PyTorch
    has no read-only tensor flag, so callers must not mutate them in place.  The
    frozen DTO prevents replacing fields and lets the runtime pass this object
    between its ordered L3 preparation and beamforming stages without copying a
    320 ms spectrum back through host memory.
    """

    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    context_start_sample: int
    context_end_sample: int
    sample_rate: int
    mode: str
    stft: StftSettings
    config: SpatialSeparationConfig
    spectrum_fct: torch.Tensor
    frequencies_hz: torch.Tensor
    passband_f: torch.Tensor
    noise_statistics: NoiseStatistics | None
    noise_algorithm_version: str | None
    preparation_error: str | None
    stft_reused_frames: int
    covariance_rolled: bool

    def __post_init__(self) -> None:
        if (
            not self.session_id
            or min(self.stream_epoch, self.window_id, self.context_start_sample) < 0
            or self.context_end_sample != self.decision_sample
            or self.context_end_sample - self.context_start_sample != 15_360
            or self.sample_rate != 48_000
        ):
            raise ValueError("PreparedL3Context窗口身份或时间边界无效")
        if self.mode not in L3_PROCESSING_MODES:
            raise ValueError(f"PreparedL3Context处理模式无效: {self.mode}")
        if (
            self.spectrum_fct.shape != (513, 7, 33)
            or self.spectrum_fct.dtype != torch.complex64
            or self.spectrum_fct.requires_grad
            or not torch.isfinite(self.spectrum_fct).all()
        ):
            raise ValueError("PreparedL3Context STFT必须是finite complex64 [513,7,33]")
        if (
            self.frequencies_hz.shape != (513,)
            or self.frequencies_hz.dtype != torch.float32
            or self.frequencies_hz.device != self.spectrum_fct.device
            or not torch.isfinite(self.frequencies_hz).all()
        ):
            raise ValueError("PreparedL3Context频率轴无效")
        if (
            self.passband_f.shape != (513,)
            or self.passband_f.dtype != torch.bool
            or self.passband_f.device != self.spectrum_fct.device
        ):
            raise ValueError("PreparedL3Context频带mask无效")
        if not 0 <= self.stft_reused_frames <= 33:
            raise ValueError("PreparedL3Context复用帧计数无效")
        if self.mode in {L3_MODE_DS_BASELINE, L3_MODE_CONSTANT_BEAMWIDTH}:
            if self.noise_statistics is not None or self.noise_algorithm_version is not None:
                raise ValueError("固定权重对照模式准备阶段不得携带IMCRA统计")
        elif self.mode == L3_MODE_OPTIMIZED:
            if (self.noise_statistics is None) == (self.preparation_error is None):
                raise ValueError("优化模式必须恰好携带IMCRA统计或准备错误之一")
        for tensor in self._owned_tensors():
            if tensor.device != self.spectrum_fct.device or tensor.requires_grad:
                raise ValueError("PreparedL3Context张量必须同设备且禁用梯度")

    @property
    def window_key(self) -> tuple[str, int, int, int]:
        return self.session_id, self.stream_epoch, self.window_id, self.decision_sample

    @property
    def persistent_tensor_bytes(self) -> int:
        seen: set[int] = set()
        total = 0
        for tensor in self._owned_tensors():
            storage = tensor.untyped_storage()
            pointer = storage.data_ptr()
            if pointer not in seen:
                seen.add(pointer)
                total += storage.nbytes()
        return total

    def _owned_tensors(self) -> tuple[torch.Tensor, ...]:
        tensors = [self.spectrum_fct, self.frequencies_hz, self.passband_f]
        if self.noise_statistics is not None:
            tensors.extend((
                self.noise_statistics.covariance_fcc,
                self.noise_statistics.noise_confidence_f,
                self.noise_statistics.frequency_gain_f,
            ))
        return tuple(tensors)


@dataclass(frozen=True, slots=True, eq=False)
class BeamformedL3Batch:
    """Device-resident candidate-dependent spectra and their public metadata."""

    spectra_mft: torch.Tensor
    theta_degrees: tuple[float, ...]
    backends: tuple[str, ...]
    fallback_reasons: tuple[str | None, ...]
    diagnostics: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        count = len(self.theta_degrees)
        if (
            self.spectra_mft.shape != (count, 513, 33)
            or self.spectra_mft.dtype != torch.complex64
            or self.spectra_mft.requires_grad
            or not torch.isfinite(self.spectra_mft).all()
        ):
            raise Layer3Error("L3候选频谱批次无效")
        if not (
            len(self.backends) == len(self.fallback_reasons) == len(self.diagnostics) == count
        ):
            raise Layer3Error("L3候选频谱元数据长度不一致")
