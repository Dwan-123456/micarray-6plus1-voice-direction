from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as functional

from common.data_types import DecisionWindow

from .configuration import StftSettings
from .interface import Layer3Error


MAX_TEMPORAL_CACHE_HOPS = 50
_CONTEXT_HOPS = 16
_ROLLING_REUSED_FRAME_INDICES = slice(1, 30)
_PREVIOUS_REUSED_FRAME_INDICES = slice(3, 32)
_ROLLING_RECOMPUTED_FRAME_INDICES = (0, 30, 31, 32)


@lru_cache(maxsize=8)
def periodic_hann(settings: StftSettings, *, device: torch.device) -> torch.Tensor:
    """Return a bounded, device-resident STFT window cache."""
    return torch.hann_window(settings.win_length, periodic=True, dtype=torch.float32, device=device)


def _input_tensor(window: DecisionWindow, *, device: torch.device) -> torch.Tensor:
    # HardwareMix remains outside every array calculation.
    return torch.as_tensor(window.samples[:, :7].T.copy(), dtype=torch.float32, device=device)


def _selected_stft_frames(
    samples: torch.Tensor,
    settings: StftSettings,
    frame_indices: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Compute selected frames exactly as torch.stft for the locked settings."""
    if not settings.center or settings.pad_mode != "reflect":
        raise Layer3Error("rolling STFT requires the locked center=true/reflect settings")
    padded = functional.pad(
        samples.unsqueeze(0),
        (settings.n_fft // 2, settings.n_fft // 2),
        mode="reflect",
    ).squeeze(0)
    frames = padded.unfold(-1, settings.n_fft, settings.hop_length)
    selected = frames.index_select(1, frame_indices)
    window = periodic_hann(settings, device=device)
    left = (settings.n_fft - settings.win_length) // 2
    right = settings.n_fft - settings.win_length - left
    fft_window = functional.pad(window, (left, right))
    return torch.fft.rfft(selected * fft_window, n=settings.n_fft, dim=-1).permute(0, 2, 1)


@dataclass(frozen=True, slots=True)
class StftCacheSnapshot:
    max_temporal_hops: int
    temporal_hops: int
    reused_frames: int
    recomputed_frames: int
    persistent_tensor_bytes: int


class RollingStftCache:
    """One-window STFT ring with an explicit 1000 ms hard ceiling."""

    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        self._identity: tuple[str, int, int, int] | None = None
        self._settings: StftSettings | None = None
        self._spectrum: torch.Tensor | None = None
        self._recomputed_indices: torch.Tensor | None = None
        self._last_reused_frames = 0
        self._last_recomputed_frames = 33

    def clear(self) -> None:
        self._identity = None
        self._settings = None
        self._spectrum = None
        self._recomputed_indices = None
        self._last_reused_frames = 0
        self._last_recomputed_frames = 33

    def process(self, window: DecisionWindow, settings: StftSettings) -> torch.Tensor:
        identity = (
            window.session_id,
            window.stream_epoch,
            window.context_start_sample,
            window.context_end_sample,
        )
        previous = self._identity
        sequential = (
            self._spectrum is not None
            and self._settings == settings
            and previous is not None
            and identity[0:2] == previous[0:2]
            and identity[2] == previous[2] + 960
            and identity[3] == previous[3] + 960
        )
        if not sequential:
            spectrum = shared_stft(window, settings, device=self.device)
            self._last_reused_frames = 0
            self._last_recomputed_frames = 33
        else:
            samples = _input_tensor(window, device=self.device)
            if self._recomputed_indices is None:
                self._recomputed_indices = torch.tensor(
                    _ROLLING_RECOMPUTED_FRAME_INDICES,
                    dtype=torch.long,
                    device=self.device,
                )
            recomputed = _selected_stft_frames(
                samples,
                settings,
                self._recomputed_indices,
                device=self.device,
            )
            spectrum = torch.empty_like(self._spectrum)
            spectrum[:, :, _ROLLING_REUSED_FRAME_INDICES] = self._spectrum[
                :, :, _PREVIOUS_REUSED_FRAME_INDICES
            ]
            spectrum[:, :, self._recomputed_indices] = recomputed
            self._last_reused_frames = 29
            self._last_recomputed_frames = 4
        if spectrum.shape != (7, 513, 33) or not torch.isfinite(spectrum).all():
            self.clear()
            raise Layer3Error("rolling STFT output is invalid")
        # Only the current 320 ms/16-hop window persists. This is deliberately
        # below the project-wide 50-hop/1000 ms temporal-cache ceiling.
        self._identity = identity
        self._settings = settings
        self._spectrum = spectrum
        return spectrum

    def snapshot(self) -> StftCacheSnapshot:
        tensors = tuple(item for item in (self._spectrum, self._recomputed_indices) if item is not None)
        tensor_bytes = sum(item.numel() * item.element_size() for item in tensors)
        return StftCacheSnapshot(
            MAX_TEMPORAL_CACHE_HOPS,
            0 if self._spectrum is None else _CONTEXT_HOPS,
            self._last_reused_frames,
            self._last_recomputed_frames,
            tensor_bytes,
        )


def shared_stft(window: DecisionWindow, settings: StftSettings, *, device: torch.device) -> torch.Tensor:
    if window.samples.shape != (15_360, 8):
        raise Layer3Error(f"L3输入必须是48 kHz逻辑8通道 [15360,8]，实际为{window.samples.shape}")
    # HardwareMix is preserved by the public input contract but is never an
    # array microphone: steering, covariance and beamforming use PhysicalAudio.
    samples = _input_tensor(window, device=device)
    spectrum = torch.stft(
        samples, n_fft=settings.n_fft, hop_length=settings.hop_length, win_length=settings.win_length,
        window=periodic_hann(settings, device=device), center=settings.center, pad_mode=settings.pad_mode,
        normalized=settings.normalized, onesided=settings.onesided, return_complex=True,
    )
    if spectrum.shape != (7, 513, 33) or spectrum.dtype != torch.complex64 or not torch.isfinite(spectrum).all():
        raise Layer3Error(f"共享STFT输出无效: {tuple(spectrum.shape)} {spectrum.dtype}")
    return spectrum


def inverse_stft(spectrum: torch.Tensor, settings: StftSettings, *, length: int = 15_360) -> torch.Tensor:
    """Invert one or more spectra, preserving any leading batch dimensions."""
    waveform = torch.istft(
        spectrum, n_fft=settings.n_fft, hop_length=settings.hop_length, win_length=settings.win_length,
        window=periodic_hann(settings, device=spectrum.device), center=settings.center,
        normalized=settings.normalized, onesided=settings.onesided, length=length,
    )
    expected_shape = (*spectrum.shape[:-2], length)
    if waveform.shape != expected_shape or not torch.isfinite(waveform).all():
        raise Layer3Error("ISTFT试听音频无效")
    return waveform
