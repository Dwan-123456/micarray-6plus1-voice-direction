from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as functional

from common.data_types import DecisionWindow
from common.timing import CONTEXT_SAMPLES

from .configuration import StftSettings
from .interface import Layer3Error


MAX_TEMPORAL_CACHE_HOPS = 50
_HOP_SAMPLES = 960
_STFT_FRAMES_PER_HOP = 2


@lru_cache(maxsize=8)
def periodic_hann(settings: StftSettings, *, device: torch.device) -> torch.Tensor:
    """Return a bounded, device-resident STFT window cache."""
    return torch.hann_window(settings.win_length, periodic=True, dtype=torch.float32, device=device)


def _input_tensor(
    window: DecisionWindow, settings: StftSettings, *, device: torch.device,
) -> torch.Tensor:
    # HardwareMix remains outside every array calculation.
    selected = window.samples[-settings.window_samples:, :7]
    return torch.as_tensor(selected.T.copy(), dtype=torch.float32, device=device)


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
        self._recomputed_key: tuple[int, int] | None = None
        self._last_reused_frames = 0
        self._last_recomputed_frames = 0

    def clear(self) -> None:
        self._identity = None
        self._settings = None
        self._spectrum = None
        self._recomputed_indices = None
        self._recomputed_key = None
        self._last_reused_frames = 0
        self._last_recomputed_frames = 0

    def process(self, window: DecisionWindow, settings: StftSettings) -> torch.Tensor:
        identity = (
            window.session_id,
            window.stream_epoch,
            window.context_start_sample,
            window.context_end_sample,
        )
        previous = self._identity
        sample_delta = 0 if previous is None else identity[2] - previous[2]
        hop_gap = sample_delta // _HOP_SAMPLES if sample_delta > 0 else 0
        has_overlapping_history = (
            self._spectrum is not None
            and self._settings == settings
            and previous is not None
            and identity[0:2] == previous[0:2]
            and identity[3] - previous[3] == sample_delta
            and sample_delta % _HOP_SAMPLES == 0
            and 1 <= hop_gap < settings.window_hops
        )
        if not has_overlapping_history:
            spectrum = shared_stft(window, settings, device=self.device)
            self._last_reused_frames = 0
            self._last_recomputed_frames = settings.frame_count
        else:
            samples = _input_tensor(window, settings, device=self.device)
            # The locked 480-sample STFT hop advances two frames per 20 ms.
            # Frame 0 of the current window and the last frame of the previous one
            # depend on their respective reflected boundary, so only the
            # aligned interior frames are reusable across the overlap.
            last_frame = settings.frame_count - 1
            first_new_frame = last_frame - _STFT_FRAMES_PER_HOP * hop_gap
            recomputed_frame_indices = (0, *range(first_new_frame, settings.frame_count))
            recomputed_key = (hop_gap, settings.frame_count)
            if self._recomputed_indices is None or self._recomputed_key != recomputed_key:
                self._recomputed_indices = torch.tensor(
                    recomputed_frame_indices,
                    dtype=torch.long,
                    device=self.device,
                )
                self._recomputed_key = recomputed_key
            recomputed = _selected_stft_frames(
                samples,
                settings,
                self._recomputed_indices,
                device=self.device,
            )
            spectrum = torch.empty_like(self._spectrum)
            current_reused = slice(1, first_new_frame)
            previous_reused = slice(1 + _STFT_FRAMES_PER_HOP * hop_gap, last_frame)
            spectrum[:, :, current_reused] = self._spectrum[
                :, :, previous_reused
            ]
            spectrum[:, :, self._recomputed_indices] = recomputed
            self._last_reused_frames = settings.frame_count - 2 - _STFT_FRAMES_PER_HOP * hop_gap
            self._last_recomputed_frames = len(recomputed_frame_indices)
        if spectrum.shape != (7, 513, settings.frame_count) or not torch.isfinite(spectrum).all():
            self.clear()
            raise Layer3Error("rolling STFT output is invalid")
        # Only the configured 80/160 ms window persists. This is deliberately
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
            0 if self._spectrum is None or self._settings is None else self._settings.window_hops,
            self._last_reused_frames,
            self._last_recomputed_frames,
            tensor_bytes,
        )


def shared_stft(window: DecisionWindow, settings: StftSettings, *, device: torch.device) -> torch.Tensor:
    if window.samples.shape != (CONTEXT_SAMPLES, 8):
        raise Layer3Error(f"L3输入必须是48 kHz逻辑8通道 [7680,8]，实际为{window.samples.shape}")
    # HardwareMix is preserved by the public input contract but is never an
    # array microphone: steering, covariance and beamforming use PhysicalAudio.
    samples = _input_tensor(window, settings=settings, device=device)
    spectrum = torch.stft(
        samples, n_fft=settings.n_fft, hop_length=settings.hop_length, win_length=settings.win_length,
        window=periodic_hann(settings, device=device), center=settings.center, pad_mode=settings.pad_mode,
        normalized=settings.normalized, onesided=settings.onesided, return_complex=True,
    )
    if spectrum.shape != (7, 513, settings.frame_count) or spectrum.dtype != torch.complex64 or not torch.isfinite(spectrum).all():
        raise Layer3Error(f"共享STFT输出无效: {tuple(spectrum.shape)} {spectrum.dtype}")
    return spectrum


def inverse_stft(spectrum: torch.Tensor, settings: StftSettings, *, length: int | None = None) -> torch.Tensor:
    """Invert one or more spectra, preserving any leading batch dimensions."""
    length = settings.window_samples if length is None else int(length)
    waveform = torch.istft(
        spectrum, n_fft=settings.n_fft, hop_length=settings.hop_length, win_length=settings.win_length,
        window=periodic_hann(settings, device=spectrum.device), center=settings.center,
        normalized=settings.normalized, onesided=settings.onesided, length=length,
    )
    expected_shape = (*spectrum.shape[:-2], length)
    if waveform.shape != expected_shape or not torch.isfinite(waveform).all():
        raise Layer3Error("ISTFT试听音频无效")
    return waveform
