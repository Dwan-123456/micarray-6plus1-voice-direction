from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from common.config import Layer1PreDenoiseConfig, ProjectConfig
from common.data_types import ImcraHopSnapshot, IngestedAudioBlock


@dataclass(frozen=True, slots=True)
class PreDenoiseHop:
    """One delayed 20 ms hop in both original and IMCRA-denoised form."""

    raw: IngestedAudioBlock
    denoised: IngestedAudioBlock


class ImcraWienerPreDenoiser:
    """Seven independent Wiener masks with causal 40 ms/20 ms WOLA synthesis."""

    def __init__(self, config: Layer1PreDenoiseConfig, *, sample_rate: int = 48_000) -> None:
        if sample_rate != 48_000:
            raise ValueError("L1 IMCRA pre-denoiser only supports 48 kHz")
        self.config = config
        self.sample_rate = sample_rate
        self.hop_samples = config.hop_samples
        self.frame_samples = config.frame_samples
        periodic_hann = np.hanning(self.frame_samples + 1)[:-1]
        self._window = np.sqrt(np.maximum(periodic_hann, 0.0)).astype(np.float64)
        self._frequencies = np.fft.rfftfreq(config.n_fft, 1.0 / sample_rate)
        self._output_band = self._frequencies <= config.frequency_max_hz
        self._minimum_gain = 10.0 ** (config.minimum_gain_db / 20.0)
        self._identity: tuple[str, int] | None = None
        self._previous_block: IngestedAudioBlock | None = None
        self._ola_tail = np.zeros((self.hop_samples, 7), dtype=np.float64)
        self._previous_gain = np.ones((self._frequencies.size, 7), dtype=np.float64)
        self._next_sample = 0
        self.last_mean_gain_db = 0.0

    @classmethod
    def from_project(cls, config: ProjectConfig) -> "ImcraWienerPreDenoiser":
        return cls(config.layer1_pre_denoise, sample_rate=config.device.sample_rate)

    def reset(self) -> None:
        self._identity = None
        self._previous_block = None
        self._ola_tail.fill(0.0)
        self._previous_gain.fill(1.0)
        self._next_sample = 0
        self.last_mean_gain_db = 0.0

    @staticmethod
    def _smooth_frequency(gain: np.ndarray) -> np.ndarray:
        padded = np.pad(gain, ((1, 1), (0, 0)), mode="edge")
        return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]

    def _gain(self, hop: ImcraHopSnapshot | None) -> np.ndarray:
        target = np.ones_like(self._previous_gain)
        if hop is not None and hop.state == "ready":
            prior_snr = np.asarray(hop.prior_snr, dtype=np.float64).T
            spp = np.asarray(hop.spp, dtype=np.float64).T
            wiener = prior_snr / (1.0 + prior_snr)
            protected = spp + (1.0 - spp) * wiener
            protected = np.clip(self._smooth_frequency(protected), self._minimum_gain, 1.0)
            if protected.shape != (int(np.count_nonzero(self._output_band)), 7):
                raise ValueError("IMCRA PSD/SPP frequency axis does not match the pre-denoiser")
            target[self._output_band] = protected
        alpha = self.config.gain_smoothing
        gain = alpha * self._previous_gain + (1.0 - alpha) * target
        gain = np.clip(gain, self._minimum_gain, 1.0)
        self._previous_gain = gain
        active = gain[self._output_band]
        self.last_mean_gain_db = float(20.0 * np.log10(np.maximum(np.mean(active), 1.0e-12)))
        return gain

    def _render_frame(
        self, left: np.ndarray, right: np.ndarray, hop: ImcraHopSnapshot | None
    ) -> np.ndarray:
        frame = np.concatenate((left[:, :7], right[:, :7]), axis=0).astype(np.float64, copy=False)
        spectrum = np.fft.rfft(frame * self._window[:, None], n=self.config.n_fft, axis=0)
        spectrum *= self._gain(hop)
        reconstructed = np.fft.irfft(spectrum, n=self.config.n_fft, axis=0)[: self.frame_samples]
        reconstructed *= self._window[:, None]
        output = self._ola_tail + reconstructed[: self.hop_samples]
        self._ola_tail = reconstructed[self.hop_samples :].copy()
        if not np.isfinite(output).all():
            raise ValueError("L1 pre-denoiser produced non-finite audio")
        return np.ascontiguousarray(output, dtype=np.float32)

    @staticmethod
    def _replace_physical(block: IngestedAudioBlock, physical: np.ndarray) -> IngestedAudioBlock:
        logical = np.array(block.samples, dtype=np.float32, order="C", copy=True)
        logical[:, :7] = physical
        # Logical channel 7 is the untouched hardware-generated mix. Native
        # samples also remain untouched so formal storage retains device truth.
        return replace(block, samples=logical)

    def _flush_previous(self) -> tuple[PreDenoiseHop, ...]:
        previous = self._previous_block
        if previous is None:
            return ()
        zeros = np.zeros((self.hop_samples, 8), dtype=np.float32)
        physical = self._render_frame(previous.samples, zeros, previous.imcra_hop)
        result = PreDenoiseHop(previous, self._replace_physical(previous, physical))
        self._previous_block = None
        return (result,)

    def process(self, block: IngestedAudioBlock) -> tuple[PreDenoiseHop, ...]:
        if block.samples.shape != (self.hop_samples, 8):
            raise ValueError("L1 pre-denoiser requires exact 20 ms [960,8] input blocks")
        identity = (block.session_id, block.stream_epoch)
        output: list[PreDenoiseHop] = []
        if identity != self._identity:
            output.extend(self._flush_previous())
            self._identity = identity
            self._previous_block = None
            self._ola_tail.fill(0.0)
            self._previous_gain.fill(1.0)
            self._next_sample = block.start_sample
        if block.start_sample != self._next_sample:
            raise ValueError("L1 pre-denoiser received a discontinuity without a new stream epoch")

        previous = self._previous_block
        if previous is None:
            zeros = np.zeros_like(block.samples)
            self._render_frame(zeros, block.samples, block.imcra_hop)
        else:
            physical = self._render_frame(previous.samples, block.samples, block.imcra_hop)
            output.append(PreDenoiseHop(previous, self._replace_physical(previous, physical)))
        self._previous_block = block
        self._next_sample = block.end_sample
        return tuple(output)

    def flush(self) -> tuple[PreDenoiseHop, ...]:
        output = self._flush_previous()
        self.reset()
        return output
