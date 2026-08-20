from __future__ import annotations

import numpy as np
import torch

from common.config import ProjectConfig
from common.data_types import CandidateDirection, DecisionWindow, EnhancedAudio
from common.geometry import MicGeometry
from common.timing import CONTEXT_SAMPLES

from .configuration import SpatialSeparationConfig, StftSettings
from .hybrid import ImcraSpatialSeparationBeamformer
from .interface import L3_MODE_OPTIMIZED, L3_PROCESSING_MODES, Layer3Output
from .prepared import BeamformedL3Batch, PreparedL3Context
from .shared_stft import inverse_stft


class Layer3Processor:
    def __init__(
        self, config: ProjectConfig, *, device: str | torch.device = "cpu",
    ) -> None:
        self.stft = StftSettings.from_project(config)
        self.window_spec = config.downstream_audio_window
        self.beamforming = SpatialSeparationConfig.from_project(config)
        self.beamformer = ImcraSpatialSeparationBeamformer(device=device)

    def clear_cache(self) -> None:
        self.beamformer.clear_cache()

    def cache_snapshot(self):
        """Expose bounded internal-cache metrics without publishing cached tensors."""
        return self.beamformer.cache_snapshot()

    def process(
        self, window: DecisionWindow, candidates: tuple[CandidateDirection, ...], geometry: MicGeometry,
        *, mode: str = L3_MODE_OPTIMIZED,
    ) -> Layer3Output:
        self._validate_input(window)
        if mode not in L3_PROCESSING_MODES:
            raise ValueError(f"未知L3处理模式: {mode}")
        if not candidates:
            return Layer3Output(())
        prepared = self.prepare(window, mode=mode)
        return self.process_prepared(prepared, candidates, geometry)

    def prepare(
        self,
        window: DecisionWindow,
        *,
        mode: str = L3_MODE_OPTIMIZED,
    ) -> PreparedL3Context:
        """Compute ordered, candidate-independent work for one timeline window."""
        self._validate_input(window)
        if mode not in L3_PROCESSING_MODES:
            raise ValueError(f"未知L3处理模式: {mode}")
        return self.beamformer.prepare_context(window, self.beamforming, self.stft, mode=mode)

    def process_prepared(
        self,
        prepared: PreparedL3Context,
        candidates: tuple[CandidateDirection, ...],
        geometry: MicGeometry,
    ) -> Layer3Output:
        """Finish steering/BF and one batched ISTFT without a device round trip."""
        batch = self.beamformer.process_prepared_batch(prepared, candidates, geometry)
        return Layer3Output(self._synthesize_prepared(prepared, batch))

    def _validate_input(self, window: DecisionWindow) -> None:
        if window.sample_rate != 48_000 or window.samples.shape != (CONTEXT_SAMPLES, 8):
            raise RuntimeError(
                f"L3输入必须是48 kHz逻辑8通道 [7680,8]，实际为"
                f"{window.sample_rate} Hz {window.samples.shape}"
            )
        if self.window_spec.samples > len(window.samples):
            raise RuntimeError("L3下游音频窗口超过DecisionWindow可用上下文")

    def _synthesize_prepared(
        self,
        prepared: PreparedL3Context,
        batch: BeamformedL3Batch,
    ) -> tuple[EnhancedAudio, ...]:
        if not batch.theta_degrees:
            return ()
        # Apply the shared passband on-device, invert all candidates in one
        # torch.istft call, and transfer the completed waveform batch once.
        band_limited = batch.spectra_mft * prepared.passband_f[None, :, None]
        waveforms = inverse_stft(
            band_limited, prepared.stft, length=self.window_spec.samples,
        )
        host = np.ascontiguousarray(
            waveforms.detach().cpu().numpy(), dtype=np.float32,
        )
        return tuple(
            EnhancedAudio(
                prepared.session_id,
                prepared.stream_epoch,
                prepared.window_id,
                prepared.decision_sample,
                prepared.context_start_sample,
                prepared.context_end_sample,
                theta,
                prepared.sample_rate,
                batch.backends[index],
                batch.fallback_reasons[index],
                batch.diagnostics[index],
                host[index],
                batch.track_ids[index],
            )
            for index, theta in enumerate(batch.theta_degrees)
        )
