from __future__ import annotations

import numpy as np
import torch

from common.config import ProjectConfig
from common.data_types import DecisionWindow, EnhancedAudio, TrackedDirection
from common.geometry import MicGeometry
from common.window_key import WindowKey
from common.timing import CONTEXT_SAMPLES

from .configuration import SpatialSeparationConfig, StftSettings
from .hybrid import ImcraSpatialSeparationBeamformer
from .interface import (
    L3_MODE_OPTIMIZED,
    L3_PROCESSING_MODES,
    Layer3Error,
    Layer3Output,
    validate_l3_directions,
)
from .prepared import BeamformedL3Batch, PreparedL3Context
from .shared_stft import inverse_stft


class Layer3Processor:
    def __init__(
        self, config: ProjectConfig, *, device: str | torch.device = "cpu",
    ) -> None:
        self.stft = StftSettings.from_project(config)
        self.beamforming = SpatialSeparationConfig.from_project(config)
        self.beamformer = ImcraSpatialSeparationBeamformer(device=device)

    def clear_cache(self) -> None:
        self.beamformer.clear_cache()

    def cache_snapshot(self):
        """Expose bounded internal-cache metrics without publishing cached tensors."""
        return self.beamformer.cache_snapshot()

    def process(
        self, window: DecisionWindow, directions: tuple[TrackedDirection, ...], geometry: MicGeometry,
        *, mode: str = L3_MODE_OPTIMIZED,
    ) -> Layer3Output:
        self._validate_input(window)
        if mode not in L3_PROCESSING_MODES:
            raise ValueError(f"未知L3处理模式: {mode}")
        window_key = WindowKey.from_window(window)
        directions = validate_l3_directions(window_key, directions)
        if not directions:
            return Layer3Output(window_key, ())
        prepared = self.prepare(window, mode=mode)
        return self.process_prepared(prepared, directions, geometry)

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
        directions: tuple[TrackedDirection, ...],
        geometry: MicGeometry,
    ) -> Layer3Output:
        """Finish steering/BF and one batched ISTFT without a device round trip."""
        directions = validate_l3_directions(prepared.window_key, directions)
        batch = self.beamformer.process_prepared_batch(prepared, directions, geometry)
        audio = self._synthesize_prepared(prepared, batch)
        self._validate_output_alignment(prepared.window_key, directions, batch, audio)
        return Layer3Output(prepared.window_key, audio)

    @staticmethod
    def _validate_input(window: DecisionWindow) -> None:
        if window.sample_rate != 48_000 or window.samples.shape != (CONTEXT_SAMPLES, 8):
            raise RuntimeError(
                f"L3输入必须是48 kHz逻辑8通道 [7680,8]，实际为"
                f"{window.sample_rate} Hz {window.samples.shape}"
            )

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
        waveforms = inverse_stft(band_limited, prepared.stft)
        host = np.ascontiguousarray(
            waveforms.detach().cpu().numpy(), dtype=np.float32,
        )
        return tuple(
            EnhancedAudio(
                prepared.session_id,
                prepared.stream_epoch,
                prepared.window_id,
                prepared.decision_sample,
                batch.track_ids[index],
                batch.ranks[index],
                prepared.context_start_sample,
                prepared.context_end_sample,
                theta,
                prepared.sample_rate,
                batch.backends[index],
                batch.fallback_reasons[index],
                batch.diagnostics[index],
                host[index],
            )
            for index, theta in enumerate(batch.theta_degrees)
        )

    @staticmethod
    def _validate_output_alignment(
        window_key: WindowKey,
        directions: tuple[TrackedDirection, ...],
        batch: BeamformedL3Batch,
        audio: tuple[EnhancedAudio, ...],
    ) -> None:
        expected_ids = tuple(item.track_id for item in directions)
        expected_ranks = tuple(item.rank for item in directions)
        expected_angles = tuple(item.theta_deg for item in directions)
        if batch.window_key != window_key or any(item.window_key != window_key for item in audio):
            raise Layer3Error("L3 output WindowKey does not match its input")
        if not (
            batch.track_ids == expected_ids
            == tuple(item.track_id for item in audio)
        ):
            raise Layer3Error("L3 output track_id set or order does not match its input")
        if not (
            batch.ranks == expected_ranks
            == tuple(item.rank for item in audio)
        ):
            raise Layer3Error("L3 output original direction order does not match its input")
        if not (
            batch.theta_degrees == expected_angles
            == tuple(item.theta_deg for item in audio)
        ):
            raise Layer3Error("L3 output angles do not match its input")
        if len(audio) != len(directions):
            raise Layer3Error("L3 output audio count does not match its input")
