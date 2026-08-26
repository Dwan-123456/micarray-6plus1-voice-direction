from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from common.config import ProjectConfig
from common.data_types import CandidateDirection, DecisionWindow, EnhancedAudio
from common.geometry import MicGeometry
from common.timing import CONTEXT_SAMPLES
from spatial_separability import P_TABLE_VERSION

from .configuration import SpatialSeparationConfig, StftSettings
from .hybrid import ImcraSpatialSeparationBeamformer
from .interface import L3_MODE_OPTIMIZED, L3_PROCESSING_MODES, Layer3Output
from .prepared import BeamformedL3Batch, PreparedL3Context
from .shared_stft import inverse_stft


@dataclass(frozen=True, slots=True, eq=False)
class PendingLayer3Output:
    """Device-resident L3 waveforms awaiting one batched host transfer."""

    prepared: PreparedL3Context
    batch: BeamformedL3Batch
    waveforms: torch.Tensor


@dataclass(frozen=True, slots=True, eq=False)
class PendingHostLayer3Output:
    """Pinned host waveforms whose non-blocking CUDA copy may still be running."""

    prepared: PreparedL3Context
    batch: BeamformedL3Batch
    waveforms: torch.Tensor


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
        defer_device_validation: bool = False,
    ) -> PreparedL3Context:
        """Compute ordered, candidate-independent work for one timeline window."""
        self._validate_input(window)
        if mode not in L3_PROCESSING_MODES:
            raise ValueError(f"未知L3处理模式: {mode}")
        return self.beamformer.prepare_context(
            window,
            self.beamforming,
            self.stft,
            mode=mode,
            defer_device_validation=defer_device_validation,
        )

    def process_prepared(
        self,
        prepared: PreparedL3Context,
        candidates: tuple[CandidateDirection, ...],
        geometry: MicGeometry,
    ) -> Layer3Output:
        """Finish steering/BF and one batched ISTFT without a device round trip."""
        pending = self.process_prepared_device(prepared, candidates, geometry)
        return self.finalize_host_output(self.stage_host_transfer(pending))

    def process_prepared_device(
        self,
        prepared: PreparedL3Context,
        candidates: tuple[CandidateDirection, ...],
        geometry: MicGeometry,
    ) -> PendingLayer3Output:
        """Keep beamforming and ISTFT on the selected compute device."""
        batch = self.beamformer.process_prepared_batch(prepared, candidates, geometry)
        inverse_kwargs = {"length": self.window_spec.samples}
        if prepared.device_validation_deferred:
            inverse_kwargs["validate_values"] = False
        # The beamformer publishes an already passband-masked spectrum.  Reuse
        # it directly so each 20 ms window avoids a second complex tensor.
        waveforms = inverse_stft(batch.spectra_mft, prepared.stft, **inverse_kwargs)
        return PendingLayer3Output(prepared, batch, waveforms.detach())

    @staticmethod
    def stage_host_transfer(pending: PendingLayer3Output) -> PendingHostLayer3Output:
        """Queue one non-blocking device-to-pinned-host waveform transfer."""
        waveforms = pending.waveforms
        if waveforms.device.type == "cuda":
            host = torch.empty_like(waveforms, device="cpu", pin_memory=True)
            host.copy_(waveforms, non_blocking=True)
        else:
            host = waveforms.to(device="cpu")
        return PendingHostLayer3Output(pending.prepared, pending.batch, host)

    def finalize_host_output(self, pending: PendingHostLayer3Output) -> Layer3Output:
        """Build public numpy DTOs after the owning stream has completed."""
        return Layer3Output(self._synthesize_host(pending))

    def _validate_input(self, window: DecisionWindow) -> None:
        if window.sample_rate != 48_000 or window.samples.shape != (CONTEXT_SAMPLES, 8):
            raise RuntimeError(
                f"L3输入必须是48 kHz逻辑8通道 [7680,8]，实际为"
                f"{window.sample_rate} Hz {window.samples.shape}"
            )
        if self.window_spec.samples > len(window.samples):
            raise RuntimeError("L3下游音频窗口超过DecisionWindow可用上下文")

    def _synthesize_host(
        self,
        pending: PendingHostLayer3Output,
    ) -> tuple[EnhancedAudio, ...]:
        prepared = pending.prepared
        batch = pending.batch
        if not batch.theta_degrees:
            return ()
        host = np.ascontiguousarray(
            pending.waveforms.numpy(), dtype=np.float32,
        )
        if not np.isfinite(host).all():
            raise RuntimeError("L3 CUDA微批次回传了非有限音频")
        diagnostics = batch.diagnostics
        fallback_reasons = batch.fallback_reasons
        if batch.deferred_diagnostics is not None:
            values = batch.deferred_diagnostics.detach().cpu().tolist()
            count = len(batch.theta_degrees)
            lcmv_bins, soft_bins, mvdr_bins = (int(item) for item in values[:3])
            fallback_bins = tuple(int(item) for item in values[3:3 + count])
            rho_min, rho_max, gain_min = (
                float(item) for item in values[count + 4:count + 7]
            )
            fallback_reason = (
                f"per-bin DAS fallback counts={fallback_bins}"
                if any(fallback_bins)
                else None
            )
            spatial_p = (
                "single_candidate" if count == 1
                else P_TABLE_VERSION if count == 2
                else "independent_loaded_mvdr"
            )
            diagnostics = tuple(
                (
                    "backend=imcra_spatial_separation",
                    f"imcra={prepared.noise_algorithm_version}:"
                    f"{prepared.stft.window_hops}x20ms",
                    f"spatial_p={spatial_p}",
                    f"rho_thresholds={prepared.config.rho_lcmv_max:.3f}/"
                    f"{prepared.config.rho_soft_null_max:.3f}",
                    f"rho_range={rho_min:.4f}..{rho_max:.4f}",
                    f"cache:stft_reused={prepared.stft_reused_frames},"
                    f"covariance_rolled={prepared.covariance_rolled}",
                    f"bins:lcmv={lcmv_bins},soft_null_mvdr={soft_bins},"
                    f"loaded_mvdr={mvdr_bins},das_fallback={fallback_bins[index]}",
                    f"frequency_gain_min={gain_min:.4f}",
                )
                for index in range(count)
            )
            fallback_reasons = tuple(fallback_reason for _item in range(count))
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
                fallback_reasons[index],
                diagnostics[index],
                host[index],
                batch.track_ids[index],
            )
            for index, theta in enumerate(batch.theta_degrees)
        )

    def _synthesize_prepared(
        self,
        prepared: PreparedL3Context,
        batch: BeamformedL3Batch,
    ) -> tuple[EnhancedAudio, ...]:
        """Compatibility helper used by the standalone benchmark."""
        pending = PendingLayer3Output(
            prepared,
            batch,
            inverse_stft(
                batch.spectra_mft,
                prepared.stft,
                length=self.window_spec.samples,
            ).detach(),
        )
        return self._synthesize_host(self.stage_host_transfer(pending))
