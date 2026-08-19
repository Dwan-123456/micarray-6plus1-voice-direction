from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from common.data_types import CandidateDirection, DecisionWindow, DirectionalSignal
from common.geometry import MicGeometry
from spatial_separability import (
    P_FREQUENCY_BIN_INDICES,
    P_TABLE_VERSION,
    lookup_p,
    validate_p_table_context,
)

from .adaptive_separation import AdaptiveStaticData, adaptive_separation_weights, adaptive_static_data
from .configuration import SpatialSeparationConfig, StftSettings
from .constant_beamwidth import constant_beamwidth_weights
from .das import apply_weights, das_weights
from .interface import (
    L3_MODE_CONSTANT_BEAMWIDTH,
    L3_MODE_DS_BASELINE,
    L3_MODE_OPTIMIZED,
    Layer3Error,
)
from .noise_context import BeamformerNoiseContext, RollingNoiseStatisticsCache
from .prepared import BeamformedL3Batch, PreparedL3Context
from .shared_stft import RollingStftCache
from .steering import steering_vectors


@dataclass(frozen=True, slots=True)
class BeamformerCacheSnapshot:
    max_temporal_hops: int
    stft_temporal_hops: int
    imcra_temporal_hops: int
    stft_reused_frames: int
    stft_recomputed_frames: int
    covariance_rolled: bool
    steering_entries: int
    p_entries: int
    persistent_tensor_bytes: int
    prepared_entries: int
    prepared_entry_limit: int
    prepared_tensor_bytes: int


class ImcraSpatialSeparationBeamformer:
    """IMCRA-controlled per-bin LCMV / soft-null MVDR / loaded MVDR."""

    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.last_diagnostics: tuple[tuple[str, ...], ...] = ()
        self._stft_cache = RollingStftCache(device=self.device)
        self._noise_cache = RollingNoiseStatisticsCache()
        self._frequency_key: tuple[int, int] | None = None
        self._frequencies: torch.Tensor | None = None
        self._adaptive_static_key: tuple[object, ...] | None = None
        self._adaptive_static: AdaptiveStaticData | None = None
        self._steering_cache: OrderedDict[tuple[object, ...], torch.Tensor] = OrderedDict()
        self._p_cache: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()
        self._constant_beamwidth_cache: OrderedDict[
            tuple[object, ...], tuple[torch.Tensor, int, float]
        ] = OrderedDict()
        self._prepared_cache: OrderedDict[tuple[object, ...], PreparedL3Context] = OrderedDict()
        self._static_cache_limit = 16
        self._prepared_cache_limit = 2
        self._p_frequency_indices: torch.Tensor | None = None

    def clear_cache(self) -> None:
        self.last_diagnostics = ()
        self._stft_cache.clear()
        self._noise_cache.clear()
        self._frequency_key = None
        self._frequencies = None
        self._adaptive_static_key = None
        self._adaptive_static = None
        self._steering_cache.clear()
        self._p_cache.clear()
        self._constant_beamwidth_cache.clear()
        self._prepared_cache.clear()
        self._p_frequency_indices = None

    def cache_snapshot(self) -> BeamformerCacheSnapshot:
        stft = self._stft_cache.snapshot()
        noise = self._noise_cache.snapshot()
        static_tensors = ([] if self._frequencies is None else [self._frequencies])
        if self._adaptive_static is not None:
            static_tensors.extend((
                self._adaptive_static.speech_band_f,
                self._adaptive_static.identity_cc,
                self._adaptive_static.alias_multiplier_f,
            ))
        static_tensors.extend(self._steering_cache.values())
        static_tensors.extend(self._p_cache.values())
        static_tensors.extend(item[0] for item in self._constant_beamwidth_cache.values())
        if self._p_frequency_indices is not None:
            static_tensors.append(self._p_frequency_indices)
        static_bytes = sum(item.numel() * item.element_size() for item in static_tensors)
        prepared_bytes = sum(item.persistent_tensor_bytes for item in self._prepared_cache.values())
        return BeamformerCacheSnapshot(
            min(stft.max_temporal_hops, noise.max_temporal_hops),
            stft.temporal_hops,
            noise.temporal_hops,
            stft.reused_frames,
            stft.recomputed_frames,
            noise.rolled,
            len(self._steering_cache),
            len(self._p_cache),
            stft.persistent_tensor_bytes + noise.persistent_tensor_bytes + static_bytes + prepared_bytes,
            len(self._prepared_cache),
            self._prepared_cache_limit,
            prepared_bytes,
        )

    def _frequency_axis(self, sample_rate: int, stft: StftSettings) -> torch.Tensor:
        key = (sample_rate, stft.n_fft)
        if self._frequency_key != key or self._frequencies is None:
            self._frequency_key = key
            self._frequencies = torch.fft.rfftfreq(
                stft.n_fft, d=1.0 / sample_rate, device=self.device,
            )
            self._adaptive_static_key = None
            self._adaptive_static = None
            self._steering_cache.clear()
            self._p_cache.clear()
            self._constant_beamwidth_cache.clear()
            self._prepared_cache.clear()
            self._p_frequency_indices = None
        return self._frequencies

    def _adaptive_static_data(
        self,
        frequencies: torch.Tensor,
        config: SpatialSeparationConfig,
    ) -> AdaptiveStaticData:
        key = (self._frequency_key, config)
        if self._adaptive_static_key != key or self._adaptive_static is None:
            self._adaptive_static_key = key
            self._adaptive_static = adaptive_static_data(frequencies, config)
        return self._adaptive_static

    @staticmethod
    def _bounded_put(cache: OrderedDict, key: object, value: torch.Tensor, limit: int) -> torch.Tensor:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def _steering_vectors(
        self,
        frequencies: torch.Tensor,
        candidates: tuple[CandidateDirection, ...],
        geometry: MicGeometry,
    ) -> torch.Tensor:
        geometry_key = (
            geometry.version,
            float(geometry.speed_of_sound_mps),
            geometry.positions_m.tobytes(),
            self._frequency_key,
        )
        vectors: list[torch.Tensor] = []
        for candidate in candidates:
            key = (*geometry_key, float(candidate.theta_deg))
            cached = self._steering_cache.get(key)
            if cached is None:
                theta = torch.tensor(
                    [candidate.theta_deg], dtype=torch.float32, device=self.device,
                )
                cached = self._bounded_put(
                    self._steering_cache,
                    key,
                    steering_vectors(frequencies, theta, geometry)[0],
                    self._static_cache_limit,
                )
            else:
                self._steering_cache.move_to_end(key)
            vectors.append(cached)
        return torch.stack(vectors, dim=0)

    def _spatial_p(
        self,
        candidates: tuple[CandidateDirection, ...],
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        # lookup_p quantizes to nearest degree and is symmetric.  Use the same
        # canonical key so sub-degree tracker jitter and candidate order do not
        # create duplicate device tensors.
        quantized = tuple(
            int(np.floor((float(item.theta_deg) % 360.0) + 0.5)) % 360
            for item in candidates
        )
        key = tuple(sorted(quantized))
        cached = self._p_cache.get(key)
        if cached is not None:
            self._p_cache.move_to_end(key)
            return cached
        spatial_p = torch.ones_like(frequencies, dtype=torch.float32)
        table_values = torch.from_numpy(lookup_p(*key).copy()).to(self.device)
        if self._p_frequency_indices is None:
            self._p_frequency_indices = torch.tensor(
                P_FREQUENCY_BIN_INDICES.copy(), dtype=torch.long, device=self.device,
            )
        spatial_p[self._p_frequency_indices] = table_values
        return self._bounded_put(self._p_cache, key, spatial_p, self._static_cache_limit)

    def _constant_beamwidth_weights(
        self,
        frequencies: torch.Tensor,
        candidates: tuple[CandidateDirection, ...],
        geometry: MicGeometry,
        config: SpatialSeparationConfig,
    ) -> tuple[torch.Tensor, tuple[int, ...], tuple[float, ...]]:
        geometry_key = (
            geometry.version,
            float(geometry.speed_of_sound_mps),
            geometry.positions_m.tobytes(),
            self._frequency_key,
            config.constant_beamwidth_fnbw_deg,
            config.constant_beamwidth_design_grid_deg,
            config.constant_beamwidth_regularization,
            config.constant_beamwidth_min_wng_db,
            config.frequency_min_hz,
            config.frequency_max_hz,
        )
        output: list[torch.Tensor] = []
        fallback: list[int] = []
        minimum_wng: list[float] = []
        for candidate in candidates:
            angle_step = config.constant_beamwidth_design_grid_deg
            quantized_theta = (
                np.floor((float(candidate.theta_deg) % 360.0) / angle_step + 0.5) * angle_step
            ) % 360.0
            key = (*geometry_key, float(quantized_theta))
            cached = self._constant_beamwidth_cache.get(key)
            if cached is None:
                theta = torch.tensor(
                    [quantized_theta], dtype=torch.float32, device=self.device,
                )
                design_steering = steering_vectors(frequencies, theta, geometry)
                solved = constant_beamwidth_weights(
                    frequencies, design_steering, theta, geometry, config,
                )
                cached = (solved.weights_mfc[0], solved.fallback_bins[0], solved.minimum_wng_db[0])
                self._constant_beamwidth_cache[key] = cached
                self._constant_beamwidth_cache.move_to_end(key)
                while len(self._constant_beamwidth_cache) > self._static_cache_limit:
                    self._constant_beamwidth_cache.popitem(last=False)
            else:
                self._constant_beamwidth_cache.move_to_end(key)
            output.append(cached[0])
            fallback.append(cached[1])
            minimum_wng.append(cached[2])
        return torch.stack(output), tuple(fallback), tuple(minimum_wng)

    def prepare_context(
        self,
        window: DecisionWindow,
        config: SpatialSeparationConfig,
        stft: StftSettings,
        *,
        mode: str = L3_MODE_OPTIMIZED,
    ) -> PreparedL3Context:
        """Prepare all candidate-independent L3 work in timeline order.

        This method owns the rolling STFT/IMCRA state and must therefore be
        called by one FIFO worker.  The returned frozen context can wait in a
        bounded runtime queue while L2 finishes the same window.
        """
        self._validate_window(window)
        if mode not in {
            L3_MODE_OPTIMIZED, L3_MODE_DS_BASELINE, L3_MODE_CONSTANT_BEAMWIDTH,
        }:
            raise ValueError(f"未知L3处理模式: {mode}")
        key = (
            window.session_id,
            window.stream_epoch,
            window.window_id,
            window.decision_sample,
            window.context_start_sample,
            window.context_end_sample,
            id(window.samples),
            tuple(id(item) for item in window.imcra_hops),
            mode,
            stft,
            config,
        )
        cached = self._prepared_cache.get(key)
        if cached is not None:
            self._prepared_cache.move_to_end(key)
            return cached

        spectrum_cft = self._stft_cache.process(window, stft)
        spectrum_fct = spectrum_cft.permute(1, 0, 2).contiguous().detach()
        frequencies = self._frequency_axis(window.sample_rate, stft)
        static = self._adaptive_static_data(frequencies, config)
        noise = None
        noise_version = None
        preparation_error = None
        covariance_rolled = False
        reused_frames = self._stft_cache.snapshot().reused_frames
        if mode == L3_MODE_OPTIMIZED:
            try:
                noise, noise_version = self._noise_cache.estimate_window(
                    window,
                    spectrum_fct,
                    frequencies,
                    config,
                    stft,
                    allow_rolling=reused_frames == 29,
                )
                covariance_rolled = self._noise_cache.snapshot().rolled
            except (Layer3Error, RuntimeError) as exc:
                preparation_error = str(exc)

        prepared = PreparedL3Context(
            window.session_id,
            window.stream_epoch,
            window.window_id,
            window.decision_sample,
            window.context_start_sample,
            window.context_end_sample,
            window.sample_rate,
            mode,
            stft,
            config,
            spectrum_fct,
            frequencies,
            static.speech_band_f,
            noise,
            noise_version,
            preparation_error,
            reused_frames,
            covariance_rolled,
        )
        self._prepared_cache[key] = prepared
        self._prepared_cache.move_to_end(key)
        while len(self._prepared_cache) > self._prepared_cache_limit:
            self._prepared_cache.popitem(last=False)
        return prepared

    def process_prepared_batch(
        self,
        prepared: PreparedL3Context,
        candidates: tuple[CandidateDirection, ...],
        geometry: MicGeometry,
    ) -> BeamformedL3Batch:
        """Run only candidate-dependent steering, BF weights and application."""
        self._validate_prepared_candidates(prepared, candidates)
        if not candidates:
            self.last_diagnostics = ()
            empty = torch.empty(
                (0, 513, 33), dtype=torch.complex64, device=prepared.spectrum_fct.device,
            )
            return BeamformedL3Batch(empty, (), (), (), ())
        prepared_device = prepared.spectrum_fct.device
        if (
            prepared_device.type != self.device.type
            or (
                self.device.index is not None
                and prepared_device.index != self.device.index
            )
        ):
            raise Layer3Error("PreparedL3Context与beamformer设备不一致")

        steering = self._steering_vectors(prepared.frequencies_hz, candidates, geometry)
        fallback_reason: str | None = None
        if prepared.mode == L3_MODE_DS_BASELINE:
            output = apply_weights(das_weights(steering), prepared.spectrum_fct)
            diagnostics = tuple(
                (
                    "backend=ds_baseline",
                    "comparison_only=true",
                    "physical_channels=7",
                    "imcra=unused",
                    "spatial_p=unused",
                )
                for _item in candidates
            )
            backends = tuple("ds_baseline" for _item in candidates)
            fallback_reasons = tuple(None for _item in candidates)
        elif prepared.mode == L3_MODE_CONSTANT_BEAMWIDTH:
            weights, fallback_bins, minimum_wng = self._constant_beamwidth_weights(
                prepared.frequencies_hz, candidates, geometry, prepared.config,
            )
            output = apply_weights(weights, prepared.spectrum_fct)
            diagnostics = tuple(
                (
                    "backend=constant_beamwidth_baseline",
                    "comparison_only=true",
                    "geometry=uca_6plus1_numerical_adaptation",
                    f"target_fnbw_deg={prepared.config.constant_beamwidth_fnbw_deg:.1f}",
                    f"design_grid_deg={prepared.config.constant_beamwidth_design_grid_deg:.1f}",
                    f"min_wng_limit_db={prepared.config.constant_beamwidth_min_wng_db:.1f}",
                    f"minimum_designed_wng_db={minimum_wng[index]:.2f}",
                    f"das_fallback_bins={fallback_bins[index]}",
                    "imcra=unused",
                    "spatial_p=unused",
                )
                for index in range(len(candidates))
            )
            backends = tuple("constant_beamwidth_baseline" for _item in candidates)
            fallback_reasons = tuple(
                None if count == 0 else f"WNG-protected DAS fallback bins={count}"
                for count in fallback_bins
            )
        elif prepared.noise_statistics is None:
            output = apply_weights(das_weights(steering), prepared.spectrum_fct)
            fallback_reason = f"IMCRA adaptive BF unavailable: {prepared.preparation_error}"
            speech_bins = int(prepared.passband_f.sum().item())
            diagnostics = tuple(
                ("backend=das_fallback", fallback_reason, f"fallback_bins={speech_bins}")
                for _item in candidates
            )
            backends = tuple("das" for _item in candidates)
            fallback_reasons = tuple(fallback_reason for _item in candidates)
        else:
            try:
                spatial_p = None
                if len(candidates) == 2:
                    validate_p_table_context(
                        sample_rate=prepared.sample_rate,
                        n_fft=prepared.stft.n_fft,
                        frequency_min_hz=prepared.config.frequency_min_hz,
                        frequency_max_hz=prepared.config.frequency_max_hz,
                        geometry=geometry,
                    )
                    spatial_p = self._spatial_p(candidates, prepared.frequencies_hz)
                noise = prepared.noise_statistics
                solved = adaptive_separation_weights(
                    noise.covariance_fcc,
                    steering,
                    prepared.frequencies_hz,
                    noise.noise_confidence_f,
                    prepared.config,
                    spatial_p_f=spatial_p,
                    static=self._adaptive_static_data(prepared.frequencies_hz, prepared.config),
                )
                output = (
                    apply_weights(solved.weights_mfc, prepared.spectrum_fct)
                    * noise.frequency_gain_f[None, :, None]
                )
                diagnostics = tuple(
                    (
                        "backend=imcra_spatial_separation",
                        f"imcra={prepared.noise_algorithm_version}:16x20ms",
                        f"spatial_p={('independent_loaded_mvdr' if len(candidates) == 3 else 'single_candidate') if spatial_p is None else P_TABLE_VERSION}",
                        f"rho_thresholds={prepared.config.rho_lcmv_max:.3f}/"
                        f"{prepared.config.rho_soft_null_max:.3f}",
                        f"rho_range={float(solved.rho_f.min().item()):.4f}.."
                        f"{float(solved.rho_f.max().item()):.4f}",
                        f"cache:stft_reused={prepared.stft_reused_frames},"
                        f"covariance_rolled={prepared.covariance_rolled}",
                        f"bins:lcmv={solved.lcmv_bins},soft_null_mvdr={solved.soft_null_bins},"
                        f"loaded_mvdr={solved.loaded_mvdr_bins},das_fallback={solved.fallback_bins[index]}",
                        f"frequency_gain_min={float(noise.frequency_gain_f.min().item()):.4f}",
                    )
                    for index in range(len(candidates))
                )
                if any(solved.fallback_bins):
                    fallback_reason = f"per-bin DAS fallback counts={solved.fallback_bins}"
                backends = tuple("imcra_spatial_separation" for _item in candidates)
                fallback_reasons = tuple(fallback_reason for _item in candidates)
            except (Layer3Error, RuntimeError) as exc:
                output = apply_weights(das_weights(steering), prepared.spectrum_fct)
                fallback_reason = f"IMCRA adaptive BF unavailable: {exc}"
                speech_bins = int(prepared.passband_f.sum().item())
                diagnostics = tuple(
                    ("backend=das_fallback", fallback_reason, f"fallback_bins={speech_bins}")
                    for _item in candidates
                )
                backends = tuple("das" for _item in candidates)
                fallback_reasons = tuple(fallback_reason for _item in candidates)

        if output.shape != (len(candidates), 513, 33) or not torch.isfinite(output).all():
            raise Layer3Error("方向分离频谱输出无效")
        self.last_diagnostics = diagnostics
        return BeamformedL3Batch(
            output.detach(),
            tuple(float(item.theta_deg) for item in candidates),
            backends,
            fallback_reasons,
            diagnostics,
        )

    def process_batch(
        self,
        window: DecisionWindow,
        candidates: tuple[CandidateDirection, ...],
        geometry: MicGeometry,
        config: SpatialSeparationConfig,
        stft: StftSettings,
    ) -> tuple[DirectionalSignal, ...]:
        self._validate_candidates(window, candidates)
        if not candidates:
            self.last_diagnostics = ()
            return ()
        spectrum_cft = self._stft_cache.process(window, stft)
        return self.process_batch_from_spectrum(
            window, candidates, geometry, config, stft, spectrum_cft,
            stft_reused_frames=self._stft_cache.snapshot().reused_frames,
        )

    def process_ds_baseline_batch(
        self,
        window: DecisionWindow,
        candidates: tuple[CandidateDirection, ...],
        geometry: MicGeometry,
        stft: StftSettings,
    ) -> tuple[DirectionalSignal, ...]:
        """Pure seven-microphone delay-and-sum baseline for Test UI comparison."""
        self._validate_candidates(window, candidates)
        if not candidates:
            self.last_diagnostics = ()
            return ()
        spectrum_cft = self._stft_cache.process(window, stft)
        spectrum_fct = spectrum_cft.permute(1, 0, 2).contiguous()
        frequencies = self._frequency_axis(window.sample_rate, stft)
        steering = self._steering_vectors(frequencies, candidates, geometry)
        output = apply_weights(das_weights(steering), spectrum_fct)
        if output.shape != (len(candidates), 513, 33) or not torch.isfinite(output).all():
            raise Layer3Error("DS baseline方向频谱输出无效")
        diagnostics = tuple(
            (
                "backend=ds_baseline",
                "comparison_only=true",
                "physical_channels=7",
                "imcra=unused",
                "spatial_p=unused",
            )
            for _item in candidates
        )
        signals = tuple(
            DirectionalSignal(
                window.session_id,
                window.stream_epoch,
                window.window_id,
                window.decision_sample,
                window.context_start_sample,
                window.context_end_sample,
                candidate.theta_deg,
                window.sample_rate,
                "ds_baseline",
                None,
                np.ascontiguousarray(output[index].detach().cpu().numpy(), dtype=np.complex64),
            )
            for index, candidate in enumerate(candidates)
        )
        self.last_diagnostics = diagnostics
        return signals

    def process_batch_from_spectrum(
        self,
        window: DecisionWindow,
        candidates: tuple[CandidateDirection, ...],
        geometry: MicGeometry,
        config: SpatialSeparationConfig,
        stft: StftSettings,
        spectrum_cft: torch.Tensor,
        *,
        stft_reused_frames: int = 0,
    ) -> tuple[DirectionalSignal, ...]:
        self._validate_candidates(window, candidates)
        if not candidates:
            self.last_diagnostics = ()
            return ()
        if spectrum_cft.shape != (7, 513, 33) or spectrum_cft.dtype != torch.complex64:
            raise Layer3Error("precomputed STFT must be complex64 [7,513,33]")

        spectrum_fct = spectrum_cft.permute(1, 0, 2).contiguous()
        frequencies = self._frequency_axis(window.sample_rate, stft)
        steering = self._steering_vectors(frequencies, candidates, geometry)
        fallback_reason: str | None = None

        try:
            spatial_p = None
            if len(candidates) == 2:
                validate_p_table_context(
                    sample_rate=window.sample_rate,
                    n_fft=stft.n_fft,
                    frequency_min_hz=config.frequency_min_hz,
                    frequency_max_hz=config.frequency_max_hz,
                    geometry=geometry,
                )
                spatial_p = self._spatial_p(candidates, frequencies)
            noise_context = BeamformerNoiseContext.from_window(window)
            noise = self._noise_cache.estimate(
                noise_context,
                spectrum_fct,
                frequencies,
                config,
                stft,
                allow_rolling=stft_reused_frames == 29,
            )
            solved = adaptive_separation_weights(
                noise.covariance_fcc, steering, frequencies, noise.noise_confidence_f, config,
                spatial_p_f=spatial_p,
                static=self._adaptive_static_data(frequencies, config),
            )
            weights = solved.weights_mfc
            output = apply_weights(weights, spectrum_fct) * noise.frequency_gain_f[None, :, None]
            diagnostics = tuple(
                (
                    "backend=imcra_spatial_separation",
                    f"imcra={noise_context.algorithm_version}:16x20ms",
                    f"spatial_p={('independent_loaded_mvdr' if len(candidates) == 3 else 'single_candidate') if spatial_p is None else P_TABLE_VERSION}",
                    f"rho_thresholds={config.rho_lcmv_max:.3f}/{config.rho_soft_null_max:.3f}",
                    f"rho_range={float(solved.rho_f.min().item()):.4f}..{float(solved.rho_f.max().item()):.4f}",
                    f"cache:stft_reused={stft_reused_frames},"
                    f"covariance_rolled={self._noise_cache.snapshot().rolled}",
                    f"bins:lcmv={solved.lcmv_bins},soft_null_mvdr={solved.soft_null_bins},"
                    f"loaded_mvdr={solved.loaded_mvdr_bins},das_fallback={solved.fallback_bins[index]}",
                    f"frequency_gain_min={float(noise.frequency_gain_f.min().item()):.4f}",
                )
                for index in range(len(candidates))
            )
            fallback_counts = solved.fallback_bins
            if any(fallback_counts):
                fallback_reason = f"per-bin DAS fallback counts={fallback_counts}"
        except (Layer3Error, RuntimeError) as exc:
            weights = das_weights(steering)
            output = apply_weights(weights, spectrum_fct)
            fallback_reason = f"IMCRA adaptive BF unavailable: {exc}"
            speech_bins = int(
                ((frequencies >= config.frequency_min_hz) & (frequencies <= config.frequency_max_hz)).sum().item()
            )
            diagnostics = tuple(
                ("backend=das_fallback", fallback_reason, f"fallback_bins={speech_bins}")
                for _item in candidates
            )

        if output.shape != (len(candidates), 513, 33) or not torch.isfinite(output).all():
            raise Layer3Error("方向分离频谱输出无效")
        signals = tuple(
            DirectionalSignal(
                window.session_id,
                window.stream_epoch,
                window.window_id,
                window.decision_sample,
                window.context_start_sample,
                window.context_end_sample,
                candidate.theta_deg,
                window.sample_rate,
                "das" if diagnostics[index][0] == "backend=das_fallback" else "imcra_spatial_separation",
                fallback_reason,
                np.ascontiguousarray(output[index].detach().cpu().numpy(), dtype=np.complex64),
            )
            for index, candidate in enumerate(candidates)
        )
        self.last_diagnostics = diagnostics
        return signals

    @staticmethod
    def _validate_candidates(
        window: DecisionWindow, candidates: tuple[CandidateDirection, ...],
    ) -> None:
        if len(candidates) > 3:
            raise Layer3Error("L3只接受0、1、2或3个候选方向")
        identity = (window.session_id, window.stream_epoch, window.window_id, window.decision_sample)
        if any(
            (item.session_id, item.stream_epoch, item.window_id, item.decision_sample) != identity
            for item in candidates
        ):
            raise Layer3Error("候选与DecisionWindow不属于同一窗口")
        if len({item.theta_deg for item in candidates}) != len(candidates):
            raise Layer3Error("L3候选方向不能重复")

    @staticmethod
    def _validate_window(window: DecisionWindow) -> None:
        if window.sample_rate != 48_000 or window.samples.shape != (15_360, 8):
            raise Layer3Error("L3输入必须是48 kHz逻辑8通道 [15360,8]")

    @staticmethod
    def _validate_prepared_candidates(
        prepared: PreparedL3Context,
        candidates: tuple[CandidateDirection, ...],
    ) -> None:
        if len(candidates) > 3:
            raise Layer3Error("L3只接受0、1、2或3个候选方向")
        if any(
            (
                item.session_id,
                item.stream_epoch,
                item.window_id,
                item.decision_sample,
            )
            != prepared.window_key
            for item in candidates
        ):
            raise Layer3Error("候选与PreparedL3Context不属于同一窗口")
        if len({item.theta_deg for item in candidates}) != len(candidates):
            raise Layer3Error("L3候选方向不能重复")
