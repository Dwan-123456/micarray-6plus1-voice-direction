"""GI-DOAEnet adapter for the common L2 direction-scanner contract.

The upstream model is loaded from a local, Git-ignored installation.  This
module contains only project-owned integration code; use the installer under
``scripts/`` to acquire the pinned upstream revision and checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.signal import find_peaks, resample_poly

from common.angle import circular_distance_deg
from common.data_types import CandidateDirection, DecisionWindow, ModelOrderEstimate, SpatialResponse
from common.geometry import MicGeometry

from .configuration import DirectionScanConfig
from .interface import DirectionScanError
from .music import MusicDiagnostics, MusicPeakEvidence, MusicStateDiagnostic


UPSTREAM_REVISION = "af865978c783f309fc929f0f2499769a1c5499d5"
CHECKPOINT_SHA256 = "d465d2ccf0b7f2d1603186db3667e8c7b7a21c7eb0b8a126173b3292441f9fe8"


@dataclass(frozen=True, slots=True)
class GiDoaEnetPaths:
    source_root: Path
    checkpoint: Path

    @classmethod
    def discover(cls, project_root: str | Path | None = None) -> "GiDoaEnetPaths":
        override = os.environ.get("GI_DOAENET_ROOT")
        base = Path(override) if override else Path(project_root or Path.cwd()) / "models" / "gi_doaenet_pm_v1"
        base = base.resolve()
        candidates = tuple(base.glob("upstream/*/model/main.py")) + tuple(base.glob("upstream/model/main.py"))
        if len(candidates) != 1:
            raise DirectionScanError(
                "GI-DOAEnet未安装；请运行 scripts/install_gi_doaenet.py --acknowledge-upstream-terms"
            )
        checkpoints = tuple(base.glob("upstream/*/pretrained/GI_DOAEnet_PM.tar")) + tuple(
            base.glob("upstream/pretrained/GI_DOAEnet_PM.tar")
        )
        if len(checkpoints) != 1:
            raise DirectionScanError("GI-DOAEnet PM权重缺失")
        return cls(candidates[0].parents[1], checkpoints[0])


class GiDoaEnetScanner:
    """Lazy CUDA/CPU inference adapter producing the existing 360-point DTO."""

    algorithm_version = f"gi_doaenet_pm_{UPSTREAM_REVISION[:12]}_adapter_v1"

    def __init__(self, *, project_root: str | Path | None = None, device: str = "auto") -> None:
        self._project_root = None if project_root is None else Path(project_root)
        self._requested_device = device
        self._model = None
        self._torch = None
        self._device = "unloaded"
        self._stream_key: tuple[str, int] | None = None
        self._last_sample: int | None = None
        self.last_state_diagnostic: MusicStateDiagnostic | None = None
        self._last_model_order: ModelOrderEstimate | None = None

    def reset(self) -> None:
        self._stream_key = None
        self._last_sample = None
        self.last_state_diagnostic = None
        self._last_model_order = None

    def _load(self) -> None:
        if self._model is not None:
            return
        paths = GiDoaEnetPaths.discover(self._project_root)
        digest = hashlib.sha256(paths.checkpoint.read_bytes()).hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise DirectionScanError("GI-DOAEnet权重SHA-256校验失败")
        try:
            import torch
        except ImportError as exc:
            raise DirectionScanError("GI-DOAEnet需要PyTorch") from exc
        package_name = "_micarray_gi_doaenet_upstream"
        package = sys.modules.get(package_name)
        if package is None:
            package_spec = importlib.util.spec_from_file_location(
                package_name, paths.source_root / "model" / "__init__.py",
                submodule_search_locations=[str(paths.source_root / "model")],
            )
            if package_spec is None or package_spec.loader is None:
                raise DirectionScanError("无法加载GI-DOAEnet外部模型包")
            package = importlib.util.module_from_spec(package_spec)
            sys.modules[package_name] = package
            package_spec.loader.exec_module(package)
        main_name = f"{package_name}.main"
        main_spec = importlib.util.spec_from_file_location(main_name, paths.source_root / "model" / "main.py")
        if main_spec is None or main_spec.loader is None:
            raise DirectionScanError("无法加载GI-DOAEnet模型入口")
        main = importlib.util.module_from_spec(main_spec)
        sys.modules[main_name] = main
        main_spec.loader.exec_module(main)
        model = main.GI_DOAEnet(MPE_type="PM")
        state = torch.load(paths.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        device = self._requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise DirectionScanError("GI-DOAEnet配置为CUDA，但CUDA当前不可用")
        self._model = model.eval().to(device)
        self._torch = torch
        self._device = device

    @staticmethod
    def _choose_peaks(scores: np.ndarray, config: DirectionScanConfig) -> tuple[list[int], int]:
        tiled = np.tile(scores, 3)
        peaks, properties = find_peaks(tiled, prominence=config.peak_prominence)
        middle = [(int(p - 360), float(prom)) for p, prom in zip(peaks, properties["prominences"], strict=True)
                  if 360 <= p < 720 and scores[int(p - 360)] >= config.direction_threshold]
        ordered = sorted(middle, key=lambda item: (-float(scores[item[0]]), item[0]))
        selected: list[int] = []
        for index, _ in ordered:
            if all(circular_distance_deg(float(index), float(other)) >= config.min_peak_distance_deg
                   for other in selected):
                selected.append(index)
                if len(selected) >= config.effective_order_limit:
                    break
        return selected, len(ordered)

    def scan_detailed(self, window: DecisionWindow, geometry: MicGeometry,
                      config: DirectionScanConfig, config_revision: int = 0):
        started = perf_counter()
        self._load()
        assert self._torch is not None and self._model is not None
        stream_key = (window.session_id, window.stream_epoch)
        previous = self._last_sample if self._stream_key == stream_key else None
        contiguous = previous is not None and window.decision_sample - previous == 960
        audio48 = np.asarray(window.physical_samples, dtype=np.float32)
        audio16 = np.ascontiguousarray(resample_poly(audio48, 1, 3, axis=0).astype(np.float32, copy=False))
        coordinates = np.column_stack((geometry.positions_m, np.zeros(7, dtype=np.float64))).astype(np.float32)
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(audio16.T[None]).to(self._device)
            mic = torch.from_numpy(coordinates[None]).to(self._device)
            output = self._model(x, mic)
            spectrum = output[:, -1, :, -min(5, output.shape[-1]):].mean(dim=-1)[0]
            scores = spectrum.detach().float().cpu().numpy()
        scores = np.ascontiguousarray(np.clip(scores, 0.0, 1.0).astype(np.float32))
        if scores.shape != (360,) or not np.isfinite(scores).all():
            raise DirectionScanError("GI-DOAEnet输出不是finite [360]方向概率")
        chosen, eligible = self._choose_peaks(scores, config)
        self._last_model_order = ModelOrderEstimate(
            len(chosen), 360, int(output.shape[-1]), 1.0 if chosen else 0.0, 0, "ready"
        )
        state = MusicStateDiagnostic(
            "incremental" if contiguous else "rebuilt", previous, window.decision_sample,
            0 if previous is None else window.decision_sample - previous, 0,
            int(output.shape[-1]), 0, False,
            "contiguous_20ms_context" if contiguous else "stream_start_or_gap",
        )
        self.last_state_diagnostic = state
        self._stream_key, self._last_sample = stream_key, window.decision_sample
        response = SpatialResponse(
            window.session_id, window.stream_epoch, window.window_id, window.decision_sample,
            window.doa_start_sample, window.doa_end_sample, np.arange(360, dtype=np.float32),
            scores, scores, self._last_model_order, 360, "ready", self.algorithm_version,
        )
        candidates = tuple(CandidateDirection(
            window.session_id, window.stream_epoch, window.window_id, window.decision_sample,
            window.doa_start_sample, window.doa_end_sample, float(index), float(scores[index]), float(scores[index]),
        ) for index in chosen)
        evidence = tuple(MusicPeakEvidence(c.theta_deg, 0, c.raw_score, c.normalized_score, 7, 360)
                         for c in candidates)
        diagnostics = MusicDiagnostics(
            "gi_doaenet", self.algorithm_version, config_revision, self._last_model_order, state, 360, "ready",
            stop_reason="neural_probability_peak_search", evidence=evidence,
            eligible_peak_count=eligible, candidate_limit=config.effective_order_limit,
            candidate_limit_applied=eligible > len(chosen), effective_model_order=len(chosen),
            births_allowed=bool(chosen), spectrum_ms=(perf_counter() - started) * 1000.0,
            total_ms=(perf_counter() - started) * 1000.0,
        )
        return response, candidates, diagnostics

    def scan(self, window: DecisionWindow, geometry: MicGeometry, config: DirectionScanConfig):
        response, candidates, _ = self.scan_detailed(window, geometry, config)
        return response, candidates

    @property
    def model_order(self) -> ModelOrderEstimate | None:
        return self._last_model_order


class SwitchableDoaScanner:
    """Route each complete window atomically to MUSIC or GI-DOAEnet."""

    def __init__(self, music, neural: GiDoaEnetScanner) -> None:
        self.music = music
        self.neural = neural
        self._active = "frequency_normalized_music"

    def reset(self) -> None:
        self.music.reset()
        self.neural.reset()

    def _scanner(self, config: DirectionScanConfig):
        self._active = config.scanner_backend
        return self.neural if self._active == "gi_doaenet" else self.music

    def scan_detailed(self, window, geometry, config, config_revision=0):
        return self._scanner(config).scan_detailed(window, geometry, config, config_revision)

    def observe_covariance(self, window, config):
        """Keep MUSIC spatial covariance warm while its probability Gate is closed."""

        if config.scanner_backend != "frequency_normalized_music":
            return None
        self._active = config.scanner_backend
        return self.music.observe_covariance(window, config)

    @property
    def model_order(self):
        return self.neural.model_order if self._active == "gi_doaenet" else self.music.model_order

    @property
    def last_state_diagnostic(self):
        scanner = self.neural if self._active == "gi_doaenet" else self.music
        return scanner.last_state_diagnostic
