from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np
from numpy.typing import NDArray
import torch
from safetensors.torch import load_file as load_safetensors

from .contracts import L4_MODEL_SAMPLE_RATE, Layer4CandidatePair, SpeakerCountDecision


def _load_manifest(artifact: Path, expected_kind: str) -> tuple[dict[str, object], Path]:
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_kind") != expected_kind:
        raise ValueError(f"expected {expected_kind} artifact")
    model_path = artifact / str(manifest["model_file"])
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual != manifest.get("model_sha256"):
        raise ValueError("Layer 4 model hash does not match its manifest")
    return manifest, model_path


class DirectionCountSpeakerClassifier:
    """Derive 1/2 speakers from the maximum recorded L2 direction count."""

    classifier_id = "l2_direction_count_max_v1"

    def classify(self, source) -> SpeakerCountDecision:
        maximum = max(count for _, count in source.l2_direction_counts)
        speaker_count = min(2, maximum)
        return SpeakerCountDecision(
            source.asset_id, speaker_count, 1.0, self.classifier_id,
            {
                "aggregation": "min(2, maximum)",
                "observation_count": len(source.l2_direction_counts),
                "maximum_l2_direction_count": maximum,
                "effective_speaker_count": speaker_count,
            },
        )


class TorchScriptSeparationBackend:
    """Strict common runtime for official MossFormer2/TIGER TorchScript exports."""

    sample_rate = L4_MODEL_SAMPLE_RATE
    source_count = 2

    def __init__(
        self,
        artifact: str | Path,
        *,
        backend: str,
        device: str = "cpu",
        model: Callable[[torch.Tensor], object] | None = None,
    ) -> None:
        expected = {
            "mossformer2_ss_16k": "l4_mossformer2_torchscript_v1",
            "tiger_speech_16k": "l4_tiger_torchscript_v1",
        }
        if backend not in expected:
            raise ValueError("unsupported Layer 4 separation backend")
        self.backend = backend
        self.artifact = Path(artifact)
        self.manifest, path = _load_manifest(self.artifact, expected[backend])
        if int(self.manifest.get("sample_rate", 0)) != 16_000 or int(self.manifest.get("source_count", 0)) != 2:
            raise ValueError("Layer 4 artifact must declare 16 kHz and two sources")
        self.model_id = str(self.manifest["model_id"])
        self.model_revision = str(self.manifest["model_revision"])
        self.device = torch.device(device)
        self.model = model or torch.jit.load(str(path), map_location=self.device).eval()

    @staticmethod
    def _normalize(output: object, expected: int) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(output, (tuple, list)) and len(output) == 2:
            value = torch.stack(tuple(torch.as_tensor(item).squeeze() for item in output))
        else:
            value = torch.as_tensor(output).squeeze()
        if value.ndim != 2 or value.shape[0] != 2:
            raise RuntimeError("separation model must return [2,time] or [1,2,time]")
        value = value.float().detach().cpu().numpy()
        if value.shape[1] < expected:
            value = np.pad(value, ((0, 0), (0, expected - value.shape[1])))
        value = value[:, :expected]
        if not np.isfinite(value).all():
            raise RuntimeError("separation model returned non-finite audio")
        return tuple(np.ascontiguousarray(item, dtype=np.float32) for item in value)  # type: ignore[return-value]

    def separate(self, request_id: str, waveform_16k: NDArray[np.float32]) -> Layer4CandidatePair:
        audio = np.asarray(waveform_16k)
        if audio.ndim != 1 or audio.dtype != np.float32 or not audio.flags.c_contiguous:
            raise ValueError("separation input must be C-contiguous float32 mono audio")
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("separation input must be non-empty and finite")
        with torch.inference_mode():
            output = self.model(torch.from_numpy(audio.copy())[None].to(self.device))
        sources = self._normalize(output, len(audio))
        return Layer4CandidatePair(
            request_id, self.model_id, self.model_revision, 16_000, sources,
        )


class _OfficialModelBackend(TorchScriptSeparationBackend):
    """Long-audio adapter with overlap-add and anonymous-source permutation repair."""

    def __init__(self, *, backend: str, manifest: dict[str, object], artifact: Path, device: str) -> None:
        self.backend = backend
        self.manifest = manifest
        self.artifact = artifact
        self.model_id = str(manifest["model_id"])
        self.model_revision = str(manifest["model_revision"])
        self.device = torch.device(device)
        self.chunk_samples = int(manifest.get("chunk_seconds", 30)) * 16_000
        self.overlap_samples = int(manifest.get("overlap_seconds", 1)) * 16_000
        if self.chunk_samples <= self.overlap_samples or self.overlap_samples <= 0:
            raise ValueError("Layer 4 model chunk/overlap settings are invalid")

    def _forward(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            output = self.model(torch.from_numpy(np.ascontiguousarray(audio))[None].to(self.device))
        return self._normalize(output, len(audio))

    @staticmethod
    def _similarity(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return 0.0 if denominator <= np.finfo(np.float32).eps else float(np.dot(left, right) / denominator)

    def _chunked(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(audio) <= self.chunk_samples:
            return self._forward(audio)
        outputs = np.zeros((2, len(audio)), dtype=np.float64)
        weights = np.zeros(len(audio), dtype=np.float64)
        step = self.chunk_samples - self.overlap_samples
        for chunk_index, start in enumerate(range(0, len(audio), step)):
            end = min(len(audio), start + self.chunk_samples)
            pair = list(self._forward(audio[start:end]))
            overlap = min(self.overlap_samples, end - start)
            if chunk_index and overlap and np.any(weights[start:start + overlap] > 0.0):
                previous = outputs[:, start:start + overlap] / np.maximum(
                    weights[start:start + overlap], np.finfo(np.float64).eps,
                )
                straight = self._similarity(previous[0], pair[0][:overlap]) + self._similarity(
                    previous[1], pair[1][:overlap]
                )
                swapped = self._similarity(previous[0], pair[1][:overlap]) + self._similarity(
                    previous[1], pair[0][:overlap]
                )
                if swapped > straight:
                    pair.reverse()
            weight = np.ones(end - start, dtype=np.float64)
            if chunk_index:
                weight[:overlap] = np.linspace(0.0, 1.0, overlap, endpoint=False)
            if end < len(audio):
                weight[-overlap:] = np.minimum(
                    weight[-overlap:], np.linspace(1.0, 0.0, overlap, endpoint=False),
                )
            outputs[:, start:end] += np.stack(pair).astype(np.float64) * weight
            weights[start:end] += weight
            if end == len(audio):
                break
        normalized = outputs / np.maximum(weights, np.finfo(np.float64).eps)
        return tuple(np.ascontiguousarray(item, dtype=np.float32) for item in normalized)  # type: ignore[return-value]

    def separate(self, request_id: str, waveform_16k: NDArray[np.float32]) -> Layer4CandidatePair:
        audio = np.asarray(waveform_16k)
        if (
            audio.ndim != 1 or audio.dtype != np.float32 or not audio.flags.c_contiguous
            or not len(audio) or not np.isfinite(audio).all()
        ):
            raise ValueError("separation input must be finite C-contiguous float32 mono audio")
        return Layer4CandidatePair(
            request_id, self.model_id, self.model_revision, 16_000, self._chunked(audio),
        )


class MossFormer2Backend(_OfficialModelBackend):
    def __init__(self, artifact: str | Path, *, device: str = "cpu") -> None:
        root = Path(artifact)
        manifest, weights_path = _load_manifest(root, "l4_mossformer2_official_v1")
        source = root / "source"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        module = importlib.import_module("clearvoice.models.mossformer2_ss.mossformer2")
        model = module.MossFormer(
            in_channels=512, out_channels=512, num_blocks=24, kernel_size=16,
            norm="ln", num_spks=2, skip_around_intra=True,
            use_global_pos_enc=True, max_length=20_000,
        )
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        super().__init__(
            backend="mossformer2_ss_16k", manifest=manifest, artifact=root, device=device,
        )
        self.model = model.to(self.device).eval()


class TigerBackend(_OfficialModelBackend):
    def __init__(self, artifact: str | Path, *, device: str = "cpu") -> None:
        root = Path(artifact)
        manifest, weights_path = _load_manifest(root, "l4_tiger_official_v1")
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        source = root / "source"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        module = importlib.import_module("look2hear.models.tiger")
        model = module.TIGER(**config)
        model.load_state_dict(load_safetensors(str(weights_path)), strict=True)
        super().__init__(
            backend="tiger_speech_16k", manifest=manifest, artifact=root, device=device,
        )
        self.model = model.to(self.device).eval()
