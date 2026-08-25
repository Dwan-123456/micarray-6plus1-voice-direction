from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .campplus import CAMPPlus


def _verified_file(artifact: Path, expected_schema: str) -> tuple[dict[str, object], Path]:
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != expected_schema:
        raise ValueError(f"unexpected model artifact schema in {artifact}")
    path = artifact / str(manifest["weights_file"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError(f"model hash mismatch in {artifact}")
    return manifest, path


class CampPlusEmbedder:
    def __init__(self, artifact: str | Path, *, device: str = "cpu", batch_size: int = 64) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("CAMPPlus device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CAMPPlus CUDA was requested but is unavailable")
        if batch_size <= 0:
            raise ValueError("CAMPPlus batch size must be positive")
        self.artifact = Path(artifact)
        self.device = device
        self.batch_size = int(batch_size)
        self.manifest, weights = _verified_file(self.artifact, "speaker_embedding_artifact_v1")
        self.model = CAMPPlus(80, int(self.manifest["embedding_size"]))
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @staticmethod
    def _features(waveform: np.ndarray) -> np.ndarray:
        import kaldi_native_fbank as knf

        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16_000
        opts.frame_opts.frame_length_ms = 25.0
        opts.frame_opts.frame_shift_ms = 10.0
        opts.frame_opts.dither = 0.0
        opts.frame_opts.snip_edges = True
        opts.frame_opts.window_type = "hamming"
        opts.mel_opts.num_bins = 80
        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(16_000, (np.asarray(waveform, np.float32) * 32768.0).tolist())
        fbank.input_finished()
        frames = np.stack([fbank.get_frame(index) for index in range(fbank.num_frames_ready)])
        frames = np.asarray(frames, dtype=np.float32)
        return np.ascontiguousarray(frames - frames.mean(axis=0, keepdims=True), dtype=np.float32)

    def embed(self, waveform_16k: np.ndarray) -> np.ndarray:
        return self.embed_many((waveform_16k,))[0]

    def embed_many(self, waveforms_16k: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        waveforms = tuple(np.ascontiguousarray(item, dtype=np.float32) for item in waveforms_16k)
        if not waveforms:
            return ()
        if any(len(item) < 8_000 for item in waveforms):
            raise ValueError("CAMPPlus requires at least 500 ms of speech")
        features = tuple(self._features(item) for item in waveforms)
        if len({item.shape for item in features}) != 1:
            raise ValueError("batched CAMPPlus inputs must have one aligned duration")
        outputs = []
        with torch.inference_mode():
            for first in range(0, len(features), self.batch_size):
                batch = torch.from_numpy(np.stack(features[first:first + self.batch_size])).to(self.device)
                outputs.append(self.model(batch).float().cpu().numpy())
        embeddings = np.concatenate(outputs, axis=0)
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        return tuple(np.ascontiguousarray(item, dtype=np.float32) for item in embeddings)


class DnsMosScorer:
    def __init__(self, artifact: str | Path) -> None:
        import onnxruntime as ort

        self.artifact = Path(artifact)
        self.manifest, weights = _verified_file(self.artifact, "audio_quality_artifact_v1")
        self.session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def score(self, waveform_16k: np.ndarray) -> tuple[float, float, float]:
        audio = np.asarray(waveform_16k, dtype=np.float32)
        required = 9 * 16_000 + 160
        if not len(audio):
            return 1.0, 1.0, 1.0
        while len(audio) < required:
            audio = np.concatenate((audio, audio))
        rows = []
        for start in range(0, len(audio) - required + 1, 16_000):
            output = self.session.run(None, {self.input_name: audio[start:start + required][None, :]})[0]
            rows.append(np.asarray(output, dtype=np.float32).reshape(-1)[:3])
        raw_sig, raw_bak, raw_ovrl = np.mean(np.stack(rows), axis=0)
        sig = np.polyval([-0.08397278, 1.22083953, 0.00524390], raw_sig)
        bak = np.polyval([-0.13166888, 1.60915514, -0.39604546], raw_bak)
        ovrl = np.polyval([-0.06766283, 1.11546468, 0.04602535], raw_ovrl)
        return tuple(float(np.clip(value, 1.0, 5.0)) for value in (sig, bak, ovrl))
