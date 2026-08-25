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
    def __init__(self, artifact: str | Path) -> None:
        self.artifact = Path(artifact)
        self.manifest, weights = _verified_file(self.artifact, "speaker_embedding_artifact_v1")
        self.model = CAMPPlus(80, int(self.manifest["embedding_size"]))
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

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
        return self.embed_batch((waveform_16k,))[0]

    def embed_batch(self, waveforms_16k: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        """Embed variable-length speech in same-shape inference batches."""

        waveforms = tuple(
            np.ascontiguousarray(waveform, dtype=np.float32)
            for waveform in waveforms_16k
        )
        if not waveforms:
            return ()
        if any(len(waveform) < 8_000 for waveform in waveforms):
            raise ValueError("CAMPPlus requires at least 500 ms of speech")
        features = tuple(self._features(waveform) for waveform in waveforms)
        groups: dict[tuple[int, int], list[int]] = {}
        for index, value in enumerate(features):
            groups.setdefault(value.shape, []).append(index)
        outputs: list[np.ndarray | None] = [None] * len(features)
        with torch.inference_mode():
            for indices in groups.values():
                batch = torch.from_numpy(np.stack([features[index] for index in indices]))
                embeddings = self.model(batch).float().numpy()
                embeddings /= np.maximum(
                    np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12,
                )
                for index, embedding in zip(indices, embeddings, strict=True):
                    outputs[index] = np.ascontiguousarray(embedding, dtype=np.float32)
        if any(output is None for output in outputs):
            raise RuntimeError("CAMPPlus batch did not produce every requested embedding")
        return tuple(np.asarray(output, dtype=np.float32) for output in outputs)


class DnsMosScorer:
    def __init__(self, artifact: str | Path) -> None:
        import onnxruntime as ort

        self.artifact = Path(artifact)
        self.manifest, weights = _verified_file(self.artifact, "audio_quality_artifact_v1")
        options = ort.SessionOptions()
        # DNSMOS is sidecar work that must not let ORT's default thread pool
        # contend with the CPU-resident L1-L3 realtime graph.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(weights),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
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
