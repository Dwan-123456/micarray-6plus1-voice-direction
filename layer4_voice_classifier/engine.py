from __future__ import annotations

import json
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np
import torch
from scipy.signal import resample_poly

from common.config import DownstreamAudioWindowSpec

from .contracts import Layer4AudioSegment, Layer4Result, ModelPrediction, VoiceDetection
from .gain_compensation import InputGainCompensationSettings, compensate_l4_input
from .marblenet import NvidiaFrameVadMarbleNet


class VoiceModelPlugin(Protocol):
    model_id: str

    def predict(self, waveforms_48k: np.ndarray) -> ModelPrediction: ...


def max_contiguous_frame_mean(
    frame_probabilities: torch.Tensor,
    lengths: torch.Tensor,
    *,
    window_frames: int = 3,
    recent_frames: int | None = None,
) -> torch.Tensor:
    """Return the strongest contiguous probability region for each item.

    MarbleNet emits one value about every 20 ms. A three-frame window requires
    a sustained peak of about 60 ms while preventing silence elsewhere in the
    configured 80/160 ms input from diluting the decision.
    """
    if frame_probabilities.ndim != 2 or lengths.shape != (frame_probabilities.shape[0],):
        raise ValueError("frame probabilities must be [batch,time] with one length per item")
    if window_frames <= 0:
        raise ValueError("aggregation window must be positive")
    valid_frames = (
        torch.arange(frame_probabilities.shape[1], device=frame_probabilities.device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    short_means = (frame_probabilities * valid_frames).sum(dim=1) / lengths.clamp_min(1)
    if frame_probabilities.shape[1] < window_frames:
        return short_means
    rolling = frame_probabilities.unfold(1, window_frames, 1).mean(dim=-1)
    window_ends = torch.arange(
        window_frames, frame_probabilities.shape[1] + 1, device=frame_probabilities.device
    )
    valid_windows = window_ends.unsqueeze(0) <= lengths.unsqueeze(1)
    if recent_frames is not None:
        if recent_frames <= 0:
            raise ValueError("recent frame horizon must be positive")
        window_starts = window_ends - window_frames
        valid_windows &= window_starts.unsqueeze(0) >= (
            lengths - recent_frames
        ).clamp_min(0).unsqueeze(1)
    peaks = rolling.masked_fill(~valid_windows, float("-inf")).max(dim=1).values

    # Both supported production inputs have enough audio for three frames.
    # This fallback keeps the helper total for shorter test or
    # future streaming inputs without inventing padded probabilities.
    return torch.where(lengths >= window_frames, peaks, short_means)


class NvidiaMarbleNetPlugin:
    def __init__(
        self,
        model_id: str,
        artifact: str | Path,
        *,
        device: str = "cpu",
        window_spec: DownstreamAudioWindowSpec | None = None,
    ) -> None:
        self.model_id = model_id
        self.artifact = Path(artifact)
        self.manifest = json.loads((self.artifact / "manifest.json").read_text(encoding="utf-8"))
        weights_path = self.artifact / self.manifest["weights_file"]
        actual_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if actual_hash != self.manifest["weights_sha256"]:
            raise ValueError("MarbleNet weight hash does not match its manifest")
        self.device = torch.device(device)
        self.window_spec = window_spec or DownstreamAudioWindowSpec(160, 7_680, 8, 17, 2_560)
        self.model = NvidiaFrameVadMarbleNet.from_artifact(self.artifact, self.device)

    def predict(self, waveforms_48k: np.ndarray) -> ModelPrediction:
        waveforms = np.asarray(waveforms_48k)
        if (
            waveforms.ndim != 2
            or waveforms.shape[1] < 960
            or waveforms.shape[1] % 960
            or waveforms.dtype != np.float32
            or not waveforms.flags.c_contiguous
            or not np.isfinite(waveforms).all()
        ):
            raise ValueError(
                "L4 continuous input must be finite float32 [M,T] at 48 kHz "
                "with complete 20 ms hops"
            )
        started = perf_counter()
        if len(waveforms) == 0:
            probabilities = np.empty((0,), dtype=np.float32)
        else:
            audio_16k = resample_poly(waveforms, up=1, down=3, axis=1).astype(np.float32, copy=False)
            audio = torch.from_numpy(np.ascontiguousarray(audio_16k)).to(self.device)
            with torch.inference_mode():
                logits, lengths = self.model(audio)
                frame_probabilities = logits.softmax(dim=-1)[..., 1]
                probabilities = max_contiguous_frame_mean(
                    frame_probabilities, lengths, window_frames=3, recent_frames=4
                ).float().cpu().numpy()
        latency_ms = (perf_counter() - started) * 1_000.0
        return ModelPrediction(
            self.model_id,
            probabilities,
            latency_ms,
            {
                "architecture": self.manifest["architecture_id"],
                "source_model": self.manifest["source_model"],
                "input_adapter": (
                    "48k_continuous_to_16k_polyphase_v2"
                ),
                "resampled_samples": waveforms.shape[1] // 3,
                "input_samples_48k": waveforms.shape[1],
                "frame_shift_ms": 20,
                "aggregation": "latest_80ms_max_contiguous_3frame_mean_v2",
            },
        )


class Layer4Engine:
    """Runs one primary plugin plus optional shadow plugins on the same immutable audio batch."""

    def __init__(
        self,
        primary: VoiceModelPlugin,
        shadows: tuple[VoiceModelPlugin, ...] = (),
        *,
        threshold: float = 0.70,
        input_gain_compensation: InputGainCompensationSettings | None = None,
        window_spec: DownstreamAudioWindowSpec | None = None,
    ):
        plugins = (primary, *shadows)
        ids = tuple(item.model_id for item in plugins)
        if len(ids) != len(set(ids)):
            raise ValueError("L4 model ids must be unique")
        if not 0 <= threshold <= 1:
            raise ValueError("L4 threshold must be in [0,1]")
        self.primary = primary
        self.shadows = tuple(shadows)
        self.threshold = float(threshold)
        self.input_gain_compensation = input_gain_compensation or InputGainCompensationSettings()
        self.window_spec = window_spec or DownstreamAudioWindowSpec(160, 7_680, 8, 17, 2_560)

    def process(self, inputs: tuple[Layer4AudioSegment, ...]) -> Layer4Result:
        inputs = tuple(inputs)
        identities = {
            (item.session_id, item.stream_epoch, item.window_id, item.decision_sample)
            for item in inputs
        }
        if len(identities) > 1:
            raise ValueError("L4 audio inputs must belong to one window")
        angles = tuple(item.theta_deg for item in inputs)
        if len(angles) != len(set(angles)):
            raise ValueError("L4 audio input angles must be unique")
        track_ids = tuple(item.track_id for item in inputs)
        if any(item is not None for item in track_ids):
            if any(item is None for item in track_ids) or len(set(track_ids)) != len(track_ids):
                raise ValueError("L4 audio track IDs must be complete and unique")
        compensated = tuple(
            (item.waveform, item.gain_compensation_diagnostic)
            if item.gain_compensated
            else compensate_l4_input(
                item.waveform,
                item.array_source_probabilities_20ms,
                self.input_gain_compensation,
                segment_count=len(item.waveform) // 960,
            )
            for item in inputs
        )
        waveforms = tuple(item[0] for item in compensated)
        grouped_batches = tuple(
            (
                tuple(index for index, item in enumerate(waveforms) if len(item) == length),
                np.ascontiguousarray(np.stack([item for item in waveforms if len(item) == length])),
            )
            for length in tuple(dict.fromkeys(len(item) for item in waveforms))
        )
        for _, batch in grouped_batches:
            batch.setflags(write=False)

        def predict_grouped(plugin: VoiceModelPlugin) -> ModelPrediction:
            if not waveforms:
                return plugin.predict(np.empty((0, self.window_spec.samples), np.float32))
            values = np.empty(len(waveforms), np.float32)
            latency_ms = 0.0
            group_metadata: list[dict[str, object]] = []
            for indices, batch in grouped_batches:
                prediction = plugin.predict(batch)
                if len(prediction.probabilities) != len(indices):
                    raise RuntimeError("every L4 model must return one probability per audio input")
                values[list(indices)] = prediction.probabilities
                latency_ms += prediction.latency_ms
                group_metadata.append(dict(prediction.metadata))
            metadata = dict(group_metadata[-1])
            metadata["continuous_input_lengths_48k"] = tuple(len(item) for item in waveforms)
            metadata["group_count"] = len(group_metadata)
            return ModelPrediction(plugin.model_id, values, latency_ms, metadata)

        predictions = tuple(predict_grouped(plugin) for plugin in (self.primary, *self.shadows))
        if any(len(item.probabilities) != len(inputs) for item in predictions):
            raise RuntimeError("every L4 model must return one probability per audio input")
        primary = predictions[0]
        detections = tuple(
            VoiceDetection(
                item.session_id, item.stream_epoch, item.window_id, item.decision_sample,
                item.theta_deg, float(probability), bool(probability >= self.threshold), self.primary.model_id,
                item.track_id,
            )
            for item, probability in zip(inputs, primary.probabilities)
        )
        return Layer4Result(
            detections,
            predictions,
            self.primary.model_id,
            self.threshold,
            tuple(item[1] for item in compensated),
        )

    def rethreshold(self, result: Layer4Result, threshold: float) -> Layer4Result:
        """Recompute decisions from existing probabilities; never reruns L3 or a model."""
        if not 0 <= threshold <= 1:
            raise ValueError("L4 threshold must be in [0,1]")
        detections = tuple(
            VoiceDetection(item.session_id, item.stream_epoch, item.window_id, item.decision_sample,
                           item.theta_deg, item.probability, item.probability >= threshold, item.model_id,
                           item.track_id)
            for item in result.detections
        )
        return Layer4Result(
            detections,
            result.predictions,
            result.primary_model_id,
            float(threshold),
            result.input_gain_compensation,
        )
