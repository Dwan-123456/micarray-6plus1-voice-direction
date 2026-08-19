from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .interface import DecodedAudio


@dataclass(slots=True)
class ChannelDiagnostic:
    channel: int
    rms_dbfs: float
    peak_dbfs: float
    dc_offset: float
    clipping_ratio: float
    silent: bool


@dataclass(slots=True)
class DiagnosticReport:
    sample_rate: int
    frames: int
    channels: list[ChannelDiagnostic]
    correlation_matrix: list[list[float]]
    likely_duplicate_pairs: list[tuple[int, int, float]]
    likely_polarity_pairs: list[tuple[int, int, float]]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def diagnose_samples(samples: np.ndarray, sample_rate: int) -> DiagnosticReport:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 2 or not len(data):
        raise ValueError("诊断输入必须是非空二维音频")
    eps = 1e-12
    rms, peak, dc = np.sqrt(np.mean(data * data, axis=0)), np.max(np.abs(data), axis=0), np.mean(data, axis=0)
    clipping = np.mean(np.abs(data) >= 0.999, axis=0)
    centered = data - dc
    safe = centered / np.maximum(np.std(centered, axis=0), eps)
    corr = np.clip((safe.T @ safe) / len(data), -1, 1)
    channels = [
        ChannelDiagnostic(
            i,
            float(20 * np.log10(max(rms[i], eps))),
            float(20 * np.log10(max(peak[i], eps))),
            float(dc[i]),
            float(clipping[i]),
            bool(rms[i] < 1e-5),
        )
        for i in range(data.shape[1])
    ]
    duplicates, polarity = [], []
    for left in range(data.shape[1]):
        for right in range(left + 1, data.shape[1]):
            value = float(corr[left, right])
            if value > 0.995:
                duplicates.append((left, right, value))
            elif value < -0.95:
                polarity.append((left, right, value))
    return DiagnosticReport(sample_rate, len(data), channels, corr.tolist(), duplicates, polarity)


class StreamingDiagnostics:
    def __init__(self, max_frames: int = 480_000):
        self.max_frames, self._chunks, self._frames, self.sample_rate = max_frames, [], 0, 0

    def add(self, frame: DecodedAudio) -> None:
        remaining = self.max_frames - self._frames
        if remaining <= 0:
            return
        chunk = frame.samples[:remaining].copy()
        self._chunks.append(chunk)
        self._frames += len(chunk)
        self.sample_rate = frame.sample_rate

    def samples(self) -> np.ndarray:
        if not self._chunks:
            raise ValueError("尚未收到音频")
        return np.concatenate(self._chunks)

    def report(self) -> DiagnosticReport:
        return diagnose_samples(self.samples(), self.sample_rate)


def save_diagnostic_plots(samples: np.ndarray, sample_rate: int, output_dir: str | Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("图表需要 matplotlib") from exc
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, axes = plt.subplots(samples.shape[1], 1, sharex=True, figsize=(12, 1.5 * samples.shape[1]))
    time_axis = np.arange(len(samples)) / sample_rate
    for channel, axis in enumerate(np.atleast_1d(axes)):
        axis.plot(time_axis, samples[:, channel], linewidth=0.5)
        axis.set_ylabel(f"CH{channel}")
    path = destination / "waveforms.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(np.corrcoef(samples, rowvar=False), vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(image, ax=axis)
    path = destination / "correlation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    return paths
