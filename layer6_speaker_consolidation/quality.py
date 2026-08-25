from __future__ import annotations

import numpy as np

from .contracts import Layer6QualityScore
from .models import DnsMosScorer


def score_quality(
    waveform: np.ndarray,
    probabilities: tuple[float, ...],
    speaker_similarity: float,
    noise_rms: float,
    dnsmos: DnsMosScorer,
) -> Layer6QualityScore:
    audio = np.asarray(waveform, dtype=np.float32)
    voice = float(np.median(probabilities))
    speaker = float(np.clip(speaker_similarity, 0.0, 1.0))
    sig, bak, ovrl = dnsmos.score(audio)
    mos = float(np.clip(((0.25 * sig + 0.25 * bak + 0.50 * ovrl) - 1.0) / 4.0, 0.0, 1.0))
    signal_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))
    snr_db = 20.0 * np.log10(signal_rms / max(noise_rms, 1e-6))
    snr = float(np.clip((snr_db + 5.0) / 35.0, 0.0, 1.0))
    clipping = float(np.mean(np.abs(audio) >= 0.995))
    silence = float(np.mean(np.abs(audio) < 1e-5))
    jumps = float(np.mean(np.abs(np.diff(audio)) > 0.5)) if len(audio) > 1 else 0.0
    continuity = float(np.clip(1.0 - 8.0 * clipping - 0.6 * silence - 8.0 * jumps, 0.0, 1.0))
    total = float(np.clip(
        0.30 * voice + 0.30 * speaker + 0.20 * mos + 0.10 * snr + 0.10 * continuity,
        0.0,
        1.0,
    ))
    return Layer6QualityScore(voice, speaker, mos, snr, continuity, total, sig, bak, ovrl)
