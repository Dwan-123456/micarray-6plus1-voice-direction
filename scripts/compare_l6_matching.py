from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import wave

import numpy as np

from layer5_voice_classifier import NvidiaMarbleNetPlugin
from layer6_speaker_consolidation.matching import hungarian_track_features
from layer6_speaker_consolidation.models import CampPlusEmbedder
from layer6_speaker_consolidation.pipeline import _cluster, _retain_consistent_segments


SEGMENT_SAMPLES = 32_000
MINIMUM_SAMPLES = 8_000


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getsampwidth() != 2
            or reader.getframerate() != 16_000
        ):
            raise ValueError(f"expected mono PCM16 16 kHz WAV: {path}")
        audio = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    usable = len(audio) // 320 * 320
    return np.ascontiguousarray(audio[:usable].astype(np.float32) / 32768.0)


def _greedy_score(left: np.ndarray, right: np.ndarray) -> float:
    available = np.asarray(left @ right.T, dtype=np.float32).copy()
    matched = []
    for _ in range(min(available.shape)):
        flat = int(np.argmax(available))
        row, column = np.unravel_index(flat, available.shape)
        matched.append(float(available[row, column]))
        available[row, :] = -np.inf
        available[:, column] = -np.inf
    required = max(2, int(np.ceil(0.30 * min(len(left), len(right)))))
    return -1.0 if len(matched) < required else matched[required - 1]


def _embeddings(
    waveform: np.ndarray,
    vad: NvidiaMarbleNetPlugin,
    embedder: CampPlusEmbedder,
    threshold: float,
) -> tuple[np.ndarray, int, int]:
    prediction = vad.predict_16k_20ms(waveform)
    voice = waveform.reshape(-1, 320)[prediction.probabilities_20ms >= threshold].reshape(-1)
    segments = tuple(
        np.ascontiguousarray(voice[start:start + SEGMENT_SAMPLES], dtype=np.float32)
        for start in range(0, len(voice), SEGMENT_SAMPLES)
        if len(voice[start:start + SEGMENT_SAMPLES]) >= MINIMUM_SAMPLES
    )
    if not segments:
        return np.empty((0, 0), np.float32), len(voice), 0
    values = np.ascontiguousarray(np.stack(embedder.embed_batch(segments)), dtype=np.float32)
    retained = _retain_consistent_segments(values)
    return retained, len(voice), len(values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare current greedy and experimental Hungarian L6 matching",
    )
    parser.add_argument("cache", type=Path)
    parser.add_argument("--campplus", type=Path, default=Path("models/campplus_zh_en_16k_v1"))
    parser.add_argument("--marblenet", type=Path, default=Path("models/nv_marblenet_baseline_v1"))
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--vad-threshold", type=float, default=0.70)
    parser.add_argument("--matching-repeats", type=int, default=200)
    args = parser.parse_args()

    paths = tuple(sorted(args.cache.glob("*.wav")))
    if len(paths) < 2:
        raise ValueError("comparison requires at least two cached WAV files")
    started = perf_counter()
    vad = NvidiaMarbleNetPlugin("nv_marblenet_baseline_v1", args.marblenet, device="cpu")
    embedder = CampPlusEmbedder(args.campplus)
    tracks = []
    for path in paths:
        waveform = _read_wav(path)
        values, voice_samples, raw_count = _embeddings(
            waveform, vad, embedder, args.vad_threshold,
        )
        if len(values):
            tracks.append((path.stem, values, voice_samples, raw_count))

    old_matrix = np.eye(len(tracks), dtype=np.float32)
    new_matrix = np.eye(len(tracks), dtype=np.float32)
    pairs = []
    embedding_pairs = []
    for left in range(len(tracks)):
        for right in range(left + 1, len(tracks)):
            left_values = tracks[left][1]
            right_values = tracks[right][1]
            embedding_pairs.append((left_values, right_values))
            old = _greedy_score(left_values, right_values)
            features = hungarian_track_features(
                left_values,
                right_values,
                threshold=args.threshold,
                minimum_match_count=2,
                required_coverage=0.30,
            )
            old_matrix[left, right] = old_matrix[right, left] = old
            new_matrix[left, right] = new_matrix[right, left] = features.decision_score
            pairs.append({
                "left": tracks[left][0],
                "right": tracks[right][0],
                "greedy_score": old,
                "hungarian": features.as_dict(),
            })
    benchmark_started = perf_counter()
    for _ in range(args.matching_repeats):
        for left_values, right_values in embedding_pairs:
            _greedy_score(left_values, right_values)
    greedy_benchmark_ms = (perf_counter() - benchmark_started) * 1_000.0
    benchmark_started = perf_counter()
    for _ in range(args.matching_repeats):
        for left_values, right_values in embedding_pairs:
            hungarian_track_features(
                left_values,
                right_values,
                threshold=args.threshold,
                minimum_match_count=2,
                required_coverage=0.30,
            )
    hungarian_benchmark_ms = (perf_counter() - benchmark_started) * 1_000.0
    calls = max(1, args.matching_repeats * len(embedding_pairs))
    payload = {
        "cache": str(args.cache.resolve()),
        "tracks": [
            {
                "name": name,
                "voice_samples": voice_samples,
                "raw_segments": raw_count,
                "retained_segments": len(values),
            }
            for name, values, voice_samples, raw_count in tracks
        ],
        "pairs": pairs,
        "greedy_assignments": _cluster(old_matrix, args.threshold, 3),
        "hungarian_assignments": _cluster(new_matrix, args.threshold, 3),
        "elapsed_ms": (perf_counter() - started) * 1_000.0,
        "matching_benchmark_ms_per_pair": {
            "greedy": greedy_benchmark_ms / calls,
            "hungarian_with_features": hungarian_benchmark_ms / calls,
        },
        "probability_calibration": "not_fitted_no_speaker_identity_labels",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
