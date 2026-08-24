from __future__ import annotations

import hashlib
from types import SimpleNamespace
import wave

import numpy as np

from layer4_speech_separation import (
    Layer4LongAudioInput,
    Layer4OfflineResult,
    SpeakerCountDecision,
)
from gui.dev_test_ui.offline_l6_store import OfflineLayer6UiStore
from layer6_speaker_consolidation import (
    Layer6Result,
    Layer6SpeakerAudio,
    OfflineLayer6Pipeline,
)


class _Embedder:
    def embed(self, waveform: np.ndarray) -> np.ndarray:
        return np.array([1.0, 0.0], np.float32) if float(np.mean(waveform)) >= 0 else np.array([0.0, 1.0], np.float32)


class _DnsMos:
    def score(self, _waveform: np.ndarray) -> tuple[float, float, float]:
        return 4.0, 4.0, 4.0


def _sha(audio: np.ndarray) -> str:
    return hashlib.sha256(audio.tobytes()).hexdigest()


def _results(track_id: int) -> tuple[Layer4OfflineResult, Layer4OfflineResult]:
    samples = 96_000
    source_audio = np.zeros(samples, np.float32)
    source = Layer4LongAudioInput(
        f"asset-{track_id}", _sha(source_audio), "session", 0, track_id, float(track_id * 90),
        0, 48_000, source_audio, ((0, 2),),
    )
    values = (
        np.full(samples // 3, 0.10, np.float32),
        np.full(samples // 3, -0.10, np.float32),
    )
    probabilities = ((0.90,) * 100, (0.60,) * 100)
    count = SpeakerCountDecision(source.asset_id, 2, 1.0, "test", {})
    return tuple(Layer4OfflineResult(
        f"request-{track_id}", source, count, "two_speaker_separation", None,
        max(probabilities[index]), True, "l5", probabilities[index], (True,) * 100,
        f"asset-{track_id}:branch-{index}", _sha(values[index]),
        {"l5_threshold": 0.5, "output_waveform_16k": values[index]},
        ("candidate_0", "candidate_1")[index],
    ) for index in range(2))  # type: ignore[return-value]


def test_l6_clusters_candidates_and_keeps_one_quality_winner_per_speaker_timeline() -> None:
    config = SimpleNamespace(
        maximum_speakers=3,
        speaker_similarity_threshold=0.62,
        minimum_embedding_speech_ms=1_500,
        minimum_voice_fragment_ms=200,
        merge_voice_gap_ms=200,
        selection_switch_margin=0.05,
        crossfade_ms=2,
    )
    result = OfflineLayer6Pipeline(_Embedder(), _DnsMos(), config).process((*_results(1), *_results(2)))
    assert result.speaker_count == 2
    assert tuple(item.label for item in result.outputs) == ("Speaker A", "Speaker B")
    assert all(item.sample_rate == 16_000 and len(item.waveform_16k) == 32_000 for item in result.outputs)
    assert len(result.fragments) == 4
    assert all(np.max(np.abs(item.waveform_16k)) <= 0.1 for item in result.outputs)


def test_l6_ui_aligns_each_final_id_to_one_shared_absolute_timeline() -> None:
    first = Layer6SpeakerAudio(
        1, "Speaker A", 16_000, 0, 1_920,
        np.full(640, 0.25, np.float32), (7,), ("a",), 0.8,
    )
    second = Layer6SpeakerAudio(
        2, "Speaker B", 16_000, 1_920, 3_840,
        np.full(640, -0.25, np.float32), (9,), ("b",), 0.7,
    )
    store = OfflineLayer6UiStore()
    try:
        store.set_result(Layer6Result("session", 2, (first, second), (), {}))
        rendered = []
        for speaker_id in (1, 2):
            path = store.audio_path(speaker_id)
            assert path is not None
            with wave.open(str(path), "rb") as reader:
                assert reader.getframerate() == 16_000
                assert reader.getnframes() == 1_280
                rendered.append(np.frombuffer(reader.readframes(1_280), dtype="<i2"))
        assert np.any(rendered[0][:640]) and not np.any(rendered[0][640:])
        assert not np.any(rendered[1][:640]) and np.any(rendered[1][640:])
        snapshots = store.snapshots()
        assert tuple(item.audio_sample_count for item in snapshots) == (3_840, 3_840)
        assert snapshots[0].display_label == "L6 ID 1 · Speaker A · 来源L2 ID 7 · Q0.80"
        assert snapshots[1].display_label == "L6 ID 2 · Speaker B · 来源L2 ID 9 · Q0.70"
    finally:
        store.close()
