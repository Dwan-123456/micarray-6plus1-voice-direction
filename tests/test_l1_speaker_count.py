from __future__ import annotations

from time import monotonic, sleep

import numpy as np

from common.config import load_config
from common.data_types import IngestedAudioBlock
from layer1_input.interface import SpeakerCountAnnotation
from layer1_input.speaker_count import AsyncSpeakerCounter, TorchScriptCountNet


CONFIG = __import__("pathlib").Path(__file__).parents[1] / "config" / "config.yaml"


class _Model:
    model_id = "fake-countnet"
    model_hash = "a" * 64

    def __init__(self, outputs=((0.1, 0.2, 0.7),)):
        self.outputs = tuple(np.asarray(item, np.float32) for item in outputs)
        self.inputs: list[np.ndarray] = []

    def predict(self, waveform_16k):
        self.inputs.append(np.asarray(waveform_16k).copy())
        return self.outputs[min(len(self.inputs) - 1, len(self.outputs) - 1)]


def _block(index: int, *, epoch: int = 0, sequence: int | None = None, timestamp: float | None = None):
    samples = np.zeros((960, 8), np.float32)
    samples[:, 6] = 0.25
    samples[:, 7] = 0.95
    start = index * 960
    return IngestedAudioBlock(
        "count-session", epoch, start, start + 960, 48_000,
        index if sequence is None else sequence,
        index * 0.02 if timestamp is None else timestamp,
        samples,
    )


def _wait(counter: AsyncSpeakerCounter, predicate, timeout=4.0) -> SpeakerCountAnnotation:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        value = counter.latest()
        if value is not None and predicate(value):
            return value
        sleep(0.002)
    raise AssertionError("speaker-count worker did not publish the expected annotation")


def _counter(model=None, *, missing=False):
    project = load_config(CONFIG, environ={})
    updates = {"enabled": True, "queue_blocks": 400}
    if missing:
        updates["model_sha256"] = None
    config = project.layer1_speaker_count.model_copy(update=updates)
    return AsyncSpeakerCounter(
        config, project_root=CONFIG.parents[1], model=model, timestamp_tolerance_ms=5.0
    )


def test_annotation_contract_does_not_invent_non_ready_probabilities():
    warming = SpeakerCountAnnotation(
        "s", 0, 0, 4_800, None, None, "countnet", None, "warming_up"
    )
    assert warming.speaker_count is warming.probabilities is None
    ready = SpeakerCountAnnotation(
        "s", 0, 0, 4_800, 2, np.asarray((0.1, 0.2, 0.7), np.float32),
        "countnet", "a" * 64, "ready",
    )
    assert not ready.probabilities.flags.writeable


def test_five_second_warmup_exact_resampling_and_five_block_alignment():
    model = _Model()
    counter = _counter(model)
    try:
        for index in range(245):
            block = _block(index)
            samples = np.array(block.samples, copy=True)
            phase = np.arange(960, dtype=np.float32) + index * 960
            samples[:, 6] = 0.25 * np.sin(2.0 * np.pi * 440.0 * phase / 48_000.0)
            assert counter.submit(__import__("dataclasses").replace(block, samples=samples))
        warm = _wait(counter, lambda item: item.end_sample >= 235_200)
        assert warm.status == "warming_up"
        for index in range(245, 250):
            assert counter.submit(_block(index))
        ready = _wait(counter, lambda item: item.status == "ready")
        assert (ready.start_sample, ready.end_sample) == (235_200, 240_000)
        assert ready.speaker_count == 2
        assert len(model.inputs) == 1 and model.inputs[0].shape == (80_000,)
        # The Center carried a tone while HardwareMix stayed constant.
        assert float(np.std(model.inputs[0][-16_000:])) > 0.15
    finally:
        counter.close()


def test_epoch_sequence_and_timestamp_gaps_reset_without_smoothing():
    model = _Model(((0.8, 0.1, 0.1), (0.1, 0.8, 0.1)))
    counter = _counter(model)
    try:
        for index in range(250):
            counter.submit(_block(index))
        first = _wait(counter, lambda item: item.status == "ready")
        assert first.speaker_count == 0
        for index in range(250, 255):
            counter.submit(_block(index))
        second = _wait(counter, lambda item: item.status == "ready" and item.end_sample == 244_800)
        assert second.speaker_count == 1  # immediate model output; no temporal smoothing

        counter.set_enabled(False)
        assert counter.latest() is None
        counter.set_enabled(True)
        for index in range(255, 260):
            counter.submit(_block(index))
        restarted = _wait(counter, lambda item: item.end_sample == 249_600)
        assert restarted.status == "warming_up"

        counter.submit(_block(260, sequence=999))
        for index in range(261, 265):
            counter.submit(_block(index, sequence=999 + index - 260))
        invalid = _wait(counter, lambda item: item.end_sample == 254_400)
        assert invalid.status == "invalid" and "continuity" in invalid.reason

        # A new epoch is another hard reset and must start a fresh five-block annotation.
        for index in range(5):
            counter.submit(_block(index, epoch=1))
        epoch_reset = _wait(counter, lambda item: item.stream_epoch == 1)
        assert epoch_reset.status == "invalid"

        # Timestamp discontinuity independently resets the state.
        for index in range(5, 10):
            counter.submit(_block(index, epoch=1, timestamp=9.0 + index * 0.02))
        timestamp_reset = _wait(counter, lambda item: item.stream_epoch == 1 and item.end_sample == 9_600)
        assert timestamp_reset.status == "invalid"
    finally:
        counter.close()


def test_missing_model_is_invalid_and_submit_never_waits_for_inference():
    counter = _counter(None, missing=True)
    try:
        started = monotonic()
        for index in range(250):
            assert counter.submit(_block(index))
        elapsed = monotonic() - started
        result = _wait(counter, lambda item: item.end_sample == 240_000)
        assert elapsed < 0.20
        assert result.status == "invalid"
        assert result.speaker_count is result.probabilities is None
        assert "SHA-256" in result.reason
    finally:
        counter.close()


def test_bundled_countnet_asset_hash_output_and_steady_cpu_gate():
    config = load_config(CONFIG, environ={}).layer1_speaker_count
    artifact = CONFIG.parents[1] / config.model_artifact
    model = TorchScriptCountNet(
        artifact, model_id=config.model_id, expected_hash=config.model_sha256
    )
    waveform = np.zeros(80_000, np.float32)
    model.predict(waveform)  # one-time TorchScript warm-up is not steady inference
    durations, result = [], None
    for _ in range(3):
        started = monotonic()
        result = model.predict(waveform)
        durations.append(monotonic() - started)
    assert result.shape == (3,) and float(result.sum()) == __import__("pytest").approx(1.0)
    assert int(np.argmax(result)) == 0
    assert float(np.median(durations)) < 0.10


def test_quiet_microphone_level_is_adapted_before_inference_and_reported():
    model = _Model(((0.2, 0.7, 0.1),))
    counter = _counter(model)
    try:
        for index in range(250):
            block = _block(index)
            samples = np.array(block.samples, copy=True)
            phase = np.arange(960, dtype=np.float32) + index * 960
            samples[:, 6] = 0.003 * np.sin(2.0 * np.pi * 440.0 * phase / 48_000.0)
            counter.submit(__import__("dataclasses").replace(block, samples=samples))
        ready = _wait(counter, lambda item: item.status == "ready")
        assert ready.speaker_count == 1
        assert ready.input_rms_dbfs == __import__("pytest").approx(-53.47, abs=0.3)
        assert ready.input_gain_db == __import__("pytest").approx(30.0, abs=0.1)
        assert float(np.sqrt(np.mean(model.inputs[0] ** 2))) > 0.06
    finally:
        counter.close()


def test_digital_silence_is_not_amplified_into_a_false_signal():
    model = _Model(((0.9, 0.05, 0.05),))
    counter = _counter(model)
    try:
        for index in range(250):
            block = _block(index)
            samples = np.array(block.samples, copy=True)
            samples[:, 6] = 0.0
            counter.submit(__import__("dataclasses").replace(block, samples=samples))
        ready = _wait(counter, lambda item: item.status == "ready")
        # The constant test signal becomes zero after DC removal and remains zero.
        assert ready.input_gain_db == 0.0
        assert np.count_nonzero(model.inputs[0]) == 0
    finally:
        counter.close()
