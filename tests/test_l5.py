from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample_poly

import torch

from common.config import DownstreamAudioWindowSpec

from layer5_voice_classifier import (
    FrameModelPrediction,
    Layer5AudioSegment,
    Layer5Engine,
    ModelPrediction,
    NvidiaMarbleNetPlugin,
    InputGainCompensationSettings,
    compensate_l5_input,
    max_contiguous_frame_mean,
)


ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "models" / "nv_marblenet_baseline_v1"


def _constant_level(dbfs: float) -> np.ndarray:
    return np.full(7_680, 10.0 ** (dbfs / 20.0), np.float32)


@pytest.mark.parametrize(
    ("probability", "expected_weight"),
    ((0.30, 0.0), (0.55, 0.5), (0.80, 1.0)),
)
def test_input_gain_probability_breakpoints(probability, expected_weight):
    source = _constant_level(-45.0)
    output, diagnostic = compensate_l5_input(
        source, (probability,) * 8, InputGainCompensationSettings()
    )
    segment = diagnostic.segments[0]
    assert segment.probability_weight == pytest.approx(expected_weight)
    assert segment.requested_gain_db == pytest.approx(22.0 * expected_weight, abs=1e-5)
    assert segment.applied_gain_db == pytest.approx(segment.requested_gain_db, abs=1e-5)
    assert np.shares_memory(source, output) is False


def test_input_above_target_is_not_amplified_and_source_is_unchanged():
    source = _constant_level(-20.0)
    original = source.copy()
    output, diagnostic = compensate_l5_input(
        source, (1.0,) * 8, InputGainCompensationSettings()
    )
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(output, source)
    assert all(item.full_gain_db == 0.0 for item in diagnostic.segments)


def test_peak_protection_caps_added_gain_without_attenuating_hot_input():
    source = np.full(7_680, 1.0e-4, np.float32)
    source[::960] = 10.0 ** (-4.0 / 20.0)
    output, diagnostic = compensate_l5_input(
        source, (1.0,) * 8, InputGainCompensationSettings()
    )
    assert 20.0 * np.log10(np.max(np.abs(output))) <= -3.0 + 1.0e-5
    assert diagnostic.peak_protection_trigger_count == 8

    hot = source.copy()
    hot[::960] = 10.0 ** (-2.0 / 20.0)
    hot_output, _ = compensate_l5_input(
        hot, (1.0,) * 8, InputGainCompensationSettings()
    )
    assert np.max(np.abs(hot_output)) == pytest.approx(np.max(np.abs(hot)))


def test_missing_probability_silence_and_nonfinite_probability_handling():
    source = _constant_level(-60.0)
    output, diagnostic = compensate_l5_input(
        source, (None,) * 8, InputGainCompensationSettings()
    )
    np.testing.assert_array_equal(output, source)
    assert diagnostic.compensated_segment_count == 0

    silence_output, silence_diagnostic = compensate_l5_input(
        np.zeros(7_680, np.float32), (1.0,) * 8, InputGainCompensationSettings()
    )
    assert not np.any(silence_output)
    assert all(item.silent for item in silence_diagnostic.segments)
    with pytest.raises(ValueError, match="IMCRA probabilities"):
        compensate_l5_input(
            source, (float("nan"),) * 8, InputGainCompensationSettings()
        )


def test_linear_transition_remains_peak_safe_when_next_segment_is_hotter():
    source = _constant_level(-60.0)
    source[960:1920] = 10.0 ** (-4.0 / 20.0)
    output, diagnostic = compensate_l5_input(
        source, (1.0,) * 8, InputGainCompensationSettings()
    )
    assert 20.0 * np.log10(np.max(np.abs(output[960:1920]))) <= -3.0 + 1.0e-5
    assert diagnostic.segments[1].peak_protection_triggered is True


def test_engine_sends_one_compensated_immutable_copy_to_primary_and_shadow():
    source = _constant_level(-45.0)
    original = source.copy()
    observations = []

    class _GainObservingPlugin:
        def __init__(self, model_id):
            self.model_id = model_id

        def predict(self, waveforms):
            observations.append((self.model_id, id(waveforms), waveforms.copy(), waveforms.flags.writeable))
            return ModelPrediction(self.model_id, np.asarray((0.5,), np.float32), 0.0, {})

    segment = Layer5AudioSegment(
        "session", 0, 1, 7_680, 10.0, 48_000, source, (0.8,) * 8
    )
    result = Layer5Engine(
        _GainObservingPlugin("primary"), (_GainObservingPlugin("shadow"),)
    ).process((segment,))

    np.testing.assert_array_equal(source, original)
    assert observations[0][1] == observations[1][1]
    assert observations[0][3] is observations[1][3] is False
    assert 20.0 * np.log10(np.sqrt(np.mean(observations[0][2] ** 2))) == pytest.approx(
        -23.0, abs=1e-4
    )
    assert len(result.input_gain_compensation[0].segments) == 8


def test_compensated_distant_official_speech_remains_voice_and_low_probability_noise_is_unchanged():
    plugin = NvidiaMarbleNetPlugin("nv_marblenet_baseline_v1", ARTIFACT, device="cpu")
    sample_rate, audio = wavfile.read(ARTIFACT / "source" / "smoke_speech.wav")
    audio = audio.astype(np.float32) / 32768.0
    # Use a continuous 640 ms region whose latest 80 ms contains speech. The
    # model receives the earlier samples as convolutional context.
    speech = resample_poly(audio, 48_000, sample_rate).astype(np.float32)[17_280:48_000]
    speech = np.ascontiguousarray(
        speech * (10.0 ** (-45.0 / 20.0) / np.sqrt(np.mean(speech.astype(np.float64) ** 2))),
        dtype=np.float32,
    )
    compensated, _ = compensate_l5_input(
        speech, (0.9,) * 32, InputGainCompensationSettings(), segment_count=32
    )
    probability = plugin.predict(compensated[None, :]).probabilities[0]
    assert probability > 0.70

    rng = np.random.default_rng(4)
    noise = np.ascontiguousarray(rng.normal(0.0, 1.0e-3, 7_680), dtype=np.float32)
    noise_output, _ = compensate_l5_input(
        noise, (0.3,) * 8, InputGainCompensationSettings()
    )
    np.testing.assert_array_equal(noise_output, noise)


def test_official_marblenet_weights_detect_official_speech_and_reject_silence():
    plugin = NvidiaMarbleNetPlugin("nv_marblenet_baseline_v1", ARTIFACT, device="cpu")
    sample_rate, audio = wavfile.read(ARTIFACT / "source" / "smoke_speech.wav")
    audio = audio.astype(np.float32) / 32768.0
    audio_48k = resample_poly(audio, 48_000, sample_rate).astype(np.float32)
    speech_window = audio_48k[17_280:48_000]
    probabilities = plugin.predict(
        np.stack((np.zeros(30_720, np.float32), speech_window))
    ).probabilities
    assert probabilities.shape == (2,)
    assert probabilities[0] < 0.01
    assert probabilities[1] > 0.70


def test_official_marblenet_long_audio_returns_finite_frame_aligned_probabilities():
    plugin = NvidiaMarbleNetPlugin("nv_marblenet_baseline_v1", ARTIFACT, device="cpu")
    sample_rate, audio = wavfile.read(ARTIFACT / "source" / "smoke_speech.wav")
    audio = audio.astype(np.float32) / 32768.0
    long_audio = np.ascontiguousarray(
        resample_poly(audio, 48_000, sample_rate).astype(np.float32)[17_280:48_000]
    )
    prediction = plugin.predict_20ms(long_audio)

    assert len(prediction.probabilities_20ms) == len(long_audio) // 960
    assert np.isfinite(prediction.probabilities_20ms).all()
    assert np.max(prediction.probabilities_20ms) > 0.70


def test_official_marblenet_accepts_the_configured_80ms_window():
    spec = DownstreamAudioWindowSpec(80, 3_840, 4, 9, 1_280)
    plugin = NvidiaMarbleNetPlugin(
        "nv_marblenet_baseline_v1", ARTIFACT, device="cpu", window_spec=spec,
    )
    sample_rate, audio = wavfile.read(ARTIFACT / "source" / "smoke_speech.wav")
    audio = audio.astype(np.float32) / 32768.0
    audio_48k = resample_poly(audio, 48_000, sample_rate).astype(np.float32)
    speech_window = np.ascontiguousarray(audio_48k[57_600:61_440])

    prediction = plugin.predict(
        np.stack((np.zeros(spec.samples, np.float32), speech_window))
    )

    assert prediction.probabilities.shape == (2,)
    assert np.isfinite(prediction.probabilities).all()
    assert np.all((prediction.probabilities >= 0.0) & (prediction.probabilities <= 1.0))
    assert prediction.metadata["resampled_samples"] == spec.resampled_16k_samples


def test_marblenet_adapter_downsamples_the_complete_160ms_batch_to_16khz():
    observed = {}

    class _SpyModel:
        def __call__(self, audio):
            observed["shape"] = tuple(audio.shape)
            logits = torch.zeros((audio.shape[0], 17, 2), device=audio.device)
            lengths = torch.full((audio.shape[0],), 17, dtype=torch.long, device=audio.device)
            return logits, lengths

    plugin = NvidiaMarbleNetPlugin.__new__(NvidiaMarbleNetPlugin)
    plugin.model_id = "spy"
    plugin.manifest = {
        "architecture_id": "spy",
        "source_model": "spy",
        "aggregation": "max_contiguous_3frame_mean_v1",
    }
    plugin.device = torch.device("cpu")
    plugin.model = _SpyModel()
    result = plugin.predict(np.zeros((2, 7_680), dtype=np.float32))
    assert observed["shape"] == (2, 2_560)
    assert result.probabilities.shape == (2,)


def test_nvidia_long_audio_adapter_returns_exactly_one_probability_per_20ms() -> None:
    observed = {}

    class _FrameSpyModel:
        def __call__(self, audio):
            observed["shape"] = tuple(audio.shape)
            probabilities = torch.tensor((0.1, 0.2, 0.3, 0.4, 0.5, 0.99))
            logits = torch.stack((torch.zeros_like(probabilities), torch.logit(probabilities)), dim=1)
            return logits.unsqueeze(0), torch.tensor((6,), dtype=torch.long)

    plugin = NvidiaMarbleNetPlugin.__new__(NvidiaMarbleNetPlugin)
    plugin.model_id = "nvidia-frame-spy"
    plugin.manifest = {"architecture_id": "spy", "source_model": "NVIDIA frame VAD"}
    plugin.device = torch.device("cpu")
    plugin.model = _FrameSpyModel()
    prediction = plugin.predict_20ms(np.zeros(5 * 960, dtype=np.float32))

    assert observed["shape"] == (1, 5 * 320)
    assert len(prediction.probabilities_20ms) == 5
    np.testing.assert_allclose(
        prediction.probabilities_20ms,
        np.asarray((0.1, 0.2, 0.3, 0.4, 0.5), np.float32),
        atol=1e-6,
    )
    assert prediction.metadata["model_frame_count"] == 6
    assert prediction.metadata["output_frame_count"] == 5


def test_layer5_long_audio_engine_preserves_frame_probabilities_and_thresholds_each_hop() -> None:
    class _FramePlugin:
        model_id = "nvidia-frame"

        def predict(self, waveforms):
            return ModelPrediction(self.model_id, np.full(len(waveforms), 0.5, np.float32), 0.0, {})

        def predict_20ms(self, waveform):
            return FrameModelPrediction(
                self.model_id,
                np.asarray((0.1, 0.8, 0.9, 0.2), np.float32),
                1.0,
                {"frame_shift_ms": 20},
            )

    item = Layer5AudioSegment(
        "session", 0, 4, 3_840, 10.0, 48_000,
        np.zeros(4 * 960, np.float32),
    )
    result = Layer5Engine(
        _FramePlugin(), threshold=0.7,
        input_gain_compensation=InputGainCompensationSettings(enabled=False),
    ).process_long_audio_20ms(item)

    np.testing.assert_array_equal(
        result.probabilities_20ms, np.asarray((0.1, 0.8, 0.9, 0.2), np.float32),
    )
    assert result.is_voice_20ms == (False, True, True, False)
    assert result.summary_probability == pytest.approx((0.8 + 0.9 + 0.2) / 3)
    assert result.summary_is_voice is False


@pytest.mark.parametrize(
    ("spec", "model_samples"),
    (
        (DownstreamAudioWindowSpec(80, 3_840, 4, 9, 1_280), 1_280),
        (DownstreamAudioWindowSpec(160, 7_680, 8, 17, 2_560), 2_560),
    ),
)
def test_marblenet_adapter_accepts_both_configured_window_lengths(spec, model_samples):
    observed = {}

    class _SpyModel:
        def __call__(self, audio):
            observed["shape"] = tuple(audio.shape)
            logits = torch.zeros((audio.shape[0], 5, 2), device=audio.device)
            lengths = torch.full((audio.shape[0],), 5, dtype=torch.long, device=audio.device)
            return logits, lengths

    plugin = NvidiaMarbleNetPlugin.__new__(NvidiaMarbleNetPlugin)
    plugin.model_id = "spy"
    plugin.manifest = {
        "architecture_id": "spy",
        "source_model": "spy",
        "aggregation": "max_contiguous_3frame_mean_v1",
    }
    plugin.device = torch.device("cpu")
    plugin.window_spec = spec
    plugin.model = _SpyModel()
    result = plugin.predict(np.zeros((2, spec.samples), dtype=np.float32))
    assert observed["shape"] == (2, model_samples)
    assert result.metadata["resampled_samples"] == model_samples


@pytest.mark.parametrize(
    "spec",
    (
        DownstreamAudioWindowSpec(80, 3_840, 4, 9, 1_280),
        DownstreamAudioWindowSpec(160, 7_680, 8, 17, 2_560),
    ),
)
def test_l5_engine_enforces_configured_window_and_probability_count(spec):
    class _Plugin:
        model_id = "fixed"

        def predict(self, waveforms):
            return ModelPrediction("fixed", np.full(len(waveforms), 0.5, np.float32), 0.0, {})

    segment = Layer5AudioSegment(
        "session", 0, 1, 7_680, 10.0, 48_000,
        np.zeros(spec.samples, np.float32), (0.5,) * spec.decision_hops,
    )
    result = Layer5Engine(_Plugin(), window_spec=spec).process((segment,))
    assert len(result.detections) == 1
    assert len(result.input_gain_compensation[0].segments) == spec.decision_hops


def test_contiguous_peak_aggregation_is_not_diluted_by_pauses():
    frames = torch.tensor(
        [[0.92, 0.88, 0.85, 0.02, 0.01, 0.01, 0.01, 0.01]], dtype=torch.float32
    )
    probability = max_contiguous_frame_mean(frames, torch.tensor([8]), window_frames=3)
    assert torch.allclose(probability, torch.tensor([(0.92 + 0.88 + 0.85) / 3]))
    assert probability.item() > 0.70
    assert frames.mean().item() < 0.40


def test_contiguous_peak_requires_more_than_one_spurious_frame():
    frames = torch.tensor([[0.01, 0.02, 0.99, 0.02, 0.01]], dtype=torch.float32)
    probability = max_contiguous_frame_mean(frames, torch.tensor([5]), window_frames=3)
    assert probability.item() < 0.40


class _FixedPlugin:
    model_id = "fixed"

    def predict(self, waveforms):
        return ModelPrediction(self.model_id, np.array((0.69, 0.71), np.float32), 0.0, {})


def test_layer5_public_contract_and_rethreshold_do_not_rerun_model():
    inputs = tuple(
        Layer5AudioSegment(
            "session", 0, 1, 7_680, theta, 48_000, np.zeros(7_680, np.float32)
        )
        for theta in (10.0, 20.0)
    )
    engine = Layer5Engine(_FixedPlugin(), threshold=0.70)
    result = engine.process(inputs)
    assert tuple(item.is_voice for item in result.detections) == (False, True)
    adjusted = engine.rethreshold(result, 0.68)
    assert adjusted.predictions is result.predictions
    assert tuple(item.is_voice for item in adjusted.detections) == (True, True)


def test_layer5_contract_rejects_old_spectrogram_shape_and_wrong_sample_rate():
    with pytest.raises(ValueError, match="20 ms"):
        Layer5AudioSegment(
            "session", 0, 1, 7_680, 10.0, 48_000, np.zeros((17, 169), np.float32)
        )
    with pytest.raises(ValueError, match="48 kHz"):
        Layer5AudioSegment(
            "session", 0, 1, 7_680, 10.0, 16_000, np.zeros(7_680, np.float32)
        )


def test_primary_and_shadow_receive_the_same_immutable_waveform_batch():
    observations = []

    class _ObservingPlugin:
        def __init__(self, model_id):
            self.model_id = model_id

        def predict(self, waveforms):
            observations.append((self.model_id, id(waveforms), waveforms.flags.writeable))
            return ModelPrediction(
                self.model_id, np.full((len(waveforms),), 0.5, np.float32), 0.0, {}
            )

    inputs = (
        Layer5AudioSegment(
            "session", 0, 1, 7_680, 10.0, 48_000, np.zeros(7_680, np.float32)
        ),
    )
    Layer5Engine(_ObservingPlugin("primary"), (_ObservingPlugin("shadow"),)).process(inputs)
    assert observations == [
        ("primary", observations[0][1], False),
        ("shadow", observations[0][1], False),
    ]


def test_layer5_rejects_incomplete_model_output():
    class _IncompletePlugin:
        model_id = "incomplete"

        def predict(self, waveforms):
            return ModelPrediction(self.model_id, np.empty((0,), np.float32), 0.0, {})

    inputs = (
        Layer5AudioSegment(
            "session", 0, 1, 7_680, 10.0, 48_000, np.zeros(7_680, np.float32)
        ),
    )
    with pytest.raises(RuntimeError, match="one probability per audio input"):
        Layer5Engine(_IncompletePlugin()).process(inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_marblenet_cpu_and_cuda_probabilities_are_consistent():
    sample_rate, audio = wavfile.read(ARTIFACT / "source" / "smoke_speech.wav")
    audio = audio.astype(np.float32) / 32768.0
    audio_48k = resample_poly(audio, 48_000, sample_rate).astype(np.float32)
    batch = np.ascontiguousarray(
        np.stack((np.zeros(7_680, np.float32), audio_48k[53_760:61_440]))
    )
    cpu = NvidiaMarbleNetPlugin("cpu", ARTIFACT, device="cpu").predict(batch).probabilities
    cuda = NvidiaMarbleNetPlugin("cuda", ARTIFACT, device="cuda").predict(batch).probabilities
    np.testing.assert_allclose(cuda, cpu, rtol=1e-4, atol=2e-5)
