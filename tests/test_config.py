import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from common.config import calibration_config_hash, config_hash, load_config
from layer1_input.configuration import AudioConfig, CalibrationConfig, CdcConfig


CONFIG = Path(__file__).parents[1] / "config" / "config.yaml"
CALIBRATION_REPORT = Path(__file__).parents[1] / "docs" / "L1_HARDWARE_CALIBRATION_2026-08-24.json"


def test_root_config_is_valid_and_builds_layer1_adapters():
    config = load_config(CONFIG, environ={})
    assert config.device.physical_channel_map == (0, 1, 2, 3, 4, 5, 7)
    assert config.device.logical_channel_map == (0, 1, 2, 3, 4, 5, 7, 6)
    assert config.layer2.max_candidates == 3
    assert config.layer2.probability_gate.backend == "mean_2x20ms_v1"
    assert config.layer2.probability_gate.threshold == 0.60
    assert config.layer2.music.context_ms == 240
    assert config.layer2.music.comparison_context_ms == (160, 240, 320)
    assert config.layer2.music.max_history_ms == 320
    assert config.layer2.scanner_backend == "frequency_normalized_music"
    assert config.layer2.direction_id_tracking.backend == "circular_imm_jpda_v1"
    assert not hasattr(config.layer2.direction_id_tracking, "enabled")
    assert config.layer2.direction_id_tracking.association_gate_deg == 50.0
    assert config.layer2.direction_id_tracking.association_chi2 == 20.0
    assert config.layer2.direction_id_tracking.survival_probability_per_second == pytest.approx(
        0.97**50
    )
    assert config.layer2.direction_id_tracking.confirmation_observations == 5
    assert config.layer2.direction_id_tracking.confirmation_window_ms == 200
    assert config.layer2.direction_id_tracking.coasting_ttl_ms == 2_000
    assert config.layer2.effective_order_limit == 3
    assert not hasattr(config.layer2, "mdl_max_age_ms")
    assert not hasattr(config.layer2, "min_cross_frequency_consistency")
    assert config.layer2.dpd_rank1_enabled is False
    assert config.layer2.dpd_peak_fusion_distance_deg == 40.0
    assert config.layer2.dpd_peak_fusion_min_normalized_score == 0.70
    assert config.layer2.noise_whitening_enabled is True
    assert config.layer2.direction_id_tracking.stationary_velocity_half_life_seconds == 0.15
    assert config.layer2.direction_id_tracking.moving_velocity_half_life_seconds == 0.5
    assert config.layer2.direction_id_tracking.max_active_tracks == 4
    assert config.layer2.min_peak_distance_deg == 50.0
    assert config.runtime.max_candidate_batch == 3
    assert config.runtime.l3_device == "cpu"
    assert config.runtime.l3_cuda_microbatch_windows == 4
    assert config.runtime.l3_cuda_batch_wait_ms == 0.0
    assert config.runtime.l4_device == "cuda"
    assert config.runtime.l5_device == "cpu"
    assert config.runtime.layer456_resident_memory_budget_bytes == 128 * 1024 * 1024
    assert config.runtime.layer456_spool_min_free_bytes == 5 * 1024**3
    assert AudioConfig.from_project(config).block_size == 960
    assert AudioConfig.from_project(config).handoff_blocks == 500
    assert CdcConfig.from_project(config).required is False
    calibration = CalibrationConfig.from_project(config)
    assert calibration.delay_samples == (2, 1, 2, 0, 1, 0, 1)
    assert calibration.version == "office_overhead_1m_20260824_v2"
    assert calibration.gains == pytest.approx(
        (1.069754, 1.034156, 1.100770, 1.056586, 1.093528, 1.056039, 0.673006)
    )
    assert calibration.status == "verified"
    assert calibration.report_hash is not None
    assert len(calibration.report_hash) == 64
    assert hashlib.sha256(CALIBRATION_REPORT.read_bytes()).hexdigest() == calibration.report_hash
    assert calibration.calibration_hash == calibration_config_hash(config.calibration)
    assert len(config_hash(config)) == 64
    assert config.layer1_imcra.algorithm_version == "cohen_imcra_2003_l1_v4"
    assert config.layer1_imcra.output_frequency_min_hz == 0.0
    assert config.layer1_imcra.output_frequency_max_hz == 10_000.0
    assert config.layer1_imcra.hop_samples == 960
    assert config.layer1_pre_denoise.enabled is False
    assert config.layer1_pre_denoise.algorithm_version == "imcra_wiener_wola_v3"
    assert config.layer1_pre_denoise.frame_samples == 1920
    assert config.layer1_pre_denoise.hop_samples == 960
    assert config.layer1_pre_denoise.frequency_max_hz == 10_000.0
    assert config.layer1_pre_denoise.minimum_gain_db == -18.0
    assert config.layer1_speaker_count.enabled is False
    assert config.layer1_speaker_count.input_sample_rate == 16_000
    assert config.layer1_speaker_count.context_seconds == 5
    assert config.layer1_speaker_count.inference_hop_ms == 100
    assert config.layer1_speaker_count.algorithm_version == "countnet_crnn_5s_100ms_v3"
    assert config.layer1_speaker_count.input_level_target_dbfs == -20.0
    assert config.layer1_speaker_count.input_level_floor_dbfs == -70.0
    assert config.layer1_speaker_count.maximum_input_gain_db == 30.0
    assert config.layer4.enabled is True
    assert config.layer4.default_backend == "mossformer2_ss_16k"
    assert config.layer4.streaming.enabled is True
    assert config.layer4.streaming.chunk_seconds == 4
    assert config.layer4.streaming.overlap_seconds == 1
    assert config.layer4.streaming.queue_chunks == 2
    assert config.layer6.maximum_speakers == 5
    assert config.layer6.embedding_cache_max_segments == 600
    assert config.layer1_speaker_count.model_sha256 == "f655f168bbd9091efd18b950e63484825ba68052a911331cab1e845e27e505e4"
    assert config.recording.runtime.record_imcra is True
    assert config.recording.runtime.record_noise_spectrum is True
    gain = config.layer5.input_gain_compensation
    assert gain.enabled is False
    assert gain.algorithm_version == "imcra_probability_rms_v1"
    assert gain.target_rms_dbfs == -23.0
    assert gain.no_compensation_probability == 0.30
    assert gain.full_compensation_probability == 0.80
    assert gain.peak_ceiling_dbfs == -3.0
    assert config.layer5.continuous_context_ms == 3_200
    assert config.layer5.models[0].backend == "nvidia_marblenet_continuous_v2"
    assert config.downstream_audio_window.duration_ms == 40
    assert config.downstream_audio_window.samples == 1_920
    assert config.downstream_audio_window.decision_hops == 2
    assert config.downstream_audio_window.stft_frames == 5
    assert config.downstream_audio_window.resampled_16k_samples == 640


@pytest.mark.parametrize(
    ("duration_ms", "samples", "hops", "frames", "resampled"),
    ((40, 1_920, 2, 5, 640), (80, 3_840, 4, 9, 1_280), (160, 7_680, 8, 17, 2_560)),
)
def test_downstream_audio_window_derives_every_length(
    tmp_path, duration_ms, samples, hops, frames, resampled,
):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "downstream_audio_window_ms: 40",
        f"downstream_audio_window_ms: {duration_ms}",
    )
    candidate = tmp_path / f"downstream-{duration_ms}.yaml"
    candidate.write_text(text, encoding="utf-8")
    spec = load_config(candidate, environ={}).downstream_audio_window
    assert (spec.samples, spec.decision_hops, spec.stft_frames, spec.resampled_16k_samples) == (
        samples, hops, frames, resampled,
    )


@pytest.mark.parametrize("duration_ms", (60, 100, 200, 81))
def test_downstream_audio_window_rejects_unsupported_values(tmp_path, duration_ms):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "downstream_audio_window_ms: 40",
        f"downstream_audio_window_ms: {duration_ms}",
    )
    candidate = tmp_path / f"bad-downstream-{duration_ms}.yaml"
    candidate.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(candidate, environ={})


def test_only_allowed_deployment_variables_override():
    config = load_config(
        CONFIG,
        environ={
            "MIC_DEVICE_NAME": "Board",
            "MIC_SERIAL_REQUIRED": "true",
            "MIC_SAMPLE_RATE": "44100",
            "MIC_CHANNELS": "2",
        },
    )
    assert config.device.device_name == "Board"
    assert config.device.serial_required is True
    assert (config.device.sample_rate, config.device.device_channels) == (48_000, 8)


def test_unknown_config_field_is_rejected(tmp_path):
    text = CONFIG.read_text(encoding="utf-8") + "\nunknown_section: true\n"
    candidate = tmp_path / "bad.yaml"
    candidate.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(candidate, environ={})


@pytest.mark.parametrize("context_ms", (160, 240, 320))
def test_music_history_comparison_context_is_configurable(tmp_path, context_ms):
    text = CONFIG.read_text(encoding="utf-8").replace("context_ms: 240", f"context_ms: {context_ms}", 1)
    candidate = tmp_path / f"music-{context_ms}.yaml"
    candidate.write_text(text, encoding="utf-8")
    assert load_config(candidate, environ={}).layer2.music.context_ms == context_ms


def test_music_history_rejects_values_outside_comparison_set(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace("context_ms: 240", "context_ms: 200", 1)
    candidate = tmp_path / "music-invalid.yaml"
    candidate.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(candidate, environ={})


def test_candidate_limit_is_fixed_to_three(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace("max_candidates: 3", "max_candidates: 4")
    candidate = tmp_path / "bad-candidate-limit.yaml"
    candidate.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(candidate, environ={})


def test_layer2_minimum_peak_distance_is_fixed_to_50_degrees(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "min_peak_distance_deg: 50.0", "min_peak_distance_deg: 30.0"
    )
    candidate = tmp_path / "bad-minimum-peak-distance.yaml"
    candidate.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(candidate, environ={})


def test_layer2_srp_band_is_fixed_to_2000_4000_hz(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "frequency_min_hz: 2000.0", "frequency_min_hz: 500.0", 1
    )
    candidate = tmp_path / "bad-layer2-srp-band.yaml"
    candidate.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError, match="2000.0"):
        load_config(candidate, environ={})


def test_unknown_layer2_probability_gate_backend_is_rejected(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "backend: mean_2x20ms_v1", "backend: unknown_gate_v9", 1
    )
    candidate = tmp_path / "unimplemented-probability-gate.yaml"
    candidate.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(candidate, environ={})


def test_probability_gate_threshold_must_be_in_unit_interval(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace("threshold: 0.60", "threshold: 1.01", 1)
    candidate = tmp_path / "invalid-probability-gate-threshold.yaml"
    candidate.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(candidate, environ={})


@pytest.mark.parametrize(
    ("source", "replacement"),
    (
        ("backend: circular_imm_jpda_v1", "backend: unknown_tracker_v9"),
        ("association_gate_deg: 50.0", "association_gate_deg: 181.0"),
        ("measurement_std_deg: 5.0", "measurement_std_deg: 0.0"),
        ("probability_track: 0.80", "probability_track: 0.90"),
        ("tentative_ttl_ms: 500", "tentative_ttl_ms: -1"),
    ),
)
def test_direction_postprocessing_configuration_is_strict(tmp_path, source, replacement):
    text = CONFIG.read_text(encoding="utf-8").replace(source, replacement, 1)
    candidate = tmp_path / "invalid-direction-smoothing.yaml"
    candidate.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(candidate, environ={})


def test_standalone_kalman_configuration_was_removed() -> None:
    config = load_config(CONFIG, environ={})
    assert not hasattr(config.layer2, "direction_kalman")


@pytest.mark.parametrize(
    ("source", "replacement"),
    (
        ("chunk_seconds: 60", "chunk_seconds: 0"),
        ("audio_queue_seconds: 10", "audio_queue_seconds: 0"),
        ("result_queue_capacity: 256", "result_queue_capacity: 0"),
        ("pre_roll_seconds: 2", "pre_roll_seconds: -1"),
        ("post_roll_seconds: 3", "post_roll_seconds: -1"),
        ("retention_days: 30", "retention_days: 0"),
        ("max_storage_gb: 200", "max_storage_gb: 0"),
        ("min_free_storage_gb: 5", "min_free_storage_gb: -1"),
    ),
)
def test_recording_queue_and_storage_limits_are_strict(
    tmp_path, source, replacement
):
    text = CONFIG.read_text(encoding="utf-8").replace(source, replacement, 1)
    candidate = tmp_path / "invalid-recording-limit.yaml"
    candidate.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(candidate, environ={})


def test_recording_minimum_free_space_must_be_below_total_budget(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "min_free_storage_gb: 5", "min_free_storage_gb: 200", 1
    )
    candidate = tmp_path / "invalid-recording-storage-budget.yaml"
    candidate.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError, match="min_free_storage_gb"):
        load_config(candidate, environ={})


def test_recording_result_queue_cannot_exceed_memory_budget(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "result_queue_capacity: 256", "result_queue_capacity: 257", 1
    )
    candidate = tmp_path / "oversized-recording-result-queue.yaml"
    candidate.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(candidate, environ={})
