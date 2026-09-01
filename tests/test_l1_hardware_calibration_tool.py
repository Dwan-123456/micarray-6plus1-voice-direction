from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from common.config import load_config
from scripts.calibrate_l1_array import SAMPLE_RATE, _lag_correlation, build_stimulus, generate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKED_CALIBRATION_REPORT = (
    PROJECT_ROOT / "docs" / "v1.4.3_existing_docs" / "L1_HARDWARE_CALIBRATION_2026-08-24.json"
)


def test_verified_hardware_calibration_hash_matches_the_tracked_report():
    config = load_config(PROJECT_ROOT / "config" / "config.yaml", environ={})
    actual = hashlib.sha256(TRACKED_CALIBRATION_REPORT.read_bytes()).hexdigest()

    assert config.hardware.hardware_calibration_status == "verified"
    assert config.hardware.hardware_calibration_report_hash == actual


def test_calibration_stimulus_is_deterministic_loud_and_unclipped(tmp_path):
    first, metadata = build_stimulus()
    second, _ = build_stimulus()

    assert np.array_equal(first, second)
    assert first.dtype == np.float32
    assert len(first) == 41 * SAMPLE_RATE
    assert float(np.max(np.abs(first))) < 1.0
    pink = first[3 * SAMPLE_RATE : 23 * SAMPLE_RATE]
    pink_rms_dbfs = 20.0 * np.log10(np.sqrt(np.mean(np.square(pink, dtype=np.float64))))
    assert pink_rms_dbfs == pytest.approx(-18.0, abs=0.01)
    assert metadata["chirps"]["count"] == 20

    generate(tmp_path)
    wav = tmp_path / "l1_array_overhead_calibration.wav"
    manifest = json.loads((tmp_path / "l1_array_overhead_calibration.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(wav.read_bytes()).hexdigest() == manifest["stimulus_sha256"]


def test_integer_lag_correlation_reports_delay_and_polarity():
    rng = np.random.default_rng(7)
    reference = rng.standard_normal(4096)
    delayed = np.concatenate((np.zeros(3), reference[:-3]))

    assert _lag_correlation(delayed, reference) == pytest.approx((3, 1.0))
    lag, correlation = _lag_correlation(-delayed, reference)
    assert lag == 3
    assert correlation == pytest.approx(-1.0)
