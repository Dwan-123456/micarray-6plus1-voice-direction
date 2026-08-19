from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .manifests import atomic_json, sha256_file, utc_now


def analyze_audio(samples: np.ndarray) -> dict[str, Any]:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] not in {7, 8}:
        raise ValueError("QA音频必须为[N,7]或[N,8]")
    rms = np.sqrt(np.mean(np.square(data, dtype=np.float64), axis=0)) if len(data) else np.zeros(data.shape[1])
    peak = np.max(np.abs(data), axis=0) if len(data) else np.zeros(data.shape[1])
    dc = np.mean(data, axis=0) if len(data) else np.zeros(data.shape[1])
    clipping = np.mean(np.abs(data) >= 1, axis=0) if len(data) else np.zeros(data.shape[1])
    silence = np.mean(np.abs(data) < 10 ** (-60 / 20), axis=0) if len(data) else np.ones(data.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(data.T) if len(data) > 1 else np.eye(data.shape[1])
    corr = np.nan_to_num(corr)
    failures = []
    if np.any(clipping > 0.001):
        failures.append("clipping_ratio")
    if np.any(np.abs(dc) > 0.02):
        failures.append("dc_offset")
    active = silence < 0.99
    if np.any(active & (rms <= 10 ** (-60 / 20))):
        failures.append("low_rms")
    duplicates = []
    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            if abs(corr[i, j]) > 0.9999:
                duplicates.append([i, j])
    return {
        "sample_count": len(data),
        "channel_count": data.shape[1],
        "rms": rms.tolist(),
        "rms_dbfs": (20 * np.log10(np.maximum(rms, 1e-12))).tolist(),
        "peak": peak.tolist(),
        "dc_offset": dc.tolist(),
        "clipping_ratio": clipping.tolist(),
        "silence_ratio": silence.tolist(),
        "correlation_matrix": corr.tolist(),
        "suspected_duplicate_channels": duplicates,
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }


def qa_recording(recording_root: str | Path) -> dict[str, Any]:
    root = Path(recording_root)
    manifest = json.loads((root / "recording_manifest.json").read_text(encoding="utf-8"))
    reports = []
    failures = []
    for asset in manifest.get("assets", []):
        path = root / asset["path"]
        if not path.exists() or sha256_file(path) != asset["sha256"]:
            failures.append(f"hash:{asset['path']}")
        if asset.get("kind") == "physical_float" and path.exists():
            report = analyze_audio(np.load(path, allow_pickle=False))
            reports.append(report)
            failures.extend(report["failures"])
    result = {
        "schema_version": "qa_report_v1",
        "recording_id": manifest["recording_id"],
        "created_at_utc": utc_now(),
        "status": "passed" if not failures else "failed",
        "failures": sorted(set(failures)),
        "audio_reports": reports,
    }
    atomic_json(root / "qa_report.json", result)
    return result


def leakage_check(recordings: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("capture_session_id", "room_id")
    owners: dict[tuple[str, str], set[str]] = {}
    for item in recordings:
        split = item.get("split", "unset")
        for key in keys:
            value = item.get(key)
            if value:
                owners.setdefault((key, str(value)), set()).add(split)
        for speaker in item.get("speaker_ids_anonymous", []):
            owners.setdefault(("speaker", str(speaker)), set()).add(split)
    leaks = [
        {"field": key[0], "value": key[1], "splits": sorted(values)}
        for key, values in owners.items()
        if len(values - {"unset"}) > 1
    ]
    return {"passed": not leaks, "leaks": leaks}
