from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter_ns
from typing import Callable, TypeVar

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import ProjectConfig, load_config  # noqa: E402
from common.data_types import (  # noqa: E402
    CandidateDirection,
    DecisionWindow,
    ImcraHopSnapshot,
)
from common.geometry import MicGeometry, physical_6plus1_geometry  # noqa: E402
from layer3_direction_signal import Layer3Processor  # noqa: E402
from layer4_voice_classifier import (  # noqa: E402
    InputGainCompensationSettings,
    Layer4AudioSegment,
    Layer4Engine,
    NvidiaMarbleNetPlugin,
)


_T = TypeVar("_T")


def _summary(values_ms: list[float]) -> dict[str, float | int]:
    values = np.asarray(values_ms, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("benchmark values must be a non-empty finite vector")
    average = float(values.mean())
    return {
        "n": int(values.size),
        "avg_ms": average,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "std_ms": float(values.std()),
        "throughput_hz": 1_000.0 / average,
    }


def _resolve_device(requested: str, config: ProjectConfig) -> str:
    value = config.runtime.preferred_device if requested == "auto" else requested
    value = value.casefold()
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if value not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported benchmark device: {value}")
    return value


def _timed(call: Callable[[], _T], stream: torch.cuda.Stream | None) -> tuple[_T, float]:
    started = perf_counter_ns()
    if stream is None:
        result = call()
    else:
        with torch.cuda.stream(stream):
            result = call()
        stream.synchronize()
    elapsed_ms = (perf_counter_ns() - started) / 1_000_000.0
    return result, elapsed_ms


def _imcra_frequencies() -> np.ndarray:
    frequencies = np.fft.rfftfreq(2_048, 1.0 / 48_000).astype(np.float32)
    return frequencies[(frequencies >= 80.0) & (frequencies <= 8_000.0)]


def _hop(index: int, frequencies: np.ndarray) -> ImcraHopSnapshot:
    shape = (7, len(frequencies))
    scale = np.float32(1.0 + index * 0.001)
    probability = np.float32(0.2 + (index % 8) * 0.05)
    noise = np.full(shape, scale, np.float32)
    ones = np.ones(shape, np.float32)
    spp = np.full(shape, probability, np.float32)
    return ImcraHopSnapshot(
        "benchmark-session",
        0,
        index * 960,
        (index + 1) * 960,
        (index,),
        "cohen_imcra_2003_l1_v1",
        "ready",
        frequencies,
        noise,
        ones * (2.0 + scale),
        ones * (1.5 + scale),
        ones * 0.5,
        ones * 0.4,
        spp,
        1.0 - spp,
        ones * (4.0 + index * 0.002),
        ones * (3.0 + index * 0.001),
        np.ones((7, 4), np.float32),
        np.full(7, np.float32(index * 0.001), np.float32),
        np.full(7, probability, np.float32),
        float(probability),
    )


def _l3_windows(total: int, seed: int) -> tuple[DecisionWindow, ...]:
    rng = np.random.default_rng(seed)
    continuous = rng.normal(0.0, 0.02, ((total + 15) * 960, 8)).astype(np.float32)
    frequencies = _imcra_frequencies()
    hops = tuple(_hop(index, frequencies) for index in range(total + 15))
    return tuple(
        DecisionWindow(
            "benchmark-session",
            0,
            index,
            (index + 16) * 960,
            (index + 14) * 960,
            (index + 16) * 960,
            index * 960,
            (index + 16) * 960,
            48_000,
            continuous[index * 960 : (index + 16) * 960],
            tuple(range(index, index + 16)),
            hops[index : index + 16],
        )
        for index in range(total)
    )


def _candidates(window: DecisionWindow, count: int) -> tuple[CandidateDirection, ...]:
    return tuple(
        CandidateDirection(
            window.session_id,
            window.stream_epoch,
            window.window_id,
            window.decision_sample,
            window.doa_start_sample,
            window.doa_end_sample,
            theta,
            1.0,
            0.8,
        )
        for theta in (20.0, 120.0)[:count]
    )


def _combined_runs(runs: list[list[float]]) -> dict[str, object]:
    return {
        "runs": [_summary(run) for run in runs],
        "combined": _summary([value for run in runs for value in run]),
    }


def _benchmark_l3(
    config: ProjectConfig,
    geometry: MicGeometry,
    device: str,
    candidate_count: int,
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    elapsed_runs: list[list[float]] = []
    prepare_runs: list[list[float]] = []
    beamform_runs: list[list[float]] = []
    synthesize_runs: list[list[float]] = []
    last_snapshot = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for repeat in range(repeats):
        windows = _l3_windows(warmup + iterations, seed + repeat)
        processor = Layer3Processor(config, device=device)
        stream = torch.cuda.Stream() if device == "cuda" else None
        elapsed: list[float] = []
        for index, window in enumerate(windows):
            candidates = _candidates(window, candidate_count)
            _, duration = _timed(
                lambda window=window, candidates=candidates: processor.process(
                    window, candidates, geometry
                ),
                stream,
            )
            if index >= warmup:
                elapsed.append(duration)
        elapsed_runs.append(elapsed)
        last_snapshot = processor.cache_snapshot()

        # A separate continuous run adds synchronization boundaries only to
        # attribute the cost. End-to-end numbers above remain authoritative.
        windows = _l3_windows(warmup + iterations, seed + 10_000 + repeat)
        processor = Layer3Processor(config, device=device)
        stream = torch.cuda.Stream() if device == "cuda" else None
        prepare_values: list[float] = []
        beamform_values: list[float] = []
        synthesize_values: list[float] = []
        for index, window in enumerate(windows):
            candidates = _candidates(window, candidate_count)
            prepared, prepare_ms = _timed(lambda window=window: processor.prepare(window), stream)
            batch, beamform_ms = _timed(
                lambda prepared=prepared, candidates=candidates: (
                    processor.beamformer.process_prepared_batch(prepared, candidates, geometry)
                ),
                stream,
            )
            _, synthesize_ms = _timed(
                lambda prepared=prepared, batch=batch: processor._synthesize_prepared(  # noqa: SLF001
                    prepared, batch
                ),
                stream,
            )
            if index >= warmup:
                prepare_values.append(prepare_ms)
                beamform_values.append(beamform_ms)
                synthesize_values.append(synthesize_ms)
        prepare_runs.append(prepare_values)
        beamform_runs.append(beamform_values)
        synthesize_runs.append(synthesize_values)

    output = _combined_runs(elapsed_runs)
    output["detail_with_extra_sync_boundaries"] = {
        "prepare": _combined_runs(prepare_runs),
        "beamform": _combined_runs(beamform_runs),
        "istft_host_dto": _combined_runs(synthesize_runs),
    }
    output["cache"] = asdict(last_snapshot) if last_snapshot is not None else None
    output["peak_cuda_bytes"] = (
        int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    )
    return output


def _primary_l4_engine(config: ProjectConfig, root: Path, device: str) -> Layer4Engine:
    model = next(
        item
        for item in config.layer4.models
        if item.enabled and item.model_id == config.layer4.primary_model_id
    )
    artifact = Path(model.model_artifact)
    if not artifact.is_absolute():
        artifact = root / artifact
    plugin = NvidiaMarbleNetPlugin(model.model_id, artifact, device=device)
    return Layer4Engine(
        plugin,
        threshold=config.layer4.voice_probability_limit,
        input_gain_compensation=InputGainCompensationSettings(
            **config.layer4.input_gain_compensation.model_dump()
        ),
    )


def _l4_batches(total: int, count: int, seed: int) -> tuple[tuple[Layer4AudioSegment, ...], ...]:
    rng = np.random.default_rng(seed)
    batches: list[tuple[Layer4AudioSegment, ...]] = []
    for window_id in range(total):
        waveforms = np.ascontiguousarray(
            rng.normal(0.0, 0.01, (count, 15_360)), dtype=np.float32
        )
        batches.append(
            tuple(
                Layer4AudioSegment(
                    "benchmark-session",
                    0,
                    window_id,
                    (window_id + 16) * 960,
                    float(20 + index * 100),
                    48_000,
                    waveforms[index],
                    (0.9,) * 16,
                )
                for index in range(count)
            )
        )
    return tuple(batches)


def _benchmark_l4(
    engine: Layer4Engine,
    device: str,
    candidate_count: int,
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    elapsed_runs: list[list[float]] = []
    prediction_runs: list[list[float]] = []
    overhead_runs: list[list[float]] = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stream = torch.cuda.Stream() if device == "cuda" else None
    for repeat in range(repeats):
        batches = _l4_batches(
            warmup + iterations, candidate_count, seed + repeat
        )
        elapsed: list[float] = []
        prediction: list[float] = []
        overhead: list[float] = []
        for index, batch in enumerate(batches):
            result, duration = _timed(lambda batch=batch: engine.process(batch), stream)
            if index >= warmup:
                model_ms = float(result.predictions[0].latency_ms)
                elapsed.append(duration)
                prediction.append(model_ms)
                overhead.append(max(0.0, duration - model_ms))
        elapsed_runs.append(elapsed)
        prediction_runs.append(prediction)
        overhead_runs.append(overhead)
    output = _combined_runs(elapsed_runs)
    output["plugin_predict"] = _combined_runs(prediction_runs)
    output["compensation_stack_and_dto_estimate"] = _combined_runs(overhead_runs)
    output["peak_cuda_bytes"] = (
        int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeatable warm-cache benchmark for the current L3 and L4 contracts."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/config.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20_260_818)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) <= 0:
        parser.error("warmup, iterations, and repeats must be positive")
    return args


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path, environ={})
    device = _resolve_device(args.device, config)
    root = config_path.parent.parent
    geometry = physical_6plus1_geometry(
        config.hardware.speed_of_sound_mps,
        config.hardware.geometry_version,
        config.hardware.ring_radius_m,
    )
    report: dict[str, object] = {
        "schema_version": "l3_l4_benchmark_v1",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "config": str(config_path),
        "method": {
            "warmup_windows": args.warmup,
            "measured_windows_per_repeat": args.iterations,
            "repeats": args.repeats,
            "continuous_hop_samples": 960,
            "input_generation_excluded": True,
            "cuda_stream_synchronized_per_window": device == "cuda",
        },
        "l3_single_candidate": _benchmark_l3(
            config,
            geometry,
            device,
            1,
            args.warmup,
            args.iterations,
            args.repeats,
            args.seed,
        ),
        "l3_double_candidate": _benchmark_l3(
            config,
            geometry,
            device,
            2,
            args.warmup,
            args.iterations,
            args.repeats,
            args.seed + 1_000,
        ),
    }
    engine = _primary_l4_engine(config, root, device)
    report["l4_single_candidate"] = _benchmark_l4(
        engine,
        device,
        1,
        args.warmup,
        args.iterations,
        args.repeats,
        args.seed + 2_000,
    )
    report["l4_double_candidate"] = _benchmark_l4(
        engine,
        device,
        2,
        args.warmup,
        args.iterations,
        args.repeats,
        args.seed + 3_000,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
