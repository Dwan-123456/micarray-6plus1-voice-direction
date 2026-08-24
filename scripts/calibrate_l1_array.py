from __future__ import annotations

import argparse
import hashlib
import json
import queue
import time
import wave
from pathlib import Path

import numpy as np
from scipy import signal


SAMPLE_RATE = 48_000
PHYSICAL_CHANNELS = (0, 1, 2, 3, 4, 5, 7)
HARDWARE_MIX_CHANNEL = 6
STIMULUS_SEED = 20260824


def _pink_noise(samples: int, rng: np.random.Generator) -> np.ndarray:
    frequencies = np.fft.rfftfreq(samples, 1.0 / SAMPLE_RATE)
    spectrum = rng.standard_normal(frequencies.size) + 1j * rng.standard_normal(frequencies.size)
    scale = np.zeros_like(frequencies)
    active = (frequencies >= 100.0) & (frequencies <= 8_000.0)
    scale[active] = 1.0 / np.sqrt(frequencies[active])
    waveform = np.fft.irfft(spectrum * scale, n=samples)
    waveform /= max(float(np.max(np.abs(waveform))), np.finfo(np.float64).tiny)
    return waveform.astype(np.float32)


def build_stimulus() -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(STIMULUS_SEED)

    def silence(seconds: float) -> np.ndarray:
        return np.zeros(round(seconds * SAMPLE_RATE), np.float32)

    pink = _pink_noise(20 * SAMPLE_RATE, rng)
    pink *= np.float32(10.0 ** (-18.0 / 20.0) / np.sqrt(np.mean(np.square(pink, dtype=np.float64))))
    pink_peak = float(np.max(np.abs(pink)))
    maximum_peak = 10.0 ** (-3.0 / 20.0)
    if pink_peak > maximum_peak:
        pink *= np.float32(maximum_peak / pink_peak)
    chirp_samples = round(0.25 * SAMPLE_RATE)
    t = np.arange(chirp_samples, dtype=np.float64) / SAMPLE_RATE
    chirp = signal.chirp(t, f0=200.0, f1=10_000.0, t1=t[-1], method="logarithmic")
    chirp *= signal.windows.tukey(chirp_samples, alpha=0.2)
    chirp = (0.45 * chirp / np.max(np.abs(chirp))).astype(np.float32)
    interval = np.concatenate((chirp, silence(0.35)))
    parts = (silence(3.0), pink, silence(2.0), np.tile(interval, 20), silence(4.0))
    waveform = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
    metadata = {
        "schema_version": "l1_calibration_stimulus_v1",
        "sample_rate_hz": SAMPLE_RATE,
        "duration_seconds": len(waveform) / SAMPLE_RATE,
        "seed": STIMULUS_SEED,
        "pink_noise": {
            "start_seconds": 3.0,
            "duration_seconds": 20.0,
            "band_hz": [100, 8000],
            "target_rms_dbfs": -18.0,
            "maximum_peak_dbfs": -3.0,
        },
        "chirps": {
            "start_seconds": 25.0,
            "count": 20,
            "duration_ms": 250,
            "interval_ms": 600,
            "band_hz": [200, 10000],
        },
    }
    return waveform, metadata


def _write_pcm16(path: Path, waveform: np.ndarray) -> None:
    pcm = np.rint(np.clip(waveform, -1.0, 32767.0 / 32768.0) * 32768.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    waveform, metadata = build_stimulus()
    wav_path = output_dir / "l1_array_overhead_calibration.wav"
    json_path = output_dir / "l1_array_overhead_calibration.json"
    _write_pcm16(wav_path, waveform)
    metadata["stimulus_sha256"] = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(wav_path.resolve())
    print(json_path.resolve())
    print(json.dumps(metadata, ensure_ascii=False))


def _find_input_device(name_fragment: str, host_api: str) -> int:
    import sounddevice as sd

    matches = []
    for index, device in enumerate(sd.query_devices()):
        host = sd.query_hostapis(device["hostapi"])["name"]
        if (
            device["max_input_channels"] >= 8
            and name_fragment.casefold() in device["name"].casefold()
            and host_api.casefold() in host.casefold()
        ):
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one 8-channel {host_api} {name_fragment} device, found {matches}")
    return matches[0]


def capture(output_dir: Path, duration_seconds: float) -> None:
    import sounddevice as sd

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "native_8ch_capture.wav"
    device = _find_input_device("MicArray", "Windows WDM-KS")
    chunks: queue.Queue[bytes] = queue.Queue()
    status_messages: list[str] = []

    def callback(indata, _frames, _time_info, status) -> None:
        if status:
            status_messages.append(str(status))
        chunks.put(bytes(indata))

    frame_count = round(duration_seconds * SAMPLE_RATE)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(8)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=960,
            device=device,
            channels=8,
            dtype="int16",
            callback=callback,
        ):
            print("READY", flush=True)
            written = 0
            deadline = time.monotonic() + duration_seconds + 3.0
            while written < frame_count and time.monotonic() < deadline:
                try:
                    payload = chunks.get(timeout=0.5)
                except queue.Empty:
                    continue
                remaining_bytes = (frame_count - written) * 8 * 2
                payload = payload[:remaining_bytes]
                wav.writeframesraw(payload)
                written += len(payload) // (8 * 2)
    if written != frame_count:
        raise RuntimeError(f"capture ended early: {written}/{frame_count} frames")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({
        "capture": str(output.resolve()),
        "duration_seconds": written / SAMPLE_RATE,
        "sha256": digest,
        "status": status_messages,
    }, ensure_ascii=False), flush=True)


def _read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels, rate, width, frames = (
            wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getnframes()
        )
        if width != 2:
            raise RuntimeError("calibration capture must be PCM16")
        data = np.frombuffer(wav.readframes(frames), dtype="<i2").reshape(-1, channels)
    return np.ascontiguousarray(data.astype(np.float32) / 32768.0), rate


def _lag_correlation(left: np.ndarray, right: np.ndarray, maximum_lag: int = 8) -> tuple[int, float]:
    left = np.asarray(left, np.float64) - float(np.mean(left))
    right = np.asarray(right, np.float64) - float(np.mean(right))
    values = []
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag >= 0:
            x, y = left[lag:], right[: len(right) - lag]
        else:
            x, y = left[: len(left) + lag], right[-lag:]
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        values.append(0.0 if denominator == 0.0 else float(np.dot(x, y) / denominator))
    index = int(np.argmax(np.abs(values)))
    return index - maximum_lag, values[index]


def analyze(output_dir: Path) -> None:
    capture_path = output_dir / "native_8ch_capture.wav"
    captured, rate = _read_pcm16(capture_path)
    if rate != SAMPLE_RATE or captured.shape[1] != 8:
        raise RuntimeError(f"expected native 48 kHz [N,8], got {rate} Hz {captured.shape}")
    physical = captured[:, PHYSICAL_CHANNELS]
    chirp_sos = signal.butter(6, (200.0, 10_000.0), btype="bandpass", fs=SAMPLE_RATE, output="sos")
    stimulus, _ = build_stimulus()
    chirp_samples = round(0.25 * SAMPLE_RATE)
    template = signal.sosfilt(
        chirp_sos, stimulus[25 * SAMPLE_RATE : 25 * SAMPLE_RATE + chirp_samples].astype(np.float64)
    )
    reference = signal.sosfilt(chirp_sos, physical[:, 6].astype(np.float64))
    numerator = signal.correlate(reference, template, mode="valid", method="fft")
    window_energy = signal.convolve(
        reference * reference, np.ones(chirp_samples), mode="valid", method="fft"
    )
    normalized = numerator / np.sqrt(np.maximum(window_energy * np.dot(template, template), 1.0e-24))
    peaks, _ = signal.find_peaks(
        np.abs(normalized), height=0.20, distance=round(0.50 * SAMPLE_RATE)
    )
    runs: list[list[int]] = []
    for peak in peaks:
        if not runs or not 0.55 * SAMPLE_RATE <= peak - runs[-1][-1] <= 0.65 * SAMPLE_RATE:
            runs.append([int(peak)])
        else:
            runs[-1].append(int(peak))
    starts = max(runs, key=len, default=[])
    if len(starts) != 20:
        raise RuntimeError(f"expected 20 periodic calibration chirps, found best run of {len(starts)}")

    trial_lags: list[list[int]] = [[] for _ in PHYSICAL_CHANNELS]
    trial_corrs: list[list[float]] = [[] for _ in PHYSICAL_CHANNELS]
    signal_levels: list[np.ndarray] = []
    signal_to_noise: list[np.ndarray] = []
    for start in starts:
        segment = signal.sosfiltfilt(chirp_sos, physical[start : start + chirp_samples], axis=0)
        quiet = signal.sosfiltfilt(
            chirp_sos,
            physical[start + round(0.30 * SAMPLE_RATE) : start + round(0.55 * SAMPLE_RATE)],
            axis=0,
        )
        observed_power = np.mean(np.square(segment, dtype=np.float64), axis=0)
        quiet_power = np.mean(np.square(quiet, dtype=np.float64), axis=0)
        net_power = np.maximum(observed_power - quiet_power, np.finfo(np.float64).tiny)
        signal_levels.append(10.0 * np.log10(net_power))
        signal_to_noise.append(10.0 * np.log10(observed_power / np.maximum(quiet_power, 1.0e-24)))
        for channel in range(7):
            lag, correlation = _lag_correlation(segment[:, channel], segment[:, 6])
            trial_lags[channel].append(lag)
            trial_corrs[channel].append(correlation)
    level_trials = np.asarray(signal_levels)
    snr_trials = np.asarray(signal_to_noise)
    levels_dbfs = np.median(level_trials, axis=0)
    target_dbfs = float(np.mean(levels_dbfs))
    gains = np.power(10.0, (target_dbfs - levels_dbfs) / 20.0)
    relative_level_std = np.std(level_trials - np.mean(level_trials, axis=1, keepdims=True), axis=0)
    median_snr = np.median(snr_trials, axis=0)
    if float(np.min(median_snr)) < 20.0:
        raise RuntimeError(f"chirp signal-to-noise gate failed: minimum median {np.min(median_snr):.2f} dB")
    offsets = np.asarray([int(np.median(values)) for values in trial_lags], dtype=int)
    correlation_medians = np.asarray([np.median(np.abs(values)) for values in trial_corrs])
    polarity = np.asarray([1 if np.median(values) >= 0.0 else -1 for values in trial_corrs], dtype=int)
    stable_trials = np.asarray([
        sum(lag == offsets[index] and np.sign(corr) == polarity[index] for lag, corr in zip(trial_lags[index], trial_corrs[index]))
        for index in range(7)
    ])
    delays = int(np.max(offsets)) - offsets
    clipped = np.sum(np.abs(captured) >= (32767.0 / 32768.0), axis=0)
    report = {
        "schema_version": "l1_hardware_calibration_analysis_v1",
        "sample_rate_hz": SAMPLE_RATE,
        "physical_channel_map": list(PHYSICAL_CHANNELS),
        "hardware_mix_channel": HARDWARE_MIX_CHANNEL,
        "stimulus_sha256": hashlib.sha256((output_dir / "l1_array_overhead_calibration.wav").read_bytes()).hexdigest(),
        "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        "chirp_detection": {
            "start_samples": starts,
            "interval_samples_median": float(np.median(np.diff(starts))),
            "template_correlation_median": float(np.median(np.abs(normalized[starts]))),
        },
        "clipped_samples_native_8ch": clipped.astype(int).tolist(),
        "gain_measurement": {
            "method": "noise-power-subtracted chirp energy",
            "signal_level_dbfs": np.round(levels_dbfs, 4).tolist(),
            "median_signal_to_noise_db": np.round(median_snr, 4).tolist(),
            "relative_level_std_db": np.round(relative_level_std, 4).tolist(),
        },
        "timing_measurement": {
            "reference_channel": "Center/native-7",
            "arrival_offset_samples_vs_center": offsets.tolist(),
            "stable_chirp_results": stable_trials.astype(int).tolist(),
            "chirp_trials": 20,
            "median_normalized_correlation": np.round(correlation_medians, 4).tolist(),
        },
        "result": {
            "gains": np.round(gains, 6).tolist(),
            "polarity": polarity.tolist(),
            "delay_samples": delays.tolist(),
        },
    }
    analysis_path = output_dir / "analysis.json"
    analysis_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(analysis_path.resolve())
    print(json.dumps(report, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or capture the L1 overhead array calibration stimulus.")
    parser.add_argument("command", choices=("generate", "capture", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/calibration/current"))
    parser.add_argument("--duration-seconds", type=float, default=50.0)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.output_dir)
    elif args.command == "capture":
        capture(args.output_dir, args.duration_seconds)
    elif args.command == "analyze":
        analyze(args.output_dir)


if __name__ == "__main__":
    main()
