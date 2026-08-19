from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from common.data_types import CalibrationMetadata

FloatArray = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]


@dataclass(slots=True, frozen=True, eq=False)
class NoiseSpectrumRecord:
    """Read-only dynamic noise PSD snapshot aligned to one audio block."""

    psd: FloatArray  # shape: (7, n_fft // 2 + 1), power/bin
    frequencies_hz: FloatArray
    sample_rate: int
    n_fft: int
    sequence_id: int
    timestamp: float
    estimator: str = "mcra_record_v1"
    state: str = "warming_up"
    noise_level_db: FloatArray | None = None
    source_probability_per_mic: FloatArray | None = None
    array_source_probability_20ms: float | None = None

    def __post_init__(self) -> None:
        psd = np.asarray(self.psd)
        frequencies = np.asarray(self.frequencies_hz)
        bins = self.n_fft // 2 + 1
        if self.sample_rate != 48_000 or self.n_fft <= 0 or self.n_fft % 2:
            raise ValueError("noise spectrum requires 48 kHz and a positive even n_fft")
        if psd.shape != (7, bins) or frequencies.shape != (bins,):
            raise ValueError("noise spectrum shape does not match n_fft")
        if self.sequence_id < 0 or not np.isfinite(self.timestamp):
            raise ValueError("noise spectrum sequence/timestamp is invalid")
        if not np.isfinite(psd).all() or np.any(psd < 0.0) or not np.isfinite(frequencies).all():
            raise ValueError("noise spectrum must contain finite non-negative values")
        expected = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate).astype(np.float32)
        if not np.array_equal(frequencies.astype(np.float32), expected):
            raise ValueError("noise spectrum frequency axis is invalid")
        immutable_psd = np.frombuffer(
            np.ascontiguousarray(psd, dtype=np.float32).tobytes(), dtype=np.float32
        ).reshape(psd.shape)
        immutable_frequencies = np.frombuffer(expected.tobytes(), dtype=np.float32)
        object.__setattr__(self, "psd", immutable_psd)
        object.__setattr__(self, "frequencies_hz", immutable_frequencies)
        object.__setattr__(self, "sequence_id", int(self.sequence_id))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        if self.state not in {"warming_up", "ready", "invalid"}:
            raise ValueError("IMCRA state must be warming_up, ready, or invalid")
        if self.noise_level_db is not None:
            levels = np.asarray(self.noise_level_db)
            probabilities = np.asarray(self.source_probability_per_mic)
            if levels.shape != (7,) or probabilities.shape != (7,):
                raise ValueError("IMCRA summaries must be [7]")
            if not np.isfinite(levels).all() or not np.isfinite(probabilities).all():
                raise ValueError("IMCRA summaries must be finite")
            if np.any((probabilities < 0.0) | (probabilities > 1.0)):
                raise ValueError("IMCRA probabilities must be in [0,1]")
            object.__setattr__(
                self, "noise_level_db",
                np.frombuffer(np.ascontiguousarray(levels, dtype=np.float32).tobytes(), dtype=np.float32),
            )
            object.__setattr__(
                self, "source_probability_per_mic",
                np.frombuffer(
                    np.ascontiguousarray(probabilities, dtype=np.float32).tobytes(), dtype=np.float32
                ),
            )
        elif self.source_probability_per_mic is not None:
            raise ValueError("IMCRA probability summary requires noise levels")
        probability = self.array_source_probability_20ms
        if probability is not None and (
            not np.isfinite(probability) or not 0.0 <= probability <= 1.0
        ):
            raise ValueError("array source probability must be finite and in [0,1]")
        if self.state == "ready" and probability is None:
            raise ValueError("ready IMCRA record requires an array probability")
        if self.state != "ready" and probability is not None:
            raise ValueError("non-ready IMCRA record cannot publish an array probability")
        if probability is not None:
            object.__setattr__(self, "array_source_probability_20ms", float(probability))


@dataclass(frozen=True, slots=True)
class InputHealthEvent:
    event_id: int
    timestamp: float
    kind: str
    last_sequence_id_before_gap: int | None
    first_sequence_id_after_gap: int | None
    lost_sample_count: int | None
    message: str

    def __post_init__(self) -> None:
        if self.event_id < 0 or not np.isfinite(self.timestamp):
            raise ValueError("InputHealthEvent标识和timestamp无效")
        if self.kind not in {"input_overflow", "handoff_drop", "device_restart", "source_error"}:
            raise ValueError("InputHealthEvent kind无效")
        for name in ("last_sequence_id_before_gap", "first_sequence_id_after_gap", "lost_sample_count"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name}不能为负")


@dataclass(slots=True, frozen=True, eq=False)
class CdcHotmapFrame:
    """One immutable 16x16 hotmap frame received from the CDC interface.

    ``timestamp`` is captured with ``time.monotonic()`` when the complete CDC
    frame is decoded; ``received_at`` optionally records Unix wall time for
    persisted/replayed metadata. It can be compared with live audio timestamps,
    but the two USB interfaces are not hardware-synchronised.
    """

    matrix: UInt8Array
    sequence_id: int
    timestamp: float
    received_at: float | None = None

    def __post_init__(self) -> None:
        raw = np.asarray(self.matrix)
        if raw.shape != (16, 16):
            raise ValueError("CDC hotmap matrix 必须是 16x16")
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("CDC hotmap matrix 必须是整数数组")
        if np.any(raw < 0) or np.any(raw > 255):
            raise ValueError("CDC hotmap matrix 数值必须在 0..255 范围内")
        if isinstance(self.sequence_id, bool) or not isinstance(self.sequence_id, (int, np.integer)):
            raise TypeError("CDC hotmap sequence_id 必须是整数")
        sequence_id = int(self.sequence_id)
        if sequence_id < 0:
            raise ValueError("CDC hotmap sequence_id 不能小于 0")
        timestamp = float(self.timestamp)
        received_at = None if self.received_at is None else float(self.received_at)
        if not np.isfinite(timestamp):
            raise ValueError("CDC hotmap timestamp 必须是有限数值")
        if received_at is not None and not np.isfinite(received_at):
            raise ValueError("CDC hotmap received_at 必须是有限数值")
        # Back with immutable ``bytes`` rather than merely clearing NumPy's
        # write flag (an owning ndarray can otherwise make itself writable
        # again). This snapshot is safe to share across layer/thread queues.
        matrix = np.frombuffer(
            np.ascontiguousarray(raw, dtype=np.uint8).tobytes(),
            dtype=np.uint8,
        ).reshape(16, 16)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "received_at", received_at)

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence_id": self.sequence_id,
            "timestamp": self.timestamp,
            "received_at": self.received_at,
            "width": 16,
            "height": 16,
            "matrix": self.matrix.tolist(),
        }


@dataclass(slots=True, frozen=True)
class DeviceAudioFormat:
    sample_rate: int
    channels: int
    sample_format: str = "s16-le"
    layout: str = "interleaved"


@dataclass(slots=True)
class DecodedAudio:
    """Layer-1 logical audio ordered MIC0..MIC5, Center, HardwareMix."""

    samples: FloatArray  # shape: (num_samples, 8 logical channels)
    sample_rate: int
    sequence_id: int
    timestamp: float
    # Exact float32 representation of the native device channels before the
    # physical-microphone mapping/calibration.  Live MA-USB8 input carries 8
    # channels here so Session recording can retain the original S16 stream;
    # offline seven-channel inputs may leave it unset.
    native_samples: FloatArray | None = None
    # Latest CDC hotmap snapshot available when this audio block left Layer 1.
    # The CDC and UAC interfaces have independent clocks/rates, so snapshots
    # may repeat across audio blocks and are never interpolated.
    hotmap: CdcHotmapFrame | None = None
    # Deprecated block-level field. Formal v0.3 IMCRA hops are emitted after
    # IngestCoordinator assigns the authoritative sample interval.
    noise_spectrum: NoiseSpectrumRecord | None = None
    # Set by ChannelCalibrator. Sources may leave it unset before correction.
    calibration: CalibrationMetadata | None = None

    def __post_init__(self) -> None:
        data = np.asarray(self.samples)
        if data.ndim != 2 or data.shape[0] <= 0 or data.shape[1] != 8:
            raise ValueError("samples必须是非空逻辑8通道数组[N,8]")
        if self.sample_rate != 48_000:
            raise ValueError("sample_rate 必须为 48000")
        if isinstance(self.sequence_id, bool) or self.sequence_id < 0:
            raise ValueError("sequence_id 不能小于 0")
        if not np.isfinite(self.timestamp) or not np.isfinite(data).all():
            raise ValueError("samples 包含 NaN 或 Inf")
        self.samples = np.frombuffer(
            np.ascontiguousarray(data, dtype=np.float32).tobytes(), dtype=np.float32
        ).reshape(data.shape)
        if self.native_samples is not None:
            native = np.asarray(self.native_samples)
            if native.ndim != 2 or native.shape != (data.shape[0], 8):
                raise ValueError("native_samples 必须是与 samples 等帧数的 [N,8] 数组")
            if not np.isfinite(native).all():
                raise ValueError("native_samples 包含 NaN 或 Inf")
            self.native_samples = np.frombuffer(
                np.ascontiguousarray(native, dtype=np.float32).tobytes(), dtype=np.float32
            ).reshape(native.shape)
        if self.hotmap is not None and not isinstance(self.hotmap, CdcHotmapFrame):
            raise TypeError("hotmap 必须是 CdcHotmapFrame 或 None")
        if self.noise_spectrum is not None:
            if not isinstance(self.noise_spectrum, NoiseSpectrumRecord):
                raise TypeError("noise_spectrum must be NoiseSpectrumRecord or None")
            if (
                self.noise_spectrum.sequence_id != self.sequence_id
                or self.noise_spectrum.timestamp != float(self.timestamp)
            ):
                raise ValueError("noise_spectrum must align with the audio block")
        if self.calibration is not None and not isinstance(self.calibration, CalibrationMetadata):
            raise TypeError("calibration must be CalibrationMetadata or None")
        self.sequence_id = int(self.sequence_id)
        self.timestamp = float(self.timestamp)

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[0])

    @property
    def native_channels(self) -> int:
        return 0 if self.native_samples is None else int(self.native_samples.shape[1])

    @property
    def sequence(self) -> int:
        """Compatibility property; public interface name is sequence_id."""
        return self.sequence_id

    @property
    def physical_samples(self) -> FloatArray:
        return self.samples[:, :7]

    @property
    def hardware_mix(self) -> FloatArray:
        return self.samples[:, 7]
