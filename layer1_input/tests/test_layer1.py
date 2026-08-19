import tempfile
import unittest
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from layer1_input.algorithms import Pcm16InterleavedDecoder
from layer1_input.calibration import ChannelCalibrator
from layer1_input.capture import AudioCapture
from layer1_input.configuration import CalibrationConfig
from layer1_input.interface import CdcHotmapFrame, DecodedAudio
from layer1_input.http_hotmap import HttpHotmapSource
from layer1_input.protocols import beam_direction_command, led_command
from layer1_input.pipeline import InputPipeline
from layer1_input.pcm import float_to_pcm16
from layer1_input.serial_device import SerialDevice
from layer1_input.sources import MultichannelWavRecorder, WavAudioSource, map_physical_channels


class Layer1Tests(unittest.TestCase):

    @staticmethod
    def _calibration(**overrides):
        values = {
            "gains": (1.0,) * 7,
            "polarity": (1,) * 7,
            "delay_samples": (0,) * 7,
        }
        values.update(overrides)
        return CalibrationConfig(**values)

    @staticmethod
    def _fake_sounddevice(stream):
        return SimpleNamespace(
            query_hostapis=lambda: [{"name": "Windows WDM-KS"}],
            query_devices=lambda: [{"name": "MicArray", "hostapi": 0, "max_input_channels": 8}],
            InputStream=lambda **_kwargs: stream,
        )

    def test_capture_closes_stream_when_start_fails(self):
        class FailingStartStream:
            samplerate = 48_000

            def __init__(self):
                self.stop_called = self.close_called = False

            def start(self):
                raise RuntimeError("start failed")

            def stop(self):
                self.stop_called = True

            def close(self):
                self.close_called = True

        stream = FailingStartStream()
        capture = AudioCapture("MicArray", "Windows WDM-KS", 48_000, 8, 960)
        with patch.dict("sys.modules", {"sounddevice": self._fake_sounddevice(stream)}):
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                capture.start()
        self.assertTrue(stream.stop_called)
        self.assertTrue(stream.close_called)
        self.assertFalse(capture.running)

    def test_capture_closes_stream_when_negotiated_rate_is_wrong(self):
        class WrongRateStream:
            samplerate = 44_100

            def __init__(self):
                self.stop_called = self.close_called = False

            def start(self):
                pass

            def stop(self):
                self.stop_called = True

            def close(self):
                self.close_called = True

        stream = WrongRateStream()
        capture = AudioCapture("MicArray", "Windows WDM-KS", 48_000, 8, 960)
        with patch.dict("sys.modules", {"sounddevice": self._fake_sounddevice(stream)}):
            with self.assertRaisesRegex(RuntimeError, "44100 Hz"):
                capture.start()
        self.assertTrue(stream.stop_called)
        self.assertTrue(stream.close_called)
        self.assertFalse(capture.running)

    def test_input_pipeline_outputs_decoded_audio(self):
        class StubSource:
            def start(self): pass
            def stop(self): pass
            def read(self, timeout=None):
                del timeout
                return DecodedAudio(np.zeros((8, 8), dtype=np.float32), 48000, 5, 1.25)

        pipeline = InputPipeline(StubSource(), ChannelCalibrator(self._calibration()))
        result = pipeline.read()
        self.assertIsInstance(result, DecodedAudio)
        self.assertEqual(result.sequence_id, 5)
        self.assertEqual(result.samples.shape, (8, 8))

    def test_pcm_decoder_contract(self):
        pcm = np.arange(32, dtype="<i2")
        decoded = Pcm16InterleavedDecoder().decode(pcm.tobytes(), 8)
        self.assertEqual(decoded.shape, (4, 8))
        self.assertEqual(decoded.dtype, np.float32)

    def test_every_pcm16_code_roundtrips_exactly(self):
        original = np.arange(-32768, 32768, dtype=np.int32).astype("<i2")
        decoded = Pcm16InterleavedDecoder().decode(original.tobytes(), 1).reshape(-1)
        encoded = float_to_pcm16(decoded)
        np.testing.assert_array_equal(encoded, original)

    def test_physical_mapping_excludes_ch6(self):
        mapped = map_physical_channels(np.tile(np.arange(8, dtype=np.float32), (2, 1)), (0, 1, 2, 3, 4, 5, 7))
        self.assertEqual(mapped[0].tolist(), [0, 1, 2, 3, 4, 5, 7])

    def test_audio_frame_shape(self):
        frame = DecodedAudio(np.zeros((960, 8)), 48000, 0, 0.0)
        self.assertEqual((frame.frame_count, frame.channels), (960, 8))
        self.assertEqual(frame.sequence_id, 0)

    def test_calibration(self):
        calibrator = ChannelCalibrator(self._calibration(gains=(2, 1, 1, 1, 1, 1, 1), polarity=(-1, 1, 1, 1, 1, 1, 1), delay_samples=(0, 2, 0, 0, 0, 0, 0)))
        samples = np.zeros((4, 8), dtype=np.float32)
        samples[:, 0] = [1, 2, 3, 4]
        samples[:, 1] = [1, 2, 3, 4]
        result = calibrator.process(DecodedAudio(samples, 48000, 0, 0)).samples
        np.testing.assert_array_equal(result[:, 0], [-2, -4, -6, -8])
        np.testing.assert_array_equal(result[:, 1], [0, 0, 1, 2])

    def test_wav_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logical-eight.wav"
            expected = np.linspace(-.8, .8, 160, dtype=np.float32).reshape(20, 8)
            with MultichannelWavRecorder(path, 48000) as recorder:
                recorder.write(DecodedAudio(expected, 48000, 0, 0))
            with WavAudioSource(path, block_size=20) as source:
                actual = source.read().samples
            np.testing.assert_allclose(actual, expected, atol=1 / 32768)

    def test_hotmap_and_protocol(self):
        device = SerialDevice("TEST", 115200)
        device._publish(b"\xff" * 16 + bytes(range(256)))
        self.assertEqual(device.latest_hotmap()["matrix"][15], list(range(240, 256)))
        frame = device.latest_hotmap_frame()
        self.assertEqual((frame.matrix.shape, frame.matrix.dtype, frame.sequence_id), ((16, 16), np.dtype("uint8"), 0))
        self.assertFalse(frame.matrix.flags.writeable)
        with self.assertRaises(ValueError):
            frame.matrix.setflags(write=True)
        self.assertEqual(led_command(False), b"e")
        self.assertEqual(beam_direction_command("a"), b"A")

    def test_hotmap_frame_normalizes_numpy_scalars_for_json(self):
        frame = CdcHotmapFrame(
            np.zeros((16, 16), dtype=np.uint8),
            np.int64(2),
            np.float64(1.25),
            np.float64(10.5),
        )
        self.assertEqual(json.loads(json.dumps(frame.as_dict()))["sequence_id"], 2)

    def test_http_hotmap_source_reads_layer1_contract_without_using_serial(self):
        payload = json.dumps({
            "available": True,
            "sequence_id": 4,
            "timestamp": 1.5,
            "received_at": 20.0,
            "matrix": np.arange(256, dtype=np.uint8).reshape(16, 16).tolist(),
        }).encode("utf-8")

        class Response:
            def read(self): return payload
            def __enter__(self): return self
            def __exit__(self, *_args): return False

        source = HttpHotmapSource("http://127.0.0.1:8765/lights/set", poll_interval=60)
        with patch("layer1_input.http_hotmap.urlopen", return_value=Response()) as opened:
            source.start()
            frame = source.latest_hotmap_frame()
            source.stop()
        self.assertEqual(opened.call_args.args[0], "http://127.0.0.1:8765/hotmap/latest")
        self.assertEqual((frame.sequence_id, int(frame.matrix[15, 15])), (4, 255))

    def test_serial_disconnect_clears_hotmap_and_reports_not_running(self):
        class FailingPort:
            is_open = True
            in_waiting = 0
            def read(self, _size): raise OSError("unplugged")
            def close(self): self.is_open = False

        device = SerialDevice("TEST", 115200)
        device._publish(b"\xff" * 16 + bytes(range(256)))
        port, stop_event = FailingPort(), threading.Event()
        device._serial = port
        device._thread = threading.current_thread()
        device._read_loop(port, stop_event)
        self.assertIsNone(device.latest_hotmap_frame())
        self.assertFalse(device.running)

    def test_serial_thread_start_failure_closes_open_port(self):
        port = MagicMock(is_open=True)
        fake_serial_module = SimpleNamespace(Serial=lambda *_args, **_kwargs: port)
        device = SerialDevice("TEST", 115200)
        with patch.dict("sys.modules", {"serial": fake_serial_module}), patch(
            "layer1_input.serial_device.threading.Thread.start",
            side_effect=RuntimeError("thread failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "thread failed"):
                device.start()
        port.close.assert_called_once()
        self.assertFalse(device.running)

    def test_hotmap_parser_handles_split_and_joined_frames_with_independent_sequence(self):
        device = SerialDevice("TEST", 115200)
        first = b"\xff" * 16 + bytes(range(256))
        second = b"\xff" * 16 + bytes(reversed(range(256)))
        device._publish(b"noise" + first[:37])
        self.assertIsNone(device.latest_hotmap_frame())
        device._publish(first[37:] + second)
        latest = device.latest_hotmap_frame()
        self.assertEqual(latest.sequence_id, 1)
        self.assertEqual(int(latest.matrix[0, 0]), 255)
        self.assertEqual(device.latest_hotmap()["frame_count"], 2)
        pending = device.take_hotmap_frames()
        self.assertEqual([frame.sequence_id for frame in pending], [0, 1])
        self.assertEqual(device.take_hotmap_frames(), ())

    def test_calibration_keeps_same_hotmap_object(self):
        hotmap = CdcHotmapFrame(np.arange(256, dtype=np.uint8).reshape(16, 16), 7, 1.5)
        audio = DecodedAudio(np.zeros((8, 8), dtype=np.float32), 48_000, 0, 1.5, hotmap=hotmap)
        output = ChannelCalibrator(self._calibration()).process(audio)
        self.assertIs(output.hotmap, hotmap)

    def test_input_pipeline_attaches_latest_hotmap_without_processing(self):
        class StubSource:
            def start(self): pass
            def stop(self): pass
            def read(self, timeout=None):
                del timeout
                return DecodedAudio(np.zeros((8, 8), dtype=np.float32), 48_000, 0, 0.0)

        class StubHotmap:
            def __init__(self, frame): self.frame, self.started, self.stopped = frame, 0, 0
            def start(self): self.started += 1
            def stop(self): self.stopped += 1
            def latest_hotmap_frame(self): return self.frame

        hotmap = CdcHotmapFrame(np.zeros((16, 16), dtype=np.uint8), 3, 2.0)
        provider = StubHotmap(hotmap)
        pipeline = InputPipeline(StubSource(), ChannelCalibrator(self._calibration()), provider)
        pipeline.start()
        first = pipeline.read()
        second = pipeline.read()
        pipeline.stop()
        self.assertIs(first.hotmap, hotmap)
        self.assertIs(second.hotmap, hotmap)
        self.assertEqual((provider.started, provider.stopped), (1, 1))

    def test_input_pipeline_optional_cdc_failure_outputs_audio_without_hotmap(self):
        class StubSource:
            def __init__(self): self.started = self.stopped = 0
            def start(self): self.started += 1
            def stop(self): self.stopped += 1
            def read(self, timeout=None):
                del timeout
                return DecodedAudio(np.zeros((8, 8), dtype=np.float32), 48_000, 0, 0.0)

        class FailingHotmap:
            def __init__(self): self.started = self.latest_calls = 0
            def start(self):
                self.started += 1
                raise OSError("CDC unavailable")
            def stop(self): raise AssertionError("an unstarted CDC source must not be stopped")
            def latest_hotmap_frame(self):
                self.latest_calls += 1
                raise AssertionError("an unstarted CDC source must not be read")

        source, provider = StubSource(), FailingHotmap()
        pipeline = InputPipeline(
            source,
            ChannelCalibrator(self._calibration()),
            provider,
            hotmap_required=False,
        )
        pipeline.start()
        result = pipeline.read()
        pipeline.stop()
        self.assertIsNone(result.hotmap)
        self.assertEqual((source.started, source.stopped), (1, 1))
        self.assertEqual((provider.started, provider.latest_calls), (1, 0))

    def test_input_pipeline_required_cdc_failure_does_not_start_audio(self):
        class StubSource:
            def __init__(self): self.started = 0
            def start(self): self.started += 1
            def stop(self): pass
            def read(self, timeout=None): return None

        class FailingHotmap:
            def start(self): raise OSError("CDC unavailable")
            def stop(self): pass
            def latest_hotmap_frame(self): return None

        source = StubSource()
        pipeline = InputPipeline(
            source,
            ChannelCalibrator(self._calibration()),
            FailingHotmap(),
            hotmap_required=True,
        )
        with self.assertRaisesRegex(OSError, "CDC unavailable"):
            pipeline.start()
        self.assertEqual(source.started, 0)

    def test_input_pipeline_does_not_stop_shared_cdc_source(self):
        class StubSource:
            def start(self): pass
            def stop(self): pass
            def read(self, timeout=None): return None

        class SharedHotmap:
            def __init__(self): self.stopped = 0
            def start(self): pass
            def stop(self): self.stopped += 1
            def latest_hotmap_frame(self): return None

        provider = SharedHotmap()
        pipeline = InputPipeline(
            StubSource(),
            ChannelCalibrator(self._calibration()),
            provider,
            owns_hotmap_source=False,
        )
        pipeline.start()
        pipeline.stop()
        self.assertEqual(provider.stopped, 0)

    def test_input_pipeline_owned_cdc_is_stopped_after_optional_read_failure(self):
        class StubSource:
            def start(self): pass
            def stop(self): pass
            def read(self, timeout=None):
                return DecodedAudio(np.zeros((8, 8), dtype=np.float32), 48_000, 0, 0.0)

        class FailingReadHotmap:
            def __init__(self): self.stopped = 0
            def start(self): pass
            def stop(self): self.stopped += 1
            def latest_hotmap_frame(self): raise OSError("CDC disconnected")

        provider = FailingReadHotmap()
        pipeline = InputPipeline(
            StubSource(),
            ChannelCalibrator(self._calibration()),
            provider,
            hotmap_required=False,
        )
        pipeline.start()
        self.assertIsNone(pipeline.read().hotmap)
        pipeline.stop()
        self.assertEqual(provider.stopped, 1)


if __name__ == "__main__":
    unittest.main()
