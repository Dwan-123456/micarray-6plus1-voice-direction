from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .api_settings import settings
from .capture import AudioCapture
from .protocols import beam_direction_command, led_command, restore_defaults_command, threshold_command
from .serial_device import SerialDevice

audio = AudioCapture(
    settings.device_name, settings.host_api, settings.sample_rate, settings.channels, settings.block_size
)
serial_device = SerialDevice(settings.serial_port, settings.serial_baud)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    audio.stop()
    serial_device.stop()


app = FastAPI(title="Layer 1 - Sipeed Device I/O", version="0.2.0", lifespan=lifespan)


class RawWrite(BaseModel):
    hex: str = Field(description="发送到 CDC 串口的十六进制字节")


class LightRequest(BaseModel):
    enabled: bool


class BeamDirectionRequest(BaseModel):
    direction: str = Field(description="0..9、A、B，每步 30 度")


class ThresholdRequest(BaseModel):
    increase: bool


def _write_complete(packet: bytes, *, label: str) -> int:
    """Write one device command and reject silent/partial serial writes."""
    try:
        bytes_written = serial_device.write(packet)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if bytes_written != len(packet):
        raise HTTPException(
            status_code=503,
            detail=f"{label}未完整写入：{bytes_written}/{len(packet)} 字节",
        )
    return bytes_written


@app.get("/health")
def health():
    return {"ok": True, "audio": audio.running, "serial": serial_device.running}


@app.get("/device")
def device():
    return {
        "usb": {"vid": "359F", "pid": "3400"},
        "audio_endpoint_id": settings.endpoint_id,
        "audio": audio.status(),
        "serial": serial_device.status(),
        "controller": "Sipeed MA-USB8",
        "protocol": "UAC2.0 + CDC ACM",
    }


@app.post("/audio/start")
def audio_start():
    try:
        return audio.start()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/audio/stop")
def audio_stop():
    return audio.stop()


@app.get("/audio/status")
def audio_status():
    return audio.status()


@app.get("/audio/latest")
def audio_latest(blocks: int = 1, channel: int | None = None):
    payload, channels = audio.latest(blocks), settings.channels
    if channel is not None:
        try:
            payload, channels = audio.select_channel(payload, channel), 1
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "encoding": "base64",
        "sample_format": "s16-le",
        "channels": channels,
        "selected_channel": channel,
        "sample_rate": settings.sample_rate,
        "data": base64.b64encode(payload).decode(),
    }


async def _audio_socket(websocket: WebSocket, channel: int | None = None):
    await websocket.accept()
    if channel is not None and not 0 <= channel < settings.channels:
        await websocket.close(code=1008, reason="channel out of range")
        return
    receiver = None
    try:
        if not audio.running:
            audio.start()
        await websocket.send_json(
            {
                "sample_format": "s16-le",
                "channels": 1 if channel is not None else settings.channels,
                "source_channel": channel,
                "sample_rate": settings.sample_rate,
                "layout": "interleaved",
            }
        )
        receiver = audio.subscribe()
        while True:
            chunk = await asyncio.to_thread(receiver.get)
            await websocket.send_bytes(audio.select_channel(chunk, channel) if channel is not None else chunk)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await websocket.close(code=1011, reason=str(exc)[:120])
    finally:
        if receiver is not None:
            audio.unsubscribe(receiver)


@app.websocket("/audio/stream")
async def audio_stream(websocket: WebSocket):
    await _audio_socket(websocket)


@app.websocket("/audio/channel/{channel}")
async def audio_channel_stream(websocket: WebSocket, channel: int):
    await _audio_socket(websocket, channel)


@app.post("/serial/start")
def serial_start():
    try:
        return serial_device.start()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/serial/stop")
def serial_stop():
    return serial_device.stop()


@app.get("/serial/status")
def serial_status():
    return serial_device.status()


@app.get("/serial/latest")
def serial_latest():
    data = serial_device.latest()
    return {"hex": data.hex(), "base64": base64.b64encode(data).decode()}


@app.websocket("/serial/stream")
async def serial_stream(websocket: WebSocket):
    await websocket.accept()
    receiver = None
    try:
        if not serial_device.running:
            serial_device.start()
        receiver = serial_device.subscribe()
        while True:
            await websocket.send_bytes(await asyncio.to_thread(receiver.get))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await websocket.close(code=1011, reason=str(exc)[:120])
    finally:
        if receiver is not None:
            serial_device.unsubscribe(receiver)


@app.get("/hotmap/latest")
def hotmap_latest():
    if not serial_device.running:
        try:
            serial_device.start()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return serial_device.latest_hotmap()


@app.post("/serial/write")
def serial_write(request: RawWrite):
    try:
        data = bytes.fromhex(request.hex)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="无效十六进制") from exc
    return {"bytes_written": _write_complete(data, label="串口数据"), "hex": data.hex()}


@app.post("/lights/set")
def lights_set(request: LightRequest):
    packet = led_command(request.enabled)
    bytes_written = _write_complete(packet, label="定位指示灯指令")
    return {
        "enabled": request.enabled,
        "official_command": packet.decode(),
        "bytes_written": bytes_written,
    }


@app.post("/lights/raw")
def lights_raw(request: RawWrite):
    return serial_write(request)


@app.post("/lights/on")
def lights_on():
    return lights_set(LightRequest(enabled=True))


@app.post("/lights/off")
def lights_off():
    return lights_set(LightRequest(enabled=False))


@app.post("/beam/direction")
def beam_direction(request: BeamDirectionRequest):
    try:
        packet = beam_direction_command(request.direction)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "direction": packet.decode(),
        "angle_degrees": int(packet.decode(), 12) * 30,
        "beamformed_output_channel": 6,
        "bytes_written": _write_complete(packet, label="波束方向指令"),
    }


@app.post("/hotmap/threshold")
def hotmap_threshold(request: ThresholdRequest):
    packet = threshold_command(request.increase)
    return {
        "change": 50 if request.increase else -50,
        "official_command": packet.decode(),
        "bytes_written": _write_complete(packet, label="热力图阈值指令"),
    }


@app.post("/device/restore-defaults")
def restore_defaults():
    packet = restore_defaults_command()
    return {"official_command": "R", "bytes_written": _write_complete(packet, label="恢复默认指令")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
