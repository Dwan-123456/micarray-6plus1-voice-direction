"""No-hardware contract tests for the MA-USB8 LED HTTP endpoint."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from layer1_input import api


class _FakeSerialDevice:
    def __init__(self, written: int):
        self.written = written
        self.packets: list[bytes] = []

    def write(self, packet: bytes) -> int:
        self.packets.append(packet)
        return self.written


class LightsApiTests(unittest.TestCase):
    @staticmethod
    def post_lights_set(enabled: bool) -> dict[str, object]:
        """Invoke the endpoint registered for POST /lights/set without HTTP I/O."""
        route = next(
            route
            for route in api.app.routes
            if getattr(route, "path", None) == "/lights/set"
            and "POST" in getattr(route, "methods", set())
        )
        return route.endpoint(api.LightRequest(enabled=enabled))

    def test_set_light_maps_enabled_to_official_e_and_E_commands(self):
        serial = _FakeSerialDevice(written=1)
        with patch.object(api, "serial_device", serial):
            on = self.post_lights_set(True)
            off = self.post_lights_set(False)

        self.assertEqual(on, {"enabled": True, "official_command": "E", "bytes_written": 1})
        self.assertEqual(off, {"enabled": False, "official_command": "e", "bytes_written": 1})
        self.assertEqual(serial.packets, [b"E", b"e"])

    def test_set_light_rejects_a_short_write(self):
        serial = _FakeSerialDevice(written=0)
        with patch.object(api, "serial_device", serial):
            with self.assertRaises(HTTPException) as raised:
                self.post_lights_set(True)

        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("0/1", raised.exception.detail)
        self.assertEqual(serial.packets, [b"E"])


if __name__ == "__main__":
    unittest.main()
