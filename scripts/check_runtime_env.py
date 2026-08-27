from __future__ import annotations

import importlib.util
import json
import sys


def main() -> int:
    required = ("numpy", "scipy", "yaml", "sounddevice", "serial", "pydantic", "PySide6")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    forbidden = [name for name in ("torch", "onnxruntime", "safetensors", "spectralcluster")
                 if importlib.util.find_spec(name) is not None]
    report = {
        "python": sys.version.split()[0],
        "required_modules_available": not missing,
        "missing": missing,
        "neural_network_packages_present": forbidden,
        "profile": "v1.4-l1-l2-only",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if missing or forbidden else 0


if __name__ == "__main__":
    raise SystemExit(main())
