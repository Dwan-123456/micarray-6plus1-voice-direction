"""Install the pinned GI-DOAEnet source/checkpoint into a Git-ignored directory."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
import zipfile


REVISION = "af865978c783f309fc929f0f2499769a1c5499d5"
CHECKPOINT_SHA256 = "d465d2ccf0b7f2d1603186db3667e8c7b7a21c7eb0b8a126173b3292441f9fe8"
ARCHIVE_URL = f"https://github.com/BaekMS/GI-DOAEnet/archive/{REVISION}.zip"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acknowledge-upstream-terms", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_upstream_terms:
        parser.error("upstream has no LICENSE file; pass --acknowledge-upstream-terms for local installation")
    root = Path(__file__).resolve().parents[1]
    destination = root / "models" / "gi_doaenet_pm_v1" / "upstream"
    destination.mkdir(parents=True, exist_ok=True)
    with urlopen(ARCHIVE_URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        safe_root = destination.resolve()
        for item in archive.infolist():
            target = (destination / item.filename).resolve()
            if safe_root not in target.parents and target != safe_root:
                raise RuntimeError("unsafe path in upstream archive")
        archive.extractall(destination)
    checkpoints = tuple(destination.glob("*/pretrained/GI_DOAEnet_PM.tar"))
    if len(checkpoints) != 1:
        raise RuntimeError("pinned archive did not contain exactly one PM checkpoint")
    digest = hashlib.sha256(checkpoints[0].read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {digest}")
    print(f"Installed GI-DOAEnet {REVISION} at {checkpoints[0].parents[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
