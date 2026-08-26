from __future__ import annotations

import hashlib
import json
from pathlib import Path

from layer6_speaker_consolidation.models import (
    _MODEL_HASH_CHUNK_BYTES,
    _verified_file,
)


def test_l6_model_hashing_uses_bounded_reads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = b"v" * (_MODEL_HASH_CHUNK_BYTES + 29)
    weights = tmp_path / "weights.bin"
    weights.write_bytes(payload)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "test_l6_artifact_v1",
                "weights_file": weights.name,
                "weights_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    original_open = Path.open
    read_sizes: list[int] = []

    class _BoundedReader:
        def __init__(self, source) -> None:
            self._source = source

        def __enter__(self):
            self._source.__enter__()
            return self

        def __exit__(self, *args):
            return self._source.__exit__(*args)

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return self._source.read(size)

    def bounded_open(path: Path, *args, **kwargs):
        source = original_open(path, *args, **kwargs)
        if path == weights and args and args[0] == "rb":
            return _BoundedReader(source)
        return source

    monkeypatch.setattr(Path, "open", bounded_open)

    manifest, model_path = _verified_file(tmp_path, "test_l6_artifact_v1")

    assert model_path == weights
    assert manifest["weights_sha256"] == hashlib.sha256(payload).hexdigest()
    assert read_sizes
    assert set(read_sizes) == {_MODEL_HASH_CHUNK_BYTES}
