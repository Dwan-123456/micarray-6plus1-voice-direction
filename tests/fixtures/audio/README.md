# Curated audio fixtures

This directory contains only short, reviewed recordings required by automated
tests. Runtime sessions, scratch recordings and the full local Test Corpus must
remain under `data/` and must not be copied here automatically.

Before adding a fixture:

1. Remove unrelated speech and private information.
2. Keep only the shortest segment needed by one regression test.
3. Record the channel layout, sample rate, expected angle/class and usage rights
   in `manifest.json`.
4. Compute and store the file SHA-256.
5. Add or update an automated test that consumes the fixture.
6. Treat an accepted fixture as immutable; create a new fixture ID/version when
   its content changes.

WAV and FLAC files in this directory are stored through Git LFS.
