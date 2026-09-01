# Git version-management workflow

> **历史归档快照**：本文记录旧完整架构时期的Git流程，包含`.venv`、CNN、数据管理和L3–L6命令，不适用于当前v1.4.3精简分支。当前操作以根`AGENTS.md`、`README.md`、`ENVIRONMENT.md`和`.venv-v1.4`为准。

## Repository boundary

The repository contains source code, configuration, documentation, tests,
curated regression audio and the released CNN model. Git LFS stores audio,
NumPy tables and model weights.

The complete `data/` directory, `.venv/`, caches, logs, catalogs, runtime
recordings and `legacy_reference_only/` remain local and are excluded by
`.gitignore`.

## Daily development

```powershell
git switch main
git pull --ff-only
git switch -c fix/short-description

# edit and test
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check app common config data_management gui ingest layer2_source_detection layer3_direction_signal layer4_speech_separation layer5_voice_classifier windowing scripts tests

git status
git add <reviewed paths>
git diff --cached --stat
git commit -m "fix(scope): concise description"
git push -u origin fix/short-description
```

Keep `main` runnable. Use short-lived `feat/`, `fix/`, `refactor/`, `test/` or
`docs/` branches and merge only after the full test suite passes.

## Adding curated test audio

Place reviewed clips under `tests/fixtures/audio/<layer>/`, update
`tests/fixtures/audio/manifest.json`, add a consuming test, then verify that Git
LFS recognizes the file:

```powershell
git check-attr filter -- tests/fixtures/audio/l2/example.wav
git lfs status
```

Never move a complete runtime session or Test Corpus into the repository.

## Updating the CNN

Update model weights, model manifest, preprocessing contract, license and model
tests together. Increase the model version rather than overwriting an unrelated
released model. Confirm weight files appear in `git lfs status` before commit.

## Rebuilding on another computer

```powershell
git clone <private-repository-url>
cd micarray-6plus1-voice-direction
git lfs pull
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock
.venv\Scripts\python.exe -m pytest -q
```

CUDA, PyTorch runtime installation and local recording data are restored
separately according to `ENVIRONMENT.md`; they are not Git assets.
