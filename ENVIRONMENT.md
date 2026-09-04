# v1.5.1精简环境

使用Python 3.12和项目专用`.venv-v1.4`。运行依赖仅为NumPy、SciPy、PyYAML、sounddevice、pyserial、Pydantic和PySide6；测试额外使用pytest、pytest-cov、ruff。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_vscode_env.ps1
```

VS Code解释器由`.vscode/settings.json`固定。`scripts/check_runtime_env.py`会拒绝torch、onnxruntime、safetensors和spectralcluster，避免误用旧L4～L6环境。
