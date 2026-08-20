from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the project Python/CUDA runtime.")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    expected_python = project_root / ".venv" / "Scripts" / "python.exe"
    active_python = Path(sys.executable).resolve()
    if active_python != expected_python.resolve():
        fail(f"必须使用项目专用解释器 {expected_python}，当前为 {active_python}")
    if sys.version_info[:2] != (3, 12):
        fail(f"需要Python 3.12.x，当前为 {platform.python_version()}")

    try:
        import numpy as np
        import scipy
        import torch
        from PySide6 import QtCore
        import sounddevice
        import yaml
        import safetensors
        from common.config import load_config
    except Exception as exc:
        fail(f"运行依赖导入失败: {exc}")

    cuda_available = bool(torch.cuda.is_available())
    project_config = load_config(project_root / "config" / "config.yaml", environ={})
    if torch.__version__ != "2.12.1+cu132":
        fail(f"需要PyTorch 2.12.1+cu132，当前为 {torch.__version__}")
    if torch.version.cuda != "13.2":
        fail(f"需要PyTorch CUDA runtime 13.2，当前为 {torch.version.cuda}")
    if args.require_cuda and not cuda_available:
        fail("PyTorch无法使用CUDA；请检查NVIDIA驱动和cu132 wheel")

    report: dict[str, object] = {
        "python": platform.python_version(),
        "python_executable": str(active_python),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "pyside": QtCore.__version__,
        "sounddevice": sounddevice.__version__,
        "pyyaml": yaml.__version__,
        "safetensors": safetensors.__version__,
    }

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        query = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        report["nvidia_smi"] = query.stdout.strip() or query.stderr.strip()
        if args.require_cuda:
            if query.returncode != 0 or not query.stdout.strip():
                fail(f"nvidia-smi查询失败: {query.stderr.strip()}")
            driver_text = query.stdout.split(",", maxsplit=2)[1].strip()
            try:
                driver_major = int(driver_text.split(".", maxsplit=1)[0])
            except ValueError:
                fail(f"无法解析NVIDIA驱动版本: {driver_text}")
            if driver_major < 580:
                fail(f"CUDA 13.x要求NVIDIA驱动>=580，当前为 {driver_text}")
    elif args.require_cuda:
        fail("找不到nvidia-smi，无法验证NVIDIA驱动")

    if cuda_available:
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        report.update(
            {
                "gpu": props.name,
                "gpu_memory_gib": round(props.total_memory / 1024**3, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )
        target_arch = f"sm_{props.major}{props.minor}"
        compiled_arches = torch.cuda.get_arch_list()
        report["torch_cuda_arches"] = compiled_arches
        if target_arch not in compiled_arches:
            fail(f"PyTorch wheel不包含当前GPU架构 {target_arch}: {compiled_arches}")

        # Exercise the operations used by the real pipeline, not only device discovery.
        window_spec = project_config.downstream_audio_window
        waveform = torch.randn((7, window_spec.samples), device=device, dtype=torch.float32)
        window = torch.hann_window(960, periodic=True, device=device)
        spectrum = torch.stft(
            waveform,
            n_fft=1024,
            hop_length=480,
            win_length=960,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        if tuple(spectrum.shape) != (7, 513, window_spec.stft_frames) or spectrum.dtype != torch.complex64:
            fail(f"CUDA STFT契约错误: {tuple(spectrum.shape)} / {spectrum.dtype}")

        matrix = torch.randn((513, 7, 7), device=device, dtype=torch.complex64)
        covariance = matrix @ matrix.mH + 1e-2 * torch.eye(7, device=device)[None, :, :]
        steering = torch.randn((513, 7, 1), device=device, dtype=torch.complex64)
        solved = torch.linalg.solve(covariance, steering)
        if not bool(torch.isfinite(solved).all()):
            fail("CUDA complex64线性求解产生NaN/Inf")

        from layer4_voice_classifier import NvidiaMarbleNetPlugin
        l4 = NvidiaMarbleNetPlugin(
            "nv_marblenet_baseline_v1",
            project_root / "models" / "nv_marblenet_baseline_v1",
            device="cuda",
            window_spec=project_config.downstream_audio_window,
        )
        l4_output = l4.predict(np.zeros(
            (5, project_config.downstream_audio_window.samples), dtype=np.float32,
        ))
        if l4_output.probabilities.shape != (5,) or not np.isfinite(l4_output.probabilities).all():
            fail(f"CUDA MarbleNet波形前向契约错误: {l4_output.probabilities.shape}")
        torch.cuda.synchronize(device)
        report["cuda_marblenet_waveform_smoke"] = "PASS"

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("[PASS] 项目运行环境检查完成")


if __name__ == "__main__":
    main()
