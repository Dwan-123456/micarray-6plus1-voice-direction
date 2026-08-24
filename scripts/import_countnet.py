from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import h5py
import numpy as np
import torch

from layer1_input.countnet_crnn import CountNetCrnn


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(upstream: Path, output: Path) -> str:
    source_model = upstream / "models" / "CRNN.h5"
    source_scaler = upstream / "models" / "scaler.npz"
    model = CountNetCrnn().eval()
    with h5py.File(source_model, "r") as source, torch.no_grad():
        weights = source["model_weights"]
        for name in ("conv1", "conv2", "conv3", "conv4"):
            layer = getattr(model, name)
            layer.weight.copy_(torch.from_numpy(np.asarray(weights[name][f"{name}_W"])))
            layer.bias.copy_(torch.from_numpy(np.asarray(weights[name][f"{name}_b"])))
        for gate in ("i", "f", "c", "o"):
            getattr(model, f"lstm_w_{gate}").copy_(
                torch.from_numpy(np.asarray(weights["lstm_1"][f"lstm_1_W_{gate}"]))
            )
            getattr(model, f"lstm_u_{gate}").copy_(
                torch.from_numpy(np.asarray(weights["lstm_1"][f"lstm_1_U_{gate}"]))
            )
            getattr(model, f"lstm_b_{gate}").copy_(
                torch.from_numpy(np.asarray(weights["lstm_1"][f"lstm_1_b_{gate}"]))
            )
        model.dense.weight.copy_(
            torch.from_numpy(np.asarray(weights["dense_1"]["dense_1_W"]).T.copy())
        )
        model.dense.bias.copy_(torch.from_numpy(np.asarray(weights["dense_1"]["dense_1_b"])))
    with np.load(source_scaler) as scaler, torch.no_grad():
        model.scaler_mean.copy_(torch.from_numpy(np.asarray(scaler["arr_0"], np.float32)))
        model.scaler_scale.copy_(torch.from_numpy(np.asarray(scaler["arr_1"], np.float32)))
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    scripted.save(str(output))
    loaded = torch.jit.load(io.BytesIO(output.read_bytes()), map_location="cpu").eval()
    with torch.inference_mode():
        smoke = loaded(torch.zeros((1, 80_000), dtype=torch.float32))
    if smoke.shape != (1, 11) or not torch.isfinite(smoke).all():
        raise RuntimeError("converted CountNet failed its TorchScript smoke test")
    print(f"upstream CRNN.h5 sha256={sha256(source_model)}")
    print(f"upstream scaler.npz sha256={sha256(source_scaler)}")
    print(f"converted model.pt sha256={sha256(output)}")
    return sha256(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    convert(arguments.upstream.resolve(), arguments.output.resolve())
