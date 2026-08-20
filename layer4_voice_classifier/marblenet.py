from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F


class _ConvBn(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, *, stride: int = 1,
                 dilation: int = 1, separable: bool = True) -> None:
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        if separable:
            self.depthwise = nn.Conv1d(in_channels, in_channels, kernel, stride=stride,
                                       padding=padding, dilation=dilation, groups=in_channels, bias=False)
            self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        else:
            self.depthwise = None
            self.pointwise = nn.Conv1d(in_channels, out_channels, kernel, stride=stride,
                                       padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)

    def forward(self, x: Tensor) -> Tensor:
        if self.depthwise is not None:
            x = self.depthwise(x)
        return self.bn(self.pointwise(x))


class _JasperBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, repeat: int, *,
                 stride: int = 1, dilation: int = 1, residual: bool = False,
                 separable: bool = True) -> None:
        super().__init__()
        self.units = nn.ModuleList([
            _ConvBn(in_channels if index == 0 else out_channels, out_channels, kernel,
                    stride=stride, dilation=dilation, separable=separable)
            for index in range(repeat)
        ])
        self.residual = _ConvBn(in_channels, out_channels, 1, separable=False) if residual else None

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        for index, unit in enumerate(self.units):
            x = unit(x)
            if index + 1 < len(self.units):
                x = F.relu(x)
        if self.residual is not None:
            x = x + self.residual(residual)
        return F.relu(x)


class NvidiaFrameVadMarbleNet(nn.Module):
    """Dependency-light inference port of NVIDIA's NeMo frame VAD MarbleNet.

    It preserves the official feature extractor and network tensors. The public
    application adapter supplies configured 48 kHz / 80 or 160 ms audio and this module consumes
    the internally resampled 16 kHz waveform.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("stft_window", torch.empty(400))
        self.register_buffer("mel_filterbank", torch.empty(1, 80, 257))
        self.blocks = nn.ModuleList([
            _JasperBlock(80, 128, 11, 1, stride=2),
            _JasperBlock(128, 64, 13, 2, residual=True),
            _JasperBlock(64, 64, 15, 2, residual=True),
            _JasperBlock(64, 64, 17, 2, residual=True),
            _JasperBlock(64, 128, 29, 1, dilation=2),
            _JasperBlock(128, 128, 1, 1, separable=False),
        ])
        self.decoder = nn.Linear(128, 2)

    @classmethod
    def from_artifact(cls, artifact: str | Path, device: str | torch.device = "cpu") -> "NvidiaFrameVadMarbleNet":
        artifact = Path(artifact)
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        if manifest["architecture_id"] != "nvidia_frame_vad_marblenet_v2.0_native_v1":
            raise ValueError("unsupported MarbleNet artifact architecture")
        tensors = load_file(str(artifact / manifest["weights_file"]), device="cpu")
        model = cls()
        model._load_nemo_tensors(tensors)
        return model.to(device).eval()

    @staticmethod
    def _copy(target: Tensor, source: Tensor) -> None:
        if target.shape != source.shape:
            raise ValueError(f"weight shape mismatch: expected {tuple(target.shape)}, got {tuple(source.shape)}")
        target.copy_(source)

    def _load_unit(self, unit: _ConvBn, tensors: dict[str, Tensor], prefix: str,
                   conv_index: int, bn_index: int) -> None:
        with torch.no_grad():
            if unit.depthwise is not None:
                self._copy(unit.depthwise.weight, tensors[f"{prefix}.mconv.{conv_index}.conv.weight"])
                self._copy(unit.pointwise.weight, tensors[f"{prefix}.mconv.{conv_index + 1}.conv.weight"])
            else:
                self._copy(unit.pointwise.weight, tensors[f"{prefix}.mconv.{conv_index}.conv.weight"])
            bn_prefix = f"{prefix}.mconv.{bn_index}"
            for name in ("weight", "bias", "running_mean", "running_var"):
                self._copy(getattr(unit.bn, name), tensors[f"{bn_prefix}.{name}"])

    def _load_nemo_tensors(self, tensors: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self._copy(self.stft_window, tensors["preprocessor.featurizer.window"])
            self._copy(self.mel_filterbank, tensors["preprocessor.featurizer.fb"])
        for block_index, block in enumerate(self.blocks):
            prefix = f"encoder.encoder.{block_index}"
            separable = block.units[0].depthwise is not None
            width = 5 if separable else 4
            for repeat_index, unit in enumerate(block.units):
                base = repeat_index * width
                self._load_unit(unit, tensors, prefix, base, base + (2 if separable else 1))
            if block.residual is not None:
                residual = block.residual
                with torch.no_grad():
                    self._copy(residual.pointwise.weight, tensors[f"{prefix}.res.0.0.conv.weight"])
                    for name in ("weight", "bias", "running_mean", "running_var"):
                        self._copy(getattr(residual.bn, name), tensors[f"{prefix}.res.0.1.{name}"])
        with torch.no_grad():
            self._copy(self.decoder.weight, tensors["decoder.layer0.weight"])
            self._copy(self.decoder.bias, tensors["decoder.layer0.bias"])

    def preprocess(self, audio_16k: Tensor) -> tuple[Tensor, Tensor]:
        if audio_16k.ndim != 2:
            raise ValueError("MarbleNet input must be [batch,samples]")
        lengths = torch.full((audio_16k.shape[0],), audio_16k.shape[1], dtype=torch.long, device=audio_16k.device)
        audio_16k = torch.cat((audio_16k[:, :1], audio_16k[:, 1:] - 0.97 * audio_16k[:, :-1]), dim=1)
        spectrum = torch.stft(audio_16k.float(), n_fft=512, hop_length=160, win_length=400,
                              center=True, window=self.stft_window.float(), return_complex=True)
        power = spectrum.abs().pow(2)
        features = torch.matmul(self.mel_filterbank.to(power.dtype), power)
        features = torch.log(features + 2**-24)
        feature_lengths = torch.div(lengths, 160, rounding_mode="floor") + 1
        if features.shape[-1] % 2:
            features = F.pad(features, (0, 1))
        return features, feature_lengths

    def forward(self, audio_16k: Tensor) -> tuple[Tensor, Tensor]:
        x, lengths = self.preprocess(audio_16k)
        for block in self.blocks:
            # NeMo conv_mask zeros samples outside each valid sequence. All L4
            # windows have equal length, so only the preprocessor efficiency pad
            # needs masking and it cannot reach a complete output frame here.
            x = block(x)
        output_lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        logits = self.decoder(x.transpose(1, 2))
        return logits, output_lengths

