"""CAMPPlus architecture adapted from ModelScope 3D-Speaker (Apache-2.0)."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as functional
from torch import nn


def _nonlinear(spec: str, channels: int) -> nn.Sequential:
    result = nn.Sequential()
    for name in spec.split("-"):
        if name == "relu":
            result.add_module("relu", nn.ReLU(inplace=True))
        elif name == "batchnorm":
            result.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            result.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f"unsupported CAMPPlus nonlinearity: {name}")
    return result


class _BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = functional.relu(self.bn1(self.conv1(value)))
        output = self.bn2(self.conv2(output)) + self.shortcut(value)
        return functional.relu(output)


class _FCM(nn.Module):
    def __init__(self, feat_dim: int = 80, channels: int = 32) -> None:
        super().__init__()
        self.in_planes = channels
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer1 = self._make_layer(channels, 2, 2)
        self.layer2 = self._make_layer(channels, 2, 2)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.out_channels = channels * (feat_dim // 8)

    def _make_layer(self, planes: int, count: int, stride: int) -> nn.Sequential:
        layers = []
        for item_stride in [stride, *([1] * (count - 1))]:
            layers.append(_BasicResBlock(self.in_planes, planes, item_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = functional.relu(self.bn1(self.conv1(value.unsqueeze(1))))
        output = self.layer2(self.layer1(output))
        output = functional.relu(self.bn2(self.conv2(output)))
        return output.reshape(output.shape[0], output.shape[1] * output.shape[2], output.shape[3])


class _TDNN(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel: int, *, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        padding = (kernel - 1) // 2 * dilation
        self.linear = nn.Conv1d(inputs, outputs, kernel, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.nonlinear = _nonlinear("batchnorm-relu", outputs)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(value))


class _CAMLayer(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.linear_local = nn.Conv1d(inputs, outputs, kernel, padding=(kernel - 1) // 2 * dilation, dilation=dilation, bias=False)
        self.linear1 = nn.Conv1d(inputs, inputs // 2, 1)
        self.linear2 = nn.Conv1d(inputs // 2, outputs, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        local = self.linear_local(value)
        pooled = functional.avg_pool1d(value, 100, stride=100, ceil_mode=True)
        pooled = pooled.unsqueeze(-1).expand(*pooled.shape, 100).reshape(*pooled.shape[:-2], -1)[..., : value.shape[-1]]
        context = functional.relu(self.linear1(value.mean(-1, keepdim=True) + pooled))
        return local * torch.sigmoid(self.linear2(context))


class _CAMDenseLayer(nn.Module):
    def __init__(self, inputs: int, outputs: int, bottleneck: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.nonlinear1 = _nonlinear("batchnorm-relu", inputs)
        self.linear1 = nn.Conv1d(inputs, bottleneck, 1, bias=False)
        self.nonlinear2 = _nonlinear("batchnorm-relu", bottleneck)
        self.cam_layer = _CAMLayer(bottleneck, outputs, kernel, dilation)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.cam_layer(self.nonlinear2(self.linear1(self.nonlinear1(value))))


class _CAMDenseBlock(nn.ModuleList):
    def __init__(self, count: int, inputs: int, outputs: int, bottleneck: int, kernel: int, dilation: int) -> None:
        super().__init__()
        for index in range(count):
            self.add_module(
                f"tdnnd{index + 1}",
                _CAMDenseLayer(inputs + index * outputs, outputs, bottleneck, kernel, dilation),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self:
            value = torch.cat((value, layer(value)), dim=1)
        return value


class _Transit(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.nonlinear = _nonlinear("batchnorm-relu", inputs)
        self.linear = nn.Conv1d(inputs, outputs, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(self.nonlinear(value))


class _StatsPool(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.cat((value.mean(-1), value.std(-1)), dim=-1)


class _Dense(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.linear = nn.Conv1d(inputs, outputs, 1, bias=False)
        self.nonlinear = _nonlinear("batchnorm_", outputs)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(value.unsqueeze(-1)).squeeze(-1))


class CAMPPlus(nn.Module):
    def __init__(self, feat_dim: int = 80, embedding_size: int = 192) -> None:
        super().__init__()
        self.head = _FCM(feat_dim)
        channels = self.head.out_channels
        modules: OrderedDict[str, nn.Module] = OrderedDict((
            ("tdnn", _TDNN(channels, 128, 5, stride=2)),
        ))
        channels = 128
        for index, (count, dilation) in enumerate(((12, 1), (24, 2), (16, 2)), start=1):
            modules[f"block{index}"] = _CAMDenseBlock(count, channels, 32, 128, 3, dilation)
            channels += count * 32
            modules[f"transit{index}"] = _Transit(channels, channels // 2)
            channels //= 2
        modules["out_nonlinear"] = _nonlinear("batchnorm-relu", channels)
        modules["stats"] = _StatsPool()
        modules["dense"] = _Dense(channels * 2, embedding_size)
        self.xvector = nn.Sequential(modules)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.xvector(self.head(features.permute(0, 2, 1)))
