from __future__ import annotations

import torch
from torch import nn


class CountNetCrnn(nn.Module):
    """PyTorch inference port of faroit/CountNet's Keras-1.2.2 CRNN."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 128, 3)
        self.conv4 = nn.Conv2d(128, 64, 3)
        self.dense = nn.Linear(1040, 11)
        self.register_buffer("scaler_mean", torch.zeros(201))
        self.register_buffer("scaler_scale", torch.ones(201))
        self.register_buffer("stft_window", torch.hann_window(400, periodic=True))
        for gate in ("i", "f", "c", "o"):
            self.register_parameter(f"lstm_w_{gate}", nn.Parameter(torch.empty(1280, 40)))
            self.register_parameter(f"lstm_u_{gate}", nn.Parameter(torch.empty(40, 40)))
            self.register_parameter(f"lstm_b_{gate}", nn.Parameter(torch.empty(40)))

    @staticmethod
    def _hard_sigmoid(value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(value * 0.2 + 0.5, 0.0, 1.0)

    def forward(self, waveform_16k: torch.Tensor) -> torch.Tensor:
        if waveform_16k.dim() != 2 or waveform_16k.size(1) != 80_000:
            raise RuntimeError("CountNet requires [batch,80000] 16 kHz waveform")
        spectrum = torch.stft(
            waveform_16k,
            n_fft=400,
            hop_length=160,
            win_length=400,
            window=self.stft_window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        features = spectrum.abs().transpose(1, 2)[:, :500, :]
        features = (features - self.scaler_mean) / self.scaler_scale
        mean_norm = torch.linalg.vector_norm(features, dim=2).mean(dim=1, keepdim=True)
        features = features / torch.clamp(mean_norm.unsqueeze(2), min=2.220446049250313e-16)
        value = features.unsqueeze(1)
        value = torch.relu(self.conv1(value))
        value = torch.relu(self.conv2(value))
        value = torch.max_pool2d(value, 3, 3)
        value = torch.relu(self.conv3(value))
        value = torch.relu(self.conv4(value))
        value = torch.max_pool2d(value, 3, 3)
        value = value.permute(0, 2, 1, 3).reshape(value.size(0), 53, 1280)

        hidden = torch.zeros((value.size(0), 40), dtype=value.dtype, device=value.device)
        cell = torch.zeros_like(hidden)
        outputs = torch.jit.annotate(list[torch.Tensor], [])
        for index in range(53):
            item = value[:, index, :]
            input_gate = self._hard_sigmoid(item @ self.lstm_w_i + hidden @ self.lstm_u_i + self.lstm_b_i)
            forget_gate = self._hard_sigmoid(item @ self.lstm_w_f + hidden @ self.lstm_u_f + self.lstm_b_f)
            candidate = torch.tanh(item @ self.lstm_w_c + hidden @ self.lstm_u_c + self.lstm_b_c)
            output_gate = self._hard_sigmoid(item @ self.lstm_w_o + hidden @ self.lstm_u_o + self.lstm_b_o)
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * torch.tanh(cell)
            outputs.append(hidden)
        recurrent = torch.stack(outputs, dim=1).transpose(1, 2)
        pooled = torch.max_pool1d(recurrent, 2, 2).transpose(1, 2).reshape(value.size(0), 1040)
        return self.dense(pooled)
