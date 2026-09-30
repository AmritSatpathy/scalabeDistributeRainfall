"""1D-CNN / Temporal Convolution sequence model (PyTorch, GPU).

A convolutional alternative to the LSTM: stacked 1D convolutions over the
rainfall sequence capture local temporal patterns and are cheaper/faster to
train than a recurrent net. Same input/output contract as the LSTM so it slots
into the same deep runner and benchmark.
"""
from __future__ import annotations

import torch
from torch import nn

from .base import ModelSpec


class TCNNet(nn.Module):
    def __init__(self, n_static: int, channels: int = 32, seq_feat: int = 1,
                 head: int = 32, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(seq_feat, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),          # global temporal pooling -> [B, C, 1]
        )
        self.head = nn.Sequential(
            nn.Linear(channels + n_static, head),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head, 1),
        )

    def forward(self, x_seq, x_stat):
        # x_seq: [B, L, seq_feat] -> conv wants [B, seq_feat, L]
        x = x_seq.transpose(1, 2)
        c = self.conv(x).squeeze(-1)          # [B, channels]
        z = torch.cat([c, x_stat], dim=1)
        return self.head(z)


def build_torch(n_static: int, **kw) -> nn.Module:
    return TCNNet(n_static=n_static, **kw)


SPEC = ModelSpec(
    name="tcn",
    framework="torch",
    description="1D-CNN / temporal conv over rainfall sequence (GPU)",
    build_torch=build_torch,
)
