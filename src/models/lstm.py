"""LSTM sequence model (PyTorch, GPU) for rainfall nowcasting.

Consumes the last L 5-minute rainfall readings per (station, time) as a
sequence, plus static context features (cyclical time, monsoon flags, lat/lon)
and the query threshold. An LSTM encodes the sequence; its final hidden state
is concatenated with the static features and passed through an MLP head to a
single logit -> P(next-30-min accumulation > threshold).

This is a non-MLlib model (framework="torch"); it is run by task2_deep.py, not
the MLlib orchestrator, but it is registered in the same models registry so the
benchmark study can reference it uniformly.
"""
from __future__ import annotations

from torch import nn
import torch

from .base import ModelSpec


class LSTMNet(nn.Module):
    def __init__(self, n_static: int, hidden: int = 32, num_layers: int = 1,
                 seq_feat: int = 1, head: int = 32, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_feat, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, head),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head, 1),
        )

    def forward(self, x_seq, x_stat):
        # x_seq: [B, L, seq_feat]; x_stat: [B, n_static]
        _, (h_n, _) = self.lstm(x_seq)
        h_last = h_n[-1]                      # [B, hidden]
        z = torch.cat([h_last, x_stat], dim=1)
        return self.head(z)                   # [B, 1] logit


def build_torch(n_static: int, **kw) -> nn.Module:
    return LSTMNet(n_static=n_static, **kw)


SPEC = ModelSpec(
    name="lstm",
    framework="torch",
    description="LSTM over recent rainfall sequence (GPU)",
    build_torch=build_torch,
)
