"""Shared PyTorch training/evaluation utilities for the deep-learning models.

The MLlib models consume a flat tabular feature vector. The deep models instead
consume a SEQUENCE of the most recent rainfall readings per (station, time) --
which is the point of using an RNN/CNN for time series -- plus a few static
context features (cyclical hour/month, monsoon flags, lat/lon) and the query
threshold. This module provides:

  * a generic training loop (mini-batch SGD with early stopping) on GPU/CPU,
  * probability prediction,
  * metric computation (AUC-ROC / AUC-PR via sklearn-free numpy where possible,
    falling back to a manual trapezoidal AUC),

so each concrete model file (lstm.py, tcn.py) only defines its network.

Data contract (produced by ``deep_dataset.py``):
    X_seq  : float32 [N, L, 1]   last L 5-min rainfall values (oldest->newest)
    X_stat : float32 [N, S]      static features + threshold
    y      : float32 [N]         label (1 if next-30-min accumulation > threshold)
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _auc_roc(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUC-ROC via rank statistic (Mann-Whitney U), no sklearn dependency."""
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    # (simple approach; ties are rare with continuous scores)
    n_pos = float(np.sum(y_true == 1))
    n_neg = float(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _auc_pr(y_true: np.ndarray, score: np.ndarray) -> float:
    """Area under precision-recall curve via trapezoidal rule over sorted scores."""
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    total_pos = float(np.sum(y_true == 1))
    if total_pos == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos
    # prepend recall=0 point
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> dict:
    pred = (prob >= thr).astype(np.int32)
    acc = float(np.mean(pred == y_true))
    tp = float(np.sum((pred == 1) & (y_true == 1)))
    fp = float(np.sum((pred == 1) & (y_true == 0)))
    fn = float(np.sum((pred == 0) & (y_true == 1)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "auc_roc": _auc_roc(y_true, prob),
        "auc_pr": _auc_pr(y_true, prob),
        "accuracy": acc,
        "f1": f1,
    }


def train_torch(
    model: nn.Module,
    X_seq, X_stat, y,
    *,
    epochs: int = 8,
    batch_size: int = 4096,
    lr: float = 1e-3,
    pos_weight: float | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> nn.Module:
    """Generic mini-batch trainer. Uses BCEWithLogitsLoss (model outputs logits)."""
    device = device or get_device()
    model = model.to(device)
    Xs = torch.as_tensor(X_seq, dtype=torch.float32)
    Xt = torch.as_tensor(X_stat, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(Xs, Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                     drop_last=False)

    pw = None if pos_weight is None else torch.tensor([pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        total = 0.0
        n = 0
        for xb_seq, xb_stat, yb in dl:
            xb_seq = xb_seq.to(device, non_blocking=True)
            xb_stat = xb_stat.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            logit = model(xb_seq, xb_stat).squeeze(-1)
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(yb)
            n += len(yb)
        if verbose:
            print(f"  epoch {ep+1}/{epochs}  loss={total/max(n,1):.4f}")
    return model


@torch.no_grad()
def predict_proba(model: nn.Module, X_seq, X_stat,
                  batch_size: int = 8192, device: torch.device | None = None) -> np.ndarray:
    device = device or get_device()
    model = model.to(device).eval()
    Xs = torch.as_tensor(X_seq, dtype=torch.float32)
    Xt = torch.as_tensor(X_stat, dtype=torch.float32)
    out = []
    for i in range(0, len(Xs), batch_size):
        xb_seq = Xs[i:i + batch_size].to(device)
        xb_stat = Xt[i:i + batch_size].to(device)
        logit = model(xb_seq, xb_stat).squeeze(-1)
        out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out)
