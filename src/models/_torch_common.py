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
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_ranks = np.arange(1, len(score) + 1, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score) != 0) + 1]
    ends = np.r_[starts[1:], len(score)]
    for start, end in zip(starts, ends):
        sorted_ranks[start:end] = (start + 1 + end) / 2.0
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    n_pos = float(np.sum(y_true == 1))
    n_neg = float(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _auc_pr(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average precision with tied scores evaluated as one threshold block."""
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    total_pos = float(np.sum(y_true == 1))
    if total_pos == 0:
        return float("nan")
    score_sorted = score[order]
    block_end = np.r_[np.flatnonzero(np.diff(score_sorted) != 0), len(score_sorted) - 1]
    precision = tp[block_end] / np.maximum(tp[block_end] + fp[block_end], 1)
    recall = tp[block_end] / total_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> dict:
    pred = (prob >= thr).astype(np.int32)
    acc = float(np.mean(pred == y_true))
    tp = float(np.sum((pred == 1) & (y_true == 1)))
    fp = float(np.sum((pred == 1) & (y_true == 0)))
    fn = float(np.sum((pred == 0) & (y_true == 1)))
    tn = float(np.sum((pred == 0) & (y_true == 0)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "auc_roc": _auc_roc(y_true, prob),
        "auc_pr": _auc_pr(y_true, prob),
        "accuracy": acc,
        "prevalence": float(np.mean(y_true)),
        "cutoff": float(thr), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision_positive": prec, "recall_positive": rec, "f1_positive": f1,
        "brier": float(np.mean((prob - y_true) ** 2)),
        "log_loss": float(np.mean(-y_true * np.log(np.clip(prob, 1e-15, 1.0))
                                  - (1 - y_true) * np.log(np.clip(1 - prob, 1e-15, 1.0)))),
    }


def select_f1_cutoff(y_true: np.ndarray, prob: np.ndarray, candidates=None):
    candidates = candidates or np.arange(0.05, 1.0, 0.05)
    rows = [{"cutoff": float(c), **compute_metrics(y_true, prob, float(c))} for c in candidates]
    rows.sort(key=lambda x: (-x["f1_positive"], x["cutoff"]))
    return rows[0]["cutoff"], rows


def fit_isotonic(y_true: np.ndarray, score: np.ndarray):
    """Pool-adjacent-violators isotonic calibration with tie aggregation."""
    order = np.argsort(score, kind="mergesort")
    x, y = score[order], y_true[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse, weights=y)
    weights = np.bincount(inverse).astype(np.float64)
    means = sums / weights
    blocks = [[i, i, weights[i], means[i]] for i in range(len(unique_x))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][3] <= blocks[i + 1][3]:
            i += 1; continue
        left, right = blocks[i], blocks[i + 1]
        weight = left[2] + right[2]
        mean = (left[2] * left[3] + right[2] * right[3]) / weight
        blocks[i:i + 2] = [[left[0], right[1], weight, mean]]
        i = max(0, i - 1)
    fitted = np.empty(len(unique_x), dtype=np.float64)
    for start, end, _, mean in blocks: fitted[start:end + 1] = mean
    return unique_x.astype(np.float64), fitted


def apply_isotonic(score: np.ndarray, calibration_x: np.ndarray, calibration_y: np.ndarray):
    return np.clip(np.interp(score, calibration_x, calibration_y,
                             left=calibration_y[0], right=calibration_y[-1]), 0.0, 1.0)


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
