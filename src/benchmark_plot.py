"""
Turn ``output/tables/benchmark_results.csv`` into report figures:

  1. model_train_time.png   - train time per model (bars), for a chosen compute
  2. model_auc.png          - AUC-ROC & AUC-PR per model (bars)
  3. time_heatmap_train.png - compute-config x model heatmap of train time
  4. time_heatmap_predict.png - compute-config x model heatmap of predict time
  5. scaling_train_time.png - train time vs compute config for each model
                              (the scale-out / scale-up study)

If a run appears multiple times (re-runs), the most recent row per
(model, compute_config) is used.

Usage
-----
    python src/benchmark_plot.py
    python src/benchmark_plot.py --primary-compute local-all
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

RESULTS = os.path.join(config.TABLES_DIR, "benchmark_results.csv")


def fig_path(name: str) -> str:
    return os.path.join(config.FIGURES_DIR, name)


def load() -> pd.DataFrame:
    if not os.path.exists(RESULTS):
        raise SystemExit(f"No benchmark results at {RESULTS}. Run src/benchmark.py first.")
    df = pd.read_csv(RESULTS)
    # keep the most recent row per (model, compute_config)
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["model", "compute_config"], keep="last"
    )
    return df


def bar_train_time(df, primary):
    d = df[df["compute_config"] == primary].sort_values("train_secs")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(d["model"], d["train_secs"], color="#457b9d")
    ax.set_ylabel("Train time (s)")
    ax.set_xlabel("Model")
    ax.set_title(f"Training time by model ({primary})")
    for x, v in zip(d["model"], d["train_secs"]):
        ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path("model_train_time.png"), dpi=130)
    plt.close(fig)


def bar_auc_all_models(df):
    """AUC comparison across ALL models regardless of compute config.

    Picks one row per model (highest AUC-ROC) so MLlib (CPU) and deep (GPU)
    models appear together - the headline model-comparison figure.
    """
    d = df.sort_values("auc_roc", ascending=False).drop_duplicates(
        subset=["model"], keep="first"
    ).sort_values("auc_pr", ascending=False)
    x = np.arange(len(d))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, d["auc_roc"], w, label="AUC-ROC", color="#1d3557")
    ax.bar(x + w / 2, d["auc_pr"], w, label="AUC-PR", color="#e63946")
    ax.set_xticks(x)
    labels = [f"{m}\n({c})" for m, c in zip(d["model"], d["compute_config"])]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Area under curve")
    ax.set_xlabel("Model (compute)")
    ax.set_title("All models: AUC-ROC vs AUC-PR (best config per model)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path("model_auc_all.png"), dpi=130)
    plt.close(fig)


def bar_train_time_all_models(df):
    """Train time across ALL models (best/only config per model)."""
    d = df.sort_values("auc_roc", ascending=False).drop_duplicates(
        subset=["model"], keep="first"
    ).sort_values("train_secs")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2a9d8f" if c == "gpu" else "#457b9d" for c in d["compute_config"]]
    ax.bar(d["model"], d["train_secs"], color=colors)
    ax.set_ylabel("Train time (s)")
    ax.set_xlabel("Model")
    ax.set_title("Training time by model (green = GPU deep model, blue = MLlib CPU)")
    for xi, v in zip(range(len(d)), d["train_secs"]):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path("model_train_time_all.png"), dpi=130)
    plt.close(fig)


def bar_auc(df, primary):
    d = df[df["compute_config"] == primary].sort_values("auc_pr", ascending=False)
    if d.empty:
        return
    x = np.arange(len(d))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, d["auc_roc"], w, label="AUC-ROC", color="#1d3557")
    ax.bar(x + w / 2, d["auc_pr"], w, label="AUC-PR", color="#e63946")
    ax.set_xticks(x)
    ax.set_xticklabels(d["model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Area under curve")
    ax.set_xlabel("Model")
    ax.set_title(f"Model discrimination ({primary}): AUC-ROC vs AUC-PR")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path("model_auc.png"), dpi=130)
    plt.close(fig)


def heatmap(df, value_col, title, out):
    piv = df.pivot_table(index="compute_config", columns="model", values=value_col)
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(1.4 * len(piv.columns) + 3, 0.8 * len(piv.index) + 2))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Model")
    ax.set_ylabel("Compute config")
    ax.set_title(title)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(fig_path(out), dpi=130)
    plt.close(fig)


def scaling(df):
    """Train time vs compute config, one line per model (scale study)."""
    # order compute configs by parallelism if available
    order = (
        df[["compute_config", "default_parallelism"]]
        .drop_duplicates()
        .sort_values("default_parallelism")["compute_config"]
        .tolist()
    )
    if len(order) < 2:
        return  # nothing to scale over
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, g in df.groupby("model"):
        g = g.set_index("compute_config").reindex(order).reset_index()
        ax.plot(g["compute_config"], g["train_secs"], "o-", label=model)
    ax.set_xlabel("Compute config (increasing parallelism ->)")
    ax.set_ylabel("Train time (s)")
    ax.set_title("Training time vs compute config (scaling study)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path("scaling_train_time.png"), dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-compute", default="local-all",
                    help="Compute config to use for the per-model bar charts")
    args = ap.parse_args()

    config.ensure_dirs()
    df = load()
    print(f"loaded {len(df)} unique (model, compute) rows")

    bar_train_time(df, args.primary_compute)
    bar_auc(df, args.primary_compute)
    bar_auc_all_models(df)
    bar_train_time_all_models(df)
    heatmap(df, "train_secs", "Train time (s): compute-config x model", "time_heatmap_train.png")
    heatmap(df, "predict_secs", "Predict time (s): compute-config x model", "time_heatmap_predict.png")
    scaling(df)
    print("wrote figures to", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
