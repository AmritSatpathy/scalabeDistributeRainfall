"""Create report-ready calibration and positive-class evaluation figures."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_artifacts import RunContext, latest_artifact  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=None, help="Tracked training tables directory")
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="full")
    args = ap.parse_args()
    args.tables = args.tables or latest_artifact("train", "tables")
    if not args.tables:
        raise SystemExit("No tracked training tables. Run task2_train.py first.")
    run = RunContext("evaluation_plots", vars(args), hash_mode=args.hash_mode)
    run.record_input("training_tables", args.tables)
    out = run.artifact_path("figures"); os.makedirs(out, exist_ok=True)

    raw = pd.read_csv(os.path.join(args.tables, "reliability_raw.csv"))
    cal = pd.read_csv(os.path.join(args.tables, "reliability_isotonic.csv"))
    thresholds = sorted(cal["threshold"].unique())
    cols = 4; rows = math.ceil(len(thresholds) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.4 * rows), squeeze=False)
    for ax, threshold in zip(axes.flat, thresholds):
        for frame, label, style in ((raw, "Raw", "o--"), (cal, "Isotonic", "s-")):
            part = frame[frame["threshold"] == threshold]
            ax.plot(part["mean_probability"], part["observed_frequency"], style, label=label)
        ax.plot([0, 1], [0, 1], color="0.5", linewidth=1, linestyle=":")
        ax.set_title(f"q = {threshold:g} mm"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    for ax in axes.flat[len(thresholds):]: ax.axis("off")
    axes.flat[0].legend(); fig.suptitle("Reliability by rainfall threshold", y=1.01)
    fig.tight_layout(); fig.savefig(os.path.join(out, "reliability_by_threshold.png"), dpi=180, bbox_inches="tight"); plt.close(fig)

    metrics = pd.read_csv(os.path.join(args.tables, "metrics_by_threshold.csv"))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(metrics["threshold"], metrics["precision_positive"], "o-", label="Positive precision")
    ax.plot(metrics["threshold"], metrics["recall_positive"], "s-", label="Positive recall")
    ax.plot(metrics["threshold"], metrics["f1_positive"], "^-", label="Positive F1")
    ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("Rainfall threshold q (mm)")
    ax.set_ylabel("Metric"); ax.set_title("Positive-class performance at validation-selected cutoff")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(out, "positive_class_metrics.png"), dpi=180); plt.close(fig)

    baselines = pd.read_csv(os.path.join(args.tables, "baseline_metrics.csv"))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(metrics["threshold"], metrics["brier"], "o-", label="Calibrated GBT")
    for name, part in baselines.groupby("baseline"):
        ax.plot(part["threshold"], part["brier"], "o--", label=f"{name.title()} baseline")
    ax.set_xscale("log"); ax.set_xlabel("Rainfall threshold q (mm)"); ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Probability quality against simple baselines"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "brier_vs_baselines.png"), dpi=180); plt.close(fig)

    run.record_artifact("figures", out); run.finalize()
    print(f"evaluation figures -> {run.path}")


if __name__ == "__main__":
    main()
