"""
Task 2 - plot model evaluation: AUC-ROC and AUC-PR versus rainfall threshold,
comparing the GBT model against the Logistic Regression baseline.

GBT per-threshold metrics are read from
``output/tables/model_metrics_by_threshold.csv`` (written by task2_train.py for
the most recently saved model - the GBT). The Logistic Regression baseline
numbers are passed in / recorded here so the comparison is reproducible even
though only one model's CSV is persisted at a time.

Usage
-----
    python src/task2_plot.py
    # optionally point at a different GBT metrics CSV:
    python src/task2_plot.py --gbt-csv output/tables/model_metrics_by_threshold.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

# Logistic Regression baseline (full-data run, test year = 2024).
# Recorded here because task2_train.py overwrites the single metrics CSV with
# whichever model ran last (the GBT). Update if the baseline is re-run.
LOGREG_METRICS = pd.DataFrame(
    {
        "threshold": [0.2, 1.0, 2.0, 5.0, 10.0],
        "auc_roc": [0.8791397510831742, 0.8743457831979586, 0.8650130000086687,
                    0.857024918752269, 0.861411226693028],
        "auc_pr": [0.5567194898451157, 0.4999181168881253, 0.4406600277941947,
                   0.3455543258655292, 0.25334142579015634],
    }
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gbt-csv",
        default=os.path.join(config.TABLES_DIR, "model_metrics_by_threshold.csv"),
    )
    args = ap.parse_args()

    config.ensure_dirs()
    gbt = pd.read_csv(args.gbt_csv).sort_values("threshold")
    lr = LOGREG_METRICS.sort_values("threshold")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # AUC-ROC
    axes[0].plot(gbt["threshold"], gbt["auc_roc"], "o-", color="#1d3557", label="GBT")
    axes[0].plot(lr["threshold"], lr["auc_roc"], "s--", color="#e63946", label="LogReg")
    axes[0].set_title("AUC-ROC vs rainfall threshold")
    axes[0].set_xlabel("Threshold (mm over next 30 min)")
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # AUC-PR
    axes[1].plot(gbt["threshold"], gbt["auc_pr"], "o-", color="#1d3557", label="GBT")
    axes[1].plot(lr["threshold"], lr["auc_pr"], "s--", color="#e63946", label="LogReg")
    axes[1].set_title("AUC-PR vs rainfall threshold")
    axes[1].set_xlabel("Threshold (mm over next 30 min)")
    axes[1].set_ylabel("AUC-PR (area under precision-recall)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Task 2 model evaluation (test year = 2024): GBT vs Logistic Regression")
    fig.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "model_auc_by_threshold.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote figure: {out}")

    # Also persist a combined comparison table for the report.
    comp = gbt[["threshold", "auc_roc", "auc_pr"]].rename(
        columns={"auc_roc": "gbt_auc_roc", "auc_pr": "gbt_auc_pr"}
    ).merge(
        lr[["threshold", "auc_roc", "auc_pr"]].rename(
            columns={"auc_roc": "logreg_auc_roc", "auc_pr": "logreg_auc_pr"}
        ),
        on="threshold",
    )
    comp_path = os.path.join(config.TABLES_DIR, "model_comparison_by_threshold.csv")
    comp.to_csv(comp_path, index=False)
    print(f"wrote table: {comp_path}")


if __name__ == "__main__":
    main()
