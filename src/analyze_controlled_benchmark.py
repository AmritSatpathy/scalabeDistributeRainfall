"""Summarise repeated benchmark runs and join Spark event-log evidence."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_artifacts import RunContext, latest_artifact  # noqa: E402
from progress_utils import StepProgress, driver_progress  # noqa: E402
from summarize_eventlog import summarize  # noqa: E402

CORE_COUNTS = {
    "local-2": 2, "local-4": 4, "local-4-matched": 4, "local-8": 8,
    "standalone-2x2": 4, "standalone-2x2-matched": 4,
    "standalone-4x2": 8, "docker-2w": 4, "docker-2x2-matched": 4,
    "docker-3w": 6,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=None)
    ap.add_argument("--event-logs", nargs="*", default=None)
    ap.add_argument("--scaling-baseline", default="local-2")
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="full")
    args = ap.parse_args()
    args.results = args.results or [latest_artifact("benchmark_group", "benchmark_results")]
    args.event_logs = args.event_logs or [latest_artifact("benchmark_group", "spark_event_logs")]
    args.results = [x for x in args.results if x]
    args.event_logs = [x for x in args.event_logs if x]
    if not args.results:
        raise SystemExit("No controlled benchmark results found.")
    missing_results = [path for path in args.results if not Path(path).is_file()]
    if missing_results:
        raise SystemExit(
            "Controlled benchmark result files are missing: "
            + ", ".join(missing_results)
            + ". Complete the corresponding architecture benchmark commands first."
        )
    missing_events = [path for path in args.event_logs if not Path(path).exists()]
    if missing_events:
        raise SystemExit(
            "Spark event-log directories are missing: "
            + ", ".join(missing_events)
            + ". Complete the corresponding architecture benchmark commands first."
        )
    progress = StepProgress(5, "controlled architecture analysis")
    run = RunContext("benchmark_analysis", vars(args), hash_mode=args.hash_mode)
    for i, path in enumerate(args.results): run.record_input(f"benchmark_results_{i+1}", path)
    for i, path in enumerate(args.event_logs): run.record_input(f"spark_event_logs_{i+1}", path)
    raw = pd.concat([pd.read_csv(path) for path in args.results], ignore_index=True)
    progress.update("benchmark result files loaded")
    raw["warmup"] = raw["warmup"].astype(str).str.lower().isin(["true", "1"])
    measured = raw[~raw["warmup"]].copy()
    measured["throughput_train_rows_per_s"] = measured["n_train"] / measured["train_secs"]
    invariant_cols = [
        column for column in (
            "model", "fold_id", "train_start_year", "validation_year", "test_year",
            "n_train", "n_validation", "n_test", "training_thresholds",
            "evaluation_thresholds", "feature_version", "features_sha256",
            "feature_contract_sha256",
        ) if column in measured.columns
    ]
    fairness = pd.DataFrame([
        {
            "field": column,
            "distinct_values": int(measured[column].nunique(dropna=False)),
            "controlled": bool(measured[column].nunique(dropna=False) == 1),
            "values": " | ".join(map(str, measured[column].drop_duplicates().tolist())),
        }
        for column in invariant_cols
        if column != "model"
    ])
    progress.update("fairness controls audited")
    event_rows = []
    if args.event_logs:
        with driver_progress("summarising Spark event logs"):
            for row in measured.itertuples():
                path = next((os.path.join(root, row.compute_config, row.model, f"rep_{int(row.repetition)}")
                             for root in args.event_logs
                             if os.path.exists(os.path.join(root, row.compute_config, row.model, f"rep_{int(row.repetition)}"))), None)
                if path:
                    totals, _ = summarize(path)
                    event_rows.append({"compute_config": row.compute_config, "model": row.model,
                                       "repetition": row.repetition, **totals})
    if event_rows:
        measured = measured.merge(pd.DataFrame(event_rows), on=["compute_config", "model", "repetition"], how="left")
    progress.update("Spark execution evidence joined")
    summary = measured.groupby(["compute_config", "model"], as_index=False).agg(
        repetitions=("train_secs", "count"), train_median_s=("train_secs", "median"),
        train_min_s=("train_secs", "min"), train_max_s=("train_secs", "max"),
        predict_median_s=("predict_secs", "median"),
        throughput_median_rows_s=("throughput_train_rows_per_s", "median"),
        auc_pr_median=("auc_pr", "median"), brier_median=("brier", "median"),
    )
    summary["cores"] = summary["compute_config"].map(CORE_COUNTS)
    for model in summary["model"].unique():
        base = summary[(summary.model == model) & (summary.compute_config == args.scaling_baseline)]
        if not base.empty:
            t0, p0 = base.iloc[0].train_median_s, base.iloc[0].cores
            mask = summary.model == model
            summary.loc[mask, "speedup_vs_baseline"] = t0 / summary.loc[mask, "train_median_s"]
            summary.loc[mask, "parallel_efficiency"] = summary.loc[mask, "speedup_vs_baseline"] / (summary.loc[mask, "cores"] / p0)
    metric_stability = measured.groupby("model", as_index=False).agg(
        auc_pr_min=("auc_pr", "min"), auc_pr_max=("auc_pr", "max"),
        brier_min=("brier", "min"), brier_max=("brier", "max"),
        f1_positive_min=("f1_positive", "min"),
        f1_positive_max=("f1_positive", "max"),
    )
    metric_stability["auc_pr_range"] = metric_stability.auc_pr_max - metric_stability.auc_pr_min
    metric_stability["brier_range"] = metric_stability.brier_max - metric_stability.brier_min
    metric_stability["f1_positive_range"] = (
        metric_stability.f1_positive_max - metric_stability.f1_positive_min
    )
    progress.update("runtime and metric summaries calculated")
    tables = run.artifact_path("tables"); figures = run.artifact_path("figures")
    os.makedirs(tables, exist_ok=True); os.makedirs(figures, exist_ok=True)
    measured.to_csv(os.path.join(tables, "controlled_benchmark_with_events.csv"), index=False)
    summary.to_csv(os.path.join(tables, "controlled_benchmark_summary.csv"), index=False)
    fairness.to_csv(os.path.join(tables, "architecture_fairness_audit.csv"), index=False)
    metric_stability.to_csv(os.path.join(tables, "architecture_metric_stability.csv"), index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, group in summary.groupby("model"):
        ax.plot(group["compute_config"], group["train_median_s"], "o-", label=model)
        for _, row in group.iterrows():
            ax.vlines(row.compute_config, row.train_min_s, row.train_max_s, color="0.45", linewidth=2)
    ax.set_ylabel("Training time (s), median with min–max range"); ax.set_xlabel("Compute configuration")
    ax.tick_params(axis="x", rotation=25); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(figures, "controlled_training_time.png"), dpi=180); plt.close(fig)
    run.record_artifact("tables", tables); run.record_artifact("figures", figures); run.finalize()
    progress.finish("tables, figure and manifest written")
    print(summary.to_string(index=False)); print(f"analysis -> {run.path}")


if __name__ == "__main__":
    main()
