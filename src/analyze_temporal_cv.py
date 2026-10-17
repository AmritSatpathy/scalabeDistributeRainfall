"""Summarise expanding-window Task 2 results across folds."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_artifacts import RunContext, latest_artifact, publish_tree  # noqa: E402
from progress_utils import StepProgress  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=None)
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="full")
    ap.add_argument("--publish-legacy", action="store_true")
    args = ap.parse_args()
    tracked_results = latest_artifact("temporal_cv", "temporal_cv_results")
    if args.results and not Path(args.results).is_file():
        if tracked_results and Path(tracked_results).is_file():
            print(
                f"Requested results file is absent: {args.results}\n"
                f"Using tracked immutable results instead: {tracked_results}",
                flush=True,
            )
            args.results = tracked_results
        else:
            raise SystemExit(
                f"Temporal CV results do not exist at {args.results}. "
                "Run src/task2_temporal_cv.py successfully before the analysis command."
            )
    args.results = args.results or tracked_results
    if not args.results or not Path(args.results).is_file():
        raise SystemExit(
            "No successful temporal CV result artifact was found. "
            "Run src/task2_temporal_cv.py first."
        )

    progress = StepProgress(4, "temporal CV analysis")
    run = RunContext("temporal_cv_analysis", vars(args), hash_mode=args.hash_mode)
    run.record_input("temporal_cv_results", args.results)
    raw = pd.read_csv(args.results)
    progress.update("loaded result rows")
    required = {"model", "fold_id", "auc_pr", "auc_roc", "brier", "f1_positive", "train_secs"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Temporal CV results are missing columns: {sorted(missing)}")
    if raw.duplicated(["model", "fold_id"]).any():
        raise ValueError("Temporal CV contains duplicate model-fold rows")
    progress.update("validated model-fold uniqueness")

    summary = raw.groupby("model", as_index=False).agg(
        folds=("fold_id", "nunique"),
        auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
        auc_roc_mean=("auc_roc", "mean"), brier_mean=("brier", "mean"),
        brier_std=("brier", "std"), f1_positive_mean=("f1_positive", "mean"),
        train_median_s=("train_secs", "median"), train_total_s=("train_secs", "sum"),
        predict_median_s=("predict_secs", "median"),
    )
    summary = summary.sort_values(
        ["auc_pr_mean", "brier_mean", "train_median_s"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    summary.insert(0, "predictive_rank", range(1, len(summary) + 1))
    progress.update("calculated cross-fold rankings")

    tables = run.artifact_path("tables")
    os.makedirs(tables, exist_ok=True)
    raw.to_csv(os.path.join(tables, "temporal_cv_by_fold.csv"), index=False)
    summary.to_csv(os.path.join(tables, "temporal_cv_model_summary.csv"), index=False)
    run.record_artifact("tables", tables)
    if args.publish_legacy:
        publish_tree(tables, os.path.join(os.path.dirname(args.results), "temporal_cv_analysis"))
    run.finalize()
    progress.finish("tables and manifest written")
    print(summary.to_string(index=False))
    print(f"temporal CV analysis -> {run.path}")


if __name__ == "__main__":
    main()
