"""Run expanding-window Task 2 model comparisons without temporal leakage.

Each fold trains from ``train_start_year`` through the year before validation,
uses one complete year for calibration/cutoff selection, and evaluates on the
immediately following year.  Every model receives the same deterministic rows,
features, thresholds, and fold boundaries.
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import compute as compute_registry  # noqa: E402
import config  # noqa: E402
import models as model_registry  # noqa: E402
from run_artifacts import RunContext, latest_artifact, publish_copy  # noqa: E402
from progress_utils import StepProgress, driver_progress  # noqa: E402


def parse_fold(value: str) -> tuple[int, int]:
    try:
        validation, test = (int(part) for part in value.split(":"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("fold must have form VALIDATION_YEAR:TEST_YEAR") from exc
    if test != validation + 1:
        raise argparse.ArgumentTypeError("test year must immediately follow validation year")
    return validation, test


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["gbt", "rf", "mlp"])
    ap.add_argument("--compute", default="local-4-matched")
    ap.add_argument("--features", default=None)
    ap.add_argument("--feature-contract", default=None)
    ap.add_argument("--train-start-year", type=int, default=2017)
    ap.add_argument(
        "--folds", nargs="+", type=parse_fold,
        default=[(2021, 2022), (2023, 2024)],
        metavar="VALIDATION:TEST",
    )
    ap.add_argument("--sample-fraction", type=float, default=1.0)
    ap.add_argument("--max-train-rows", type=int, default=8_000_000)
    ap.add_argument("--max-validation-contexts", type=int, default=100_000)
    ap.add_argument("--max-test-contexts", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=5208)
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="full")
    ap.add_argument("--child-hash-mode", choices=["full", "metadata"], default="metadata",
                    help="Per-model input hashing; parent run retains the full input fingerprint")
    ap.add_argument("--publish-results", default=None)
    ap.add_argument(
        "--resume-results", default=None,
        help="Existing temporal-CV CSV whose compatible completed jobs are reused",
    )
    args, extra = ap.parse_known_args()

    if not 0 < args.sample_fraction <= 1:
        ap.error("--sample-fraction must lie in (0, 1]")
    if args.train_start_year >= min(validation for validation, _ in args.folds):
        ap.error("--train-start-year must precede every validation year")
    if args.resume_results and not Path(args.resume_results).is_file():
        ap.error(f"--resume-results does not exist: {args.resume_results}")
    for model in args.models:
        spec = model_registry.get_model(model)
        if spec.framework != "mllib":
            ap.error(f"{model} uses {spec.framework}; this runner compares MLlib models only")
    compute_registry.get_compute(args.compute)
    tracked_spatiotemporal = latest_artifact(
        "spatiotemporal_features", "spatiotemporal_features"
    )
    args.features = (
        args.features
        or tracked_spatiotemporal
        or latest_artifact("features", "features_base")
    )
    if not args.features or not Path(args.features).is_dir():
        raise SystemExit(
            "No model-ready feature dataset found. Pass --features or complete "
            "build_spatiotemporal_features.py first."
        )
    if args.feature_contract is None and args.features == tracked_spatiotemporal:
        args.feature_contract = latest_artifact(
            "spatiotemporal_features", "feature_contract"
        )

    group = datetime.now(timezone.utc).strftime("temporal_cv_%Y%m%dT%H%M%SZ")
    parameters = {**vars(args), "folds": [list(fold) for fold in args.folds], "extra": extra}
    run = RunContext("temporal_cv", parameters, hash_mode=args.hash_mode, run_group=group)
    with driver_progress("fingerprinting temporal-CV feature input"):
        run.record_input("features_base", args.features)
        if args.feature_contract:
            run.record_input("feature_contract", args.feature_contract)
        if args.resume_results:
            run.record_input("resume_results", args.resume_results)
    results_path = run.artifact_path("temporal_cv_results.csv")
    event_root = run.artifact_path("spark-events")

    largest_training_years = max(
        validation - args.train_start_year for validation, _ in args.folds
    )
    jobs = [(validation, test, model) for validation, test in args.folds for model in args.models]
    random.Random(args.seed).shuffle(jobs)
    reusable = pd.DataFrame()
    if args.resume_results:
        prior = pd.read_csv(args.resume_results)
        required_resume_columns = {
            "model", "fold_id", "compute_config", "train_start_year",
            "validation_year", "test_year", "sample_fraction", "max_train_rows",
            "max_validation_contexts", "max_test_contexts",
        }
        missing = required_resume_columns - set(prior.columns)
        if missing:
            ap.error(f"resume CSV is missing compatibility columns: {sorted(missing)}")
        compatible_rows = []
        for validation, test, model in jobs:
            fold_id = f"expanding_{args.train_start_year}_{validation - 1}_val{validation}_test{test}"
            training_years = validation - args.train_start_year
            fold_train_cap = (
                max(1, round(args.max_train_rows * training_years / largest_training_years))
                if args.max_train_rows else 0
            )
            match = prior[
                (prior.model == model)
                & (prior.fold_id == fold_id)
                & (prior.compute_config == args.compute)
                & (prior.train_start_year == args.train_start_year)
                & (prior.validation_year == validation)
                & (prior.test_year == test)
                & (prior.sample_fraction == args.sample_fraction)
                & (prior.max_train_rows == fold_train_cap)
                & (prior.max_validation_contexts == args.max_validation_contexts)
                & (prior.max_test_contexts == args.max_test_contexts)
            ]
            if not match.empty:
                compatible_rows.append(match.iloc[-1])
        if compatible_rows:
            reusable = pd.DataFrame(compatible_rows).drop_duplicates(
                ["model", "fold_id"], keep="last"
            )
            reusable.to_csv(results_path, index=False)
            print(
                f"Reusing {len(reusable)} compatible completed model-fold jobs "
                f"from {args.resume_results}", flush=True,
            )
    reusable_keys = set(zip(reusable.get("model", []), reusable.get("fold_id", [])))
    failures = []
    progress = StepProgress(len(jobs), "temporal model-fold jobs")
    for validation, test, model in jobs:
        fold_id = f"expanding_{args.train_start_year}_{validation - 1}_val{validation}_test{test}"
        training_years = validation - args.train_start_year
        fold_train_cap = (
            max(1, round(args.max_train_rows * training_years / largest_training_years))
            if args.max_train_rows else 0
        )
        if (model, fold_id) in reusable_keys:
            print(f"REUSE: {fold_id} / {model}", flush=True)
            progress.update(f"{fold_id} / {model} / reused")
            continue
        event_dir = os.path.join(event_root, fold_id, model)
        cmd = [
            sys.executable, os.path.join(config.PROJECT_ROOT, "src", "task2_train.py"),
            "--model", model, "--compute", args.compute, "--features", args.features,
            "--train-start-year", str(args.train_start_year),
            "--validation-year", str(validation), "--test-year", str(test),
            "--fold-id", fold_id, "--sample-fraction", str(args.sample_fraction),
            "--max-train-rows", str(fold_train_cap), "--no-save",
            "--max-validation-contexts", str(args.max_validation_contexts),
            "--max-test-contexts", str(args.max_test_contexts),
            "--benchmark-csv", results_path, "--run-group", group,
            "--event-log-dir", event_dir, "--hash-mode", args.child_hash_mode,
            *extra,
        ]
        if args.feature_contract:
            cmd.extend(["--feature-contract", args.feature_contract])
        print("RUN:", " ".join(cmd), flush=True)
        return_code = subprocess.run(cmd, cwd=config.PROJECT_ROOT).returncode
        progress.update(f"{fold_id} / {model} / {'ok' if return_code == 0 else 'failed'}")
        if return_code:
            failures.append({
                "fold_id": fold_id, "model": model, "compute": args.compute,
                "exit_code": return_code,
            })
            if not args.continue_on_error:
                break

    if not failures or args.continue_on_error:
        progress.finish("all scheduled jobs processed")

    if os.path.exists(results_path):
        run.record_artifact("temporal_cv_results", results_path)
        if args.publish_results and not failures:
            published_results = str(Path(args.publish_results).expanduser().resolve())
            publish_copy(results_path, published_results)
            run.record_artifact("published_temporal_cv_results", published_results)
    if os.path.exists(event_root):
        run.record_artifact("spark_event_logs", event_root)
    run.write_json("failures", failures)
    run.write_json(
        "resume_summary",
        {
            "source": args.resume_results,
            "jobs_requested": len(jobs),
            "jobs_reused": len(reusable_keys),
            "jobs_executed": len(jobs) - len(reusable_keys),
        },
    )
    run.finalize("complete" if not failures else "partial")
    if failures and not args.continue_on_error:
        raise SystemExit(1)
    print(f"temporal CV group: {group} -> {run.path}")


if __name__ == "__main__":
    main()
