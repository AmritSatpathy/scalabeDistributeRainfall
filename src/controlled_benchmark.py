"""Repeated, randomized benchmark harness over one frozen feature sample."""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import compute as compute_registry  # noqa: E402
import config  # noqa: E402
import models as model_registry  # noqa: E402
from run_artifacts import RunContext, latest_artifact, publish_copy  # noqa: E402
from progress_utils import StepProgress, driver_progress  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gbt"])
    ap.add_argument("--computes", nargs="+", required=True)
    ap.add_argument("--features", default=None)
    ap.add_argument("--feature-contract", default=None)
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--seed", type=int, default=5208)
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="metadata")
    ap.add_argument("--child-hash-mode", choices=["full", "metadata"], default="metadata",
                    help="Per-repetition hashing; the group run fingerprints the frozen input")
    ap.add_argument("--publish-results", default=None)
    ap.add_argument("--publish-events", default=None)
    args, extra = ap.parse_known_args()
    args.features = args.features or latest_artifact("benchmark_sample", "benchmark_features")
    if not args.features:
        raise SystemExit("Run prepare_benchmark_sample.py first or pass --features.")
    for model in args.models: model_registry.get_model(model)
    for compute in args.computes: compute_registry.get_compute(compute)
    group = datetime.now(timezone.utc).strftime("controlled_%Y%m%dT%H%M%SZ")
    run = RunContext("benchmark_group", {**vars(args), "extra": extra}, hash_mode=args.hash_mode, run_group=group)
    with driver_progress("fingerprinting frozen benchmark input"):
        run.record_input("benchmark_features", args.features)
        if args.feature_contract:
            run.record_input("feature_contract", args.feature_contract)
    csv_path = run.artifact_path("controlled_benchmark.csv")
    tasks = []
    for compute in args.computes:
        for model in args.models:
            tasks.extend((True, i, compute, model) for i in range(1, args.warmups + 1))
            tasks.extend((False, i, compute, model) for i in range(1, args.repetitions + 1))
    warmups = [x for x in tasks if x[0]]
    measured = [x for x in tasks if not x[0]]
    random.Random(args.seed).shuffle(measured)
    failures = []
    progress = StepProgress(len(warmups + measured), "controlled benchmark runs")
    for warmup, repetition, compute, model in warmups + measured:
        event_dir = run.artifact_path(f"spark-events/{compute}/{model}/{'warmup_' + str(repetition) if warmup else 'rep_' + str(repetition)}")
        cmd = [
            sys.executable, os.path.join(config.PROJECT_ROOT, "src", "task2_train.py"),
            "--model", model, "--compute", compute, "--features", args.features,
            "--sample-fraction", "1", "--max-train-rows", "0", "--no-save",
            "--benchmark-csv", csv_path, "--run-group", group,
            "--repetition", str(repetition), "--event-log-dir", event_dir,
            "--hash-mode", args.child_hash_mode, *extra,
        ]
        if args.feature_contract:
            cmd.extend(["--feature-contract", args.feature_contract])
        if warmup: cmd.append("--warmup")
        print("RUN:", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=config.PROJECT_ROOT).returncode
        progress.update(
            f"{compute} / {model} / "
            f"{'warmup' if warmup else f'repetition {repetition}'} / "
            f"{'ok' if rc == 0 else 'failed'}"
        )
        if rc:
            failures.append({"compute": compute, "model": model, "repetition": repetition, "warmup": warmup, "exit_code": rc})
            if not args.continue_on_error: break
    if not failures or args.continue_on_error:
        progress.finish("all scheduled runs processed")
    if os.path.exists(csv_path):
        run.record_artifact("benchmark_results", csv_path)
        if args.publish_results and not failures:
            published_results = os.path.abspath(args.publish_results)
            publish_copy(csv_path, published_results)
            run.record_artifact("published_benchmark_results", published_results)
    event_root = run.artifact_path("spark-events")
    if os.path.exists(event_root):
        run.record_artifact("spark_event_logs", event_root)
        if args.publish_events and not failures:
            published_events = os.path.abspath(args.publish_events)
            publish_copy(event_root, published_events)
            run.record_artifact("published_spark_event_logs", published_events)
    run.write_json("failures", failures)
    run.finalize("complete" if not failures else "partial")
    if failures and not args.continue_on_error: raise SystemExit(1)
    print(f"benchmark group: {group} -> {run.path}")


if __name__ == "__main__":
    main()
