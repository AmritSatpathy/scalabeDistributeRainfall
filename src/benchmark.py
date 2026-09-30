"""
Benchmark harness - run the (model x compute-config) grid and collect
train time, predict time, and AUC into one CSV for the study.

It drives ``task2_train.py`` as a subprocess for each cell so every run gets a
fresh JVM / Spark session with the correct compute-config environment applied
(this is important: Spark master/executor settings are fixed at SparkContext
creation, so each config needs its own process).

Each run appends a row to ``output/tables/benchmark_results.csv`` (the same
file task2_train writes), tagged with the model and compute-config names.

Usage
-----
    # all MLlib models on local[*]:
    python src/benchmark.py --models gbt rf logreg linsvc mlp fm --computes local-all

    # scaling study: GBT across local parallelism levels:
    python src/benchmark.py --models gbt --computes local-2 local-4 local-8 local-all

    # fix a small sample so each cell is fast + comparable (recommended for the
    # compute-config timing study):
    python src/benchmark.py --models gbt rf --computes local-2 local-4 local-8 \
        --sample-fraction 0.1 --max-train-rows 1000000

Notes
-----
* Standalone / Docker configs require the cluster to be up first (see the
  launch scripts); this harness only sets the env + submits.
* Uses --no-save for every cell so the canonical saved model is preserved.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import models as model_registry  # noqa: E402
import compute as compute_registry  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable  # the venv python running this harness


def run_cell(model: str, compute: str, sample_fraction: float,
             max_train_rows: int, extra_args: list[str]) -> int:
    """Run one (model, compute) cell as a subprocess. Returns exit code."""
    cmd = [
        PY, os.path.join(THIS_DIR, "task2_train.py"),
        "--model", model,
        "--compute", compute,
        "--no-save",
        "--sample-fraction", str(sample_fraction),
        "--max-train-rows", str(max_train_rows),
        *extra_args,
    ]
    print("\n" + "=" * 70)
    print(f"CELL: model={model}  compute={compute}")
    print("CMD :", " ".join(cmd))
    print("=" * 70, flush=True)
    proc = subprocess.run(cmd, cwd=os.path.dirname(THIS_DIR))
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=model_registry.mllib_models(),
                    help="Model names (see task2_train.py --list)")
    ap.add_argument("--computes", nargs="+", default=["local-all"],
                    help="Compute-config names (see task2_train.py --list)")
    ap.add_argument("--sample-fraction", type=float, default=1.0)
    ap.add_argument("--max-train-rows", type=int, default=8_000_000)
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Keep going if a cell fails (e.g. OOM) instead of stopping.")
    args, extra = ap.parse_known_args()

    # Validate names up front so we fail fast on a typo.
    for m in args.models:
        model_registry.get_model(m)
    for c in args.computes:
        compute_registry.get_compute(c)

    print(f"Grid: {len(args.computes)} compute-config(s) x {len(args.models)} model(s) "
          f"= {len(args.computes) * len(args.models)} runs")
    results = []
    for compute in args.computes:
        for model in args.models:
            rc = run_cell(model, compute, args.sample_fraction,
                          args.max_train_rows, extra)
            results.append((compute, model, rc))
            if rc != 0 and not args.continue_on_error:
                print(f"[abort] cell (compute={compute}, model={model}) failed "
                      f"with exit {rc}. Use --continue-on-error to skip failures.")
                sys.exit(rc)

    print("\n" + "=" * 70)
    print("BENCHMARK GRID COMPLETE")
    for compute, model, rc in results:
        status = "ok" if rc == 0 else f"FAIL({rc})"
        print(f"  {compute:16s} {model:8s} {status}")
    print(f"\nResults appended to: {os.path.join(config.TABLES_DIR, 'benchmark_results.csv')}")
    print("Plot with: python src/benchmark_plot.py")


if __name__ == "__main__":
    main()
