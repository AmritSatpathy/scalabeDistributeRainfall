"""Tracked supplementary LSTM/TCN experiment with the shared temporal contract."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import deep_dataset as dd  # noqa: E402
import models as model_registry  # noqa: E402
import task2_features as tf  # noqa: E402
from run_artifacts import RunContext, latest_artifact, publish_copy  # noqa: E402
from progress_utils import SparkProgressMonitor, StepProgress, driver_progress  # noqa: E402
from spark_utils import get_spark  # noqa: E402


def reliability(y, prob, threshold, bins=10):
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append({"threshold": threshold, "bin": b, "n": int(mask.sum()),
                         "mean_probability": float(prob[mask].mean()),
                         "observed_frequency": float(y[mask].mean())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "tcn"], default="lstm")
    ap.add_argument("--data", default=None)
    ap.add_argument("--validation-year", type=int, default=2023)
    ap.add_argument("--test-year", type=int, default=2024)
    ap.add_argument("--thresholds", type=float, nargs="*", default=tf.DEFAULT_THRESHOLDS)
    ap.add_argument("--seq-len", type=int, default=dd.SEQ_LEN)
    ap.add_argument("--max-rows", type=int, default=400_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--benchmark-csv", default=None)
    ap.add_argument("--publish-canonical", action="store_true")
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="full")
    args = ap.parse_args()
    args.data = args.data or latest_artifact("clean", "clean_parquet") or config.CLEAN_PARQUET
    run = RunContext("deep_train", vars(args), hash_mode=args.hash_mode)
    with driver_progress("fingerprinting sequence-model input"):
        run.record_input("clean_parquet", args.data)
    progress = StepProgress(4, f"Task 2 {args.model} workflow")

    import torch
    from models._torch_common import (
        apply_isotonic, compute_metrics, fit_isotonic, get_device,
        predict_proba, select_f1_cutoff, train_torch,
    )
    spec = model_registry.get_model(args.model)
    spark = get_spark(f"task2_deep_{run.run_id}")
    with SparkProgressMonitor(spark, "building exact-cadence sequence datasets"):
        data = dd.build_sequences(
            spark, args.data, args.thresholds, args.validation_year, args.test_year,
            seq_len=args.seq_len, max_rows_per_split=args.max_rows,
        )
    spark.stop()
    progress.update("sequence datasets ready")
    device = get_device()
    ytr = data["y_train"]
    pos, neg = float((ytr == 1).sum()), float((ytr == 0).sum())
    net = spec.build_torch(n_static=data["n_static"])
    t0 = time.perf_counter()
    net = train_torch(net, data["X_train_seq"], data["X_train_stat"], ytr,
                      epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      pos_weight=neg / max(pos, 1.0), device=device)
    train_secs = time.perf_counter() - t0
    progress.update("neural model fitted")
    val_prob = predict_proba(net, data["X_validation_seq"], data["X_validation_stat"], device=device)
    t0 = time.perf_counter()
    test_prob = predict_proba(net, data["X_test_seq"], data["X_test_stat"], device=device)
    predict_secs = time.perf_counter() - t0
    progress.update("validation and test predictions ready")
    cal_x, cal_y = fit_isotonic(data["y_validation"], val_prob)
    val_cal = apply_isotonic(val_prob, cal_x, cal_y)
    test_cal = apply_isotonic(test_prob, cal_x, cal_y)
    cutoff, cutoff_rows = select_f1_cutoff(data["y_validation"], val_cal)
    overall = compute_metrics(data["y_test"], test_cal, cutoff)

    per_threshold, reliability_rows = [], []
    for threshold in sorted(set(args.thresholds)):
        vm = data["threshold_validation"] == threshold
        tm = data["threshold_test"] == threshold
        threshold_cutoff, _ = select_f1_cutoff(data["y_validation"][vm], val_cal[vm])
        per_threshold.append({"threshold": threshold,
                              **compute_metrics(data["y_test"][tm], test_cal[tm], threshold_cutoff)})
        reliability_rows.extend(reliability(data["y_test"][tm], test_cal[tm], threshold))

    monotonic = pd.DataFrame({"sample_id": data["sample_id_test"],
                              "threshold": data["threshold_test"], "probability": test_cal})
    monotonic = monotonic.sort_values(["sample_id", "threshold"])
    monotonic["next_probability"] = monotonic.groupby("sample_id")["probability"].shift(-1)
    pairs = monotonic.dropna(subset=["next_probability"])
    monotonic_summary = {"ordered_pairs": len(pairs),
                         "violations": int((pairs.next_probability > pairs.probability + 1e-12).sum())}
    monotonic_summary["violation_rate"] = monotonic_summary["violations"] / max(1, monotonic_summary["ordered_pairs"])

    tables = run.artifact_path("tables"); os.makedirs(tables, exist_ok=True)
    pd.DataFrame([overall]).to_csv(os.path.join(tables, "metrics_overall.csv"), index=False)
    pd.DataFrame(per_threshold).to_csv(os.path.join(tables, "metrics_by_threshold.csv"), index=False)
    pd.DataFrame(cutoff_rows).to_csv(os.path.join(tables, "validation_cutoff_selection.csv"), index=False)
    pd.DataFrame(reliability_rows).to_csv(os.path.join(tables, "reliability_isotonic.csv"), index=False)
    pd.DataFrame([monotonic_summary]).to_csv(os.path.join(tables, "threshold_monotonicity.csv"), index=False)
    model_path = run.artifact_path(f"rainfall_{args.model}.pt")
    torch.save({"state_dict": net.state_dict(), "seq_len": args.seq_len,
                "n_static": data["n_static"], "model": args.model,
                "calibration_x": cal_x, "calibration_y": cal_y, "cutoff": cutoff}, model_path)
    metadata = {"run_id": run.run_id, "model": args.model, "framework": "pytorch",
                "validation_year": args.validation_year, "test_year": args.test_year,
                "thresholds": args.thresholds, "train_secs": train_secs,
                "predict_secs": predict_secs, "metrics": overall,
                "device": str(device), "exact_cadence": True}
    run.write_json("model_metadata", metadata)
    row = {"run_id": run.run_id, "model": args.model, "compute_config": str(device),
           "n_train": len(ytr), "n_validation": len(data["y_validation"]),
           "n_test": len(data["y_test"]), "train_secs": train_secs,
           "predict_secs": predict_secs, **overall}
    pd.DataFrame([row]).to_csv(os.path.join(tables, "benchmark_row.csv"), index=False)
    if args.benchmark_csv:
        pd.DataFrame([row]).to_csv(args.benchmark_csv, mode="a", header=not os.path.exists(args.benchmark_csv), index=False)
    run.record_artifact("tables", tables); run.record_artifact("model", model_path)
    if args.publish_canonical:
        publish_copy(model_path, os.path.join(config.MODELS_DIR, f"rainfall_{args.model}.pt"))
    run.finalize(); progress.finish("evaluation and artifacts complete")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
