"""
Task 2 (deep-learning path) - train/evaluate a PyTorch sequence model (LSTM or
TCN) on the GPU and record results into the SAME benchmark table as the MLlib
models, for a like-for-like comparison.

Spark builds the sequence dataset (per-station lag windows + 30-min label);
PyTorch trains on the GPU. Metrics (AUC-ROC / AUC-PR / accuracy / F1) and
train/predict wall-clock time are appended to output/tables/benchmark_results.csv
with compute_config="gpu" so the model-comparison plots include the deep models.

Usage
-----
    python src/task2_deep.py --model lstm
    python src/task2_deep.py --model tcn --max-rows 300000 --epochs 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import task2_features as tf  # noqa: E402
import models as model_registry  # noqa: E402
import deep_dataset as dd  # noqa: E402
from spark_utils import get_spark  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "tcn"], default="lstm")
    ap.add_argument("--data", default=config.CLEAN_PARQUET,
                    help="Cleaned parquet (needs raw rain_mm sequence)")
    ap.add_argument("--test-year", type=int, default=max(config.YEARS))
    ap.add_argument("--thresholds", type=float, nargs="*", default=tf.DEFAULT_THRESHOLDS)
    ap.add_argument("--seq-len", type=int, default=dd.SEQ_LEN)
    ap.add_argument("--max-rows", type=int, default=400_000,
                    help="Max sequence rows per split (bounds GPU/driver memory)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--compute-config", default="gpu")
    ap.add_argument("--benchmark-csv",
                    default=os.path.join(config.TABLES_DIR, "benchmark_results.csv"))
    args = ap.parse_args()

    import torch  # imported here so pure-MLlib runs never require torch
    from models._torch_common import (
        train_torch, predict_proba, compute_metrics, get_device,
    )

    spec = model_registry.get_model(args.model)
    if spec.framework != "torch":
        raise SystemExit(f"{args.model} is not a torch model")

    config.ensure_dirs()

    # --- build sequences with Spark ----------------------------------------
    spark = get_spark("task2_deep")
    print(f"building sequences (seq_len={args.seq_len}, max_rows={args.max_rows})...")
    data = dd.build_sequences(
        spark, args.data, args.thresholds, args.test_year,
        seq_len=args.seq_len, max_rows_per_split=args.max_rows,
    )
    spark.stop()

    device = get_device()
    print(f"device: {device}")

    # class imbalance -> positive weight for the loss
    ytr = data["y_train"]
    pos = float((ytr == 1).sum())
    neg = float((ytr == 0).sum())
    pos_weight = (neg / max(pos, 1.0)) if pos > 0 else 1.0
    print(f"train pos={pos:.0f} neg={neg:.0f} pos_weight={pos_weight:.2f}")

    # --- build + train the network -----------------------------------------
    net = spec.build_torch(n_static=data["n_static"])
    print(f"training {args.model} for {args.epochs} epochs...")
    t0 = time.perf_counter()
    net = train_torch(
        net, data["X_train_seq"], data["X_train_stat"], data["y_train"],
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        pos_weight=pos_weight, device=device,
    )
    train_secs = time.perf_counter() - t0
    print(f"train time: {train_secs:.1f}s")

    # --- predict + evaluate -------------------------------------------------
    t0 = time.perf_counter()
    prob = predict_proba(net, data["X_test_seq"], data["X_test_stat"], device=device)
    predict_secs = time.perf_counter() - t0
    metrics = compute_metrics(data["y_test"], prob)
    print(f"predict time: {predict_secs:.1f}s")
    print("OVERALL test metrics:", json.dumps(metrics, indent=2))

    # --- append benchmark row (same schema as the MLlib path) --------------
    import pandas as pd
    row = {
        "model": args.model,
        "compute_config": args.compute_config,
        "spark_master": "n/a (pytorch)",
        "default_parallelism": 1,
        "n_train": len(data["y_train"]),
        "n_test": len(data["y_test"]),
        "train_secs": round(train_secs, 2),
        "predict_secs": round(predict_secs, 2),
        "auc_roc": round(metrics["auc_roc"], 5),
        "auc_pr": round(metrics["auc_pr"], 5),
        "accuracy": round(metrics["accuracy"], 5),
        "f1": round(metrics["f1"], 5),
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(args.benchmark_csv), exist_ok=True)
    pd.DataFrame([row]).to_csv(
        args.benchmark_csv, mode="a",
        header=not os.path.exists(args.benchmark_csv), index=False,
    )
    print(f"appended benchmark row -> {args.benchmark_csv}: {row}")

    # save the torch model weights for submission
    out = os.path.join(config.MODELS_DIR, f"rainfall_{args.model}.pt")
    torch.save({"state_dict": net.state_dict(),
                "seq_len": args.seq_len, "n_static": data["n_static"],
                "model": args.model}, out)
    print(f"saved model -> {out}")


if __name__ == "__main__":
    main()
