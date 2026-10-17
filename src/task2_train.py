"""
Task 2 - Train and evaluate a rainfall-probability model with Spark MLlib.

This is a THIN ORCHESTRATOR. It selects:
  * WHICH MODEL       via the models registry   (--model,   see src/models/)
  * WHICH ARCHITECTURE via the compute registry (--compute, see src/compute/)
and runs the shared train/evaluate/benchmark flow. Model definitions and
compute configurations live in their own modules, so adding either is a
one-file change with no edits here.

Target: P(rain over next 30 min at a station > threshold). The threshold is a
model feature (see task2_features.py) so one model serves any threshold.
Evaluation uses a temporal split (train on earlier years, test on the most
recent year) and reports AUC-ROC, AUC-PR, accuracy, F1 - overall and per
threshold - plus train/predict wall-clock time for the benchmark study.

Usage
-----
    python src/task2_train.py --model gbt
    python src/task2_train.py --model rf --compute local-4 --no-save
    python src/task2_train.py --list          # show models + compute configs
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
from task2_build_features import FEATURES_BASE  # noqa: E402
from spark_utils import get_spark, save_table  # noqa: E402
import models as model_registry  # noqa: E402
import compute as compute_registry  # noqa: E402

from pyspark.ml.evaluation import (  # noqa: E402
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import functions as F  # noqa: E402
from pyspark import StorageLevel  # noqa: E402

MODEL_DIR = os.path.join(config.MODELS_DIR, "rainfall_model")
META_PATH = os.path.join(config.MODELS_DIR, "model_metadata.json")

# Assembled feature columns (base features + threshold + station index).
FEATURE_COLS = tf.BASE_FEATURE_COLS + ["threshold", "station_idx"]


def evaluate(pred, label_col="label"):
    auc = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(pred)
    aupr = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    ).evaluate(pred)
    acc = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="accuracy"
    ).evaluate(pred)
    f1 = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="f1"
    ).evaluate(pred)
    return {"auc_roc": auc, "auc_pr": aupr, "accuracy": acc, "f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gbt",
                    help="Model name (see --list). Default: gbt")
    ap.add_argument("--compute", default=None,
                    help="Compute config name (see --list). Applies env overrides "
                         "before the Spark session starts. Default: inherit env / local[*]")
    ap.add_argument("--features", default=FEATURES_BASE)
    ap.add_argument("--test-year", type=int, default=max(config.YEARS))
    ap.add_argument("--sample-fraction", type=float, default=1.0)
    ap.add_argument("--max-train-rows", type=int, default=8_000_000,
                    help="Cap on exploded training rows (downsampled if exceeded); 0 disables.")
    ap.add_argument("--thresholds", type=float, nargs="*", default=tf.DEFAULT_THRESHOLDS)
    ap.add_argument("--no-save", action="store_true",
                    help="Do not overwrite the canonical models/rainfall_model (for benchmark runs).")
    ap.add_argument("--save-as", default=None,
                    help="Save this fitted model under models/<SAVE_AS>/ (and its "
                         "metadata as models/<SAVE_AS>_metadata.json) instead of the "
                         "canonical rainfall_model. Lets every comparison model be "
                         "persisted without clobbering the graded GBT.")
    ap.add_argument("--benchmark-csv",
                    default=os.path.join(config.TABLES_DIR, "benchmark_results.csv"))
    ap.add_argument("--list", action="store_true",
                    help="Print available models and compute configs, then exit.")
    args = ap.parse_args()

    if args.list:
        print("Models:")
        for name, fw, desc in model_registry.describe():
            print(f"  {name:10s} [{fw}]  {desc}")
        print("Compute configs:")
        for name, master, desc in compute_registry.describe():
            print(f"  {name:16s} {master:28s} {desc}")
        return

    spec = model_registry.get_model(args.model)
    if spec.framework != "mllib":
        raise SystemExit(
            f"model {args.model!r} is framework={spec.framework!r}; "
            f"use its own runner (this orchestrator handles MLlib models)."
        )

    # Apply the compute config BEFORE creating the Spark session so its env
    # overrides (master, memory, executors) take effect.
    compute_label = args.compute or os.environ.get("SPARK_MASTER", "local[*]")
    if args.compute:
        cfg = compute_registry.get_compute(args.compute)
        cfg.apply()
        compute_label = cfg.name
        print(f"[compute] {cfg.name}: {cfg.description}  (master={cfg.master})")

    config.ensure_dirs()
    spark = get_spark("task2_train")

    # --- data: read materialised features, explode thresholds, temporal split
    base = spark.read.parquet(args.features)
    if args.sample_fraction < 1.0:
        base = base.sample(False, args.sample_fraction, seed=42)

    labelled = tf.explode_thresholds(base, args.thresholds)
    keep = (
        ["station_id", "year", "threshold", "label", "future_30min_mm"]
        + tf.BASE_FEATURE_COLS
    )
    labelled = labelled.select(*keep).na.drop()

    train = labelled.filter(F.col("year") < args.test_year)
    test = labelled.filter(F.col("year") == args.test_year)

    if args.max_train_rows and args.max_train_rows > 0:
        n_raw = train.count()
        if n_raw > args.max_train_rows:
            frac = args.max_train_rows / float(n_raw)
            print(f"downsampling train {n_raw:,} -> ~{args.max_train_rows:,} (frac={frac:.4f})")
            train = train.sample(False, frac, seed=42)

    train = train.persist(StorageLevel.MEMORY_AND_DISK)
    test = test.persist(StorageLevel.MEMORY_AND_DISK)
    n_train, n_test = train.count(), test.count()
    print(f"train rows: {n_train:,}  |  test rows: {n_test:,}")
    print("train positive rate:", train.select(F.avg("label")).first()[0])

    # --- model: built by the selected spec (feature handling encapsulated) ---
    pipeline = spec.build_pipeline(FEATURE_COLS)
    print(f"fitting {args.model} pipeline...")
    t0 = time.perf_counter()
    fitted = pipeline.fit(train)
    train_secs = time.perf_counter() - t0
    print(f"train time: {train_secs:.1f}s")

    print("scoring test set...")
    t0 = time.perf_counter()
    pred = fitted.transform(test).persist(StorageLevel.MEMORY_AND_DISK)
    _ = pred.count()  # force materialisation so the timer captures real work
    predict_secs = time.perf_counter() - t0
    print(f"predict time (full test set): {predict_secs:.1f}s")

    overall = evaluate(pred)
    print("OVERALL test metrics:", json.dumps(overall, indent=2))

    per_thr = []
    for t in sorted(set(args.thresholds)):
        sub = pred.filter(F.col("threshold") == float(t))
        n_pos = sub.filter(F.col("label") == 1).limit(1).count()
        n_neg = sub.filter(F.col("label") == 0).limit(1).count()
        m = evaluate(sub) if (n_pos and n_neg) else {
            "auc_roc": None, "auc_pr": None, "accuracy": None, "f1": None}
        per_thr.append({"threshold": t, **m})
        print(f"  threshold={t}: {per_thr[-1]}")

    import pandas as pd

    save_table(pd.DataFrame(per_thr),
               os.path.join(config.TABLES_DIR, f"model_metrics_by_threshold_{args.model}.csv"))
    save_table(pd.DataFrame([{"scope": "overall", **overall}]),
               os.path.join(config.TABLES_DIR, f"model_metrics_overall_{args.model}.csv"))
    if args.model == "gbt":
        save_table(pd.DataFrame(per_thr),
                   os.path.join(config.TABLES_DIR, "model_metrics_by_threshold.csv"))
        save_table(pd.DataFrame([{"scope": "overall", **overall}]),
                   os.path.join(config.TABLES_DIR, "model_metrics_overall.csv"))

    # --- benchmark row: model x compute-config x timings x AUC --------------
    sc = spark.sparkContext
    bench_row = {
        "model": args.model,
        "compute_config": compute_label,
        "spark_master": sc.master,
        "default_parallelism": sc.defaultParallelism,
        "n_train": n_train,
        "n_test": n_test,
        "train_secs": round(train_secs, 2),
        "predict_secs": round(predict_secs, 2),
        "auc_roc": round(overall["auc_roc"], 5),
        "auc_pr": round(overall["auc_pr"], 5),
        "accuracy": round(overall["accuracy"], 5),
        "f1": round(overall["f1"], 5),
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(args.benchmark_csv), exist_ok=True)
    pd.DataFrame([bench_row]).to_csv(
        args.benchmark_csv, mode="a",
        header=not os.path.exists(args.benchmark_csv), index=False,
    )
    print(f"appended benchmark row -> {args.benchmark_csv}")

    # --- persist model ------------------------------------------------------
    # Resolve the save target:
    #   --save-as NAME -> models/NAME/            (+ models/NAME_metadata.json)
    #   default (gbt)  -> models/rainfall_model/  (the canonical graded model)
    #   --no-save      -> nothing
    if args.no_save and not args.save_as:
        print("--no-save set: not persisting this model.")
    else:
        if args.save_as:
            model_dir = os.path.join(config.MODELS_DIR, args.save_as)
            meta_path = os.path.join(config.MODELS_DIR, f"{args.save_as}_metadata.json")
        else:
            model_dir = MODEL_DIR
            meta_path = META_PATH
        print(f"saving model -> {model_dir}")
        fitted.write().overwrite().save(model_dir)
        meta = {
            "model_kind": args.model,
            "model_description": spec.description,
            "horizon_minutes": config.HORIZON_MINUTES,
            "base_feature_cols": tf.BASE_FEATURE_COLS,
            "uses_threshold_as_feature": True,
            "train_thresholds": sorted(set(args.thresholds)),
            "test_year": args.test_year,
            "train_years": [y for y in config.YEARS if y < args.test_year],
            "max_train_rows": args.max_train_rows,
            "sample_fraction": args.sample_fraction,
            "compute_config": compute_label,
            "n_train": n_train,
            "n_test": n_test,
            "train_secs": round(train_secs, 2),
            "predict_secs": round(predict_secs, 2),
            "metrics_overall": overall,
            "metrics_by_threshold": per_thr,
        }
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"wrote metadata -> {meta_path}")

    spark.stop()


if __name__ == "__main__":
    main()
