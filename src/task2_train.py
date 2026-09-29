"""
Task 2 - Train and evaluate a rainfall-probability model with Spark MLlib.

Model target: P(rain over next 30 min at a station > threshold), where the
threshold is supplied as an input feature (see task2_features.py). We train a
Gradient-Boosted Trees classifier (with a Logistic Regression baseline) using
a StringIndexer + OneHotEncoder for station identity and a VectorAssembler
for the numeric features.

Evaluation uses a *temporal* train/test split (train on earlier years, test on
the most recent year) to reflect real forecasting use, and reports AUC-ROC,
area under PR curve, accuracy and F1. Per-threshold AUC is also reported so we
can see how discrimination varies with the requested threshold.

The fitted PipelineModel is written to models/rainfall_gbt_pipeline and a JSON
metadata file records the feature list, thresholds and metrics.

Usage
-----
    python src/task2_train.py                       # full data
    python src/task2_train.py --sample-fraction 0.2 # subsample base rows
    python src/task2_train.py --test-year 2024 --model gbt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import task2_features as tf  # noqa: E402
from task2_build_features import FEATURES_BASE  # noqa: E402
from spark_utils import get_spark, save_table  # noqa: E402

from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import GBTClassifier, LogisticRegression  # noqa: E402
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator  # noqa: E402
from pyspark.ml.feature import StringIndexer, VectorAssembler  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark import StorageLevel  # noqa: E402

MODEL_DIR = os.path.join(config.MODELS_DIR, "rainfall_model")
META_PATH = os.path.join(config.MODELS_DIR, "model_metadata.json")


def build_pipeline(model_kind: str) -> Pipeline:
    # Station identity is kept as a single indexed numeric column rather than a
    # one-hot vector: with ~91 stations, one-hot makes the feature vector wide
    # and forces GBT to densify it (memory heavy). Trees can still split on the
    # index, and lat/lon already carry the real spatial signal.
    feature_cols = tf.BASE_FEATURE_COLS + ["threshold", "station_idx"]

    idx = StringIndexer(
        inputCol="station_id", outputCol="station_idx", handleInvalid="keep"
    )
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    if model_kind == "gbt":
        # maxBins must be >= the number of station categories (~91) so the
        # tree can split on station identity as a categorical feature.
        clf = GBTClassifier(
            labelCol="label", featuresCol="features",
            maxIter=40, maxDepth=5, stepSize=0.1, maxBins=128,
            subsamplingRate=0.7, cacheNodeIds=True, seed=42,
        )
    elif model_kind == "logreg":
        clf = LogisticRegression(
            labelCol="label", featuresCol="features",
            maxIter=50, regParam=0.01, elasticNetParam=0.0,
        )
    else:
        raise ValueError(f"unknown model kind {model_kind!r}")

    return Pipeline(stages=[idx, assembler, clf])


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
    ap.add_argument("--features", default=FEATURES_BASE,
                    help="Materialised base feature Parquet (from task2_build_features.py)")
    ap.add_argument("--model", choices=["gbt", "logreg"], default="gbt")
    ap.add_argument("--test-year", type=int, default=max(config.YEARS))
    ap.add_argument("--sample-fraction", type=float, default=1.0,
                    help="Randomly subsample base rows before threshold explode")
    ap.add_argument("--max-train-rows", type=int, default=8_000_000,
                    help="Cap on exploded training rows (downsampled if exceeded). "
                         "Keeps GBT within memory; set 0 to disable.")
    ap.add_argument("--thresholds", type=float, nargs="*", default=tf.DEFAULT_THRESHOLDS)
    args = ap.parse_args()

    config.ensure_dirs()
    spark = get_spark("task2_train", driver_memory="20g", shuffle_partitions=96)

    # Base features were materialised once by task2_build_features.py, so this
    # is a cheap read - no per-station window recomputation here.
    base = spark.read.parquet(args.features)

    if args.sample_fraction < 1.0:
        base = base.sample(False, args.sample_fraction, seed=42)

    labelled = tf.explode_thresholds(base, args.thresholds)

    # Keep only the columns we need, and drop any residual nulls.
    keep = (
        ["station_id", "year", "threshold", "label", "future_30min_mm"]
        + tf.BASE_FEATURE_COLS
    )
    labelled = labelled.select(*keep).na.drop()

    # Temporal split: train on years < test-year, test on test-year.
    train = labelled.filter(F.col("year") < args.test_year)
    test = labelled.filter(F.col("year") == args.test_year)

    # Cap the exploded training volume so GBT stays within memory. We downsample
    # uniformly at random (label distribution preserved in expectation).
    if args.max_train_rows and args.max_train_rows > 0:
        n_raw = train.count()
        if n_raw > args.max_train_rows:
            frac = args.max_train_rows / float(n_raw)
            print(f"downsampling train {n_raw:,} -> ~{args.max_train_rows:,} (frac={frac:.4f})")
            train = train.sample(False, frac, seed=42)

    # Disk-backed persistence avoids OOM on the large exploded frames.
    train = train.persist(StorageLevel.MEMORY_AND_DISK)
    test = test.persist(StorageLevel.MEMORY_AND_DISK)
    n_train, n_test = train.count(), test.count()
    print(f"train rows: {n_train:,}  |  test rows: {n_test:,}")
    print("train positive rate:", train.select(F.avg("label")).first()[0])

    pipeline = build_pipeline(args.model)
    print(f"fitting {args.model} pipeline...")
    model = pipeline.fit(train)

    print("scoring test set...")
    pred = model.transform(test)

    overall = evaluate(pred)
    print("OVERALL test metrics:", json.dumps(overall, indent=2))

    # Per-threshold AUC (evaluate on the subset for each threshold value).
    per_thr = []
    for t in sorted(set(args.thresholds)):
        sub = pred.filter(F.col("threshold") == float(t))
        # AUC undefined if only one class present
        n_pos = sub.filter(F.col("label") == 1).limit(1).count()
        n_neg = sub.filter(F.col("label") == 0).limit(1).count()
        if n_pos and n_neg:
            m = evaluate(sub)
        else:
            m = {"auc_roc": None, "auc_pr": None, "accuracy": None, "f1": None}
        m2 = {"threshold": t, **m}
        per_thr.append(m2)
        print(f"  threshold={t}: {m}")

    import pandas as pd
    save_table(pd.DataFrame(per_thr), os.path.join(config.TABLES_DIR, "model_metrics_by_threshold.csv"))
    save_table(pd.DataFrame([{"scope": "overall", **overall}]),
               os.path.join(config.TABLES_DIR, "model_metrics_overall.csv"))

    # Persist model + metadata
    print(f"saving model -> {MODEL_DIR}")
    model.write().overwrite().save(MODEL_DIR)

    meta = {
        "model_kind": args.model,
        "horizon_minutes": config.HORIZON_MINUTES,
        "base_feature_cols": tf.BASE_FEATURE_COLS,
        "uses_threshold_as_feature": True,
        "train_thresholds": sorted(set(args.thresholds)),
        "test_year": args.test_year,
        "train_years": [y for y in config.YEARS if y < args.test_year],
        "max_train_rows": args.max_train_rows,
        "sample_fraction": args.sample_fraction,
        "n_train": n_train,
        "n_test": n_test,
        "metrics_overall": overall,
        "metrics_by_threshold": per_thr,
    }
    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote metadata -> {META_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
