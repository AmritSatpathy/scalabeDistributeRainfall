"""Score one station/time/threshold using the exact training feature contract."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import task2_features as tf  # noqa: E402
from run_artifacts import RunContext, latest_artifact  # noqa: E402
from progress_utils import SparkProgressMonitor, StepProgress  # noqa: E402
from spark_utils import get_spark  # noqa: E402

from pyspark.ml import PipelineModel  # noqa: E402
from pyspark.ml.feature import VectorAssembler  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.ml.regression import IsotonicRegressionModel  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

CANONICAL_MODEL = os.path.join(config.MODELS_DIR, "rainfall_model")
CANONICAL_CALIBRATOR = os.path.join(config.MODELS_DIR, "rainfall_calibrator")
CANONICAL_METADATA = os.path.join(config.MODELS_DIR, "model_metadata.json")


def compute_feature_row(spark, feature_path, station_id, when_ts, threshold,
                        model_feature_cols, max_staleness_minutes=0):
    query_ts = F.to_timestamp(F.lit(when_ts))
    df = spark.read.parquet(feature_path).filter(F.col("station_id") == station_id)
    missing = sorted(set(model_feature_cols) - set(df.columns))
    if missing:
        raise SystemExit(f"Feature table is missing model columns: {missing}")
    df = df.filter((F.col("ts") <= query_ts) & (F.col("ts") >= query_ts - F.expr("INTERVAL 24 HOURS")))
    candidate = (
        df.withColumn("staleness_minutes", (query_ts.cast("long") - F.col("ts").cast("long")) / 60.0)
        .filter(F.col("staleness_minutes") <= float(max_staleness_minutes))
        .orderBy(F.col("ts").desc()).limit(1)
    )
    if candidate.count() == 0:
        policy = "exact timestamp" if max_staleness_minutes == 0 else f"≤{max_staleness_minutes} min staleness"
        raise SystemExit(f"No supported reading for station {station_id} at {when_ts} under policy {policy}.")
    candidate = candidate.withColumn("threshold", F.lit(float(threshold)))
    missing_values = candidate.select(
        *[F.sum(F.col(column).isNull().cast("int")).alias(column) for column in model_feature_cols]
    ).first().asDict()
    invalid = [column for column, count in missing_values.items() if count]
    if invalid:
        raise SystemExit(f"Selected forecast row has unavailable required features: {invalid}")
    return candidate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--calibrator", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--station", required=True)
    ap.add_argument("--time", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--max-staleness-minutes", type=int, default=0)
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="metadata")
    args = ap.parse_args()
    args.features = (
        args.features
        or latest_artifact("spatiotemporal_features", "spatiotemporal_features")
        or latest_artifact("features", "features_base")
    )
    if not args.features:
        raise SystemExit("No model-ready feature table found.")
    args.model = args.model or (CANONICAL_MODEL if os.path.exists(CANONICAL_MODEL) else latest_artifact("train", "model"))
    args.calibrator = args.calibrator or (CANONICAL_CALIBRATOR if os.path.exists(CANONICAL_CALIBRATOR) else latest_artifact("train", "calibrator"))
    args.metadata = args.metadata or (CANONICAL_METADATA if os.path.isfile(CANONICAL_METADATA) else latest_artifact("train", "model_metadata"))
    if not args.model:
        raise SystemExit("No fitted model found. Run the canonical task2_train.py command first.")
    metadata = {}
    if args.metadata:
        with open(args.metadata, encoding="utf-8") as fh:
            metadata = json.load(fh)
    model_feature_cols = metadata.get("base_model_feature_cols", tf.BASE_FEATURE_COLS)
    run = RunContext("predict", vars(args), hash_mode=args.hash_mode)
    run.record_input("features", args.features)
    run.record_input("model", args.model)
    if args.calibrator:
        run.record_input("calibrator", args.calibrator)
    if args.metadata:
        run.record_input("model_metadata", args.metadata)

    progress = StepProgress(3, "single-station rainfall prediction")
    spark = get_spark("task2_predict", driver_memory="6g", shuffle_partitions=16)
    model = PipelineModel.load(args.model)
    with SparkProgressMonitor(spark, "building past-only feature row"):
        row = compute_feature_row(
            spark, args.features, args.station, args.time, args.threshold,
            model_feature_cols, args.max_staleness_minutes,
        )
    progress.update("forecast features ready")
    pred = model.transform(row)
    if "probability" in pred.columns:
        pred = pred.withColumn("raw_probability", vector_to_array("probability")[1])
    else:
        margin = vector_to_array("rawPrediction")[1]
        pred = pred.withColumn("raw_probability", 1.0 / (1.0 + F.exp(-margin)))
    if args.calibrator:
        assembler = VectorAssembler(inputCols=["raw_probability"], outputCol="calibration_features")
        calibrator = IsotonicRegressionModel.load(args.calibrator)
        pred = calibrator.transform(assembler.transform(pred))
    else:
        pred = pred.withColumn("calibrated_probability", F.col("raw_probability"))
    diagnostic_candidates = [
        "rain_mm", "lag_5m_mm", "lag_5m_available", "lag_15m_mm",
        "lag_30m_mm", "lag_60m_mm", "roll_30m_mm", "roll_180m_mm",
        "roll_1440m_mm", "intensity_change_30m_mm", "mins_since_rain",
    ]
    diagnostics = [column for column in diagnostic_candidates if column in pred.columns]
    with SparkProgressMonitor(spark, "scoring and calibrating probability"):
        result = pred.select(
            "ts", "staleness_minutes", "raw_probability", "calibrated_probability",
            *diagnostics,
        ).first().asDict()
    progress.update("probability generated")
    payload = {
        "station_id": args.station, "requested_time_sgt": args.time,
        "forecast_origin_sgt": str(result["ts"]), "staleness_minutes": result["staleness_minutes"],
        "threshold_mm": args.threshold, "raw_probability": result["raw_probability"],
        "calibrated_probability": result["calibrated_probability"],
        "model_run_id": metadata.get("run_id"),
        "feature_version": metadata.get("feature_version"),
        "feature_diagnostics": {column: result[column] for column in diagnostics},
    }
    print(json.dumps(payload, indent=2, default=str))
    run.write_json("prediction", payload)
    spark.stop(); run.finalize()
    progress.finish("prediction artifact and hashes written")


if __name__ == "__main__":
    main()
