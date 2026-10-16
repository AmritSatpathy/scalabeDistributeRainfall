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
from spark_utils import get_spark  # noqa: E402

from pyspark.ml import PipelineModel  # noqa: E402
from pyspark.ml.feature import VectorAssembler  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.ml.regression import IsotonicRegressionModel  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402


def compute_feature_row(spark, data_path, station_id, when_ts, threshold, max_staleness_minutes=0):
    query_ts = F.to_timestamp(F.lit(when_ts))
    df = spark.read.parquet(data_path).filter(F.col("station_id") == station_id)
    df = df.filter((F.col("ts") <= query_ts) & (F.col("ts") >= query_ts - F.expr("INTERVAL 24 HOURS")))
    df = tf.add_time_features(df)
    df = tf.add_history_features(df)
    candidate = (
        df.withColumn("staleness_minutes", (query_ts.cast("long") - F.col("ts").cast("long")) / 60.0)
        .filter(F.col("staleness_minutes") <= float(max_staleness_minutes))
        .orderBy(F.col("ts").desc()).limit(1)
    )
    if candidate.count() == 0:
        policy = "exact timestamp" if max_staleness_minutes == 0 else f"≤{max_staleness_minutes} min staleness"
        raise SystemExit(f"No supported reading for station {station_id} at {when_ts} under policy {policy}.")
    candidate = candidate.withColumn("threshold", F.lit(float(threshold)))
    for column in tf.BASE_FEATURE_COLS:
        candidate = candidate.withColumn(column, F.coalesce(F.col(column), F.lit(0.0)))
    return candidate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--calibrator", default=None)
    ap.add_argument("--station", required=True)
    ap.add_argument("--time", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--max-staleness-minutes", type=int, default=0)
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="metadata")
    args = ap.parse_args()
    args.data = args.data or latest_artifact("clean", "clean_parquet") or config.CLEAN_PARQUET
    args.model = args.model or latest_artifact("train", "model") or os.path.join(config.MODELS_DIR, "rainfall_model")
    args.calibrator = args.calibrator or latest_artifact("train", "calibrator")
    run = RunContext("predict", vars(args), hash_mode=args.hash_mode)
    run.record_input("clean_parquet", args.data)
    run.record_input("model", args.model)
    if args.calibrator:
        run.record_input("calibrator", args.calibrator)

    spark = get_spark("task2_predict", driver_memory="6g", shuffle_partitions=16)
    model = PipelineModel.load(args.model)
    row = compute_feature_row(spark, args.data, args.station, args.time, args.threshold, args.max_staleness_minutes)
    pred = model.transform(row).withColumn("raw_probability", vector_to_array("probability")[1])
    if args.calibrator:
        assembler = VectorAssembler(inputCols=["raw_probability"], outputCol="calibration_features")
        calibrator = IsotonicRegressionModel.load(args.calibrator)
        pred = calibrator.transform(assembler.transform(pred))
    else:
        pred = pred.withColumn("calibrated_probability", F.col("raw_probability"))
    result = pred.select(
        "ts", "staleness_minutes", "raw_probability", "calibrated_probability",
        "rain_mm", "lag_5m_mm", "lag_5m_available", "roll_30m_mm",
        "roll_60m_mm", "roll_180m_mm", "mins_since_rain",
    ).first().asDict()
    payload = {
        "station_id": args.station, "requested_time_sgt": args.time,
        "forecast_origin_sgt": str(result["ts"]), "staleness_minutes": result["staleness_minutes"],
        "threshold_mm": args.threshold, "raw_probability": result["raw_probability"],
        "calibrated_probability": result["calibrated_probability"],
        "feature_diagnostics": {k: result[k] for k in (
            "rain_mm", "lag_5m_mm", "lag_5m_available", "roll_30m_mm",
            "roll_60m_mm", "roll_180m_mm", "mins_since_rain")},
    }
    print(json.dumps(payload, indent=2, default=str))
    run.write_json("prediction", payload)
    spark.stop(); run.finalize()


if __name__ == "__main__":
    main()
