"""
Task 2 - Inference: estimate P(next-30-min rainfall > threshold) for a given
(station, prediction time, threshold).

The trained pipeline needs the same recent-rainfall context features it saw in
training. We reconstruct those from the cleaned Parquet by taking the readings
for the station up to (and including) the prediction time, computing the same
rolling look-back features, and scoring the single row for the requested
threshold.

Usage
-----
    python src/task2_predict.py --station S77 \
        --time "2024-06-15T14:00:00" --threshold 1.0

Prints the probability that the station accumulates more than <threshold> mm
of rain in the 30 minutes after the prediction time.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import task2_features as tf  # noqa: E402
from spark_utils import get_spark  # noqa: E402

from pyspark.ml import PipelineModel  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql import Window  # noqa: E402

MODEL_DIR = os.path.join(config.MODELS_DIR, "rainfall_model")


def compute_feature_row(spark, data_path, station_id, when_ts, threshold):
    """Build the single feature row for the query, using only data up to when_ts."""
    df = spark.read.parquet(data_path).filter(F.col("station_id") == station_id)
    # restrict to a look-back window (a few hours is plenty for our features)
    df = df.filter(F.col("ts") <= F.to_timestamp(F.lit(when_ts)))
    df = df.filter(F.col("ts") >= F.to_timestamp(F.lit(when_ts)) - F.expr("INTERVAL 6 HOURS"))
    if df.limit(1).count() == 0:
        raise SystemExit(
            f"No readings for station {station_id} in the 6h before {when_ts}."
        )

    # recompute calendar features from the prediction timestamp itself
    df = tf.add_time_features(df)

    ts_secs = F.col("ts").cast("long")
    w = Window.partitionBy("station_id").orderBy(ts_secs)
    for mins in tf.LAG_WINDOWS_MIN:
        wb = w.rangeBetween(-(mins * 60), 0)
        df = df.withColumn(f"roll_{mins}m_mm", F.sum("rain_mm").over(wb))
    df = df.withColumn("lag_5m_mm", F.coalesce(F.lag("rain_mm", 1).over(w), F.lit(0.0)))
    is_wet = (F.col("rain_mm") > 0).cast("int")
    df = df.withColumn("is_wet", is_wet)
    wet_ts = F.when(F.col("is_wet") == 1, ts_secs)
    df = df.withColumn(
        "last_wet_secs",
        F.max(wet_ts).over(w.rowsBetween(Window.unboundedPreceding, 0)),
    )
    df = df.withColumn(
        "mins_since_rain",
        F.when(F.col("last_wet_secs").isNull(), F.lit(1440.0))
        .otherwise(F.least(F.lit(1440.0), (ts_secs - F.col("last_wet_secs")) / 60.0)),
    )

    # take the latest available row at/just before the prediction time
    row = df.orderBy(F.col("ts").desc()).limit(1)
    row = row.withColumn("threshold", F.lit(float(threshold)))
    for c in tf.BASE_FEATURE_COLS:
        row = row.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))
    return row


def extract_probability(pred_row):
    """Pull P(label=1) out of the MLlib probability vector."""
    p = pred_row.select("probability", "prediction").first()
    prob_vec = p["probability"]
    return float(prob_vec[1]), float(p["prediction"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=config.CLEAN_PARQUET)
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--station", required=True, help="station_id, e.g. S77")
    ap.add_argument("--time", required=True,
                    help="prediction time, ISO e.g. 2024-06-15T14:00:00 (SGT)")
    ap.add_argument("--threshold", type=float, required=True, help="mm over 30 min")
    args = ap.parse_args()

    spark = get_spark("task2_predict", driver_memory="6g", shuffle_partitions=16)
    model = PipelineModel.load(args.model)

    row = compute_feature_row(spark, args.data, args.station, args.time, args.threshold)

    # --- diagnostic: show the exact feature row being scored --------------
    diag = row.select(
        "ts", "hour", "rain_mm", "lag_5m_mm",
        "roll_30m_mm", "roll_60m_mm", "roll_180m_mm", "mins_since_rain",
    ).first()

    pred = model.transform(row)
    prob, label = extract_probability(pred)

    print("=" * 60)
    print(f"Station    : {args.station}")
    print(f"Time (SGT) : {args.time}")
    print(f"Threshold  : {args.threshold} mm over next 30 min")
    print("-" * 60)
    print("Feature row actually scored (context up to prediction time):")
    print(f"  matched reading ts : {diag['ts']}  (hour={diag['hour']})")
    print(f"  rain_mm (this 5min): {diag['rain_mm']:.3f}")
    print(f"  lag_5m_mm          : {diag['lag_5m_mm']:.3f}")
    print(f"  roll_30m_mm        : {diag['roll_30m_mm']:.3f}")
    print(f"  roll_60m_mm        : {diag['roll_60m_mm']:.3f}")
    print(f"  roll_180m_mm       : {diag['roll_180m_mm']:.3f}")
    print(f"  mins_since_rain    : {diag['mins_since_rain']:.1f}")
    print("-" * 60)
    print(f"P(rain > {args.threshold} mm in next 30 min) = {prob:.4f}")
    print(f"Predicted class (0.5 cut)                = {int(label)}")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
