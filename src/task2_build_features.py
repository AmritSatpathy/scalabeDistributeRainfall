"""
Task 2 (stage 1) - Materialise the base feature table once.

The rolling look-back windows and the 30-minute-ahead label are expensive
per-station time-series computations. Computing them repeatedly (for every
count / fit / evaluate action) is wasteful and memory heavy, so we compute
them a single time here and persist the result as Parquet partitioned by
year. Training then just reads this table.

Output: data/processed/features_base (Parquet, partitioned by year)

Usage
-----
    python src/task2_build_features.py
    python src/task2_build_features.py --sample-days 20   # smoke test
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import task2_features as tf  # noqa: E402
from spark_utils import get_spark  # noqa: E402
from run_artifacts import RunContext, latest_artifact, publish_copy  # noqa: E402

from pyspark.sql import functions as F  # noqa: E402

FEATURES_BASE = os.path.join(config.PROCESSED_DIR, "features_base")
FEATURE_VERSION = "v2_exact_cadence_20261016"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=FEATURES_BASE,
                    help="Legacy publish path; immutable output always lives under output/runs.")
    ap.add_argument("--sample-days", type=int, default=None,
                    help="Keep only the first N days of each year (smoke test)")
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="full")
    ap.add_argument("--publish-legacy", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    args.data = args.data or latest_artifact("clean", "clean_parquet") or config.CLEAN_PARQUET
    run = RunContext("features", vars(args), hash_mode=args.hash_mode)
    run.record_input("clean_parquet", args.data)
    # More shuffle partitions so the per-station window sorts spread out.
    spark = get_spark("task2_build_features", driver_memory="11g", shuffle_partitions=200)

    df = spark.read.parquet(args.data)
    if args.sample_days is not None:
        df = df.filter(F.col("dayofyear") <= args.sample_days)

    base = tf.build_base_features(df)
    base = (
        base.withColumn("sample_id", F.sha2(F.concat_ws("|", "station_id", F.col("ts").cast("string")), 256))
        .withColumn("label_year", F.year("label_end_ts"))
        .withColumn("feature_version", F.lit(FEATURE_VERSION))
    )

    keep = (
        ["sample_id", "station_id", "ts", "label_end_ts", "year", "label_year",
         "feature_version", "future_30min_mm", "future_n", "future_complete"]
        + tf.BASE_FEATURE_COLS
    )
    base = base.select(*keep)

    immutable_out = run.artifact_path("features_base")
    print(f"writing base features -> {immutable_out}")
    (
        base.write.mode("overwrite")
        .partitionBy("year")
        .parquet(immutable_out)
    )

    chk = spark.read.parquet(immutable_out)
    print("feature rows per year:")
    chk.groupBy("year").count().orderBy("year").show()
    print("TOTAL feature rows:", chk.count())
    metadata = {
        "feature_version": FEATURE_VERSION,
        "history_interval": "(t-W, t]",
        "exact_lag_minutes": config.INTERVAL_MINUTES,
        "target_offsets_minutes": [config.INTERVAL_MINUTES * i for i in range(1, config.HORIZON_STEPS + 1)],
        "base_feature_cols": tf.BASE_FEATURE_COLS,
    }
    run.write_json("feature_contract", metadata)
    run.record_artifact("features_base", immutable_out)
    if args.publish_legacy:
        publish_copy(immutable_out, args.out)
    spark.stop()
    run.finalize()
    print(f"tracked feature run: {run.run_id} -> {run.path}")


if __name__ == "__main__":
    main()
