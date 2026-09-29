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

from pyspark.sql import functions as F  # noqa: E402

FEATURES_BASE = os.path.join(config.PROCESSED_DIR, "features_base")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=config.CLEAN_PARQUET)
    ap.add_argument("--out", default=FEATURES_BASE)
    ap.add_argument("--sample-days", type=int, default=None,
                    help="Keep only the first N days of each year (smoke test)")
    args = ap.parse_args()

    config.ensure_dirs()
    # More shuffle partitions so the per-station window sorts spread out.
    spark = get_spark("task2_build_features", driver_memory="11g", shuffle_partitions=200)

    df = spark.read.parquet(args.data)
    if args.sample_days is not None:
        df = df.filter(F.col("dayofyear") <= args.sample_days)

    base = tf.build_base_features(df)

    keep = (
        ["station_id", "year", "future_30min_mm"]
        + tf.BASE_FEATURE_COLS
    )
    base = base.select(*keep)

    print(f"writing base features -> {args.out}")
    (
        base.write.mode("overwrite")
        .partitionBy("year")
        .parquet(args.out)
    )

    chk = spark.read.parquet(args.out)
    print("feature rows per year:")
    chk.groupBy("year").count().orderBy("year").show()
    print("TOTAL feature rows:", chk.count())
    spark.stop()


if __name__ == "__main__":
    main()
