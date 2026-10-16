"""Create one immutable, split-balanced feature sample for all deployments."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from run_artifacts import RunContext, latest_artifact  # noqa: E402
from spark_utils import get_spark  # noqa: E402

from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None)
    ap.add_argument("--validation-year", type=int, default=2023)
    ap.add_argument("--test-year", type=int, default=2024)
    ap.add_argument("--train-contexts", type=int, default=200_000)
    ap.add_argument("--validation-contexts", type=int, default=50_000)
    ap.add_argument("--test-contexts", type=int, default=50_000)
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="full")
    args = ap.parse_args()
    args.features = args.features or latest_artifact("features", "features_base")
    if not args.features:
        raise SystemExit("No tracked feature dataset. Run task2_build_features.py first.")
    run = RunContext("benchmark_sample", vars(args), hash_mode=args.hash_mode)
    run.record_input("features_base", args.features)
    spark = get_spark("prepare_benchmark_sample", driver_memory="8g", shuffle_partitions=64)
    base = spark.read.parquet(args.features)
    val_start = F.to_timestamp(F.lit(f"{args.validation_year}-01-01 00:00:00"))
    test_start = F.to_timestamp(F.lit(f"{args.test_year}-01-01 00:00:00"))
    test_end = F.to_timestamp(F.lit(f"{args.test_year + 1}-01-01 00:00:00"))
    splits = [
        ("train", base.filter(F.col("label_end_ts") < val_start), args.train_contexts),
        ("validation", base.filter((F.col("ts") >= val_start) & (F.col("label_end_ts") < test_start)), args.validation_contexts),
        ("test", base.filter((F.col("ts") >= test_start) & (F.col("label_end_ts") < test_end)), args.test_contexts),
    ]
    selected = None
    for name, frame, limit in splits:
        w = Window.orderBy(F.xxhash64("sample_id"), "sample_id")
        part = frame.withColumn("_rank", F.row_number().over(w)).filter(F.col("_rank") <= limit).drop("_rank").withColumn("benchmark_split", F.lit(name))
        selected = part if selected is None else selected.unionByName(part)
    feature_out = run.artifact_path("benchmark_features")
    ids_out = run.artifact_path("sample_ids")
    selected.write.mode("overwrite").partitionBy("benchmark_split", "year").parquet(feature_out)
    selected.select("sample_id", "benchmark_split").write.mode("overwrite").partitionBy("benchmark_split").parquet(ids_out)
    counts = selected.groupBy("benchmark_split").count().orderBy("benchmark_split")
    counts_path = run.artifact_path("sample_counts.csv")
    counts.toPandas().to_csv(counts_path, index=False)
    spark.stop()
    run.record_artifact("benchmark_features", feature_out)
    run.record_artifact("sample_ids", ids_out)
    run.record_artifact("sample_counts", counts_path)
    run.finalize()
    print(f"tracked frozen benchmark sample: {run.run_id} -> {run.path}")


if __name__ == "__main__":
    main()
