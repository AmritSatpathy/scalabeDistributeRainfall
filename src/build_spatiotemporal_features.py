"""Materialise the reusable spatiotemporal rainfall feature table.

This stage reads the combined cleaned observations, computes temporal and
geospatial features once in Spark, and writes immutable Parquet partitioned by
year and month for the lagged storm and antecedent-condition maps.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import spatiotemporal_features as sf  # noqa: E402
from progress_utils import SparkProgressMonitor, driver_progress  # noqa: E402
from run_artifacts import RunContext, latest_artifact, publish_copy  # noqa: E402
from spark_utils import get_spark  # noqa: E402

from pyspark import StorageLevel  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

DEFAULT_OUTPUT = os.path.join(config.PROCESSED_DIR, "spatiotemporal_features")
FEATURE_VERSION = "spatiotemporal_v2_20261016"

FEATURE_PROFILES = {
    # Recommended assignment profile: a concise set of interpretable features,
    # annual station partitions and no unbounded persistence calculations.
    "minimal": {
        "exact_lags_minutes": (5, 15, 30, 60),
        "rolling_windows_minutes": (30, 180, 1440),
        "long_state_days": (),
        "rolling_statistics": ("sum", "max", "wet_fraction", "coverage"),
        "include_persistence": False,
        "partition_by_year": True,
    },
    # Required by the planned geospatial EDA figures. This is the fastest
    # complete descriptive profile.
    "eda": {
        "exact_lags_minutes": (5, 15, 30, 60),
        "rolling_windows_minutes": (30, 180, 1440),
        "long_state_days": (),
        "rolling_statistics": ("sum", "max", "wet_fraction", "coverage"),
        "include_persistence": False,
        "partition_by_year": True,
    },
    # Recommended default: strong recency, daily/weekly recurrence, all rolling
    # horizons and a seven-day state without expensive distant ablations.
    "core": {
        "exact_lags_minutes": (5, 10, 15, 30, 60, 180, 1440, 10080),
        "rolling_windows_minutes": sf.ROLLING_WINDOWS_MINUTES,
        "long_state_days": (7,),
        "rolling_statistics": sf.ALL_ROLLING_STATISTICS,
        "include_persistence": True,
        "partition_by_year": True,
    },
    # Research extension profile with every available temporal feature.
    "full": {
        "exact_lags_minutes": sf.EXACT_LAGS_MINUTES,
        "rolling_windows_minutes": sf.ROLLING_WINDOWS_MINUTES,
        "long_state_days": sf.LONG_STATE_DAYS,
        "rolling_statistics": sf.ALL_ROLLING_STATISTICS,
        "include_persistence": True,
        "partition_by_year": False,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Cleaned Parquet input")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="Legacy publish destination")
    parser.add_argument("--sample-days", type=int, default=None)
    parser.add_argument("--profile", choices=sorted(FEATURE_PROFILES), default="minimal")
    parser.add_argument(
        "--forecast-stride-minutes", type=int, default=15,
        help="Keep one forecast origin every N minutes after computing histories.",
    )
    parser.add_argument("--wet-threshold-mm", type=float, default=0.0)
    parser.add_argument("--neighbour-count", type=int, default=4)
    spatial = parser.add_mutually_exclusive_group()
    spatial.add_argument(
        "--with-neighbours", action="store_true",
        help="Add nearest-station features. This materially increases runtime.",
    )
    spatial.add_argument(
        "--without-neighbours", action="store_true",
        help="Deprecated compatibility flag; neighbours are now off by default.",
    )
    parser.add_argument("--shuffle-partitions", type=int, default=192)
    parser.add_argument("--output-partitions", type=int, default=96)
    parser.add_argument("--station-buckets", type=int, default=4)
    parser.add_argument("--hash-mode", choices=["full", "metadata"], default="full")
    parser.add_argument("--publish-legacy", action="store_true")
    args = parser.parse_args()

    if args.neighbour_count < 1:
        parser.error("--neighbour-count must be at least 1")
    if args.shuffle_partitions < 1 or args.output_partitions < 1 or args.station_buckets < 1:
        parser.error("partition and bucket counts must be positive")
    if args.forecast_stride_minutes < 5 or 60 % args.forecast_stride_minutes != 0:
        parser.error("--forecast-stride-minutes must be at least 5 and divide 60 exactly")
    config.ensure_dirs()
    args.data = args.data or latest_artifact("clean", "clean_parquet") or config.CLEAN_PARQUET
    run = RunContext("spatiotemporal_features", vars(args), hash_mode=args.hash_mode)
    with driver_progress("[1/8] Fingerprinting the cleaned Parquet input"):
        run.record_input("clean_parquet", args.data)
    spark = get_spark(
        "build_spatiotemporal_features",
        driver_memory="12g",
        shuffle_partitions=args.shuffle_partitions,
    )
    # With only 91 station histories, aggressive adaptive coalescing placed
    # multiple long histories in one task. Preserve the requested partitioning
    # so tasks finish sooner and all local cores remain fed.
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")

    print("\n[2/8] Defining temporal windows, exact lags, outcomes and spatial joins")
    observations = spark.read.parquet(args.data)
    if args.sample_days is not None:
        observations = observations.filter(F.col("dayofyear") <= F.lit(args.sample_days))

    profile = FEATURE_PROFILES[args.profile]
    series_partition_columns = (
        ("station_id", "year") if profile["partition_by_year"] else ("station_id",)
    )
    temporal, _ = sf.build_spatiotemporal_features(
        observations,
        neighbour_count=args.neighbour_count,
        wet_threshold_mm=args.wet_threshold_mm,
        include_neighbours=False,
        exact_lags_minutes=profile["exact_lags_minutes"],
        rolling_windows_minutes=profile["rolling_windows_minutes"],
        long_state_days=profile["long_state_days"],
        rolling_statistics=profile["rolling_statistics"],
        include_persistence=profile["include_persistence"],
        series_partition_columns=series_partition_columns,
    )
    temporal = (
        temporal.withColumn(
            "sample_id",
            F.sha2(F.concat_ws("|", "station_id", F.col("ts").cast("string")), 256),
        )
        .withColumn("feature_version", F.lit(f"{FEATURE_VERSION}_{args.profile}"))
        .withColumn("label_year", F.year("label_end_ts"))
    )
    # Compute histories on the full five-minute series, then retain fewer
    # forecast origins. This preserves feature correctness while reducing the
    # written dataset and downstream geographic analysis by stride/5.
    temporal = temporal.filter(
        F.pmod(F.minute("ts"), F.lit(args.forecast_stride_minutes)) == 0
    )

    neighbour_map = None
    if args.with_neighbours:
        # The temporal frame feeds both the final output and neighbour branch.
        # Materialise it once on disk to prevent Spark from recomputing every
        # temporal window on both sides of the spatial join.
        temporal = temporal.persist(StorageLevel.DISK_ONLY)
        with SparkProgressMonitor(
            spark,
            "[3/8] Materialising temporal features once for spatial reuse",
        ):
            temporal.count()
        stations = observations.select("station_id", "lat", "lon").dropDuplicates(["station_id"])
        neighbour_map = sf.build_neighbour_map(stations, args.neighbour_count)
        featured = sf.add_neighbour_features(temporal, neighbour_map, args.wet_threshold_mm)
    else:
        print("\n[3/8] Spatial materialisation skipped; neighbour features are opt-in")
        featured = temporal

    output = run.artifact_path("spatiotemporal_features")
    output_frame = featured.withColumn(
        "_station_bucket",
        F.pmod(F.xxhash64("station_id"), F.lit(args.station_buckets)),
    )
    with SparkProgressMonitor(
        spark,
        "[4/8] Computing and writing the selected spatiotemporal feature table",
    ):
        (
            output_frame.repartition(
                args.output_partitions, "year", "month", "_station_bucket"
            )
            .drop("_station_bucket")
            .write.mode("overwrite")
            .partitionBy("year", "month")
            .parquet(output)
        )

    neighbour_path = None
    if neighbour_map is not None:
        neighbour_path = run.artifact_path("nearest_neighbours.csv")
        with SparkProgressMonitor(spark, "[5/8] Exporting the nearest-station lookup"):
            neighbour_pdf = neighbour_map.orderBy("station_id", "neighbour_rank").toPandas()
        neighbour_pdf.to_csv(neighbour_path, index=False)
    else:
        print("\n[5/8] Nearest-station lookup skipped")

    contract = {
        "feature_version": f"{FEATURE_VERSION}_{args.profile}",
        "feature_profile": args.profile,
        "history_interval": "(t-W, t]",
        "future_interval": "(t, t+30 minutes]",
        "exact_lags_minutes": profile["exact_lags_minutes"],
        "rolling_windows_minutes": profile["rolling_windows_minutes"],
        "long_state_days": profile["long_state_days"],
        "rolling_statistics": profile["rolling_statistics"],
        "include_persistence": profile["include_persistence"],
        "series_partition_columns": series_partition_columns,
        "forecast_stride_minutes": args.forecast_stride_minutes,
        "analysis_feature_columns": sf.analysis_feature_columns(
            exact_lags_minutes=profile["exact_lags_minutes"],
            rolling_windows_minutes=profile["rolling_windows_minutes"],
            long_state_days=profile["long_state_days"],
            rolling_statistics=profile["rolling_statistics"],
            include_persistence=profile["include_persistence"],
            include_neighbours=args.with_neighbours,
        ),
        "neighbour_count": args.neighbour_count if args.with_neighbours else None,
        "wet_threshold_mm": args.wet_threshold_mm,
    }
    print("\n[6/8] Writing the machine-readable feature contract")
    run.write_json("feature_contract", contract)

    with driver_progress("[7/8] Hashing and recording immutable feature artifacts"):
        run.record_artifact("spatiotemporal_features", output)
        if neighbour_path is not None:
            run.record_artifact("nearest_neighbours", neighbour_path)

    if args.publish_legacy:
        with driver_progress("[8/8] Publishing the legacy-compatible feature table"):
            publish_copy(output, args.out)
    else:
        print("\n[8/8] Legacy publish skipped (enable with --publish-legacy)")

    if args.with_neighbours:
        temporal.unpersist()
    spark.stop()
    run.finalize()
    print(f"tracked spatiotemporal feature run: {run.run_id} -> {run.path}")


if __name__ == "__main__":
    main()

