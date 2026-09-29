"""
Clean and standardise the raw yearly rainfall CSVs into a single, tidy
Parquet dataset partitioned by year.

Cleaning steps
--------------
1.  Read every raw CSV under data/raw with a fixed string schema (robust to
    any stray formatting) and stamp the source year.
2.  Parse ``timestamp`` as an SGT instant and derive calendar parts
    (year, month, day, hour, minute, day-of-week, date).
3.  Cast ``reading_value`` to double (rainfall mm per 5-minute interval).
4.  Drop physically impossible / unusable rows:
      * null or non-numeric reading_value
      * negative rainfall
      * absurdly large rainfall (> 60 mm in 5 min is beyond any credible
        Singapore 5-min intensity and indicates a bad record)
      * null station_id or timestamp
5.  De-duplicate on (station_id, timestamp), keeping the record with the
    latest ``update_timestamp`` (the most recent correction).
6.  Resolve a single canonical station name + coordinates per station_id
    (station names in the raw data alternate between the real name and the
    bare id; coordinates are stable).  Broadcast-join these back on.
7.  Write Parquet partitioned by year.

Usage
-----
    python src/clean_data.py                 # clean all downloaded years
    python src/clean_data.py --sample-days 3 # quick smoke test: keep first N days/year
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from spark_utils import get_spark  # noqa: E402

from pyspark.sql import DataFrame, Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql import types as T  # noqa: E402

RAW_SCHEMA = T.StructType(
    [
        T.StructField("date", T.StringType()),
        T.StructField("timestamp", T.StringType()),
        T.StructField("update_timestamp", T.StringType()),
        T.StructField("station_id", T.StringType()),
        T.StructField("station_name", T.StringType()),
        T.StructField("station_device_id", T.StringType()),
        T.StructField("location_longitude", T.StringType()),
        T.StructField("location_latitude", T.StringType()),
        T.StructField("reading_update_timestamp", T.StringType()),
        T.StructField("reading_value", T.StringType()),
        T.StructField("reading_type", T.StringType()),
        T.StructField("reading_unit", T.StringType()),
    ]
)

# Sanity bound: the highest credible 5-minute rainfall total. Singapore's
# most intense bursts are well under this; anything above is treated as an
# erroneous record.
MAX_5MIN_MM = 60.0


def read_raw(spark, years) -> DataFrame:
    frames = []
    for year in years:
        path = os.path.join(config.RAW_DIR, f"rainfall_{year}.csv")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        df = (
            spark.read.option("header", True)
            .schema(RAW_SCHEMA)
            .csv(path)
            .withColumn("src_year", F.lit(year))
        )
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No raw CSVs found under data/raw")
    out = frames[0]
    for f_ in frames[1:]:
        out = out.unionByName(f_)
    return out


def looks_like_id(col):
    """True when a station_name is just the bare station id (e.g. 'S216')."""
    return col.rlike(r"^S\d+$")


def clean(df: DataFrame, sample_days: int | None) -> DataFrame:
    # ---- parse core fields -------------------------------------------------
    df = df.withColumn("ts", F.to_timestamp("timestamp"))
    df = df.withColumn("rain_mm", F.col("reading_value").cast(T.DoubleType()))
    df = df.withColumn("lat", F.col("location_latitude").cast(T.DoubleType()))
    df = df.withColumn("lon", F.col("location_longitude").cast(T.DoubleType()))
    df = df.withColumn("upd_ts", F.to_timestamp("update_timestamp"))

    # ---- validity filter ---------------------------------------------------
    df = df.filter(
        F.col("station_id").isNotNull()
        & F.col("ts").isNotNull()
        & F.col("rain_mm").isNotNull()
        & (F.col("rain_mm") >= 0)
        & (F.col("rain_mm") <= MAX_5MIN_MM)
    )

    # optional smoke-test subset: first N days of each year
    if sample_days is not None:
        df = df.withColumn("doy", F.dayofyear("ts")).filter(F.col("doy") <= sample_days).drop("doy")

    # ---- de-duplicate on (station_id, ts) ---------------------------------
    w = Window.partitionBy("station_id", "ts").orderBy(F.col("upd_ts").desc_nulls_last())
    df = (
        df.withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    # ---- calendar features -------------------------------------------------
    df = (
        df.withColumn("year", F.year("ts"))
        .withColumn("month", F.month("ts"))
        .withColumn("day", F.dayofmonth("ts"))
        .withColumn("hour", F.hour("ts"))
        .withColumn("minute", F.minute("ts"))
        .withColumn("dayofweek", F.dayofweek("ts"))  # 1=Sun..7=Sat
        .withColumn("dayofyear", F.dayofyear("ts"))
        .withColumn("date", F.to_date("ts"))
    )
    return df


def resolve_stations(df: DataFrame) -> DataFrame:
    """One canonical name + coordinates per station_id."""
    # Prefer a real (non-id-like) name; break ties by frequency.
    name_counts = (
        df.groupBy("station_id", "station_name")
        .count()
        .withColumn("is_real", (~looks_like_id(F.col("station_name"))).cast("int"))
    )
    wname = Window.partitionBy("station_id").orderBy(
        F.col("is_real").desc(), F.col("count").desc()
    )
    best_name = (
        name_counts.withColumn("_r", F.row_number().over(wname))
        .filter(F.col("_r") == 1)
        .select("station_id", F.col("station_name").alias("station_name_std"))
    )

    # Median coordinate per station (robust to the odd stray value).
    coords = df.groupBy("station_id").agg(
        F.expr("percentile_approx(lat, 0.5)").alias("lat_std"),
        F.expr("percentile_approx(lon, 0.5)").alias("lon_std"),
    )

    stations = best_name.join(coords, "station_id", "inner")
    return stations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=config.YEARS)
    ap.add_argument("--sample-days", type=int, default=None,
                    help="Keep only the first N days of each year (smoke test)")
    ap.add_argument("--out", default=config.CLEAN_PARQUET)
    args = ap.parse_args()

    config.ensure_dirs()
    spark = get_spark("clean_rainfall", driver_memory="11g")

    raw = read_raw(spark, args.years)
    df = clean(raw, args.sample_days)

    stations = resolve_stations(df).cache()
    n_stations = stations.count()
    print(f"  resolved {n_stations} stations")

    df = (
        df.drop("station_name", "lat", "lon")
        .join(F.broadcast(stations), "station_id", "left")
        .withColumnRenamed("station_name_std", "station_name")
        .withColumnRenamed("lat_std", "lat")
        .withColumnRenamed("lon_std", "lon")
    )

    keep = [
        "station_id", "station_name", "lat", "lon",
        "ts", "date", "year", "month", "day", "hour", "minute",
        "dayofweek", "dayofyear", "rain_mm",
    ]
    df = df.select(*keep)

    out = args.out
    print(f"  writing Parquet -> {out} (partitioned by year)")
    (
        df.repartition("year")
        .write.mode("overwrite")
        .partitionBy("year")
        .parquet(out)
    )

    # Report row counts per year for the report/README.
    check = spark.read.parquet(out)
    print("  rows per year after cleaning:")
    check.groupBy("year").count().orderBy("year").show()
    print("  TOTAL rows:", check.count())

    # Persist the station master table for reference.
    stations.orderBy("station_id").toPandas().to_csv(
        os.path.join(config.TABLES_DIR, "stations.csv"), index=False
    )
    spark.stop()


if __name__ == "__main__":
    main()
