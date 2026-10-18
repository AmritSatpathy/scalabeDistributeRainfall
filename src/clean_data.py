"""
Clean and standardise the raw yearly rainfall CSVs into a single, tidy
Parquet dataset partitioned by year.

Cleaning steps
--------------
1.  Read the canonical ``data/raw/rainfall_<year>.csv`` file for every
    requested year with a fixed string schema and stamp the source year.
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
import json
import os
import sys
import threading
import time
from contextlib import contextmanager

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from spark_utils import get_spark  # noqa: E402
from run_artifacts import RunContext, publish_copy  # noqa: E402

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
DEFAULT_GRID_TOLERANCE_SECONDS = 60


class SparkProgressMonitor:
    """Display live task progress while a blocking Spark action runs.

    Spark actions block the Python driver, so the status tracker is polled from
    a small daemon thread. The monitor is observational only and does not alter
    Spark execution. In an interactive terminal it refreshes one line; in a
    redirected log it emits periodic timestamped lines.
    """

    SPINNER = ("|", "/", "-", "\\")

    def __init__(self, spark, label: str, interval_seconds: float = 2.0, width: int = 28):
        self.spark = spark
        self.label = label
        self.interval_seconds = interval_seconds
        self.width = width
        self.started = 0.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.is_tty = sys.stdout.isatty()
        self.last_log_bucket = -1

    def __enter__(self):
        self.started = time.monotonic()
        print(f"\n  {self.label}")
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.interval_seconds + 0.5)
        elapsed = time.monotonic() - self.started
        if self.is_tty:
            sys.stdout.write("\r" + " " * 140 + "\r")
            sys.stdout.flush()
        outcome = "FAILED" if exc_type is not None else "complete"
        print(f"  [{outcome}] {self.label} ({elapsed:,.1f} s)")
        return False

    def _poll(self) -> None:
        tick = 0
        while not self.stop_event.is_set():
            elapsed = time.monotonic() - self.started
            message = self._status_message(elapsed, tick)
            if self.is_tty:
                sys.stdout.write("\r" + message[:138].ljust(138))
                sys.stdout.flush()
            else:
                # Log once every ~10 seconds so redirected output remains useful
                # without producing thousands of almost-identical lines.
                bucket = int(elapsed // 10)
                if bucket != self.last_log_bucket:
                    print(message)
                    self.last_log_bucket = bucket
            tick += 1
            self.stop_event.wait(self.interval_seconds)

    def _status_message(self, elapsed: float, tick: int) -> str:
        try:
            tracker = self.spark.sparkContext.statusTracker()
            stage_infos = [
                tracker.getStageInfo(stage_id)
                for stage_id in tracker.getActiveStageIds()
            ]
            stage_infos = [info for info in stage_infos if info is not None]
            total = sum(int(info.numTasks) for info in stage_infos)
            completed = sum(int(info.numCompletedTasks) for info in stage_infos)
            active = sum(int(info.numActiveTasks) for info in stage_infos)
            failed = sum(int(info.numFailedTasks) for info in stage_infos)
            if total > 0:
                fraction = min(1.0, completed / total)
                filled = int(round(self.width * fraction))
                bar = "#" * filled + "-" * (self.width - filled)
                stage_ids = ",".join(str(info.stageId) for info in stage_infos)
                return (
                    f"  [{bar}] {100.0 * fraction:5.1f}% | stages {stage_ids} | "
                    f"tasks {completed}/{total}, active {active}, failed {failed} | "
                    f"elapsed {elapsed:,.0f}s"
                )
        except Exception as exc:
            return f"  [{self.SPINNER[tick % len(self.SPINNER)]}] Spark status unavailable: {exc} | elapsed {elapsed:,.0f}s"
        return (
            f"  [{self.SPINNER[tick % len(self.SPINNER)]}] waiting for/planning Spark stages | "
            f"elapsed {elapsed:,.0f}s"
        )


@contextmanager
def driver_progress(label: str, interval_seconds: float = 5.0):
    """Emit elapsed-time heartbeats for non-Spark work such as hashing."""
    started = time.monotonic()
    stop_event = threading.Event()
    is_tty = sys.stdout.isatty()

    def poll():
        tick = 0
        while not stop_event.is_set():
            elapsed = time.monotonic() - started
            message = f"  [{SparkProgressMonitor.SPINNER[tick % 4]}] {label} | elapsed {elapsed:,.0f}s"
            if is_tty:
                sys.stdout.write("\r" + message[:138].ljust(138))
                sys.stdout.flush()
            elif tick % max(1, int(10 / interval_seconds)) == 0:
                print(message)
            tick += 1
            stop_event.wait(interval_seconds)

    print(f"\n  {label}")
    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=interval_seconds + 0.5)
        elapsed = time.monotonic() - started
        if is_tty:
            sys.stdout.write("\r" + " " * 140 + "\r")
            sys.stdout.flush()
        print(f"  [complete] {label} ({elapsed:,.1f} s)")


def discover_raw_files(years) -> dict[int, str]:
    """Return the canonical ``rainfall_<year>.csv`` input for each year.

    ``download_data.py`` and the mainline branch use this filename contract.
    Keeping one deterministic path per year prevents duplicate LFS objects and
    ensures cleaning, modelling, and geospatial analysis share the same source.
    """
    requested = sorted(set(int(year) for year in years))
    matches = {
        year: os.path.join(config.RAW_DIR, f"rainfall_{year}.csv")
        for year in requested
    }
    missing = [year for year, path in matches.items() if not os.path.isfile(path)]
    if missing:
        expected = ", ".join(f"rainfall_{year}.csv" for year in missing)
        raise FileNotFoundError(
            f"Missing canonical raw CSVs under {config.RAW_DIR}: {expected}"
        )
    return matches


def read_raw(spark, years) -> DataFrame:
    raw_files = discover_raw_files(years)
    frames = []
    for year, path in raw_files.items():
        print(f"  [input] {year}: {path}")
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


def clean_with_audit(
    df: DataFrame,
    sample_days: int | None,
    max_5min_mm: float = MAX_5MIN_MM,
    grid_tolerance_seconds: int = DEFAULT_GRID_TOLERANCE_SECONDS,
):
    """Apply mutually exclusive cleaning rules and return audit-ready frames."""
    # ---- parse core fields -------------------------------------------------
    df = df.withColumn("raw_ts", F.to_timestamp("timestamp"))
    df = df.withColumn("rain_mm", F.col("reading_value").cast(T.DoubleType()))
    df = df.withColumn("lat", F.col("location_latitude").cast(T.DoubleType()))
    df = df.withColumn("lon", F.col("location_longitude").cast(T.DoubleType()))
    df = df.withColumn("upd_ts", F.to_timestamp("update_timestamp"))
    df = df.withColumn("reading_upd_ts", F.to_timestamp("reading_update_timestamp"))

    # Source observations commonly arrive at hh:mm:59. Map each record to the
    # nearest canonical five-minute endpoint and retain the shift for audit.
    raw_epoch = F.col("raw_ts").cast("long")
    canonical_epoch = (F.floor((raw_epoch + F.lit(150)) / F.lit(300)) * F.lit(300)).cast("long")
    df = (
        df.withColumn("ts", canonical_epoch.cast("timestamp"))
        .withColumn("slot_shift_seconds", canonical_epoch - raw_epoch)
    )

    # optional smoke-test subset: first N days of each year
    if sample_days is not None:
        df = df.filter(F.dayofyear("raw_ts") <= sample_days)

    # Assign one priority-ordered reason so rule-level losses add to the raw total.
    df = df.withColumn(
        "drop_reason",
        F.when(F.col("station_id").isNull() | (F.trim("station_id") == ""), "missing_station")
        .when(F.col("raw_ts").isNull(), "invalid_timestamp")
        .when(F.abs(F.col("slot_shift_seconds")) > F.lit(int(grid_tolerance_seconds)), "timestamp_off_grid")
        .when(F.col("rain_mm").isNull(), "invalid_rainfall")
        .when(F.col("rain_mm") < 0, "negative_rainfall")
        .when(F.col("rain_mm") > F.lit(float(max_5min_mm)), "above_screening_bound"),
    )
    rule_audit = (
        df.groupBy(F.col("src_year").alias("year"), "drop_reason")
        .count().withColumnRenamed("count", "rows")
    )
    valid = df.filter(F.col("drop_reason").isNull()).drop("drop_reason")

    # ---- de-duplicate on (station_id, ts) ---------------------------------
    # The feed can re-issue a reading as corrections arrive. We keep exactly one
    # row per (station, 5-min timestamp): the one with the LATEST update_timestamp
    # (the most recent correction). row_number()==1 over that ordering picks it.
    # Exactly one reading per slot is essential for correct rolling sums later.
    w = Window.partitionBy("station_id", "ts").orderBy(F.col("upd_ts").desc_nulls_last())
    ranked = valid.withColumn("_rn", F.row_number().over(w))
    duplicate_audit = (
        ranked.filter(F.col("_rn") > 1)
        .groupBy(F.col("src_year").alias("year")).count()
        .withColumn("drop_reason", F.lit("duplicate_station_timestamp"))
        .withColumnRenamed("count", "rows")
        .select("year", "drop_reason", "rows")
    )
    df = ranked.filter(F.col("_rn") == 1).drop("_rn")

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
    return df, rule_audit.unionByName(duplicate_audit)


def clean(df: DataFrame, sample_days: int | None) -> DataFrame:
    """Backward-compatible cleaning entry point without returning audit data."""
    return clean_with_audit(df, sample_days)[0]


def audit_stage_counts(cleaned: DataFrame, rule_audit: DataFrame):
    raw_counts = (
        rule_audit.filter(
            F.col("drop_reason").isNull()
            | (F.col("drop_reason") != "duplicate_station_timestamp")
        ).groupBy("year").agg(F.sum("rows").alias("raw_rows"))
    )
    clean_counts = cleaned.groupBy("year").count().withColumnRenamed("count", "cleaned_rows")
    removed = (
        rule_audit.filter(F.col("drop_reason").isNotNull())
        .groupBy("year").pivot("drop_reason").sum("rows").na.fill(0)
    )
    return raw_counts.join(clean_counts, "year", "outer").join(removed, "year", "outer").na.fill(0).orderBy("year")


def cadence_audit(cleaned: DataFrame):
    w = Window.partitionBy("station_id").orderBy("ts")
    d = (
        cleaned.withColumn("prev_ts", F.lag("ts").over(w))
        .withColumn("gap_seconds", F.col("ts").cast("long") - F.col("prev_ts").cast("long"))
        .withColumn("off_grid", (F.pmod(F.col("ts").cast("long"), F.lit(300)) != 0).cast("int"))
        .withColumn("gap_over_5m", (F.col("gap_seconds") > 300).cast("int"))
        .withColumn(
            "estimated_missing_slots",
            F.when(F.col("gap_seconds") > 300, F.floor(F.col("gap_seconds") / 300) - 1).otherwise(0),
        )
    )
    return (
        d.groupBy("year").agg(
            F.count("*").alias("cleaned_rows"),
            F.sum("off_grid").alias("off_grid_timestamps"),
            F.sum("gap_over_5m").alias("gaps_over_5m"),
            F.sum("estimated_missing_slots").alias("estimated_missing_slots"),
            F.max("gap_seconds").alias("max_gap_seconds"),
            F.min("slot_shift_seconds").alias("min_slot_shift_seconds"),
            F.max("slot_shift_seconds").alias("max_slot_shift_seconds"),
        ).orderBy("year")
    )


def station_history_audit(cleaned: DataFrame):
    """Summarise device and observed-coordinate continuity by station/year."""
    return (
        cleaned.groupBy("station_id", "src_year")
        .agg(
            F.min("raw_ts").alias("first_observation"),
            F.max("raw_ts").alias("last_observation"),
            F.countDistinct("station_device_id").alias("device_ids"),
            F.countDistinct(F.struct("lat", "lon")).alias("coordinate_pairs"),
            F.expr("percentile_approx(lat, 0.5)").alias("median_lat"),
            F.expr("percentile_approx(lon, 0.5)").alias("median_lon"),
        )
        .withColumnRenamed("src_year", "year")
        .orderBy("station_id", "year")
    )


def extreme_day_audit(cleaned: DataFrame):
    daily = cleaned.groupBy("year", "date", "station_id").agg(
        F.sum("rain_mm").alias("daily_mm"),
        F.count("*").alias("observed_intervals"),
        F.max("rain_mm").alias("max_5min_mm"),
    )
    top = daily.orderBy(F.col("daily_mm").desc()).limit(25)
    trace_keys = top.select("station_id", "date")
    trace = (
        cleaned.join(F.broadcast(trace_keys), ["station_id", "date"], "inner")
        .select("station_id", "date", "ts", "rain_mm", "upd_ts")
        .orderBy(F.col("date"), F.col("station_id"), F.col("ts"))
    )
    return top, trace


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
    ap.add_argument("--max-5min-mm", type=float, default=MAX_5MIN_MM)
    ap.add_argument("--grid-tolerance-seconds", type=int, default=DEFAULT_GRID_TOLERANCE_SECONDS)
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="full")
    ap.add_argument("--publish-legacy", action="store_true",
                    help="Also copy the immutable run output to the legacy --out path.")
    args = ap.parse_args()

    config.ensure_dirs()
    run = RunContext("clean", vars(args), hash_mode=args.hash_mode)
    spark = get_spark("clean_rainfall", driver_memory="11g")

    print("\n[1/10] Discovering annual CSV inputs")
    raw = read_raw(spark, args.years)
    with driver_progress("Fingerprinting raw CSV inputs"):
        run.record_input("raw_csvs", config.RAW_DIR)

    print("\n[2/10] Defining parsing, cleaning, timestamp alignment and deduplication")
    df, rule_audit = clean_with_audit(
        raw, args.sample_days, args.max_5min_mm, args.grid_tolerance_seconds
    )

    station_history = station_history_audit(df)

    stations = resolve_stations(df).cache()
    with SparkProgressMonitor(spark, "[3/10] Resolving canonical station names and coordinates"):
        n_stations = stations.count()
    print(f"  resolved {n_stations} stations")

    df = (
        df.withColumn("observed_lat", F.col("lat"))
        .withColumn("observed_lon", F.col("lon"))
        .drop("station_name", "lat", "lon")
        .join(F.broadcast(stations), "station_id", "left")
        .withColumnRenamed("station_name_std", "station_name")
        .withColumnRenamed("lat_std", "lat")
        .withColumnRenamed("lon_std", "lon")
    )

    keep = [
        "station_id", "station_name", "station_device_id", "lat", "lon",
        "observed_lat", "observed_lon",
        "raw_ts", "slot_shift_seconds", "src_year",
        "ts", "date", "year", "month", "day", "hour", "minute",
        "dayofweek", "dayofyear", "rain_mm", "upd_ts", "reading_upd_ts",
    ]
    df = df.select(*keep)

    out = run.artifact_path("rainfall_clean")
    with SparkProgressMonitor(spark, f"[4/10] Writing combined clean Parquet by year -> {out}"):
        (
            df.repartition("year")
            .write.mode("overwrite")
            .partitionBy("year")
            .parquet(out)
        )

    # Persist all audit evidence beside the cleaned dataset.
    check = spark.read.parquet(out)
    audit = audit_stage_counts(check, rule_audit)
    cadence = cadence_audit(check)
    extreme, trace = extreme_day_audit(check)
    audit_path = run.artifact_path("cleaning_rule_audit.csv")
    cadence_path = run.artifact_path("cadence_audit.csv")
    extreme_path = run.artifact_path("extreme_station_days.csv")
    trace_path = run.artifact_path("extreme_day_trace.csv")
    history_path = run.artifact_path("station_history.csv")
    with SparkProgressMonitor(spark, "[5/10] Computing rule-level cleaning audit"):
        audit_pdf = audit.toPandas()
    audit_pdf.to_csv(audit_path, index=False)

    with SparkProgressMonitor(spark, "[6/10] Computing cadence and missing-slot audit"):
        cadence_pdf = cadence.toPandas()
    cadence_pdf.to_csv(cadence_path, index=False)

    with SparkProgressMonitor(spark, "[7/10] Finding extreme station-days and their five-minute traces"):
        extreme_pdf = extreme.toPandas()
        trace_pdf = trace.toPandas()
    extreme_pdf.to_csv(extreme_path, index=False)
    trace_pdf.to_csv(trace_path, index=False)

    with SparkProgressMonitor(spark, "[8/10] Summarising station, device and coordinate continuity"):
        station_history_pdf = station_history.toPandas()
        stations_pdf = stations.orderBy("station_id").toPandas()
    station_history_pdf.to_csv(history_path, index=False)

    # Persist the station master table for reference.
    stations_path = run.artifact_path("stations.csv")
    stations_pdf.to_csv(stations_path, index=False)
    with driver_progress("[9/10] Hashing and recording immutable output artifacts"):
        for name, path in {
            "clean_parquet": out, "cleaning_rule_audit": audit_path,
            "cadence_audit": cadence_path, "extreme_station_days": extreme_path,
            "extreme_day_trace": trace_path, "stations": stations_path,
            "station_history": history_path,
        }.items():
            run.record_artifact(name, path)
    if args.publish_legacy:
        with driver_progress("[10/10] Publishing legacy-compatible clean dataset and station table"):
            publish_copy(out, args.out)
            publish_copy(stations_path, os.path.join(config.TABLES_DIR, "stations.csv"))
    else:
        print("\n[10/10] Legacy publish skipped (enable with --publish-legacy)")
    spark.stop()
    run.finalize()
    print(f"tracked clean run: {run.run_id} -> {run.path}")


if __name__ == "__main__":
    main()
