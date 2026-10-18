"""Helpers for creating the local SparkSession used by geographic stages.

Settings can be supplied explicitly or through environment variables. The
default uses all local cores and keeps Spark scratch data inside the project.

Environment variables honoured
------------------------------
  SPARK_MASTER            e.g. local[*] (default)
  SPARK_DRIVER_MEMORY     e.g. 20g  (default 10g)
  SPARK_SHUFFLE_PARTITIONS integer  (default: the code's per-job value)
  SPARK_LOCAL_DIR         scratch dir (default: <project>/data/_spark_tmp)

CLI callers can also pass explicit values, which take precedence over env vars,
which take precedence over the built-in defaults.
"""
from __future__ import annotations

import os

from pyspark.sql import SparkSession

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_LOCAL_DIR = os.path.join(PROJECT_ROOT, "data", "_spark_tmp")


def _first(*vals):
    """Return the first value that is not None/empty."""
    for v in vals:
        if v is not None and str(v) != "":
            return v
    return None


def get_spark(
    app_name: str,
    driver_memory: str | None = None,
    shuffle_partitions: int | None = None,
    master: str | None = None,
):
    """Create a SparkSession for the rainfall geography pipeline.

    Resolution order for every setting: explicit argument > environment
    variable > built-in default. The Singapore session time zone is always
    pinned so timestamp arithmetic matches the SGT source data.
    """
    # An explicitly exported environment value wins over the script argument,
    # followed by the built-in workstation default.
    resolved_master = _first(os.environ.get("SPARK_MASTER"), master, "local[*]")
    resolved_driver_mem = _first(
        os.environ.get("SPARK_DRIVER_MEMORY"), driver_memory, "10g"
    )
    resolved_shuffle = _first(
        os.environ.get("SPARK_SHUFFLE_PARTITIONS"), shuffle_partitions, 64
    )
    resolved_local_dir = _first(os.environ.get("SPARK_LOCAL_DIR"), _DEFAULT_LOCAL_DIR)

    builder = (
        SparkSession.builder.appName(app_name)
        .master(str(resolved_master))
        .config("spark.driver.memory", str(resolved_driver_mem))
        .config("spark.sql.session.timeZone", "Asia/Singapore")
        .config("spark.sql.shuffle.partitions", str(resolved_shuffle))
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.memory.fraction", "0.6")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.local.dir", str(resolved_local_dir))
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # One-line banner records the effective processing setup in every run log.
    sc = spark.sparkContext
    print(
        f"[spark] app={app_name} master={sc.master} "
        f"defaultParallelism={sc.defaultParallelism} "
        f"driverMem={resolved_driver_mem} shuffle={resolved_shuffle}"
    )
    return spark


def save_table(pdf, path_csv: str) -> None:
    """Persist a (small) pandas DataFrame to CSV, creating parent dirs."""
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    pdf.to_csv(path_csv, index=False)
    print(f"  wrote table: {path_csv}  ({len(pdf)} rows)")
