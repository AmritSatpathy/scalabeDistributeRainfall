"""Helpers for creating a SparkSession that runs unchanged on any compute tier.

The same scripts must run on:
  * E0  local[*]                 (single process, default)
  * E1  local Standalone cluster (spark://host:7077)
  * E2  Docker Spark cluster     (spark://spark-master:7077)

Rather than hard-coding ``master("local[*]")``, every deployment knob is read
from an environment variable (with a sensible local default) so no code changes
are needed between tiers -- you just export a few vars (or use the launch
scripts / docker compose which set them for you).

Environment variables honoured
------------------------------
  SPARK_MASTER            e.g. local[*] (default) or spark://host:7077
  SPARK_DRIVER_MEMORY     e.g. 20g  (default 10g)
  SPARK_EXECUTOR_MEMORY   e.g. 4g   (cluster mode; unset -> not applied)
  SPARK_EXECUTOR_CORES    e.g. 2    (cluster mode; unset -> not applied)
  SPARK_EXECUTOR_INSTANCES e.g. 4   (cluster mode; unset -> not applied)
  SPARK_SHUFFLE_PARTITIONS integer  (default: the code's per-job value)
  SPARK_LOCAL_DIR         scratch dir (default: <project>/data/_spark_tmp)

CLI callers can also pass explicit values, which take precedence over env vars,
which take precedence over the built-in defaults.
"""
from __future__ import annotations

import os
from pathlib import Path

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
    executor_memory: str | None = None,
    executor_cores: int | None = None,
    executor_instances: int | None = None,
):
    """Create a SparkSession configured for whichever tier we're running on.

    Resolution order for every setting: explicit argument > environment
    variable > built-in default. The Singapore session time zone is always
    pinned so timestamp arithmetic matches the SGT source data.
    """
    # For deployment knobs, an explicitly exported environment variable wins
    # over the value hard-coded in a script: this lets the SAME script run on
    # local[*] (using its tuned default) and on a cluster (overridden via env /
    # the launch scripts) with no code edits. Falls back to the script arg,
    # then the built-in default.
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

    event_log_dir = os.environ.get("SPARK_EVENTLOG_DIR")
    if event_log_dir:
        Path(event_log_dir).mkdir(parents=True, exist_ok=True)
        builder = (
            builder.config("spark.eventLog.enabled", "true")
            .config("spark.eventLog.dir", Path(event_log_dir).resolve().as_uri())
        )

    # Executor settings only make sense on a real cluster; apply them only when
    # provided (arg or env), so local[*] runs are unaffected.
    # When the driver runs inside a container (Docker cluster), executors in
    # other containers must be able to connect back to it. SPARK_DRIVER_HOST
    # (set to a resolvable service name, e.g. "spark-master") makes that work;
    # without it Spark advertises the container's internal hostname and the job
    # hangs on "Initial job has not accepted any resources".
    driver_host = os.environ.get("SPARK_DRIVER_HOST")
    if driver_host:
        builder = builder.config("spark.driver.host", driver_host)
        builder = builder.config("spark.driver.bindAddress", "0.0.0.0")

    exec_mem = _first(os.environ.get("SPARK_EXECUTOR_MEMORY"), executor_memory)
    if exec_mem:
        builder = builder.config("spark.executor.memory", str(exec_mem))
    exec_cores = _first(os.environ.get("SPARK_EXECUTOR_CORES"), executor_cores)
    if exec_cores:
        builder = builder.config("spark.executor.cores", str(exec_cores))
    exec_inst = _first(
        os.environ.get("SPARK_EXECUTOR_INSTANCES"), executor_instances
    )
    if exec_inst:
        builder = builder.config("spark.executor.instances", str(exec_inst))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # One-line banner so every run's log records exactly where/how it ran -
    # useful for the benchmark study and the report.
    sc = spark.sparkContext
    print(
        f"[spark] app={app_name} master={sc.master} "
        f"defaultParallelism={sc.defaultParallelism} "
        f"driverMem={resolved_driver_mem} shuffle={resolved_shuffle}"
        + (f" execMem={exec_mem}" if exec_mem else "")
        + (f" execCores={exec_cores}" if exec_cores else "")
        + (f" execInstances={exec_inst}" if exec_inst else "")
    )
    return spark


def save_table(pdf, path_csv: str) -> None:
    """Persist a (small) pandas DataFrame to CSV, creating parent dirs."""
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    pdf.to_csv(path_csv, index=False)
    print(f"  wrote table: {path_csv}  ({len(pdf)} rows)")
