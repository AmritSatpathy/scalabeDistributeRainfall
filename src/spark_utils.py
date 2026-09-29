"""Helpers for creating a locally-tuned SparkSession and saving pandas tables."""
from __future__ import annotations

import os

from pyspark.sql import SparkSession


def get_spark(app_name: str, driver_memory: str = "10g", shuffle_partitions: int = 64) -> SparkSession:
    """Create a SparkSession tuned for a single 16-core / 15 GB machine.

    We keep everything local (no cluster) and pin the session time zone to
    Singapore so all timestamp arithmetic matches the SGT source data.
    """
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.session.timeZone", "Asia/Singapore")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.local.dir", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "_spark_tmp"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def save_table(pdf, path_csv: str) -> None:
    """Persist a (small) pandas DataFrame to CSV, creating parent dirs."""
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    pdf.to_csv(path_csv, index=False)
    print(f"  wrote table: {path_csv}  ({len(pdf)} rows)")
