"""Small deterministic validation of temporal-window and label semantics."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import task2_features as tf  # noqa: E402
from progress_utils import SparkProgressMonitor, StepProgress  # noqa: E402
from spark_utils import get_spark  # noqa: E402


def main() -> None:
    spark = get_spark("validate_feature_contract", driver_memory="2g", shuffle_partitions=2, master="local[2]")
    progress = StepProgress(2, "temporal feature-contract checks")
    start = datetime(2024, 1, 1, 0, 0)
    rows = []
    for i in range(15):
        ts = start + timedelta(minutes=5 * i)
        rows.append(("S_TEST", ts, 1.0, 1, ts.month, ts.hour, 2, 1.30, 103.80))
    columns = ["station_id", "ts", "rain_mm", "year", "month", "hour", "dayofweek", "lat", "lon"]
    regular = spark.createDataFrame(rows, columns)
    built = tf.build_base_features(regular)
    with SparkProgressMonitor(spark, "checking regular five-minute cadence"):
        row = built.filter(built.ts == start + timedelta(minutes=30)).first()
    assert row is not None
    assert row.roll_30m_mm == 6.0, f"expected six readings in (t-30,t], got {row.roll_30m_mm}"
    assert row.roll_30m_n == 6
    assert row.lag_5m_available == 1.0 and row.lag_5m_mm == 1.0
    assert row.future_30min_mm == 6.0 and row.future_n == 6
    progress.update("regular cadence passed")

    gapped = spark.createDataFrame([r for i, r in enumerate(rows) if i != 8], columns)
    gap_built = tf.add_lag_and_future(tf.add_time_features(gapped))
    with SparkProgressMonitor(spark, "checking missing-cadence safeguards"):
        before_gap = gap_built.filter(gap_built.ts == start + timedelta(minutes=35)).first()
        after_gap = gap_built.filter(gap_built.ts == start + timedelta(minutes=45)).first()
    assert before_gap is not None and not before_gap.future_complete, "missing future offset must invalidate label"
    assert after_gap is not None and after_gap.lag_5m_available == 0.0, "missing t-5 reading must be explicit"
    spark.stop()
    progress.finish("all contract checks passed")
    print("feature contract validation passed")


if __name__ == "__main__":
    main()
