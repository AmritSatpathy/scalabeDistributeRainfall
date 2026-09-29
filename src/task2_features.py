"""
Task 2 - Feature engineering for rainfall probability prediction.

Goal: estimate P(rain accumulated over the NEXT 30 minutes at a station
exceeds a given threshold), given (station, prediction time, threshold).

Because the threshold is a *user input* at prediction time, we treat it as a
model feature rather than training one model per threshold. For each base
observation row we emit one labelled example per candidate threshold; the
label is 1 if the true 30-minute-ahead accumulation exceeds that threshold.

This module builds the base per-(station, 5-min timestamp) feature frame and
exposes helpers used by both training and inference so features are computed
identically in both paths.

Base features (all computed from information available up to and including
the prediction time - no leakage):
  * calendar / cyclical: hour (sin/cos), month (sin/cos), day-of-week,
    is_weekend, monsoon-season indicators
  * station identity (encoded) and location (lat, lon)
  * recent rainfall context: rain in the current 5-min reading and rolling
    sums over the last 30 / 60 / 180 minutes, plus minutes-since-last-rain
  * the query threshold (mm)

Label:
  * future_30min_mm  = sum of rain over the next 6 five-minute intervals
  * y                = 1 if future_30min_mm > threshold else 0
"""
from __future__ import annotations

import math
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

# Rolling windows expressed in number of 5-minute steps.
STEP = config.INTERVAL_MINUTES  # 5
LAG_WINDOWS_MIN = [30, 60, 180]
HORIZON_STEPS = config.HORIZON_STEPS  # 6 (30 min)

# Candidate thresholds (mm over 30 min) used to generate training examples.
# Chosen to span "any rain" up to "heavy rain" while keeping classes usable.
DEFAULT_THRESHOLDS = [0.2, 1.0, 2.0, 5.0, 10.0]


def add_time_features(df):
    """Cyclical + categorical calendar features (leakage-free)."""
    two_pi = 2 * math.pi
    df = (
        df.withColumn("hour_sin", F.sin(F.col("hour") / 24.0 * two_pi))
        .withColumn("hour_cos", F.cos(F.col("hour") / 24.0 * two_pi))
        .withColumn("month_sin", F.sin((F.col("month") - 1) / 12.0 * two_pi))
        .withColumn("month_cos", F.cos((F.col("month") - 1) / 12.0 * two_pi))
        .withColumn("is_weekend", F.col("dayofweek").isin(1, 7).cast("double"))
        .withColumn("ne_monsoon", F.col("month").isin(12, 1, 2, 3).cast("double"))
        .withColumn("sw_monsoon", F.col("month").isin(6, 7, 8, 9).cast("double"))
    )
    return df


def add_lag_and_future(df):
    """Rolling look-back features and the 30-minute-ahead accumulation label.

    Uses a per-station time-ordered window. Rolling windows are defined over
    the event-time (seconds) range so gaps in the series do not silently
    borrow readings from far away in time.
    """
    ts_secs = F.col("ts").cast("long")
    w_station = Window.partitionBy("station_id").orderBy(ts_secs)

    # ---- backward-looking rolling sums (features) -------------------------
    for mins in LAG_WINDOWS_MIN:
        wb = w_station.rangeBetween(-(mins * 60), 0)
        df = df.withColumn(f"roll_{mins}m_mm", F.sum("rain_mm").over(wb))

    # rain in the immediately preceding reading (t-5min)
    df = df.withColumn("lag_5m_mm", F.lag("rain_mm", 1).over(w_station))
    df = df.withColumn("lag_5m_mm", F.coalesce("lag_5m_mm", F.lit(0.0)))

    # minutes since last wet reading (capped): how long dry so far
    is_wet = (F.col("rain_mm") > 0).cast("int")
    df = df.withColumn("is_wet", is_wet)
    # running "last wet ts" via max over unbounded preceding of wet timestamps
    wet_ts = F.when(F.col("is_wet") == 1, ts_secs)
    df = df.withColumn(
        "last_wet_secs",
        F.max(wet_ts).over(w_station.rowsBetween(Window.unboundedPreceding, 0)),
    )
    df = df.withColumn(
        "mins_since_rain",
        F.when(F.col("last_wet_secs").isNull(), F.lit(1440.0))
        .otherwise(F.least(F.lit(1440.0), (ts_secs - F.col("last_wet_secs")) / 60.0)),
    )

    # ---- forward-looking label: sum of next 30 minutes --------------------
    # future window is (0, +30min]; we exclude the current reading so the
    # label is strictly the NEXT 30 minutes.
    wf = w_station.rangeBetween(1, config.HORIZON_MINUTES * 60)
    df = df.withColumn("future_30min_mm", F.sum("rain_mm").over(wf))
    # count of readings actually observed in the future window (completeness)
    df = df.withColumn("future_n", F.count("rain_mm").over(wf))

    return df


def build_base_features(df):
    """Full base feature frame (one row per station-timestamp)."""
    df = add_time_features(df)
    df = add_lag_and_future(df)
    # Only keep rows where the full 30-min future is observed, otherwise the
    # label is under-counted. We expect 6 future readings (5-min cadence).
    df = df.filter(F.col("future_n") >= HORIZON_STEPS)
    # roll windows can be null for the very first readings -> zero-fill
    for mins in LAG_WINDOWS_MIN:
        df = df.withColumn(
            f"roll_{mins}m_mm", F.coalesce(F.col(f"roll_{mins}m_mm"), F.lit(0.0))
        )
    return df


# Numeric feature columns fed to the model (threshold is appended separately).
BASE_FEATURE_COLS = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_weekend", "ne_monsoon", "sw_monsoon",
    "lat", "lon",
    "rain_mm", "lag_5m_mm",
    "roll_30m_mm", "roll_60m_mm", "roll_180m_mm",
    "mins_since_rain",
]


def explode_thresholds(df, thresholds=DEFAULT_THRESHOLDS):
    """Emit one labelled row per candidate threshold."""
    thr_col = F.explode(F.array([F.lit(float(t)) for t in thresholds]))
    df = df.withColumn("threshold", thr_col)
    df = df.withColumn(
        "label", (F.col("future_30min_mm") > F.col("threshold")).cast("double")
    )
    return df
