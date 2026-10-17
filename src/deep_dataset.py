"""
Build sequence datasets for the deep-learning models (LSTM / TCN).

For each (station, 5-min timestamp) we assemble:
  * X_seq  : the last L rainfall readings (oldest -> newest), shape [N, L, 1]
  * X_stat : static context features + threshold, shape [N, S]
  * y      : 1 if the next-30-min accumulation > threshold else 0

Spark does the heavy lifting (per-station ordered windows to gather the lag
sequence and the forward label), then we collect a bounded sample to the driver
as numpy arrays for PyTorch. Because the deep models train on the GPU (4 GB
VRAM) and in-driver memory, we cap the number of sequence rows.

This reuses the SAME 30-min label and thresholds as the MLlib path so the
comparison is like-for-like.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import task2_features as tf  # noqa: E402

from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

# Sequence length: number of past 5-min readings fed to the model.
SEQ_LEN = 12  # last 60 minutes

# Static (non-sequence) context features handed to the model alongside the seq.
STATIC_COLS = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_weekend", "ne_monsoon", "sw_monsoon", "lat", "lon",
]


def build_sequences(spark, features_or_clean_path, thresholds, validation_year, test_year,
                    seq_len=SEQ_LEN, max_rows_per_split=400_000, seed=42):
    """Return exact-cadence train/validation/test numpy arrays.

    We read the CLEANED parquet (needs raw rain_mm sequence + calendar), rebuild
    the lag sequence and the 30-min label, then explode thresholds and sample.
    """
    df = spark.read.parquet(features_or_clean_path)

    # time features (reuse the shared definitions)
    df = tf.add_time_features(df)

    ts_secs = F.col("ts").cast("long")
    w = Window.partitionBy("station_id").orderBy(ts_secs)

    # Exact lag sequence: a previous row is accepted only at the requested
    # timestamp offset; gaps never collapse the sequence.
    lag_cols = []
    exact_flags = []
    for k in range(seq_len):
        c = f"lag{k}"
        lag_ts = F.lag(ts_secs, k).over(w)
        exact = lag_ts == ts_secs - k * config.INTERVAL_MINUTES * 60
        df = df.withColumn(c, F.when(exact, F.lag("rain_mm", k).over(w)))
        lag_cols.append(c)
        exact_flags.append(F.col(c).isNotNull())

    df = tf.add_future_labels(df)
    full_sequence = exact_flags[0]
    for flag in exact_flags[1:]: full_sequence = full_sequence & flag
    df = df.filter(full_sequence & F.col("future_complete"))
    df = df.withColumn("sample_id", F.sha2(F.concat_ws("|", "station_id", F.col("ts").cast("string")), 256))

    keep = ["sample_id", "station_id", "ts", "label_end_ts", "year", "future_30min_mm"] + STATIC_COLS + lag_cols
    df = df.select(*keep).na.drop()

    def collect_split(sdf, split_name):
        # Select base contexts first so every threshold copy stays together.
        max_contexts = max(1, max_rows_per_split // max(1, len(thresholds)))
        sdf = sdf.orderBy(F.xxhash64("sample_id"), "sample_id").limit(max_contexts)
        thr_col = F.explode(F.array([F.lit(float(t)) for t in thresholds]))
        e = sdf.withColumn("threshold", thr_col)
        e = e.withColumn("label", (F.col("future_30min_mm") > F.col("threshold")).cast("double"))
        pdf = e.select(*(["sample_id"] + STATIC_COLS + lag_cols + ["threshold", "label"])).toPandas()
        print(f"  {split_name}: {len(pdf):,} sequence rows")

        # lag_cols were built as [lag0, lag1, ..., lag{L-1}] where lag0 = the
        # reading AT the prediction time and lagK = K steps (5*K min) earlier.
        # An RNN/CNN expects the sequence in chronological order (oldest first),
        # so we REVERSE the lag columns -> [lag{L-1}, ..., lag0] before reshaping
        # to [batch, seq_len, 1]. Getting this order wrong would feed time
        # backwards and quietly hurt the model.
        seq = pdf[list(reversed(lag_cols))].to_numpy(dtype=np.float32)
        X_seq = seq.reshape(len(pdf), seq_len, 1)
        X_stat = pdf[STATIC_COLS + ["threshold"]].to_numpy(dtype=np.float32)
        y = pdf["label"].to_numpy(dtype=np.float32)
        return X_seq, X_stat, y, pdf["threshold"].to_numpy(dtype=np.float32), pdf["sample_id"].to_numpy()

    val_start = F.to_timestamp(F.lit(f"{validation_year}-01-01 00:00:00"))
    test_start = F.to_timestamp(F.lit(f"{test_year}-01-01 00:00:00"))
    test_end = F.to_timestamp(F.lit(f"{test_year + 1}-01-01 00:00:00"))
    train = df.filter(F.col("label_end_ts") < val_start)
    validation = df.filter((F.col("ts") >= val_start) & (F.col("label_end_ts") < test_start))
    test = df.filter((F.col("ts") >= test_start) & (F.col("label_end_ts") < test_end))
    Xtr_seq, Xtr_stat, ytr, ttr, itr = collect_split(train, "train")
    Xva_seq, Xva_stat, yva, tva, iva = collect_split(validation, "validation")
    Xte_seq, Xte_stat, yte, tte, ite = collect_split(test, "test")

    return {
        "X_train_seq": Xtr_seq, "X_train_stat": Xtr_stat, "y_train": ytr,
        "threshold_train": ttr, "sample_id_train": itr,
        "X_validation_seq": Xva_seq, "X_validation_stat": Xva_stat, "y_validation": yva,
        "threshold_validation": tva, "sample_id_validation": iva,
        "X_test_seq": Xte_seq, "X_test_stat": Xte_stat, "y_test": yte,
        "threshold_test": tte, "sample_id_test": ite,
        "seq_len": seq_len, "n_static": Xtr_stat.shape[1],
    }
