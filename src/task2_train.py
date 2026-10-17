"""Train, calibrate, evaluate and trace a Spark MLlib rainfall model.

The design uses three temporal partitions: training years, one validation year
for cutoff selection and isotonic calibration, and an untouched test year.
Every invocation writes an immutable run directory with input/output hashes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import compute as compute_registry  # noqa: E402
import config  # noqa: E402
import models as model_registry  # noqa: E402
import task2_features as tf  # noqa: E402
from run_artifacts import RunContext, latest_artifact, publish_copy  # noqa: E402
from progress_utils import SparkProgressMonitor, StepProgress, driver_progress  # noqa: E402
from spark_utils import get_spark  # noqa: E402

from pyspark import StorageLevel  # noqa: E402
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator  # noqa: E402
from pyspark.ml.feature import VectorAssembler  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.ml.regression import IsotonicRegression  # noqa: E402
from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

MODEL_DIR = os.path.join(config.MODELS_DIR, "rainfall_model")
CALIBRATOR_DIR = os.path.join(config.MODELS_DIR, "rainfall_calibrator")
META_PATH = os.path.join(config.MODELS_DIR, "model_metadata.json")
LEGACY_FEATURES_BASE = os.path.join(config.PROCESSED_DIR, "features_base")


def add_score(pred):
    """Expose a scalar positive score for probability and margin-only models."""
    if "probability" in pred.columns:
        return pred.withColumn("raw_probability", vector_to_array("probability")[1])
    margin = vector_to_array("rawPrediction")[1]
    return pred.withColumn("raw_probability", 1.0 / (1.0 + F.exp(-margin)))


def evaluate_scores(df, score_col="raw_probability", cutoff=0.5, label_col="label"):
    eps = 1e-15
    scored = df.withColumn("_decision", (F.col(score_col) >= F.lit(float(cutoff))).cast("double"))
    row = scored.agg(
        F.count("*").alias("n"), F.sum(label_col).alias("positive"),
        F.sum(((F.col(label_col) == 1) & (F.col("_decision") == 1)).cast("long")).alias("tp"),
        F.sum(((F.col(label_col) == 0) & (F.col("_decision") == 0)).cast("long")).alias("tn"),
        F.sum(((F.col(label_col) == 0) & (F.col("_decision") == 1)).cast("long")).alias("fp"),
        F.sum(((F.col(label_col) == 1) & (F.col("_decision") == 0)).cast("long")).alias("fn"),
        F.avg(F.pow(F.col(score_col) - F.col(label_col), 2)).alias("brier"),
        F.avg(-F.col(label_col) * F.log(F.greatest(F.lit(eps), F.col(score_col)))
              - (1 - F.col(label_col)) * F.log(F.greatest(F.lit(eps), 1 - F.col(score_col)))).alias("log_loss"),
    ).first().asDict()
    n, pos = int(row["n"] or 0), int(row["positive"] or 0)
    tp, tn, fp, fn = (int(row[k] or 0) for k in ("tp", "tn", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    positive_f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    specificity = tn / (tn + fp) if tn + fp else None
    auc_roc = auc_pr = weighted_f1 = None
    if pos and pos < n:
        auc_roc = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=score_col, metricName="areaUnderROC").evaluate(scored)
        auc_pr = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=score_col, metricName="areaUnderPR").evaluate(scored)
        weighted_f1 = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="_decision", metricName="f1").evaluate(scored)
    return {
        "n": n, "positive": pos, "prevalence": pos / n if n else None,
        "cutoff": float(cutoff), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision_positive": precision, "recall_positive": recall,
        "f1_positive": positive_f1, "specificity": specificity,
        "accuracy": (tp + tn) / n if n else None, "weighted_f1": weighted_f1,
        "auc_roc": auc_roc, "auc_pr": auc_pr,
        "brier": row["brier"], "log_loss": row["log_loss"],
    }


def select_cutoff(validation, score_col="raw_probability", candidates=None):
    candidates = candidates or [i / 100 for i in range(5, 100, 5)]
    expanded = validation.select("label", score_col).withColumn(
        "cutoff", F.explode(F.array(*[F.lit(float(x)) for x in candidates]))
    )
    rows = expanded.groupBy("cutoff").agg(
        F.sum(((F.col("label") == 1) & (F.col(score_col) >= F.col("cutoff"))).cast("long")).alias("tp"),
        F.sum(((F.col("label") == 0) & (F.col(score_col) >= F.col("cutoff"))).cast("long")).alias("fp"),
        F.sum(((F.col("label") == 1) & (F.col(score_col) < F.col("cutoff"))).cast("long")).alias("fn"),
    ).collect()
    scored = []
    for row in rows:
        tp, fp, fn = int(row.tp or 0), int(row.fp or 0), int(row.fn or 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scored.append({"cutoff": float(row.cutoff), "precision": precision, "recall": recall, "f1": f1})
    scored.sort(key=lambda x: (-x["f1"], x["cutoff"]))
    return scored[0]["cutoff"], scored


def fit_isotonic(validation):
    assembler = VectorAssembler(inputCols=["raw_probability"], outputCol="calibration_features")
    frame = assembler.transform(validation)
    model = IsotonicRegression(
        featuresCol="calibration_features", labelCol="label",
        predictionCol="calibrated_probability", isotonic=True,
    ).fit(frame)
    return assembler, model


def apply_calibration(df, assembler, model):
    return model.transform(assembler.transform(df)).withColumn(
        "calibrated_probability",
        F.least(F.lit(1.0), F.greatest(F.lit(0.0), F.col("calibrated_probability"))),
    )


def calibration_table(df, score_col, bins=10):
    return (
        df.withColumn("bin", F.least(F.lit(bins - 1), F.floor(F.col(score_col) * bins).cast("int")))
        .groupBy("threshold", "bin").agg(
            F.count("*").alias("n"), F.avg(score_col).alias("mean_probability"),
            F.avg("label").alias("observed_frequency"),
        ).withColumn("bin_lower", F.col("bin") / bins)
        .withColumn("bin_upper", (F.col("bin") + 1) / bins)
        .orderBy("threshold", "bin")
    )


def monotonicity_table(pred, score_col="calibrated_probability"):
    w = Window.partitionBy("sample_id").orderBy("threshold")
    pairs = pred.withColumn("next_probability", F.lead(score_col).over(w)).filter(F.col("next_probability").isNotNull())
    return pairs.agg(
        F.count("*").alias("ordered_pairs"),
        F.sum((F.col("next_probability") > F.col(score_col) + F.lit(1e-12)).cast("long")).alias("violations"),
    ).withColumn("violation_rate", F.col("violations") / F.col("ordered_pairs"))


def deterministic_sample(df, fraction: float, max_rows: int | None = None):
    if fraction < 1.0:
        df = df.filter(F.pmod(F.xxhash64("sample_id"), F.lit(1_000_000)) < int(fraction * 1_000_000))
    if max_rows and max_rows > 0:
        count = df.count()
        if count > max_rows:
            keep = max_rows / float(count)
            df = df.filter(F.pmod(F.xxhash64("sample_id"), F.lit(1_000_000)) < int(keep * 1_000_000))
    return df


def write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.toPandas().to_csv(path, index=False)


def runtime_resources(spark):
    sc = spark.sparkContext
    conf = dict(sc.getConf().getAll())
    result = {
        "spark_master": sc.master, "application_id": sc.applicationId,
        "default_parallelism": sc.defaultParallelism,
        "driver_memory": conf.get("spark.driver.memory"),
        "executor_memory": conf.get("spark.executor.memory"),
        "executor_cores": conf.get("spark.executor.cores"),
        "executor_instances_requested": conf.get("spark.executor.instances"),
        "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
    }
    try:
        infos = sc._jsc.sc().statusTracker().getExecutorInfos()
        result["executor_processes_observed"] = max(0, len(infos) - 1)
    except Exception as exc:
        result["executor_observation_error"] = str(exc)
    return result


def main() -> None:
    wall_started = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gbt")
    ap.add_argument("--compute", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--feature-contract", default=None,
                    help="JSON contract containing model_feature_columns")
    ap.add_argument("--sample-ids", default=None, help="Frozen Parquet sample_id manifest")
    ap.add_argument("--train-start-year", type=int, default=2017)
    ap.add_argument("--validation-year", type=int, default=2023)
    ap.add_argument("--test-year", type=int, default=2024)
    ap.add_argument("--fold-id", default=None,
                    help="Stable label for an expanding-window temporal fold")
    ap.add_argument("--sample-fraction", type=float, default=1.0)
    ap.add_argument("--max-train-rows", type=int, default=8_000_000,
                    help="Maximum exploded rows; base contexts are sampled before threshold expansion.")
    ap.add_argument("--max-validation-contexts", type=int, default=0,
                    help="Deterministic cap before threshold expansion; 0 keeps the complete year.")
    ap.add_argument("--max-test-contexts", type=int, default=0,
                    help="Deterministic cap before threshold expansion; 0 keeps the complete year.")
    ap.add_argument("--thresholds", type=float, nargs="*", default=tf.DEFAULT_THRESHOLDS)
    ap.add_argument("--evaluation-thresholds", type=float, nargs="*", default=[0.2, 0.5, 1, 2, 3, 5, 7, 10])
    ap.add_argument("--cutoff-candidates", type=float, nargs="*", default=[i / 100 for i in range(5, 100, 5)])
    ap.add_argument("--calibration-bins", type=int, default=10)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--publish-canonical", action="store_true")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--benchmark-csv", default=None)
    ap.add_argument("--run-group", default=None)
    ap.add_argument("--repetition", type=int, default=None)
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="full")
    ap.add_argument("--event-log-dir", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("Models:")
        for name, fw, desc in model_registry.describe():
            print(f"  {name:10s} [{fw}]  {desc}")
        print("Compute configs:")
        for name, master, desc in compute_registry.describe():
            print(f"  {name:16s} {master:28s} {desc}")
        return
    if args.train_start_year >= args.validation_year:
        raise SystemExit("--train-start-year must be earlier than --validation-year")
    if args.test_year != args.validation_year + 1:
        raise SystemExit("Yearly rolling-origin evaluation requires --test-year = --validation-year + 1")
    spec = model_registry.get_model(args.model)
    if spec.framework != "mllib":
        raise SystemExit(f"Use the dedicated runner for framework={spec.framework}")

    tracked_spatiotemporal = latest_artifact(
        "spatiotemporal_features", "spatiotemporal_features"
    )
    tracked_legacy = latest_artifact("features", "features_base")
    args.features = (
        args.features
        or tracked_spatiotemporal
        or tracked_legacy
        or (LEGACY_FEATURES_BASE if os.path.isdir(LEGACY_FEATURES_BASE) else None)
    )
    if not args.features or not os.path.isdir(args.features):
        raise SystemExit(
            "No model-ready feature dataset was found. Expected the latest "
            "spatiotemporal_features artifact or an explicit --features path."
        )
    if args.feature_contract is None and args.features == tracked_spatiotemporal:
        args.feature_contract = latest_artifact(
            "spatiotemporal_features", "feature_contract"
        )
    feature_contract = {}
    if args.feature_contract:
        with open(args.feature_contract, encoding="utf-8") as fh:
            feature_contract = json.load(fh)
    model_feature_cols = feature_contract.get(
        "model_feature_columns", tf.BASE_FEATURE_COLS
    )
    feature_cols = model_feature_cols + ["threshold", "station_idx"]
    compute_label = args.compute or os.environ.get("SPARK_MASTER", "local[*]")
    compute_description = "Environment-selected Spark configuration"
    if args.compute:
        cfg = compute_registry.get_compute(args.compute)
        cfg.apply(); compute_label = cfg.name; compute_description = cfg.description
    if args.event_log_dir:
        os.environ["SPARK_EVENTLOG_DIR"] = os.path.abspath(args.event_log_dir)

    params = vars(args).copy()
    run = RunContext("train", params, hash_mode=args.hash_mode, run_group=args.run_group)
    with driver_progress("fingerprinting Task 2 input artifacts"):
        run.record_input("features_base", args.features)
        if args.feature_contract:
            run.record_input("feature_contract", args.feature_contract)
        if args.sample_ids:
            run.record_input("sample_ids", args.sample_ids)
    event_dir = args.event_log_dir or run.artifact_path("spark-events")
    os.makedirs(event_dir, exist_ok=True)
    os.environ["SPARK_EVENTLOG_DIR"] = event_dir
    spark = get_spark(f"task2_{args.model}_{run.run_id}")
    stages = StepProgress(7, f"Task 2 {args.model} workflow")

    prep_started = time.perf_counter()
    base = spark.read.parquet(args.features)
    required = {
        "sample_id", "station_id", "ts", "label_end_ts", "feature_version",
        "future_30min_mm", *model_feature_cols,
    }
    missing = sorted(required - set(base.columns))
    if missing:
        raise SystemExit(f"Feature dataset predates the exact contract; missing columns: {missing}")
    if args.sample_ids:
        ids = spark.read.parquet(args.sample_ids).select("sample_id").distinct()
        base = base.join(F.broadcast(ids), "sample_id", "inner")

    train_start = F.to_timestamp(F.lit(f"{args.train_start_year}-01-01 00:00:00"))
    validation_start = F.to_timestamp(F.lit(f"{args.validation_year}-01-01 00:00:00"))
    validation_end = F.to_timestamp(F.lit(f"{args.validation_year + 1}-01-01 00:00:00"))
    test_start = F.to_timestamp(F.lit(f"{args.test_year}-01-01 00:00:00"))
    test_end = F.to_timestamp(F.lit(f"{args.test_year + 1}-01-01 00:00:00"))
    train_base = base.filter((F.col("ts") >= train_start) & (F.col("label_end_ts") < validation_start))
    val_base = base.filter((F.col("ts") >= validation_start) & (F.col("label_end_ts") < validation_end))
    test_base = base.filter((F.col("ts") >= test_start) & (F.col("label_end_ts") < test_end))

    max_base = args.max_train_rows // max(1, len(set(args.thresholds))) if args.max_train_rows else None
    train_base = deterministic_sample(train_base, args.sample_fraction, max_base)
    val_base = deterministic_sample(val_base, 1.0, args.max_validation_contexts)
    test_base = deterministic_sample(test_base, 1.0, args.max_test_contexts)
    train = tf.explode_thresholds(train_base, sorted(set(args.thresholds)))
    eval_thresholds = sorted(set(args.evaluation_thresholds) | set(args.thresholds))
    validation = tf.explode_thresholds(val_base, eval_thresholds)
    test = tf.explode_thresholds(test_base, eval_thresholds)
    keep = ["sample_id", "station_id", "ts", "label_end_ts", "year", "feature_version",
            "threshold", "label", "future_30min_mm"] + model_feature_cols
    train = train.select(*keep).na.drop().persist(StorageLevel.MEMORY_AND_DISK)
    validation = validation.select(*keep).na.drop().persist(StorageLevel.MEMORY_AND_DISK)
    test = test.select(*keep).na.drop().persist(StorageLevel.MEMORY_AND_DISK)
    with SparkProgressMonitor(spark, "materialising temporal train/validation/test splits"):
        n_train, n_val, n_test = train.count(), validation.count(), test.count()
    if min(n_train, n_val, n_test) == 0:
        raise RuntimeError(
            f"Temporal split produced an empty partition: train={n_train}, "
            f"validation={n_val}, test={n_test}"
        )
    feature_version = train.select("feature_version").first()[0]
    prep_secs = time.perf_counter() - prep_started
    print(f"rows train={n_train:,} validation={n_val:,} test={n_test:,}")
    stages.update("temporal splits ready")

    pipeline = spec.build_pipeline(feature_cols)
    t0 = time.perf_counter()
    with SparkProgressMonitor(spark, f"fitting {args.model} model"):
        fitted = pipeline.fit(train)
    train_secs = time.perf_counter() - t0
    stages.update("model fitted")
    t0 = time.perf_counter()
    with SparkProgressMonitor(spark, "scoring validation and test periods"):
        val_pred = add_score(fitted.transform(validation)).persist(StorageLevel.MEMORY_AND_DISK)
        test_pred = add_score(fitted.transform(test)).persist(StorageLevel.MEMORY_AND_DISK)
        val_pred.count(); test_pred.count()
    predict_secs = time.perf_counter() - t0
    stages.update("validation and test predictions ready")

    evaluation_started = time.perf_counter()
    with SparkProgressMonitor(spark, "calibrating probabilities and evaluating thresholds"):
        raw_cutoff, _ = select_cutoff(val_pred, "raw_probability", args.cutoff_candidates)
        assembler, calibrator = fit_isotonic(val_pred)
        val_cal = apply_calibration(val_pred, assembler, calibrator).persist(StorageLevel.MEMORY_AND_DISK)
        test_cal = apply_calibration(test_pred, assembler, calibrator).persist(StorageLevel.MEMORY_AND_DISK)
        val_cal.count(); test_cal.count()
        selected_cutoff, overall_cutoff_rows = select_cutoff(
            val_cal, "calibrated_probability", args.cutoff_candidates)
        cutoff_rows = [{"threshold": "overall", **row} for row in overall_cutoff_rows]
        threshold_cutoffs = {}
        for threshold in eval_thresholds:
            cutoff, rows = select_cutoff(
                val_cal.filter(F.col("threshold") == float(threshold)),
                score_col="calibrated_probability",
                candidates=args.cutoff_candidates,
            )
            threshold_cutoffs[float(threshold)] = cutoff
            cutoff_rows.extend({"threshold": threshold, **row} for row in rows)
        overall_raw = evaluate_scores(test_cal, "raw_probability", raw_cutoff)
        overall_cal = evaluate_scores(test_cal, "calibrated_probability", selected_cutoff)
        per_threshold = []
        for threshold in eval_thresholds:
            sub = test_cal.filter(F.col("threshold") == float(threshold))
            metrics = evaluate_scores(sub, "calibrated_probability", threshold_cutoffs[float(threshold)])
            per_threshold.append({"threshold": threshold, **metrics})

    # Probability baselines use validation prevalence by threshold and a simple
    # persistence score based on the current five-minute accumulation.
        prevalence = val_pred.groupBy("threshold").agg(F.avg("label").alias("prevalence_probability"))
        baseline_frame = (
            test_cal.join(F.broadcast(prevalence), "threshold", "left")
            .withColumn("persistence_probability", (F.col("rain_mm") > F.col("threshold")).cast("double"))
        )
        baseline_rows = []
        for threshold in eval_thresholds:
            sub = baseline_frame.filter(F.col("threshold") == float(threshold))
            for name, score in (("prevalence", "prevalence_probability"), ("persistence", "persistence_probability")):
                baseline_rows.append({"threshold": threshold, "baseline": name,
                                      **evaluate_scores(sub, score, threshold_cutoffs[float(threshold)])})
    evaluation_secs = time.perf_counter() - evaluation_started
    stages.update("calibration and evaluation complete")

    tables_dir = run.artifact_path("tables")
    os.makedirs(tables_dir, exist_ok=True)
    import pandas as pd
    pd.DataFrame([{"score": "raw", **overall_raw}, {"score": "isotonic", **overall_cal}]).to_csv(os.path.join(tables_dir, "metrics_overall.csv"), index=False)
    pd.DataFrame(per_threshold).to_csv(os.path.join(tables_dir, "metrics_by_threshold.csv"), index=False)
    pd.DataFrame(cutoff_rows).to_csv(os.path.join(tables_dir, "validation_cutoff_selection.csv"), index=False)
    pd.DataFrame(baseline_rows).to_csv(os.path.join(tables_dir, "baseline_metrics.csv"), index=False)
    with SparkProgressMonitor(spark, "writing reliability and monotonicity tables"):
        write_csv(calibration_table(test_cal, "raw_probability", args.calibration_bins), os.path.join(tables_dir, "reliability_raw.csv"))
        write_csv(calibration_table(test_cal, "calibrated_probability", args.calibration_bins), os.path.join(tables_dir, "reliability_isotonic.csv"))
        write_csv(monotonicity_table(test_cal), os.path.join(tables_dir, "threshold_monotonicity.csv"))
    stages.update("evaluation tables written")

    sample_manifest = (
        train.select("sample_id", "station_id", "ts").distinct()
        .withColumn("split", F.lit("train"))
        .unionByName(validation.select("sample_id", "station_id", "ts").distinct().withColumn("split", F.lit("validation")))
        .unionByName(test.select("sample_id", "station_id", "ts").distinct().withColumn("split", F.lit("test")))
    )
    sample_path = run.artifact_path("sample_membership")
    with SparkProgressMonitor(spark, "writing split-membership audit"):
        sample_manifest.write.mode("overwrite").partitionBy("split").parquet(sample_path)
    if args.save_predictions:
        pred_path = run.artifact_path("test_predictions")
        with SparkProgressMonitor(spark, "writing calibrated test predictions"):
            test_cal.select("sample_id", "station_id", "ts", "threshold", "label",
                            "raw_probability", "calibrated_probability").write.mode("overwrite").parquet(pred_path)
        run.record_artifact("test_predictions", pred_path)
    stages.update("audits and optional predictions written")

    resources = runtime_resources(spark)
    total_secs = time.perf_counter() - wall_started
    benchmark_row = {
        "run_id": run.run_id, "run_group": args.run_group, "model": args.model,
        "compute_config": compute_label, "compute_description": compute_description,
        "repetition": args.repetition,
        "warmup": args.warmup, "n_train": n_train, "n_validation": n_val, "n_test": n_test,
        "fold_id": args.fold_id, "train_start_year": args.train_start_year,
        "train_end_year": args.validation_year - 1,
        "validation_year": args.validation_year, "test_year": args.test_year,
        "preparation_secs": round(prep_secs, 3),
        "train_secs": round(train_secs, 3), "predict_secs": round(predict_secs, 3),
        "evaluation_secs": round(evaluation_secs, 3), "total_secs": round(total_secs, 3),
        "sample_fraction": args.sample_fraction, "max_train_rows": args.max_train_rows,
        "max_validation_contexts": args.max_validation_contexts,
        "max_test_contexts": args.max_test_contexts,
        "training_thresholds": json.dumps(sorted(set(args.thresholds))),
        "evaluation_thresholds": json.dumps(eval_thresholds),
        "model_parameters": json.dumps(spec.extra, sort_keys=True),
        "feature_version": feature_version,
        "features_sha256": run.manifest["inputs"]["features_base"].get("sha256"),
        "feature_contract_sha256": (
            run.manifest["inputs"].get("feature_contract", {}).get("sha256")
        ),
        "auc_roc": overall_cal["auc_roc"], "auc_pr": overall_cal["auc_pr"],
        "brier": overall_cal["brier"], "f1_positive": overall_cal["f1_positive"],
        "selected_cutoff": selected_cutoff, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **resources,
    }
    pd.DataFrame([benchmark_row]).to_csv(os.path.join(tables_dir, "benchmark_row.csv"), index=False)
    if args.benchmark_csv:
        os.makedirs(os.path.dirname(args.benchmark_csv), exist_ok=True)
        pd.DataFrame([benchmark_row]).to_csv(args.benchmark_csv, mode="a", header=not os.path.exists(args.benchmark_csv), index=False)

    model_path = calibrator_path = None
    if not args.no_save:
        model_path = run.artifact_path("model")
        calibrator_path = run.artifact_path("calibrator")
        with SparkProgressMonitor(spark, "saving fitted model and calibrator"):
            fitted.write().overwrite().save(model_path)
            calibrator.write().overwrite().save(calibrator_path)
    metadata = {
        "run_id": run.run_id, "model_kind": args.model, "model_description": spec.description,
        "model_parameters": spec.extra,
        "feature_version": feature_version,
        "feature_cols": feature_cols, "base_model_feature_cols": model_feature_cols,
        "feature_contract": feature_contract, "history_interval": "(t-W, t]",
        "fold_id": args.fold_id, "train_start_year": args.train_start_year,
        "validation_year": args.validation_year, "test_year": args.test_year,
        "training_thresholds": sorted(set(args.thresholds)), "evaluation_thresholds": eval_thresholds,
        "n_train": n_train, "n_validation": n_val, "n_test": n_test,
        "selected_cutoff_overall": selected_cutoff,
        "selected_cutoff_by_threshold": threshold_cutoffs, "metrics_raw": overall_raw,
        "metrics_calibrated": overall_cal, "resources": resources,
        "compute_description": compute_description,
        "timings_seconds": {
            "preparation": prep_secs, "model_fit": train_secs,
            "validation_and_test_inference": predict_secs,
            "calibration_and_evaluation": evaluation_secs, "wall_total": total_secs,
        },
        "model_path": model_path, "calibrator_path": calibrator_path,
    }
    metadata_path = run.write_json("model_metadata", metadata)
    run.record_artifact("tables", tables_dir)
    run.record_artifact("sample_membership", sample_path)
    if model_path:
        run.record_artifact("model", model_path)
        run.record_artifact("calibrator", calibrator_path)
    if args.publish_canonical and model_path:
        publish_copy(model_path, MODEL_DIR); publish_copy(calibrator_path, CALIBRATOR_DIR)
        publish_copy(metadata_path, META_PATH)

    spark.stop()
    run.record_artifact("spark_event_logs", event_dir)
    run.finalize()
    stages.finish("run manifest and hashes complete")
    print(json.dumps({"run_id": run.run_id, "run_dir": str(run.path), "metrics": overall_cal}, indent=2))


if __name__ == "__main__":
    main()
