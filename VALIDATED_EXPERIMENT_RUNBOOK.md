# Validated experiment runbook

Run every command from the project root. The sequence is intentional: each stage reads the latest successfully tracked upstream artifact. Every execution creates `output/runs/<UTC timestamp>_<stage>_<configuration hash>/`, containing a manifest, input hashes, code hashes, output hashes and stage artifacts. Nothing in this runbook depends on mutable result CSVs.

## 0. Activate the environment

```bash
cd /Users/whitetiger/Desktop/WhiteTiger/17_projects/21_Rainfall/scalabeDistributeRainfall
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Creates or refreshes the project Python environment. If `.venv` already exists, the first command is harmless and the install command reconciles dependencies.

## 1. Validate the temporal feature contract

```bash
./.venv/bin/python src/validate_feature_contract.py
```

Runs a tiny synthetic Spark check. It verifies that a 30-minute history contains exactly six regular five-minute readings, missing future offsets invalidate the label, and a missing `t−5` observation is represented explicitly.

## 2. Clean the complete dataset and generate audits

```bash
./.venv/bin/python src/clean_data.py \
  --years 2017 2018 2019 2020 2021 2022 2023 2024 \
  --max-5min-mm 60 \
  --hash-mode full \
  --publish-legacy
```

Creates an immutable cleaned Parquet dataset plus rule-level loss counts, cadence gaps, estimated missing slots, extreme station-days, a five-minute trace for those days, and the station master. Full SHA-256 hashes are recorded. `--publish-legacy` also refreshes `data/processed/rainfall_clean` for older scripts.

## 3. Run Task 1 with coverage controls

```bash
./.venv/bin/python src/task1_explore.py \
  --daily-completeness 0.95 \
  --stable-year-coverage 0.90 \
  --hash-mode full \
  --publish-legacy
```

Uses the latest clean run. In addition to the original descriptive analysis, it creates annual coverage rates, a stable-station cohort comparison, completeness sensitivity at 90%, 95% and 99%, and heavy station-days per 1,000 eligible station-days.

## 4. Build corrected temporal features

```bash
./.venv/bin/python src/task2_build_features.py \
  --hash-mode full \
  --publish-legacy
```

Builds `(t−W,t]` histories, exact five-minute lag availability, exact `t+5,…,t+30` labels, 24-hour dry-spell state, rolling-window coverage, stable sample IDs, prediction timestamps, label-end timestamps and the feature-version identifier.

## 5. Train, calibrate and evaluate the canonical GBT

```bash
./.venv/bin/python src/task2_train.py \
  --model gbt \
  --compute local-all \
  --validation-year 2023 \
  --test-year 2024 \
  --thresholds 0.2 1 2 5 10 \
  --evaluation-thresholds 0.2 0.5 1 2 3 5 7 10 \
  --cutoff-candidates 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 \
  --max-train-rows 8000000 \
  --save-predictions \
  --publish-canonical \
  --hash-mode full
```

Trains on data whose label window ends before 2023, uses 2023 only for cutoff selection and isotonic calibration, and evaluates once on 2024. It writes raw and calibrated Brier/log loss, ROC-AUC, PR-AUC, prevalence, confusion counts, positive precision/recall/F1, reliability bins, baselines, unseen-threshold results and monotonicity violations. The evaluated model, calibrator, sample membership, predictions, Spark event log and hashes are retained together.

## 6. Create probability-evaluation figures

```bash
./.venv/bin/python src/task2_evaluation_plot.py --hash-mode full
```

Reads the latest tracked training tables and produces reliability panels, positive-class metrics by threshold and Brier-score comparisons against persistence and prevalence baselines.

## 7. Freeze one sample for every systems run

```bash
./.venv/bin/python src/prepare_benchmark_sample.py \
  --validation-year 2023 \
  --test-year 2024 \
  --train-contexts 200000 \
  --validation-contexts 50000 \
  --test-contexts 50000 \
  --hash-mode full
```

Creates one deterministic, split-balanced feature sample. All local, Standalone and Docker runs below read the same Parquet rows and therefore have identical sample membership.

### Optional supplementary sequence models

```bash
./.venv/bin/python src/task2_deep.py \
  --model lstm --validation-year 2023 --test-year 2024 \
  --epochs 8 --max-rows 400000 --hash-mode full

./.venv/bin/python src/task2_deep.py \
  --model tcn --validation-year 2023 --test-year 2024 \
  --epochs 8 --max-rows 400000 --hash-mode full
```

These runs are supplementary. They now enforce exact sequence cadence, use the same train/validation/test years, average tied ROC ranks correctly, calibrate on validation data, and store positive-class, Brier, log-loss, reliability and threshold-consistency evidence. Keep their runtime comparison separate from MLlib unless sample membership and feature representations are also matched.

## 8. Run local scaling and matched Standalone experiments

```bash
cluster/start_standalone.sh 2 2 3g
export SPARK_MASTER_URL=spark://localhost:7077

./.venv/bin/python src/controlled_benchmark.py \
  --models gbt \
  --computes local-2 local-4 local-8 local-4-matched standalone-2x2-matched \
  --warmups 1 \
  --repetitions 3 \
  --seed 5208 \
  --hash-mode full

HOST_BENCHMARK_RESULTS=$(./.venv/bin/python src/run_artifacts.py --latest benchmark_group --artifact benchmark_results)
HOST_BENCHMARK_EVENTS=$(./.venv/bin/python src/run_artifacts.py --latest benchmark_group --artifact spark_event_logs)

cluster/stop_standalone.sh
```

Runs one warm-up and three randomized measured repetitions per configuration. The frozen sample, model, thresholds and split are unchanged. Local 2/4/8 provides scaling evidence; `local-4-matched` versus `standalone-2x2-matched` isolates deployment layout at four compute cores and 32 shuffle partitions. The two shell variables preserve this run’s artifact locations before the Docker run updates the latest pointer.

## 9. Run the matched Docker experiment

```bash
docker compose -f docker/docker-compose.yml up -d --build --scale spark-worker=2

docker compose -f docker/docker-compose.yml exec -T \
  --user "$(id -u):$(id -g)" \
  -e SPARK_DRIVER_HOST=spark-master \
  -e SPARK_LOCAL_DIR=/tmp/spark-local \
  spark-master \
  python /project/src/controlled_benchmark.py \
    --models gbt \
    --computes docker-2x2-matched \
    --warmups 1 \
    --repetitions 3 \
    --seed 5208 \
    --hash-mode full

DOCKER_BENCHMARK_RESULTS=$(./.venv/bin/python src/run_artifacts.py --latest benchmark_group --artifact benchmark_results)
DOCKER_BENCHMARK_EVENTS=$(./.venv/bin/python src/run_artifacts.py --latest benchmark_group --artifact spark_event_logs)

docker compose -f docker/docker-compose.yml down
```

Runs the driver inside the master container so worker containers can reach it and all containers see `/project`. Compose now enforces two CPU cores and 3 GB on each worker; the matched Spark configuration requests two executors with two cores, 2 GB each and 32 shuffle partitions.

## 10. Analyse all controlled systems runs together

```bash
./.venv/bin/python src/analyze_controlled_benchmark.py \
  --results "$HOST_BENCHMARK_RESULTS" "$DOCKER_BENCHMARK_RESULTS" \
  --event-logs "$HOST_BENCHMARK_EVENTS" "$DOCKER_BENCHMARK_EVENTS" \
  --scaling-baseline local-2 \
  --hash-mode full
```

Combines measured repetitions while excluding warm-ups. It reports median and min–max training time, prediction time, throughput, speedup, parallel efficiency, PR-AUC and Brier stability, and joins executor runtime, CPU time, GC, shuffle and spill measurements from Spark event logs.

## 11. Inspect the tracked artifacts

```bash
./.venv/bin/python src/run_artifacts.py --latest clean
./.venv/bin/python src/run_artifacts.py --latest task1
./.venv/bin/python src/run_artifacts.py --latest features
./.venv/bin/python src/run_artifacts.py --latest train
./.venv/bin/python src/run_artifacts.py --latest benchmark_analysis
```

Prints the latest immutable run pointer for each stage, including its manifest path and manifest SHA-256. Use the reported tables and figures to update the report; do not mix them with the earlier untracked results.
