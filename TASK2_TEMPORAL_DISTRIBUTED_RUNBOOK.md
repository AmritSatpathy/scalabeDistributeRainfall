# Task 2 temporal prediction and distributed-computing runbook

Run these commands sequentially from the project root. The workflow separates
predictive model selection from systems benchmarking. The first study uses
expanding temporal folds. The second holds the model and data constant while
changing the Spark deployment architecture.

## 1. Confirm the registered models and compute configurations

```bash
./.venv/bin/python src/task2_train.py --list
```

This is a read-only check. The reduced predictive comparison uses GBT, random
forest and Spark MLlib's multilayer perceptron. The matched systems
configurations should include local, Standalone and Docker variants.

## 2. Validate the past-only feature and future-label contract

```bash
./.venv/bin/python src/validate_feature_contract.py
```

This small synthetic validation checks exact five-minute lags, past-only
rolling windows and the six observations in the next-30-minute target. It does
not rebuild the completed feature dataset.

## 3. Compare models with expanding rolling-origin folds

```bash
./.venv/bin/python src/task2_temporal_cv.py \
  --models gbt rf mlp \
  --compute local-4-matched \
  --train-start-year 2017 \
  --folds 2021:2022 2023:2024 \
  --max-train-rows 2000000 \
  --max-validation-contexts 100000 \
  --max-test-contexts 100000 \
  --seed 5208 \
  --hash-mode full \
  --publish-results output/benchmarks/temporal_cv_results.csv
```

The earlier fold trains on 2017–2020, validates/calibrates on 2021 and tests on
2022. The final fold trains on 2017–2022, validates on 2023 and tests on 2024.
The training-row cap grows with the expanding history and reaches 2,000,000
exploded rows in the final fold. Hash-based sampling keeps contexts identical
across the three models within each fold. The MLP is the native Spark MLlib
`MultilayerPerceptronClassifier` with `[input, 32, 16, 2]` layers, 30 maximum
iterations and block size 512. A TCN is excluded because Spark MLlib does not
provide temporal convolution layers; the repository's TCN is PyTorch-based.

## 4. Summarise temporal generalisation

```bash
./.venv/bin/python src/analyze_temporal_cv.py \
  --results output/benchmarks/temporal_cv_results.csv \
  --hash-mode full
```

This ranks models primarily by mean out-of-time PR-AUC, with Brier score and
training time retained as supporting evidence. Review every fold rather than
selecting a model from one year.

## 5. Train the canonical final GBT

```bash
./.venv/bin/python src/task2_train.py \
  --model gbt \
  --compute local-all \
  --train-start-year 2017 \
  --validation-year 2023 \
  --test-year 2024 \
  --fold-id canonical_2017_2022_val2023_test2024 \
  --thresholds 0.2 1 2 5 10 \
  --evaluation-thresholds 0.2 0.5 1 2 3 5 7 10 \
  --max-train-rows 8000000 \
  --max-validation-contexts 0 \
  --max-test-contexts 0 \
  --save-predictions \
  --publish-canonical \
  --hash-mode full
```

This is the report-quality run. It uses the complete 2023 validation year and
the untouched complete 2024 test year, saves calibrated predictions and
publishes the fitted model. If the temporal study clearly selects another
model, replace only `--model gbt`; keep the split and thresholds unchanged.

## 6. Generate model evaluation figures

```bash
./.venv/bin/python src/task2_evaluation_plot.py --hash-mode full
```

This produces calibration, positive-class and baseline-comparison figures from
the canonical run.

An optional single-station prediction can then be generated with the exact
training feature contract:

```bash
./.venv/bin/python src/task2_predict.py \
  --station S07 \
  --time "2024-08-15 12:00:00" \
  --threshold 1.0 \
  --max-staleness-minutes 5 \
  --hash-mode full
```

The prediction artifact records the canonical model, calibrator and metadata
fingerprints together with the forecast features and probability.

## 7. Freeze one final-fold sample for every architecture

```bash
./.venv/bin/python src/prepare_benchmark_sample.py \
  --validation-year 2023 \
  --test-year 2024 \
  --train-contexts 200000 \
  --validation-contexts 50000 \
  --test-contexts 50000 \
  --hash-mode full \
  --publish-features output/benchmarks/frozen_features \
  --publish-sample-ids output/benchmarks/frozen_sample_ids \
  --publish-feature-contract output/benchmarks/feature_contract.json
```

The published Parquet directory is the same physical input for the local,
Standalone and Docker studies. Do not regenerate it between architectures.

## 8. Benchmark local Spark and local scaling

```bash
./.venv/bin/python src/controlled_benchmark.py \
  --models gbt \
  --computes local-2 local-4-matched local-8 \
  --features output/benchmarks/frozen_features \
  --feature-contract output/benchmarks/feature_contract.json \
  --train-start-year 2017 \
  --validation-year 2023 \
  --test-year 2024 \
  --fold-id architecture_final_fold \
  --warmups 1 \
  --repetitions 3 \
  --seed 5208 \
  --hash-mode full \
  --publish-results output/benchmarks/local_results.csv \
  --publish-events output/benchmarks/local_events
```

This measures scale-up behaviour in one JVM. `local-4-matched` is the fair
four-core comparator for the two distributed deployments.

## 9. Benchmark the matched local Spark Standalone cluster

```bash
cluster/start_standalone.sh 2 2 3g
```

```bash
export SPARK_MASTER_URL=spark://localhost:7077
```

```bash
./.venv/bin/python src/controlled_benchmark.py \
  --models gbt \
  --computes standalone-2x2-matched \
  --features output/benchmarks/frozen_features \
  --feature-contract output/benchmarks/feature_contract.json \
  --train-start-year 2017 \
  --validation-year 2023 \
  --test-year 2024 \
  --fold-id architecture_final_fold \
  --warmups 1 \
  --repetitions 3 \
  --seed 5208 \
  --hash-mode full \
  --publish-results output/benchmarks/standalone_results.csv \
  --publish-events output/benchmarks/standalone_events
```

```bash
cluster/stop_standalone.sh
```

The two worker JVMs provide four executor cores in total. Separate worker
processes introduce scheduling, serialization and shuffle overhead, which is
part of the measured architecture effect.

## 10. Benchmark the matched Dockerized Spark cluster

```bash
docker compose -f docker/docker-compose.yml up -d --build --scale spark-worker=2
```

```bash
docker compose -f docker/docker-compose.yml exec -T \
  --user "$(id -u):$(id -g)" \
  -e SPARK_DRIVER_HOST=spark-master \
  -e SPARK_LOCAL_DIR=/tmp/spark-local \
  spark-master \
  python /project/src/controlled_benchmark.py \
    --models gbt \
    --computes docker-2x2-matched \
    --features /project/output/benchmarks/frozen_features \
    --feature-contract /project/output/benchmarks/feature_contract.json \
    --train-start-year 2017 \
    --validation-year 2023 \
    --test-year 2024 \
    --fold-id architecture_final_fold \
    --warmups 1 \
    --repetitions 3 \
    --seed 5208 \
    --hash-mode full \
    --publish-results /project/output/benchmarks/docker_results.csv \
    --publish-events /project/output/benchmarks/docker_events
```

```bash
docker compose -f docker/docker-compose.yml down
```

This uses two worker containers with two cores each. The `/project` path is the
container view of the same project and frozen feature sample used by the host.

## 11. Analyse the architecture experiment

```bash
./.venv/bin/python src/analyze_controlled_benchmark.py \
  --results \
    output/benchmarks/local_results.csv \
    output/benchmarks/standalone_results.csv \
    output/benchmarks/docker_results.csv \
  --event-logs \
    output/benchmarks/local_events \
    output/benchmarks/standalone_events \
    output/benchmarks/docker_events \
  --scaling-baseline local-2 \
  --hash-mode full
```

The analysis excludes warm-ups and reports median/minimum/maximum runtime,
throughput, speedup, parallel efficiency, Spark executor evidence and metric
stability. `architecture_fairness_audit.csv` confirms that sample counts,
folds, thresholds, feature version and input fingerprint are controlled.

