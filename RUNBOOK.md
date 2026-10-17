# RUNBOOK — how to run every model on every compute architecture

> **Superseded for final evidence.** Use
> [`VALIDATED_EXPERIMENT_RUNBOOK.md`](VALIDATED_EXPERIMENT_RUNBOOK.md) for the
> corrected temporal contract, immutable hashed artifacts, calibration,
> positive-class metrics and controlled repeated systems experiments. The
> commands below describe the earlier exploratory workflow and can produce
> results that are not directly comparable across deployments.

Copy-paste commands to run the rainfall prediction (Task 2) across all models
and all compute tiers. Run everything from the project root in a WSL terminal
with the venv Python (`./.venv/bin/python`).

- **Primary models:** `gbt`, `rf` and `mlp` (Spark MLlib, CPU).
- **Compute tiers (registry names):** `local-all`, `local-2/4/8`,
  `standalone-2x2`, `standalone-4x2`, `docker-2w`, `docker-3w`.

List everything available at any time:

```bash
./.venv/bin/python src/task2_train.py --list
```

Two result files are used:
- `output/tables/benchmark_results.csv` — model-quality study (full-data +
  GPU rows).
- `output/tables/benchmark_compute.csv` — compute-config study (same small
  sample across tiers, so timings are comparable).

---

## 0. One-time prerequisites

```bash
cd ~/rainTheFall
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt   # first time only

# data + features (skip if already present)
./.venv/bin/python src/download_data.py            # raw CSVs (2017-2024)
./.venv/bin/python src/clean_data.py               # -> data/processed/rainfall_clean
./.venv/bin/python src/task2_build_features.py     # -> data/processed/features_base
```

> **Fair-comparison tip:** for the *compute-config* study use a FIXED small
> sample so every tier does the same work and timings are comparable:
> `--sample-fraction 0.15 --max-train-rows 1000000`. For the *model-quality*
> study (accuracy) use the defaults (near-full data, 8M cap).

---

## 1. LOCAL — single process (`local[*]`)

### 1a. Reduced MLlib comparison

```bash
./.venv/bin/python src/benchmark.py \
    --models gbt rf mlp \
    --computes local-all
```

### 1b. The 2 deep models on GPU

```bash
./.venv/bin/python src/task2_deep.py --model lstm --epochs 8
./.venv/bin/python src/task2_deep.py --model tcn  --epochs 8
```

### 1c. (Optional) vary local parallelism — a single-machine scaling curve

```bash
./.venv/bin/python src/benchmark.py \
    --models gbt rf logreg linsvc \
    --computes local-2 local-4 local-8 local-all \
    --sample-fraction 0.15 --max-train-rows 1000000 \
    --benchmark-csv output/tables/benchmark_compute.csv
```

### Run a single model locally

```bash
./.venv/bin/python src/task2_train.py --model gbt --compute local-all --no-save
```

---

## 2. STANDALONE — local Spark cluster (real executors)

Start a cluster (args: `NUM_WORKERS CORES_PER_WORKER MEM_PER_WORKER`), export
its master URL, run, then stop. See `cluster/README.md`.

### 2a. 2 workers × 2 cores

```bash
cluster/start_standalone.sh 2 2 3g
export SPARK_MASTER_URL=spark://localhost:7077

for M in gbt rf logreg linsvc; do
  ./.venv/bin/python src/task2_train.py --model $M --compute standalone-2x2 \
      --no-save --sample-fraction 0.15 --max-train-rows 1000000 \
      --benchmark-csv output/tables/benchmark_compute.csv
done

cluster/stop_standalone.sh
```

### 2b. 4 workers × 2 cores

```bash
cluster/start_standalone.sh 4 2 3g
export SPARK_MASTER_URL=spark://localhost:7077

for M in gbt rf logreg linsvc; do
  ./.venv/bin/python src/task2_train.py --model $M --compute standalone-4x2 \
      --no-save --sample-fraction 0.15 --max-train-rows 1000000 \
      --benchmark-csv output/tables/benchmark_compute.csv
done

cluster/stop_standalone.sh
```

> **Note:** the `standalone-*` configs request 2 GB executors; the start script
> gives workers 3 GB so they fit. Deep models (`lstm`/`tcn`) are PyTorch, not
> Spark — they do NOT run on the Standalone/Docker clusters.

---

## 3. DOCKER — containerized Spark cluster

Build once, bring the cluster up with N workers, run the sweep, tear down. See
`docker/README.md`.

### 3a. 2 worker containers

```bash
docker compose -f docker/docker-compose.yml up -d --build --scale spark-worker=2
# master UI: http://localhost:8080
docker/run_sweep.sh docker-2w          # runs gbt, rf, logreg, linsvc
docker compose -f docker/docker-compose.yml down
```

### 3b. 3 worker containers

```bash
docker compose -f docker/docker-compose.yml up -d --scale spark-worker=3
docker/run_sweep.sh docker-3w
docker compose -f docker/docker-compose.yml down
```

### Run a single model in Docker

```bash
docker compose -f docker/docker-compose.yml up -d --scale spark-worker=2
docker/submit.sh gbt docker-2w --sample-fraction 0.15 --max-train-rows 1000000
docker compose -f docker/docker-compose.yml down
```

> **Known WSL2 quirk:** benchmark rows written from inside the container may not
> reach the host CSV reliably; the train times print in the logs. See
> `docker/README.md`. Not an issue on a real cluster with a shared filesystem.

---

## 4. Generate the comparison figures

```bash
# model-quality study: -> output/figures/model_auc_all.png etc.
./.venv/bin/python src/benchmark_plot.py

# compute-config study (tiers x models): -> output/figures/*_compute.png
./.venv/bin/python src/benchmark_plot.py \
    --results output/tables/benchmark_compute.csv --suffix _compute
```

---

## 5. Run EVERYTHING locally in one go (models on local[*] + GPU + figures)

```bash
cd ~/rainTheFall
# MLlib models
./.venv/bin/python src/benchmark.py --models gbt rf mlp --computes local-all
# deep models
./.venv/bin/python src/task2_deep.py --model lstm --epochs 8
./.venv/bin/python src/task2_deep.py --model tcn  --epochs 8
# figures
./.venv/bin/python src/benchmark_plot.py
```

For the full compute-config sweep across tiers, run sections 1c, 2a, 2b, 3a, 3b
(all writing to `output/tables/benchmark_compute.csv`), then the second command
in section 4.

---

## Quick reference

| I want to… | Command |
|------------|---------|
| See models + configs | `python src/task2_train.py --list` |
| One model, local | `python src/task2_train.py --model gbt --compute local-all --no-save` |
| Reduced MLlib comparison | `python src/benchmark.py --models gbt rf mlp --computes local-all` |
| Deep model (GPU) | `python src/task2_deep.py --model lstm --epochs 8` |
| Standalone cluster | `cluster/start_standalone.sh 4 2 3g` then `--compute standalone-4x2` |
| Docker cluster | `docker compose -f docker/docker-compose.yml up -d --scale spark-worker=2` then `docker/run_sweep.sh docker-2w` |
| Predict one query | `python src/task2_predict.py --station S77 --time "2024-06-15T14:00:00" --threshold 1.0` |
| Make figures | `python src/benchmark_plot.py` |
