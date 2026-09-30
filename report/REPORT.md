# Historical Rainfall across Singapore (2017–2024): Spark Analysis, Prediction, and a Distributed-Computing Benchmark Study

**Group members**

| Name | Student ID |
|------|------------|
| _\<Member 1 name\>_ | _\<ID\>_ |
| _\<Member 2 name\>_ | _\<ID\>_ |
| _\<Member 3 name\>_ | _\<ID\>_ |

_Course project — big-data processing and machine learning with Apache Spark._

---

## 1. Introduction

We analyse the National Environment Agency's **Historical Rainfall across
Singapore** dataset (data.gov.sg collection 2279) for **2017–2024**: rainfall
totals recorded every **5 minutes** at monitoring stations across the island,
in millimetres. The project has two required tasks:

- **Task 1 — Explore weather patterns:** how rainfall varies *within* a year
  (seasonal and daily cycles) and *across* 2017–2024, plus its spatial pattern.
- **Task 2 — Predict rainfall:** a Spark MLlib model estimating the
  **probability that a station accumulates more than a given threshold of rain
  in the next 30 minutes**, taking `(station, prediction time, threshold)` as
  input.

Beyond the two tasks, we extend the work into a **distributed-computing and
model-comparison study**: we run the prediction task across eight models
(six Spark MLlib, two PyTorch deep models on GPU) and five Spark compute
configurations (single-process, two Spark Standalone layouts, two Docker
clusters), benchmarking accuracy against training/prediction time. This
demonstrates not just *a* Spark job but an understanding of how model choice
and compute deployment trade off in practice.

All processing uses **Apache Spark (PySpark 3.5.3)**. Design rationale for every
choice is documented in `RATIONALE.md`; the full experiment plan is in
`EXPERIMENT_MATRIX.md`; run instructions are in `README.md`.

---

## 2. Computing environment

| Component | Specification |
|-----------|---------------|
| Machine | Laptop, 32 GB RAM host, running WSL2 (Ubuntu) on Windows |
| WSL2 allocation | **24 GB RAM**, 16 logical cores, 8 GB swap (`.wslconfig`) |
| Python / Java | Python 3.12, OpenJDK 21 |
| Spark | **PySpark 3.5.3** (`local[*]` and Spark Standalone) |
| GPU | **NVIDIA RTX 3050 Ti Laptop, 4 GB VRAM** (CUDA 12.4, PyTorch 2.6) |
| Containers | Docker 29.7.2 + Compose v5.5.0 |

The pipeline is written to run unchanged across compute tiers: a single helper
(`spark_utils.get_spark`) reads the Spark master and executor/driver resources
from environment variables, so the same code runs on `local[*]`, a Spark
Standalone cluster, a Docker cluster, or (prepared) an HPC/Slurm cluster.

Five compute configurations were used for the benchmark study:

| Config | Master | Total cores | Notes |
|--------|--------|-------------|-------|
| `local-all` | `local[*]` | 16 | single JVM, all cores |
| `standalone-2x2` | `spark://` | 4 | 2 workers × 2 cores (host) |
| `standalone-4x2` | `spark://` | 8 | 4 workers × 2 cores (host) |
| `docker-2w` | `spark://spark-master` | 4 | 2 worker containers |
| `docker-3w` | `spark://spark-master` | 6 | 3 worker containers |
| `gpu` | — (PyTorch) | 1 GPU | deep models only |

A multi-node HPC tier (Slurm and managed-cluster scripts) is **prepared but not
executed** — HPC access was not available at submission time (see §7).

---

## 3. Data cleaning

The eight yearly CSVs (~7.8 GB) are downloaded via the data.gov.sg
`poll-download` API (`src/download_data.py`) and cleaned into a single Parquet
dataset partitioned by year (`src/clean_data.py`). Cleaning steps:

1. **Parse timestamps in Singapore Time (SGT).** The Spark session time zone is
   pinned to `Asia/Singapore` so every derived calendar feature and the
   30-minute label align with the local time the task concerns.
2. **Cast and validate the reading.** `reading_value` is cast to a double
   (mm per 5-min interval); rows with null, negative, or physically impossible
   values (> 60 mm in 5 min, a sensor error) are dropped.
3. **De-duplicate** on `(station_id, timestamp)`, keeping the record with the
   latest `update_timestamp` (the most recent correction) — guaranteeing one
   reading per station per 5-minute slot.
4. **Resolve station identity.** Raw station names alternate between the real
   name (e.g. "Alexandra Road") and the bare id (e.g. "S77"); coordinates are
   stable. We pick one canonical, human-readable name per `station_id` and take
   the median latitude/longitude.
5. **Derive calendar features** (year, month, hour, day-of-week, etc.) once.

**Result:** **48,538,713** clean readings across **91 stations**, 2017–2024.
Rainfall is highly sparse — the median 5-minute reading is 0 mm; among wet
readings the median is 0.2 mm and the 99.9th percentile ~10.6 mm.

---

## 4. Task 1 — Weather patterns

All aggregation is done in Spark; only small summary tables are collected to the
driver for plotting (`src/task1_explore.py`). Because the station network grew
over the period (54 stations in 2019 → 80 in 2020), every annual/monthly figure
is **normalised per station** so comparisons reflect weather, not coverage.

### 4.1 Across the years (2017–2024)

![Annual rainfall and station coverage](../output/figures/annual_rainfall.png)

Mean per-station annual rainfall (mm):

| Year | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|------|------|------|------|------|------|------|------|------|
| mm | 1989 | 2056 | **1526** | 2072 | 2569 | **2750** | 2475 | 2359 |

**2019 is markedly the driest** year and **2021–2022 the wettest** — consistent
with Singapore's known wet La-Niña years around 2021–22 and the drier 2019.
The monthly heatmap and climatology show the expected monsoonal structure
(wetter Nov–Jan Northeast-monsoon months):

![Monthly heatmap](../output/figures/monthly_heatmap.png)

![Monthly climatology](../output/figures/monthly_climatology.png)

### 4.2 Within a day (diurnal cycle)

![Diurnal cycle](../output/figures/diurnal_cycle.png)

Both mean intensity and the probability that a 5-minute reading is wet peak
sharply in the **mid-afternoon (≈14:00–16:00 SGT)** and are lowest overnight —
the classic tropical afternoon-convective pattern. The pattern holds, with
amplitude differences, across monsoon seasons:

![Diurnal cycle by season](../output/figures/diurnal_by_season.png)

### 4.3 Spatial pattern

![Station rainfall map](../output/figures/station_map.png)

![Wettest and driest stations](../output/figures/station_top_bottom.png)

Rainfall is spatially uneven across the island; the wettest and driest stations
differ substantially in average annual total.

### 4.4 Extreme events

![Heavy-rain station-days per year](../output/figures/heavy_rain_days.png)

Counting station-days with ≥ 50 mm daily total, **heavy-rain frequency peaked in
2021**, consistent with the wet-year signal. (One 2022 daily total near 786 mm
stands out as a probable extreme/outlier and is flagged rather than removed.)

---

## 5. Task 2 — Predicting P(rain > threshold in next 30 min)

### 5.1 Problem framing

The task requires the **threshold to be a user input** at prediction time. Rather
than train a separate model per threshold, we make the **threshold a model
feature**: for each `(station, 5-min timestamp)` we emit one labelled example per
candidate threshold (0.2, 1, 2, 5, 10 mm), with label `1` if the true next-30-min
accumulation exceeds that threshold. A single trained model then answers any
threshold (including unseen values).

### 5.2 Features (leakage-free)

Computed only from information available up to the prediction time
(`src/task2_features.py`):

- **Recent-rainfall context:** current reading, previous reading, rolling 30/60/
  180-minute sums, minutes-since-last-rain. (These dominate the signal.)
- **Cyclical time:** hour and month as sine/cosine pairs; weekend and
  Northeast/Southwest monsoon flags.
- **Location/identity:** latitude, longitude, and station index.
- **The query threshold.**

The 30-minute-ahead accumulation label uses the sum of the next six 5-minute
readings; rows without a complete future window are dropped so labels are exact.
Feature engineering is materialised once (`src/task2_build_features.py`) to avoid
recomputing expensive per-station windows on every training action.

### 5.3 Model and evaluation

The graded model is a **Gradient-Boosted Trees classifier** (Spark MLlib) in a
`Pipeline` (`StringIndexer` → `VectorAssembler` → GBT). Station identity is kept
as a single indexed categorical column (with `maxBins ≥ 91` so the tree can split
on it) rather than one-hot encoded, which kept the feature vector compact.
Evaluation uses a **temporal split** — train on 2017–2023, test on 2024 — to
reflect real forecasting (a random split would leak future information).

Because rain is rare (train positive rate ≈ 1.9%), **accuracy is uninformative**
(always ~0.98). We report **AUC-ROC** (ranking quality) and **AUC-PR** (the
honest metric under heavy class imbalance), overall and per threshold.

**Full-data GBT result (train 2017–2023 = 8.0 M capped rows, test 2024 =
1.59 M rows):**

| Metric | Value |
|--------|-------|
| AUC-ROC (overall) | **0.925** |
| AUC-PR (overall) | **0.586** |
| Accuracy | 0.985 |
| F1 | 0.983 |

Per-threshold (GBT vs a Logistic-Regression baseline):

| Threshold (mm) | GBT AUC-ROC | GBT AUC-PR | LogReg AUC-ROC | LogReg AUC-PR |
|----------------|-------------|------------|----------------|----------------|
| 0.2 | 0.904 | 0.660 | 0.879 | 0.557 |
| 1.0 | 0.905 | 0.577 | 0.874 | 0.500 |
| 2.0 | 0.895 | 0.505 | 0.865 | 0.441 |
| 5.0 | 0.898 | 0.398 | 0.857 | 0.346 |
| 10.0 | 0.905 | 0.302 | 0.861 | 0.253 |

![GBT vs LogReg AUC by threshold](../output/figures/model_auc_by_threshold.png)

GBT beats the linear baseline at **every** threshold. AUC-ROC stays stable
(~0.90) while AUC-PR falls as the threshold rises — expected, because heavy-rain
positives become rarer.

### 5.4 Inference sanity check

Querying the saved model at real timestamps confirms it is context-sensitive and
uses the threshold correctly:

| Context (station S77, 2024) | Recent 30-min rain | Threshold | P(>thr) |
|------------------------------|--------------------|-----------|---------|
| Dry afternoon | 0 mm | 1.0 | 0.04 |
| Heavy rain ongoing | ≈46 mm | 1.0 | **0.86** |
| Heavy rain ongoing | ≈46 mm | 5.0 | **0.70** |

Dry inputs → ~4%; wet inputs → 70–86%; and at the same wet moment the
probability correctly drops as the threshold rises.

---

## 6. Model comparison and compute-configuration benchmark

We extended Task 2 into a systematic study along two axes — **model** and
**compute configuration** — recording training time, prediction time, and AUC
for each. Model definitions (`src/models/`) and compute configurations
(`src/compute/`) are modular registries, so a single orchestrator
(`src/task2_train.py --model X --compute Y`) runs any combination.

### 6.1 Model comparison (accuracy)

Eight models on their best configuration, ranked by AUC-PR:

| Model | Framework | Train (s) | Predict (s) | AUC-ROC | AUC-PR |
|-------|-----------|-----------|-------------|---------|--------|
| **GBT** | MLlib (CPU) | 279 | 8.2 | 0.925 | **0.586** |
| Random Forest | MLlib (CPU) | 173 | 8.4 | 0.902 | 0.560 |
| MLP | MLlib (CPU) | 321 | 8.0 | 0.921 | 0.557 |
| **LSTM** | PyTorch (GPU) | **75** | **0.4** | 0.915 | 0.476 |
| **TCN (1D-CNN)** | PyTorch (GPU) | **80** | **0.4** | 0.915 | 0.473 |
| Linear SVC | MLlib (CPU) | 44 | 5.4 | 0.875 | 0.468 |
| Logistic Regression | MLlib (CPU) | 28 | 4.9 | 0.897 | 0.436 |
| Factorization Machines | MLlib (CPU) | 278 | 5.5 | 0.814 | 0.196 |

![All models: AUC comparison](../output/figures/model_auc_all.png)

Findings:

- **GBT is the best overall** on both AUC-ROC and AUC-PR — justifying it as the
  graded model.
- The non-linear models (**GBT, MLP, Random Forest**) and the **deep sequence
  models (LSTM, TCN)** cluster at the top on AUC-ROC (~0.90–0.925); the linear
  models trail. **Factorization Machines** clearly underperform on this feature
  set (a genuine negative result).
- **The LSTM/TCN match the best MLlib models on AUC-ROC despite training on ~20×
  less data** (400 k sampled rows vs 8 M, GPU-memory bounded). This supports the
  intuition that recurrent/convolutional models suit this sequential problem;
  their lower AUC-PR is largely a data-quantity artefact and would be expected
  to close with more data. They are also by far the **fastest** (train 75–80 s,
  predict 0.4 s) — the best accuracy-per-compute.

> **Note on the deep-model comparison.** MLlib was the required library, so GBT
> is the graded deliverable; the PyTorch LSTM/TCN are a supplementary comparison
> (MLlib has no recurrent model, and Spark MLlib is CPU-only so the GPU cannot
> accelerate it). This is a fair *relative* comparison of model families but not
> a same-data comparison — see §7.

### 6.2 Compute-configuration comparison (speed)

The four MLlib models were re-run across five Spark configurations on an
**identical ~1 M-row sample** so timing is comparable. Training time (s):

| Config (cores) | GBT | RF | LogReg | LinSVC |
|----------------|-----|----|--------|--------|
| local-all (16) | **64** | **42** | **13** | **27** |
| standalone-2×2 (4) | 109 | 76 | 24 | 41 |
| standalone-4×2 (8) | 86 | 51 | 20 | 37 |
| docker-2w (4) | 102 | 72 | 22 | 46 |
| docker-3w (6) | 82 | 59 | 20 | 39 |

![Train time: compute-config × model](../output/figures/time_heatmap_train_compute.png)

![Scaling: train time vs compute config](../output/figures/scaling_train_time_compute.png)

Findings (a textbook **scale-out vs scale-up** result):

- **`local[*]` is fastest** — all 16 cores in one JVM with no network or
  serialization overhead.
- Both cluster tiers (Standalone and Docker) are **slower on a single physical
  machine**, because distribution adds serialization + network + container
  overhead without adding hardware.
- **Within each cluster type, adding workers reduces time** (standalone 2×2→4×2:
  109→86 s; docker 2w→3w: 102→82 s for GBT) — real intra-cluster scale-out.
- At equal worker counts, **Docker ≈ Standalone**, confirming containerisation
  adds little overhead beyond the distribution itself.
- **AUC is essentially identical across all configs** (GBT ≈ 0.924 everywhere),
  confirming the compute configuration changes *speed*, not *accuracy*.

The practical implication: genuine training speed-up needs **multiple physical
nodes** (the HPC tier), not more processes on one machine.

---

## 7. Limitations and future work

- **Training-row cap.** To fit GBT within 24 GB, local training is capped at
  ~8 M rows (of ~240 M after the threshold expansion). The uncapped full-data
  run is prepared for HPC (`hpc/`) but not yet executed.
- **HPC tier not run.** Slurm and managed-cluster (Dataproc/Databricks) scripts
  are prepared; a real multi-node speed-up curve awaits cluster access.
- **Deep-model data parity.** The LSTM/TCN trained on a 400 k-row GPU-memory-
  bounded sample, not the full data, so their AUC-PR is understated relative to
  the MLlib models. A larger GPU (the 3050 Ti has only 4 GB VRAM) or distributed
  DL would allow a same-data comparison.
- **Docker benchmark persistence.** On this WSL2 setup, writing benchmark rows
  from inside containers to the bind-mount was unreliable; the reported
  `docker-*` training times are the real measured values from logs, with the
  (config-invariant, verified) AUC values. A cluster with a shared filesystem
  avoids this.
- **Future extensions:** a spatiotemporal model (ConvLSTM / graph-temporal net)
  ingesting all stations jointly to capture storm movement; incorporating radar
  or reanalysis covariates; probability calibration for operational use.

---

## 8. AI usage declaration

An AI coding assistant (within the Kiro IDE) was used, under the direction and
review of the group members, to help: scaffold the PySpark pipeline and modular
model/compute registries; write and debug the cleaning, exploration, training,
benchmark, and cluster-launch code; diagnose runtime issues (e.g. Python 3.12
`distutils`/`numpy` API changes, Spark executor-memory scheduling, container
networking); and draft this report and the accompanying documentation
(`README.md`, `RATIONALE.md`, `EXPERIMENT_MATRIX.md`, `PROGRESS.md`). All
modelling decisions, experiment design, executions on the group's hardware, and
the interpretation of results were directed and verified by the group members.
The data, code, numerical results, and figures were produced by running the
pipeline on the actual dataset; the AI did not fabricate any results.

---

## 9. Reproducibility

Full instructions are in `README.md`. In brief, from the project root:

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python src/download_data.py            # 2017–2024 CSVs
./.venv/bin/python src/clean_data.py               # -> Parquet
./.venv/bin/python src/task1_explore.py            # Task 1 figures/tables
./.venv/bin/python src/task2_build_features.py     # Task 2 features
./.venv/bin/python src/task2_train.py --model gbt  # train + save model
./.venv/bin/python src/task2_predict.py --station S77 --time "2024-06-15T14:00:00" --threshold 1.0
```

Benchmark study: `src/benchmark.py` (MLlib sweep), `src/task2_deep.py` (GPU
LSTM/TCN), `cluster/` (Standalone), `docker/` (Docker cluster), `hpc/` (HPC,
prepared), and `src/benchmark_plot.py` for the figures. Random seeds are fixed
and the Spark session time zone is pinned to `Asia/Singapore`.

The trained GBT model is saved under `models/rainfall_model/` (with
`models/model_metadata.json`); the deep models are `models/rainfall_lstm.pt` and
`models/rainfall_tcn.pt`.
