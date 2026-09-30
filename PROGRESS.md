# PROGRESS — where this project stands

> Status handoff for an AI agent or teammate resuming the work.
> Last updated: **ALL PHASES COMPLETE.** Data pipeline, Task 1, Task 2 (full-data
> GBT saved), and the full distributed-computing + model-comparison study are
> done. Report drafted (`report/REPORT.md` + self-contained `report/REPORT.html`
> for PDF export). Remaining: fill in group member names/IDs, export the PDF,
> and merge `feature/distributed-compute` → `mainline`.
>
> **Study delivered:** 8 models (6 MLlib + LSTM/TCN on GPU) × up to 5 compute
> configs (local[*], standalone-2×2/4×2, docker-2w/3w) + prepared HPC/Slurm
> tier. Master plan: [`EXPERIMENT_MATRIX.md`](EXPERIMENT_MATRIX.md).

## TL;DR

- **Done:** environment, data download (all 8 years), cleaning to Parquet,
  Task 1 exploration (figures + tables), all Task 2 code, and a verified
  train/evaluate/predict cycle on a 20-day/year smoke sample.
- **Model verified correct:** dry-context queries → ~3-4% probability,
  wet-context queries → 70-92%; probability drops as the threshold rises at the
  same moment. The threshold-as-feature design works.
- **Not done yet:** (1) one full-coverage local training run to lock in the
  reported metrics + saved model, (2) a Task 2 figure (AUC vs threshold),
  (3) the PDF report, (4) group member names/IDs, (5) the eventual HPC run on
  the full uncapped dataset.

## Environment (must read before running anything)

- Work happens inside **WSL Ubuntu**. Project root: `~/rainTheFall`
  (== `\\wsl.localhost\Ubuntu\home\amritsat\rainTheFall` from Windows).
- Run Python via the venv, e.g.:
  ```bash
  cd ~/rainTheFall && ./.venv/bin/python src/<script>.py
  ```
- Host has 32 GB RAM. **WSL2 was raised to 24 GB** via
  `C:\Users\satpa\.wslconfig`:
  ```ini
  [wsl2]
  memory=24GB
  swap=8GB
  processors=16
  ```
  Applied with `wsl --shutdown` + reopen. `free -h` now shows ~23 GB.
- Python 3.12, Java 21, PySpark 3.5.3 (`local[*]`, driver 20 GB for training).
- `setuptools<81` is in the venv to provide `distutils` under Python 3.12.
- **The user runs all train/predict commands themselves in their own WSL
  terminal** and pastes back output. Provide commands; do not run training via
  an automated terminal.

## Data facts

- Collection `2279` on data.gov.sg. Dataset ids per year in `src/config.py`.
- Raw: 8 CSVs in `data/raw/` (~7.8 GB), all downloaded.
- Cleaned: `data/processed/rainfall_clean` (Parquet, partitioned by `year`).
  - **48,538,713 rows**, **91 stations**.
  - Per-year rows: 2017≈5.26M, 2018≈4.59M, 2019≈5.03M, 2020≈6.33M,
    2021≈7.04M, 2022≈7.13M, 2023≈6.74M, 2024≈6.41M.
- Rainfall is sparse: median 5-min reading is 0; wet-reading intensity
  median 0.2 mm, p99.9 ≈ 10.6 mm.

## Task 1 — DONE and verified

`src/task1_explore.py` produced 8 figures + 8 tables in `output/`.
Key findings: annual per-station rainfall ≈ 1,500–2,750 mm; **2019 driest,
2021–2022 wettest**; strong **afternoon convective peak ~14:00–16:00 SGT**;
heavy-rain station-days (≥50 mm/day) peaked in **2021 (875)**; a ~786 mm daily
total appears in 2022 (flag as possible extreme/outlier in the report).

## Task 2 — CODE COMPLETE, VERIFIED ON SMOKE SAMPLE

Design: **threshold is a model feature**, so one model serves any threshold.

Files:
- `src/task2_features.py` — leakage-free features + threshold explode + label
  (next-30-min accumulation > threshold). `BASE_FEATURE_COLS`; default
  thresholds `[0.2, 1.0, 2.0, 5.0, 10.0]` mm.
- `src/task2_build_features.py` — **stage 1**: computes expensive per-station
  rolling/forward windows **once**, writes `data/processed/features_base`
  (Parquet, partitioned by year).
- `src/task2_train.py` — **stage 2**: reads features, samples, explodes,
  temporal split (train `year < test-year`, test `== test-year`), fits GBT
  (or `--model logreg`), evaluates, saves `models/rainfall_model` +
  `models/model_metadata.json`.
- `src/task2_predict.py` — loads pipeline, scores one `(station, time,
  threshold)` query, prints the feature row + `P(> threshold in next 30 min)`.

### Key implementation decisions (already applied)
- **No one-hot for station.** `StringIndexer` -> single `station_idx` column;
  GBT `maxBins=128` (must be ≥ ~91 station categories). One-hot made vectors
  wide + memory heavy. `lat`/`lon` carry the spatial signal too.
- **`--max-train-rows` cap (default 8,000,000)** downsamples the exploded
  training set so GBT fits locally. `--sample-fraction` subsamples base rows.
- **`MEMORY_AND_DISK`** persistence so Spark spills instead of OOM.
- GBT params: `maxIter=40, maxDepth=5, stepSize=0.1, subsamplingRate=0.7,
  cacheNodeIds=True, seed=42`.
- Two-stage split (build-features vs train) exists because computing windows
  inside training recomputed them on every Spark action → very slow / OOM.

### SMOKE-SAMPLE results (20 days/year — NOT the final reported numbers)
- Overall: AUC-ROC 0.936, AUC-PR 0.666, acc 0.964, F1 0.960.
- Per-threshold AUC-ROC rises 0.90 (0.2 mm) → 0.95 (10 mm); AUC-PR falls
  0.74 → 0.20 as positives get rarer (class imbalance — report AUC-PR honestly).

### Inference verification (real timestamps, S77, 2024)
- Dry context (roll_30m=0): P(>1mm)=0.036–0.038, class 0. ✓
- Wet context (roll_30m≈46mm, 2024-12-29 17:10): P(>1mm)=0.859, P(>5mm)=0.702. ✓
- Wet context (roll_30m≈42.6mm, 2024-11-17 13:55): P(>1mm)=0.916. ✓
- Conclusion: model is context-sensitive and uses the threshold feature
  correctly. The earlier "all ~3%" was because those inputs were genuinely dry.

## FULL-COVERAGE LOCAL RUN — DONE (final reported numbers)

Ran `task2_build_features.py` (full data) then `task2_train.py` for both models
(logreg first, GBT last so the saved model is the GBT). Test year = 2024,
train = 2017–2023, 8M-row training cap.

- Base feature rows: **47,636,919**. Train (capped): **7,997,496**;
  test (full 2024): **31,872,380**. Train positive rate ≈ **1.93 %**.
- **GBT overall:** AUC-ROC **0.924**, AUC-PR **0.584**, acc 0.986, F1 0.984.
- Per-threshold (GBT AUC-ROC / AUC-PR): 0.2→0.904/0.660, 1.0→0.905/0.577,
  2.0→0.895/0.505, 5.0→0.898/0.398, 10.0→0.905/0.302.
- **GBT beats LogReg at every threshold** on both AUC-ROC (~+0.03–0.04) and
  AUC-PR (~+0.05–0.10) — justifies the non-linear model. LogReg per-threshold
  numbers are recorded in `src/task2_plot.py` (`LOGREG_METRICS`).
- Saved: `models/rainfall_model` (GBT PipelineModel) + `models/model_metadata.json`.
- Figure: `output/figures/model_auc_by_threshold.png`; comparison table
  `output/tables/model_comparison_by_threshold.csv`.

## SCOPE EXPANSION — distributed-computing + model-comparison study

The single-process `local[*]` pipeline works but is "too easy" for the marks.
We are expanding into a **compute-environment × model matrix**: run Task 2
across several Spark deployment tiers and several models, benchmark them, and
discuss the trade-offs. The full plan (two axes, the combined matrix, the
enabling deliverables D1–D9, execution order, and verified feasibility) lives
in **[`EXPERIMENT_MATRIX.md`](EXPERIMENT_MATRIX.md)** — that is the master plan;
this section is just the running checklist.

Environments: **E0** `local[*]` (done) · **E1** local Standalone cluster ·
**E2** Docker Spark cluster · **E3** HPC multi-node · **E4** GPU (deep learning).
Models: **M0** GBT + **M1** LogReg (done) · **M2** RandomForest · **M3** LinearSVC ·
**M4** MLP (MLlib) · (**M5** FM, **M6** XGBoost-on-Spark = stretch) · **M7** LSTM (GPU).

### WHAT'S LEFT (in execution order — see matrix for detail)

1. **D1 — cluster-ready config.** Parameterise `--master`, executor cores/
   memory, shuffle partitions via CLI/env in `spark_utils.py`/`config.py` so the
   same scripts run on E0–E3 unchanged. *(unlocks everything else)*
2. **D2 — more MLlib models.** Extend `--model` to `rf`, `linsvc`, `mlp`
   (+ optional `fm`) → model-comparison table. Cheap, stays distributed.
3. **D5 + E1 — local Standalone cluster.** Launch scripts (`cluster/`), run
   Task 2, capture timing vs E0.
4. **D6 + E2 — Docker Spark cluster.** `docker/docker-compose.yml` (configurable
   worker replicas) + `submit.sh`; validate locally, capture timing.
   *(Docker 29.7.2 + Compose v5.5.0 confirmed present.)*
5. **D3 + D4 — benchmark harness + plots.** `src/benchmark.py` times runs per
   (env, model, #executors) → CSV; `src/benchmark_plot.py` → runtime/speed-up
   curves + model-AUC bars for the report.
6. **D8 + E4 — LSTM on GPU.** Install PyTorch-CUDA (~2.5 GB), build
   `src/task2_lstm.py` (same features/label/split/metrics), benchmark vs GBT.
   *(GPU: RTX 3050 Ti, 4 GB VRAM — modest model/batch sizes.)*
7. **D7 + E3 — HPC scale-out.** Submit scripts (Slurm/PBS or managed) + staging
   checklist; run uncapped full-data GBT (removes 8M cap). See HPC plan below.
8. **D9 — write the report** (`report/`): fold in Task 1, Task 2, the
   cluster-deployment + scaling + model-comparison sections, AI-usage
   declaration, limitations, and group member names + student IDs.

### Honest caveats to carry into the report
- **E1/E2 on one physical machine will likely be *slower* than E0** (network +
  serialization + container overhead on the same 16 cores / 24 GB). Real
  speed-up and the uncapped full-data training belong to **E3/HPC**. Measuring
  and explaining this scale-out-vs-scale-up distinction is itself worth marks.
- **4 GB VRAM** bounds the LSTM (E4) to a modest architecture.

### Open decisions (need user input) — mirrors matrix
1. Final scope: full matrix or trim (e.g. drop M5/M6, defer E3)?
2. HPC scheduler type: Slurm / PBS / Dataproc / Databricks / EMR?
3. Approve installing PyTorch-CUDA (~2.5 GB) for the LSTM?
4. Group member names + student IDs (for the report).

## HPC / cluster plan (full uncapped dataset — future work)

Goal: train on the **full ~240M-row exploded dataset** (48.5M base × 5
thresholds) without the `--max-train-rows` cap, for the strongest possible
model.

- **Where:** a Spark cluster — options: Google Cloud Dataproc, Databricks, or a
  university HPC with Spark/YARN. Any works because the code is plain PySpark
  with no local-only assumptions beyond paths.
- **Data staging:** upload `data/processed/rainfall_clean` (Parquet, small &
  compressed) to cluster storage (GCS/DBFS/HDFS). Do NOT re-upload raw CSVs;
  Parquet is enough. Alternatively run `download_data.py` + `clean_data.py` on
  the cluster.
- **Code changes needed:**
  - Parameterise input/output paths to the cluster FS (`gs://…`, `dbfs:/…`,
    `hdfs:/…`) instead of local `data/processed/...`. `src/config.py` is the
    single place to change; consider adding an env-var / CLI override.
  - In `spark_utils.get_spark`, drop `master("local[*]")` and the fixed
    `spark.driver.memory` so the cluster manager (YARN/K8s/Databricks) controls
    resources; keep the SGT timezone and shuffle-partition settings.
  - Run `task2_train.py` with `--max-train-rows 0` (disables the cap) and a
    larger `--shuffle-partitions` (e.g. 400–1000) suited to the cluster.
- **Suggested cluster size (starting point):** ~4–8 workers, 4 cores / 16–32 GB
  each (e.g. Dataproc `n2-standard-8`). GBT on ~240M rows is shuffle- and
  memory-heavy; scale workers if the fit is slow, and raise
  `spark.sql.shuffle.partitions` accordingly.
- **Expected differences vs local:** more positives seen per threshold → likely
  better AUC-PR on rare (high) thresholds; full station/time coverage.
- **Submit-check:** save the cluster-trained pipeline, copy `models/` and
  `model_metadata.json` back for the final submission, and update the report's
  metrics + environment section to describe the cluster config actually used.

## Git

- Repo on branch `mainline`. Commits so far:
  - `1447c2f` pipeline + Task 1 + Task 2 scaffolding
  - `df03a93` README + PROGRESS
  - `2679427` Task 2 training fixes (drop one-hot, row cap, disk persist,
    memory/maxBins) + README update
  - `049e91b` prediction feature-row diagnostic
- Remote `origin` exists (owner pushes). `.gitignore` excludes `data/`,
  `.venv/`, logs, `__pycache__`. Tracked: `src/`, `output/figures`,
  `output/tables`, `models/`, `report/`, README, PROGRESS, requirements.
- Note: `models/` + `output/tables/model_metrics_*.csv` are currently the
  smoke-run artifacts; replace with the full-run outputs before final commit.
