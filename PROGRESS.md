# PROGRESS — where this project stands

> Status handoff for an AI agent or teammate resuming the work.
> Last updated: data pipeline + Task 1 complete; Task 2 complete end-to-end —
> **full-coverage local training run done** (GBT + LogReg, final metrics below),
> model saved, AUC-vs-threshold figure produced. Remaining: the written report
> (+ member IDs), an optional GPU LSTM comparison, and the eventual uncapped
> HPC run. Plan for HPC included below.

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

## WHAT'S LEFT (in order)

1. **Write the report** (`report/`, PDF-ready markdown): approach, environment,
   Task 1 + Task 2 results with figures/tables, model evaluation (AUC etc.),
   AI-usage declaration, limitations, group member names + student IDs.
   Use the numbers above and the figures in `output/figures`.
2. **Optional (recommended per discussion):** build `task2_lstm.py` — a PyTorch
   GPU LSTM benchmarked head-to-head with the GBT (same features/label/split/
   metrics), as supplementary model-comparison + future-work evidence. The GBT
   remains the graded MLlib deliverable.
3. **Needed from the user:** group member names + student IDs.
4. **HPC full uncapped run:** see the cluster plan below (removes the 8M cap).

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
