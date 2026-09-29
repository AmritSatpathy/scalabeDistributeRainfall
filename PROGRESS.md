# PROGRESS — where this project stands

> Status handoff for an AI agent or teammate resuming the work.
> Last updated after: data pipeline + Task 1 complete; Task 2 code written but
> not yet trained end-to-end.

## TL;DR

- **Done:** environment, data download (all 8 years), cleaning to Parquet,
  Task 1 exploration (figures + tables), and all Task 2 code.
- **Not done yet:** actually running Task 2 (build features → train → predict),
  verifying model metrics, and writing the final PDF report.

## Environment (must read before running anything)

- Work happens inside **WSL Ubuntu**. Project root: `~/rainTheFall`
  (same as `\\wsl.localhost\Ubuntu\home\amritsat\rainTheFall` from Windows).
- Run Python via the venv, e.g.:
  ```bash
  wsl -d Ubuntu -- bash -lc "cd ~/rainTheFall && ./.venv/bin/python src/<script>.py"
  ```
- Machine: 16 cores, 15 GB RAM, ~900 GB disk. Python 3.12, Java 21,
  PySpark 3.5.3 (`local[*]`, driver 11 GB).
- `setuptools<81` is installed in the venv to provide `distutils` for
  PySpark under Python 3.12 (needed by `toPandas()`).
- Through PowerShell→WSL, **avoid nested quotes / `python -c`**; write code to
  files and run the file. `<` redirection also trips PowerShell's parser.
- In this shell, a trailing `echo EXIT=$?` sometimes prints `True`; **judge
  success by the script's own completion messages / output files, not the
  reported exit code.**

## Data facts

- Collection `2279` on data.gov.sg. Dataset ids per year are in
  `src/config.py` (`DATASET_IDS`).
- Raw: 8 CSVs in `data/raw/` (~7.8 GB total), all downloaded.
- Cleaned: `data/processed/rainfall_clean` (Parquet, partitioned by `year`).
  - **48,538,713 rows**, **91 stations**.
  - Per-year rows: 2017≈5.26M, 2018≈4.59M, 2019≈5.03M, 2020≈6.33M,
    2021≈7.04M, 2022≈7.13M, 2023≈6.74M, 2024≈6.41M.
  - Columns: `station_id, station_name, lat, lon, ts, date, year, month,
    day, hour, minute, dayofweek, dayofyear, rain_mm`.
- Rainfall is sparse: median 5-min reading is 0; wet-reading intensity
  median 0.2 mm, p99.9 ≈ 10.6 mm.

## Task 1 — DONE and verified

`src/task1_explore.py` produced (in `output/`):

- Figures: `annual_rainfall.png`, `monthly_heatmap.png`,
  `monthly_climatology.png`, `diurnal_cycle.png`, `diurnal_by_season.png`,
  `station_map.png`, `station_top_bottom.png`, `heavy_rain_days.png`.
- Tables: `annual_summary.csv`, `monthly_cycle.csv`, `diurnal_cycle.csv`,
  `diurnal_by_season.csv`, `station_rainfall.csv`,
  `wet_intensity_quantiles.csv`, `heavy_rain_days.csv`, `stations.csv`.

Key findings:
- Annual per-station rainfall ≈ 1,500–2,750 mm. **2019 driest**,
  **2021–2022 wettest** (consistent with La Niña years).
- Strong **afternoon convective peak at ~14:00–16:00 SGT**; overnight minimum.
- Heavy-rain station-days (daily total ≥ 50 mm) peaked in **2021 (875)**;
  a max daily total of ~786 mm appears in 2022 (flag as possible extreme /
  outlier in the report).

## Task 2 — CODE WRITTEN, NOT YET RUN END-TO-END

Design: **threshold is a model feature**, so one model serves any threshold.
Files:
- `src/task2_features.py` — leakage-free feature engineering + threshold
  explode + label (next-30-min accumulation > threshold). Base feature cols in
  `BASE_FEATURE_COLS`; default thresholds `[0.2, 1.0, 2.0, 5.0, 10.0]` mm.
- `src/task2_build_features.py` — **stage 1**: computes the expensive
  per-station rolling/forward windows **once** and writes
  `data/processed/features_base` (Parquet, partitioned by year).
- `src/task2_train.py` — **stage 2**: reads `features_base`, samples, explodes
  thresholds, temporal split (train `year < test-year`, test `== test-year`),
  fits GBT (or `--model logreg`), evaluates AUC-ROC/PR + accuracy/F1 + per
  threshold, saves model to `models/rainfall_model` and
  `models/model_metadata.json`.
- `src/task2_predict.py` — loads the saved pipeline and scores a single
  `(station, time, threshold)` query, printing `P(> threshold in next 30 min)`.

### Why the two-stage split exists (important)
The first attempt computed all window features inside `task2_train.py`. On the
full 48M rows this was very slow and memory-heavy because the per-station
`Window.partitionBy("station_id")` sorts were recomputed for every Spark action
(count, fit, transform, per-threshold eval). Splitting feature materialisation
into `task2_build_features.py` (write once, read many) fixes this.

## NEXT STEPS (in order)

1. **Validate feature build on a sample:**
   ```bash
   ./.venv/bin/python src/task2_build_features.py --sample-days 20 --out data/processed/_feat_smoke
   ```
   Confirm it finishes quickly and row counts look sane, then delete `_feat_smoke`.
2. **Full feature build:**
   ```bash
   ./.venv/bin/python src/task2_build_features.py
   ```
   (watch memory; if tight, raise `shuffle_partitions` or process in year chunks.)
3. **Train + evaluate:**
   ```bash
   ./.venv/bin/python src/task2_train.py --model gbt --test-year 2024
   ```
   Consider `--sample-fraction 0.2` first to sanity-check metrics fast.
   Also try `--model logreg` for the baseline comparison in the report.
4. **Verify inference:**
   ```bash
   ./.venv/bin/python src/task2_predict.py --station S77 --time "2024-06-15T14:00:00" --threshold 1.0
   ```
5. **Write the report** (`report/`): approach, environment, Task 1 + Task 2
   results with the figures/tables, model evaluation (AUC etc.), AI-usage
   declaration, limitations. Add group member names + student IDs.
6. **Confirm the trained model** is saved under `models/` for submission.

## Watch-outs / open questions

- **Class imbalance:** high thresholds (e.g. 10 mm/30 min) are rare — check
  positive rate per threshold; AUC-PR matters more than accuracy. May want
  class weighting or reporting calibration.
- **Label completeness:** `build_base_features` keeps only rows with a full
  30-min future (`future_n >= 6`). Verify this doesn't drop too much.
- **Station cold-start:** `StringIndexer(handleInvalid="keep")` handles unseen
  stations at predict time, but such predictions rely only on non-identity
  features.
- **Report figures for Task 2:** consider adding an AUC-vs-threshold plot and a
  reliability/calibration curve from the metrics tables.

## Git

- Repo initialised; branch `mainline`. First commit: pipeline + Task 1 +
  Task 2 scaffolding. No remote configured yet (owner will add + push).
- `.gitignore` excludes `data/`, `.venv/`, logs, `__pycache__`. Source,
  `output/figures`, `output/tables`, `models/`, `report/` are tracked.
