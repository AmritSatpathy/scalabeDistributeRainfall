# Spatiotemporal Processing and Task 1 EDA Runbook

Run these commands sequentially from a terminal. Each processing stage creates an immutable timestamped directory under `output/runs`, records its parameters and hashes, and updates a latest-successful-run pointer. The commands use full content hashes.

## 1. Enter the project and install dependencies

```bash
cd /Users/whitetiger/Desktop/WhiteTiger/17_projects/21_Rainfall/scalabeDistributeRainfall
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Creates the isolated Python environment and installs PySpark, pandas, Matplotlib and H3 support. Reusing an existing `.venv` is safe.

## 2. Combine and clean every annual CSV

```bash
./.venv/bin/python src/clean_data.py \
  --years 2017 2018 2019 2020 2021 2022 2023 2024 \
  --max-5min-mm 60 \
  --grid-tolerance-seconds 60 \
  --hash-mode full \
  --publish-legacy
```

Discovers exactly one uploaded CSV for every requested year, combines the files, parses Singapore timestamps, maps near-boundary source times such as `08:04:59` to canonical five-minute endpoints, deduplicates revisions, standardises station metadata and writes year-partitioned Parquet. It also writes cleaning-loss, cadence, extreme-day, station and device-history audits.

The script displays ten named phases. During Spark actions it shows active stage IDs, completed and active task counts, percentage, elapsed time and a live progress bar. Input/output hashing and the legacy publication copy display elapsed-time heartbeats, so long non-Spark operations do not appear stalled.

The immutable clean data are stored inside the run directory. `--publish-legacy` also refreshes `data/processed/rainfall_clean` for downstream compatibility.

## 3. Build all reusable temporal and spatial features

```bash
./.venv/bin/python src/build_spatiotemporal_features.py \
  --profile minimal \
  --forecast-stride-minutes 15 \
  --wet-threshold-mm 0 \
  --shuffle-partitions 192 \
  --output-partitions 96 \
  --station-buckets 4 \
  --hash-mode metadata
```

Reads the latest complete clean run and materialises a Parquet feature table containing:

- exact lags at 5, 15, 30 and 60 minutes, each paired with an availability flag;
- rolling sum, maximum, wet fraction and coverage over 30 minutes, 3 hours and 24 hours;
- recent 30-minute intensity change and current wet/dry state;
- cyclic time, day-of-year, weekday and monsoon-regime features;
- an exact, completeness-qualified next-30-minute outcome for later Task 2 work.

Training-only station/month/hour climatology is intentionally deferred to the temporal model-fitting stage. Computing it once from all years would leak validation and test history into model features.

The recommended `minimal` profile keeps the features needed by the selected Task 1 figures and a concise temporal Task 2 baseline. Station histories are partitioned by station and year, making each window task much smaller; the first 24 hours and final 30 minutes of each station-year naturally carry incomplete coverage/outcome indicators. Features are calculated on the full five-minute sequence, after which only 15-minute forecast origins are retained. This reduces the written contexts and downstream training work by approximately two thirds without corrupting the rolling windows.

Expensive distant-lag, long-state, wet-spell and nearest-station calculations are reserved for later ablations. The scalable-computing experiment should compare controlled execution resources using this stable feature table rather than making feature construction itself unnecessarily large.

The table is partitioned by year and month and distributed across station buckets to avoid very large output tasks. The machine-readable feature contract is stored with the run. The immutable run artifact is consumed automatically by Step 4, so a duplicate `--publish-legacy` copy is unnecessary.

The script displays eight named phases. The main feature-materialisation phase shows active Spark stages, completed and active tasks, failed tasks, percentage and elapsed time. Input/output hashing and the optional legacy copy show regular elapsed-time heartbeats.

### Optional spatial extension

After the core run succeeds, nearest-neighbour features can be evaluated separately:

```bash
./.venv/bin/python src/build_spatiotemporal_features.py \
  --profile core \
  --with-neighbours \
  --neighbour-count 4 \
  --shuffle-partitions 384 \
  --output-partitions 384 \
  --station-buckets 8 \
  --hash-mode metadata
```

This path materialises the temporal frame once on disk before constructing the spatial branch, preventing Spark from recalculating every temporal window twice.

## 4. Generate the geospatial Task 1 tables and PNG figures

```bash
./.venv/bin/python src/geospatial/task1_geospatial.py \
  --daily-completeness 0.90 \
  --h3-resolution 7 \
  --minimum-event-stations 20 \
  --hash-mode metadata \
  --publish-legacy
```

Reads the latest clean and spatiotemporal-feature runs. It selects the target month by the largest positive same-calendar-month historical anomaly and selects the event origin by the largest coverage-qualified network 30-minute accumulation.

It produces 300-dpi PNG files:

| File | Purpose |
|---|---|
| `geo_G1_network_continuity.png` | Station geography, active years, device changes and monthly network size |
| `geo_G2_seasonal_geography.png` | Four-panel H3 comparison of Singapore rainfall regimes |
| `geo_G4_multibaseline_anomaly.png` | Target month versus previous month, trailing three months, previous year and historical same-month mean |
| `geo_G10_lagged_storm_footprint.png` | Rainfall state at `t−60`, `t−30`, `t−15`, `t` and the clearly labelled `t+30` outcome |
| `geo_G8_G9_antecedent_intensification.png` | 30-minute, 3-hour and 24-hour accumulation, intensity change, wet fraction and recent peak intensity |

Tables containing the plotted values and the selection audit are stored beside the figures. `--publish-legacy` copies the completed figures into `output/figures/geospatial` and tables into `output/tables`.

## 5. Inspect the tracked outputs

```bash
./.venv/bin/python src/run_artifacts.py --latest clean
./.venv/bin/python src/run_artifacts.py --latest spatiotemporal_features
./.venv/bin/python src/run_artifacts.py --latest task1_geospatial
```

Prints the immutable run locations, manifests and manifest hashes. Use the figures from the `task1_geospatial` run recorded in the final command.

## Optional controlled rerun for a chosen case study

After inspecting the automatically selected period, a declared target month and forecast origin can be rendered without changing the processed data:

```bash
./.venv/bin/python src/geospatial/task1_geospatial.py \
  --target-month 2024-11 \
  --event-time "2024-11-15T15:30:00" \
  --daily-completeness 0.90 \
  --h3-resolution 7 \
  --minimum-event-stations 20 \
  --hash-mode full
```

Replace the example month and timestamp with a scientifically justified case. Retain the automatic-selection run as evidence that the report event was not chosen only for visual effect.

