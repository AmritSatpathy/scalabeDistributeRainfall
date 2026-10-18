# Singapore Rainfall Analysis and Prediction

**DSA5208 — Scalable Distributed Computing for Data Science**<br>
National University of Singapore

This project uses Apache Spark and PySpark to analyse five-minute rainfall
observations collected across Singapore from 2017 to 2024. It combines
distributed data preparation, exploratory analysis, geospatial rainfall
mapping, short-horizon rainfall prediction, and compute-configuration
experiments.

The submission document is:

- [`output/Historical_Rainfall_Singapore_Report.docx`](output/Historical_Rainfall_Singapore_Report.docx)

[`output/Rainfall_Report_Gap_Analysis.md`](output/Rainfall_Report_Gap_Analysis.md)
records the earlier report review and improvement plan.

## Project scope

The project addresses two tasks:

1. **Exploratory data analysis.** Describe rainfall variation across space,
   years, months, monsoon seasons, hours of the day, and heavy-rain events.
2. **Rainfall prediction.** Estimate whether rainfall accumulated during the
   next 30 minutes will exceed a selected threshold for a station and
   prediction time.

The submitted report presents the exploratory analysis in this order:

1. geospatial observation coverage and rainfall geography;
2. interannual variation;
3. monthly seasonality;
4. diurnal variation; and
5. heavy-rain station-days.

The systems component compares Spark execution layouts and model behaviour.
Results from a single physical machine are interpreted as deployment-overhead
measurements rather than evidence of multi-machine scalability.

## Data

Source: [data.gov.sg — Historical Rainfall across Singapore](https://data.gov.sg/collections/2279/view),
collection `2279`, published by Singapore's National Environment Agency.

The annual CSV files contain five-minute rainfall observations and station
metadata, including:

```text
date, timestamp, update_timestamp, station_id, station_name,
station_device_id, location_longitude, location_latitude,
reading_update_timestamp, reading_value, reading_type, reading_unit
```

The 2017–2024 input is approximately 7.8 GB. The retained cleaned dataset used
in the report contains 48,538,713 readings from 91 stations. Timestamps are
interpreted in `Asia/Singapore`.

## Repository structure

```text
scalabeDistributeRainfall/
├── assets/                         # NUS branding, author metadata and references
├── cluster/                        # local Spark Standalone launch scripts
├── data/
│   ├── raw/                        # annual source CSV files
│   ├── processed/                  # cleaned and feature Parquet datasets
│   └── singapore_map_bounary/      # official planning-region GeoJSON
├── docker/                         # Docker Spark configuration and sweep scripts
├── models/                         # retained trained-model artifacts and metadata
├── output/
│   ├── Historical_Rainfall_Singapore_Report.docx
│   ├── figures/
│   │   └── geospatial/             # published geographic PNG figures
│   ├── geojson/                    # generated geographic data layers
│   ├── interactive/                # interactive Folium HTML maps
│   ├── tables/                     # EDA, coverage and model-result tables
│   └── runs/                       # timestamped, hashed run artifacts
├── src/
│   ├── clean_data.py
│   ├── task1_explore.py
│   ├── build_spatiotemporal_features.py
│   ├── geospatial/                 # all Task 1 map-generation scripts
│   ├── task2_*.py                  # feature, training, evaluation and prediction tools
│   ├── models/                     # model registry
│   └── compute/                    # local, Standalone and Docker configurations
├── SPATIOTEMPORAL_PIPELINE_RUNBOOK.md
├── VALIDATED_EXPERIMENT_RUNBOOK.md
├── requirements.txt
└── README.md
```

Large processed datasets, virtual environments, Spark logs, and immutable run
directories are excluded from version control. Published report figures,
tables, GeoJSON, and HTML maps are kept under `output/`.

## Environment setup

Python, Java, and Spark must be available locally. The recorded report
experiments used PySpark 3.5.3 and Java 21.

```bash
cd /path/to/scalabeDistributeRainfall
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m playwright install chromium
```

`setuptools<81` is pinned because PySpark 3.5.x still imports `distutils` in
some supported Python environments.

### Basemap configuration

Folium maps can use OpenStreetMap without a key. CARTO-rendered map layers use
a local configuration file that is intentionally excluded from Git:

```json
{
  "carto": {
    "api_key": "YOUR_CARTO_BASEMAP_KEY"
  }
}
```

Save this as `basemaps.local.json` in the project root. Do not commit API keys.
Map attribution remains visible in HTML and PNG outputs.

## Running the pipeline

Run commands from the project root with the virtual-environment Python.

### 1. Download or provide the annual data

```bash
./.venv/bin/python src/download_data.py
```

The canonical raw filename is `data/raw/rainfall_<year>.csv`; this is the same
contract used by the mainline branch and by `src/download_data.py`. Cleaning,
modelling, and geospatial analysis all derive from these eight files. Do not
keep a second copy under an alternative filename. Dataset identifiers for
2017–2024 are recorded in `src/config.py`.

### 2. Clean and standardise the observations

```bash
./.venv/bin/python src/clean_data.py
```

The cleaning stage parses Singapore timestamps, casts rainfall and coordinates
to numeric values, rejects invalid readings, de-duplicates station-time rows,
and resolves canonical station metadata. Its primary output is partitioned
Parquet under `data/processed/rainfall_clean`.

For a small smoke run:

```bash
./.venv/bin/python src/clean_data.py \
  --sample-days 3 \
  --out data/processed/_smoke
```

### 3. Build the compact spatiotemporal feature dataset

```bash
./.venv/bin/python src/build_spatiotemporal_features.py
```

This stage retains a deliberately limited set of temporal and spatial
features so it remains feasible on a single workstation. Progress is printed
while Spark actions are running. Completed artifacts are registered under a
timestamped directory in `output/runs/`.

See [`SPATIOTEMPORAL_PIPELINE_RUNBOOK.md`](SPATIOTEMPORAL_PIPELINE_RUNBOOK.md)
for the exact staged workflow and resume guidance.

### 4. Generate the non-geographic EDA figures

```bash
./.venv/bin/python src/task1_explore.py
```

This produces annual, monthly, diurnal, monsoon, station-ranking, and
heavy-rain summaries under `output/figures/` and `output/tables/`.

### 5. Generate the geospatial analysis

All geographic scripts are located in `src/geospatial/`. Their detailed
responsibilities are listed in
[`src/geospatial/README.md`](src/geospatial/README.md).

First generate the coverage-qualified station-month summaries:

```bash
./.venv/bin/python src/geospatial/task1_geospatial_choropleths.py \
  --publish-legacy
```

Then generate the report maps. `--render-png` captures the interactive Folium
maps with Playwright, and `--publish-legacy` copies the selected outputs to the
stable `output/` directories used by the report.

```bash
# Active gauges, August 2017/2024, complete-year maps and change maps
./.venv/bin/python src/geospatial/task1_august_geospatial_maps.py \
  --monthly-summary output/tables/geo_station_month_summary.csv \
  --render-png \
  --publish-legacy

# Comparable August maps for 2017, 2018, 2023 and 2024
./.venv/bin/python src/geospatial/task1_august_year_maps.py \
  --years 2017 2018 2023 2024 \
  --render-png \
  --publish-legacy

# Long-period H3 mean monthly rainfall, 2017–2024
./.venv/bin/python src/geospatial/task1_long_period_average_map.py \
  --render-png \
  --publish-legacy

# Long-period official planning-region summary
./.venv/bin/python src/geospatial/task1_region_monthly_choropleth.py \
  --render-png \
  --publish-legacy

# August 2024 and complete-2024 planning-region maps
./.venv/bin/python src/geospatial/task1_region_2024_choropleths.py \
  --render-png \
  --publish-legacy
```

The geospatial outputs are separated by artifact type:

| Output | Location |
|---|---|
| Report-ready PNG maps | `output/figures/geospatial/` |
| Interactive Folium maps | `output/interactive/` |
| Generated geographic layers | `output/geojson/` |
| Coverage and regional summaries | `output/tables/` |
| Immutable execution record | `output/runs/<timestamp>_<task>_<hash>/` |

The report uses four focused geographic views: active gauges in 2017 and
2024, August 2017 versus August 2024, the 2017–2024 H3 mean monthly pattern,
and the official planning-region summary for 2024.

### 6. Task 2 model workflow

The retained modelling scripts support temporal feature materialisation,
Spark MLlib training, evaluation, and single-query prediction:

```bash
./.venv/bin/python src/task2_build_features.py
./.venv/bin/python src/task2_train.py --model gbt --test-year 2024
./.venv/bin/python src/task2_predict.py \
  --station S77 \
  --time "2024-06-15T14:00:00" \
  --threshold 1.0
```

The canonical split trains on 2017–2023 and tests on 2024. Recent rainfall,
rolling 30/60/180-minute history, time-of-day and month cycles, station
identity, coordinates, and the query threshold form the model input. The
current reading is excluded from the future 30-minute target.

For controlled experiment commands, use
[`VALIDATED_EXPERIMENT_RUNBOOK.md`](VALIDATED_EXPERIMENT_RUNBOOK.md). The
report distinguishes retained results from proposed follow-up experiments.

## Analytical conventions

- **Coverage normalisation:** station-month and region-month summaries are
  used so growth in the observation network does not directly determine the
  rainfall pattern.
- **Sensor-supported geography:** H3 polygons are shown only where station
  evidence exists. The maps do not interpolate a continuous rainfall surface
  over ungauged land.
- **Comparable map scales:** paired August maps use a shared colour scale.
- **Regional weighting:** stations are averaged within a region-month and
  qualifying months are then averaged with equal weight.
- **Temporal evaluation:** prediction training precedes the held-out test
  period; random row splitting is avoided for the canonical result.
- **Single-host limitation:** Standalone and Docker layouts use separate Spark
  processes but share the same physical CPU, memory, storage, and network.

## Run tracking and reproducibility

Pipeline stages create immutable directories under `output/runs/`. Each run
records its timestamp, parameters, command, Git state, source hash, input
fingerprints, host metadata, upstream artifacts, and output hashes. Published
files under `output/figures`, `output/tables`, `output/geojson`, and
`output/interactive` are convenient report copies; the corresponding run
directory is the provenance record.

The generated Folium HTML files embed their map configuration, so files that
contain a credentialled tile URL should be treated as local artifacts.

## Related documentation

| Document | Purpose |
|---|---|
| [`project_assignment_guidelines.md`](project_assignment_guidelines.md) | Assignment requirements |
| [`SPATIOTEMPORAL_PIPELINE_RUNBOOK.md`](SPATIOTEMPORAL_PIPELINE_RUNBOOK.md) | Current preprocessing and geospatial workflow |
| [`VALIDATED_EXPERIMENT_RUNBOOK.md`](VALIDATED_EXPERIMENT_RUNBOOK.md) | Controlled temporal and systems experiment protocol |
| [`TIME_SERIES_PREDICTION_PLAN.md`](TIME_SERIES_PREDICTION_PLAN.md) | Time-series modelling design |
| [`RATIONALE.md`](RATIONALE.md) | Design decisions and interpretation |
| [`output/Architecture_Diagram_Specification.md`](output/Architecture_Diagram_Specification.md) | Architecture figure specification |
