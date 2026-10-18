# Singapore Rainfall Geospatial Analysis

This branch contains only the geographic analysis of Singapore's five-minute
rainfall observations from 2017 to 2024. It uses PySpark for large-scale data
preparation, H3 for sensor-supported spatial aggregation, Folium for
interactive maps, and Singapore planning-region boundaries for regional
choropleths.

<p align="center">
  <img src="output/figures/geospatial/geo_rainfall_2017_2024_average_monthly_choropleth.png"
       alt="Average monthly rainfall across Singapore from 2017 to 2024"
       width="900">
</p>

<p align="center"><em>H3 average monthly rainfall across Singapore, 2017–2024.</em></p>

## Contents

1. [Analysis included](#analysis-included)
2. [Repository structure](#repository-structure)
3. [Requirements](#requirements)
4. [Required rainfall data](#required-rainfall-data)
5. [Complete sequential run](#complete-sequential-run)
6. [Generated outputs](#generated-outputs)
7. [Analytical conventions](#analytical-conventions)

## Analysis included

| Geographic analysis | Result |
|---|---|
| Gauge coverage | Active station maps for 2017 and 2024 |
| August comparison | H3 rainfall maps for August 2017, 2018, 2023, and 2024 |
| Annual comparison | Complete-year 2017 and 2024 maps and the 2024-minus-2017 change |
| Long-period pattern | Average monthly rainfall across 2017–2024 |
| Hotspots and anomalies | Coverage-qualified H3 comparisons across time baselines |
| Planning regions | Long-period, August 2024, and complete-2024 regional choropleths |
| Interactive inspection | Folium HTML maps with selectable layers and tooltips |

This branch intentionally excludes prediction models, model artifacts,
benchmarking, Docker/cluster experiments, non-geographic EDA, and the full
academic report.

## Repository structure

```text
scalabeDistributeRainfall/
├── data/
│   ├── raw/                        # rainfall_2017.csv ... rainfall_2024.csv
│   ├── processed/                  # generated clean Parquet
│   └── singapore_map_bounary/      # Singapore planning-region GeoJSON
├── output/
│   ├── figures/geospatial/         # publication-ready PNG maps
│   ├── geojson/                    # generated H3/region layers
│   ├── interactive/                # interactive Folium maps
│   ├── tables/                     # geographic source tables
│   └── runs/                       # generated run manifests and artifacts
├── src/
│   ├── download_data.py
│   ├── clean_data.py
│   ├── config.py
│   ├── spark_utils.py
│   ├── progress_utils.py
│   ├── run_artifacts.py
│   └── geospatial/                 # all map-generation scripts
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.9 or later
- Java 17 or 21 available through `java` or `JAVA_HOME`
- Git LFS when retrieving the large CSV files from Git
- Chromium installed through Playwright for Folium-to-PNG rendering
- At least 16 GB RAM and sufficient disk for approximately 7.8 GB of raw data,
  generated Parquet, and Spark temporary files

The Python environment includes PySpark 3.5.3, pandas, Matplotlib, H3, Folium,
Playwright, Requests, and NumPy. No PyTorch, model-training, Docker, or cluster
dependencies are required.

## Required rainfall data

> **The eight complete annual CSV files are required.** Large-file submission
> limits may mean that `data/raw/` is empty or contains only small Git LFS
> pointers. Before running the analysis, provide `rainfall_2017.csv` through
> `rainfall_2024.csv`. Every cleaning and map stage uses these canonical files.

Source: [data.gov.sg — Historical Rainfall across Singapore](https://data.gov.sg/collections/2279/view),
collection 2279, published by Singapore's National Environment Agency.

Expected columns:

```text
date, timestamp, update_timestamp, station_id, station_name,
station_device_id, location_longitude, location_latitude,
reading_update_timestamp, reading_value, reading_type, reading_unit
```

Do not create a second copy under a `HistoricalRainfall...` filename.

## Complete sequential run

Run the following blocks in order from one terminal session.

### 1. Clone the geospatial branch and install dependencies

```bash
git clone --branch geospatial --single-branch \
  https://github.com/AmritSatpathy/scalabeDistributeRainfall.git
cd scalabeDistributeRainfall

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m playwright install chromium

mkdir -p .cache/matplotlib
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
```

### 2. Acquire and verify the annual rainfall files

Use one acquisition method:

```bash
# Option A: materialise the Git LFS objects
git lfs install
git lfs pull --include="data/raw/rainfall_*.csv"

# Option B: download directly from data.gov.sg
./.venv/bin/python src/download_data.py \
  --years 2017 2018 2019 2020 2021 2022 2023 2024 \
  --force
```

Then validate the inputs:

```bash
for year in 2017 2018 2019 2020 2021 2022 2023 2024; do
  file="data/raw/rainfall_${year}.csv"
  if [ ! -f "$file" ] || [ "$(wc -c < "$file")" -lt 1000000 ]; then
    echo "ERROR: $file is missing or is only a Git LFS pointer"
    exit 1
  fi
done
echo "All eight rainfall CSVs are present."
```

### 3. Clean and standardise the observations

```bash
./.venv/bin/python src/clean_data.py \
  --years 2017 2018 2019 2020 2021 2022 2023 2024 \
  --max-5min-mm 60 \
  --grid-tolerance-seconds 60 \
  --hash-mode full \
  --publish-legacy
```

This produces the partitioned `data/processed/rainfall_clean` dataset and
audits timestamp alignment, invalid readings, duplicates, cadence gaps, station
coordinates, and device history.

### 4. Build the common station-month table and H3 comparisons

```bash
./.venv/bin/python src/geospatial/task1_geospatial_choropleths.py \
  --daily-completeness 0.90 \
  --minimum-eligible-days 20 \
  --h3-resolution 7 \
  --publish-legacy
```

The published `output/tables/geo_station_month_summary.csv` is reused by all
remaining scripts, avoiding repeated Spark aggregation.

### 5. Render gauge, August, annual, and change maps

```bash
./.venv/bin/python src/geospatial/task1_august_geospatial_maps.py \
  --monthly-summary output/tables/geo_station_month_summary.csv \
  --render-png \
  --publish-legacy

./.venv/bin/python src/geospatial/task1_august_year_maps.py \
  --monthly-summary output/tables/geo_station_month_summary.csv \
  --years 2017 2018 2023 2024 \
  --render-png \
  --publish-legacy
```

### 6. Render long-period and planning-region maps

```bash
./.venv/bin/python src/geospatial/task1_long_period_average_map.py \
  --monthly-summary output/tables/geo_station_month_summary.csv \
  --start-year 2017 \
  --end-year 2024 \
  --render-png \
  --publish-legacy

./.venv/bin/python src/geospatial/task1_region_monthly_choropleth.py \
  --monthly-summary output/tables/geo_station_month_summary.csv \
  --start-year 2017 \
  --end-year 2024 \
  --render-png \
  --publish-legacy

./.venv/bin/python src/geospatial/task1_region_2024_choropleths.py \
  --monthly-summary output/tables/geo_station_month_summary.csv \
  --year 2024 \
  --focus-month 8 \
  --render-png \
  --publish-legacy
```

The planning-region scripts use
`data/singapore_map_bounary/MasterPlan2019RegionBoundaryNoSeaGEOJSON.geojson`.
The directory name retains the source repository's existing spelling.

### 7. Inspect the tracked runs

```bash
./.venv/bin/python src/run_artifacts.py --latest clean
./.venv/bin/python src/run_artifacts.py --latest task1_geo_choropleths
./.venv/bin/python src/run_artifacts.py --latest task1_august_geo
./.venv/bin/python src/run_artifacts.py --latest task1_long_period_average
./.venv/bin/python src/run_artifacts.py --latest task1_region_monthly_choropleth
./.venv/bin/python src/run_artifacts.py --latest task1_region_period_choropleths
```

### Optional CARTO basemap

OpenStreetMap works without credentials. To enable CARTO raster layers, create
an untracked `basemaps.local.json` in the repository root:

```json
{
  "carto": {
    "api_key": "YOUR_CARTO_BASEMAP_KEY"
  }
}
```

Keep attribution visible and never commit the key. The generated HTML contains
the selected tile URL and should be treated as a local artifact when a key is
used.

## Generated outputs

| Artifact | Location |
|---|---|
| Clean observations | `data/processed/rainfall_clean/` |
| Geographic PNG maps | `output/figures/geospatial/` |
| Interactive Folium maps | `output/interactive/` |
| H3 and region layers | `output/geojson/` |
| Coverage and map source tables | `output/tables/` |
| Immutable run records | `output/runs/<timestamp>_<stage>_<hash>/` |

Each run manifest records the command, parameters, Git state, input
fingerprints, source hashes, upstream artifacts, and output hashes. Published
files under `output/figures/geospatial`, `output/interactive`,
`output/geojson`, and `output/tables` are stable convenience copies.

## Analytical conventions

- **Coverage-qualified rainfall:** a station-month must contain enough eligible
  days before it contributes to a comparison.
- **Sensor-supported geography:** H3 cells appear only where gauges provide
  observations; the maps do not interpolate rainfall over ungauged land.
- **Comparable scales:** paired August and annual maps share colour bins.
- **Matched change estimates:** temporal change maps use stations represented
  in both comparison periods.
- **Regional aggregation:** stations are averaged within a region-month and
  qualifying months then receive equal weight.
- **Local time:** timestamp processing is fixed to `Asia/Singapore`.

The individual map scripts are summarised in
[`src/geospatial/README.md`](src/geospatial/README.md).
