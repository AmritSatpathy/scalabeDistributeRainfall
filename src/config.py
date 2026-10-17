"""
Central configuration for the Singapore Historical Rainfall project.

All paths are resolved relative to the project root so the scripts work
regardless of the current working directory.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")            # downloaded yearly CSVs
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")  # cleaned Parquet

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
GEOSPATIAL_FIGURES_DIR = os.path.join(FIGURES_DIR, "geospatial")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")

# The cleaned dataset (Parquet, partitioned by year)
CLEAN_PARQUET = os.path.join(PROCESSED_DIR, "rainfall_clean")

# ---------------------------------------------------------------------------
# data.gov.sg dataset ids for the "Historical Rainfall across Singapore"
# collection (collection id 2279). We only use 2017-2024 per the assignment.
# ---------------------------------------------------------------------------
DATASET_IDS = {
    2017: "d_1990a5a1aeaf3dd243cf4dae294a61c4",
    2018: "d_024fb501ce7092b71bb713eaf54fa7eb",
    2019: "d_61995f092320e7155b7528050880b502",
    2020: "d_9e7de44094f876f6804b8b5bcee45c81",
    2021: "d_3b41598f74f1f11fc3430348fea51af5",
    2022: "d_42d64cc6c176ace1c52fbb40b9ede302",
    2023: "d_f864cc30d58b467db83659ad17c737bf",
    2024: "d_a0b69d3e02576a1fd0ab673e71f83507",
}

YEARS = sorted(DATASET_IDS.keys())

POLL_DOWNLOAD_URL = (
    "https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download"
)

# ---------------------------------------------------------------------------
# Reading semantics
# ---------------------------------------------------------------------------
# Rainfall is reported in mm accumulated over each 5-minute interval.
INTERVAL_MINUTES = 5
# Prediction horizon for Task 2 (minutes).
HORIZON_MINUTES = 30
# Number of 5-minute steps in the horizon.
HORIZON_STEPS = HORIZON_MINUTES // INTERVAL_MINUTES  # 6


def ensure_dirs() -> None:
    """Create all output directories if they do not exist."""
    for d in (
        RAW_DIR,
        PROCESSED_DIR,
        FIGURES_DIR,
        GEOSPATIAL_FIGURES_DIR,
        TABLES_DIR,
        RUNS_DIR,
        MODELS_DIR,
        REPORT_DIR,
    ):
        os.makedirs(d, exist_ok=True)
