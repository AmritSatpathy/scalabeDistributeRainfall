"""Create comparable August rainfall choropleths for selected years."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h3
import pandas as pd

import config
from run_artifacts import RunContext, publish_tree
from task1_august_geospatial_maps import (
    VALUE,
    DEFAULT_BASEMAP_CONFIG,
    add_choropleth,
    geojson,
    load_carto_key,
    map_page,
    render_pngs,
    safe_choropleth_bins,
    save_map,
)

EXPECTED_INTERVALS_PER_DAY = 24 * 60 // config.INTERVAL_MINUTES


def partial_august_from_raw(path, year, daily_completeness):
    """Build a clearly flagged estimate when the source contains a partial August."""
    raw_path = Path(path).expanduser().resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"Partial-year raw CSV does not exist: {raw_path}")
    columns = [
        "date", "station_id", "station_name", "location_longitude",
        "location_latitude", "reading_value",
    ]
    selected = []
    prefix = f"{year}-08-"
    for chunk in pd.read_csv(raw_path, usecols=columns, chunksize=500_000):
        part = chunk[chunk.date.astype(str).str.startswith(prefix)].copy()
        if not part.empty:
            selected.append(part)
    if not selected:
        raise ValueError(f"Raw CSV has no August {year} observations")
    readings = pd.concat(selected, ignore_index=True)
    readings["reading_value"] = pd.to_numeric(readings.reading_value, errors="coerce")
    readings = readings.dropna(
        subset=["station_id", "location_latitude", "location_longitude", "reading_value"]
    )
    daily = readings.groupby(
        ["station_id", "station_name", "location_latitude", "location_longitude", "date"],
        as_index=False,
    ).agg(
        observed_intervals=("reading_value", "count"),
        daily_rainfall_mm=("reading_value", "sum"),
    )
    daily = daily[
        daily.observed_intervals >= EXPECTED_INTERVALS_PER_DAY * daily_completeness
    ]
    station = daily.groupby(
        ["station_id", "station_name", "location_latitude", "location_longitude"],
        as_index=False,
    ).agg(
        eligible_days=("date", "nunique"),
        mean_daily_rainfall_mm=("daily_rainfall_mm", "mean"),
    )
    station = station.rename(
        columns={"location_latitude": "lat", "location_longitude": "lon"}
    )
    station["year"] = year
    station["month"] = 8
    station[VALUE] = station.mean_daily_rainfall_mm * 31
    station["coverage_note"] = (
        "partial August estimate from " + station.eligible_days.astype(str) + " eligible days"
    )
    return station


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monthly-summary",
        default=str(Path(config.TABLES_DIR) / "geo_station_month_summary.csv"),
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2017, 2018, 2023, 2024]
    )
    parser.add_argument("--h3-resolution", type=int, default=7)
    parser.add_argument("--daily-completeness", type=float, default=0.90)
    parser.add_argument(
        "--partial-raw-csv",
        help="Raw CSV used only for requested years missing from the monthly summary",
    )
    parser.add_argument("--basemap-config", default=str(DEFAULT_BASEMAP_CONFIG))
    parser.add_argument("--render-png", action="store_true")
    parser.add_argument("--png-width", type=int, default=1600)
    parser.add_argument("--png-height", type=int, default=1000)
    parser.add_argument("--png-wait-seconds", type=float, default=8.0)
    parser.add_argument("--hash-mode", choices=["full", "metadata"], default="metadata")
    parser.add_argument("--publish-legacy", action="store_true")
    args = parser.parse_args()

    args.years = list(dict.fromkeys(args.years))
    if not args.years:
        parser.error("Provide at least one year")
    if not 0 <= args.h3_resolution <= 15:
        parser.error("H3 resolution must be between 0 and 15")
    if not 0 < args.daily_completeness <= 1:
        parser.error("Daily completeness must lie in (0, 1]")
    if args.png_width < 800 or args.png_height < 500 or args.png_wait_seconds < 0:
        parser.error("PNG dimensions must be at least 800x500 and wait must be nonnegative")
    source = Path(args.monthly_summary).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Monthly summary does not exist: {source}")
    carto_key = load_carto_key(args.basemap_config)

    run = RunContext("task1_august_year_maps", vars(args), hash_mode=args.hash_mode)
    try:
        run.record_input("monthly_summary", str(source))
        run.record_input("basemap_config", args.basemap_config, mode="metadata")
        if args.partial_raw_csv:
            run.record_input("partial_raw_csv", args.partial_raw_csv, mode="metadata")
        tables = Path(run.artifact_path("tables")); tables.mkdir()
        geojson_dir = Path(run.artifact_path("geojson")); geojson_dir.mkdir()
        html_dir = Path(run.artifact_path("interactive")); html_dir.mkdir()
        figures_dir = Path(run.artifact_path("figures")); figures_dir.mkdir()

        monthly = pd.read_csv(source)
        required = {"year", "month", "station_id", "lat", "lon", VALUE}
        missing = required - set(monthly.columns)
        if missing:
            raise ValueError(f"Monthly summary is missing columns: {sorted(missing)}")
        august = monthly[
            monthly.year.isin(args.years) & (monthly.month == 8)
        ].dropna(subset=["lat", "lon", VALUE]).copy()
        available = set(int(value) for value in august.year.unique())
        absent = [year for year in args.years if year not in available]
        if absent:
            if not args.partial_raw_csv or len(absent) != 1:
                raise ValueError(
                    f"No coverage-qualified August data for years {absent}. "
                    "Provide --partial-raw-csv to create an explicitly labeled partial estimate."
                )
            partial = partial_august_from_raw(
                args.partial_raw_csv, absent[0], args.daily_completeness
            )
            august = pd.concat([august, partial], ignore_index=True, sort=False)
        if august.duplicated(["year", "station_id"]).any():
            raise ValueError("Duplicate station records exist within an August year")
        august["h3_cell"] = [
            h3.latlng_to_cell(float(lat), float(lon), args.h3_resolution)
            for lat, lon in zip(august.lat, august.lon)
        ]
        august.to_csv(tables / "august_station_years.csv", index=False)

        frames = {}
        shapes = {}
        for year in args.years:
            frames[year] = august[august.year == year].groupby(
                "h3_cell", as_index=False
            ).agg(
                rainfall_mm=(VALUE, "mean"),
                station_support=("station_id", "nunique"),
                observed_days=("eligible_days", "min"),
            )
            shapes[year] = geojson(frames[year], "rainfall_mm")
            frames[year].to_csv(
                tables / f"rainfall_august_{year}_h3.csv", index=False
            )
            (geojson_dir / f"rainfall_august_{year}.geojson").write_text(
                json.dumps(shapes[year], allow_nan=False), encoding="utf-8"
            )

        shared_bins = safe_choropleth_bins([shapes[year] for year in args.years])
        for year in args.years:
            minimum_days = int(frames[year].observed_days.min())
            qualifier = ""
            if minimum_days < 20:
                qualifier = f" — partial estimate ({minimum_days} observed days minimum)"
            map_object = map_page(
                f"Singapore rainfall — August {year}{qualifier}", carto_key
            )
            add_choropleth(
                map_object,
                shapes[year],
                f"Estimated rainfall — August {year}{qualifier}",
                shared_bins,
            )
            save_map(
                map_object,
                html_dir / f"geo_rainfall_{year}_08_choropleth.html",
            )

        if args.render_png:
            render_pngs(
                html_dir,
                figures_dir,
                args.png_width,
                args.png_height,
                args.png_wait_seconds,
            )
        run.write_json(
            "method",
            {
                "month": "August",
                "years": args.years,
                "measure": VALUE,
                "shared_scale": True,
                "h3_resolution": args.h3_resolution,
                "aggregation": "equal station mean within each H3 cell",
                "observed_days_minimum_by_year": {
                    str(year): int(frames[year].observed_days.min())
                    for year in args.years
                },
                "partial_month_warning": "A year with fewer than 20 eligible days is a scaled estimate and is not directly equivalent to a complete August",
            },
        )
        for name, directory in (
            ("tables", tables),
            ("geojson", geojson_dir),
            ("interactive_maps", html_dir),
            ("figures", figures_dir),
        ):
            run.record_artifact(name, str(directory), mode="full")
        if args.publish_legacy:
            for directory in (tables, geojson_dir, html_dir, figures_dir):
                publish_tree(
                    str(directory), str(Path(config.OUTPUT_DIR) / directory.name)
                )
        run.finalize()
        print(f"Completed: {run.path}")
    except Exception as exc:
        run.finalize("failed", str(exc))
        raise


if __name__ == "__main__":
    main()
