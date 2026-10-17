"""Average monthly Singapore rainfall geography across 2017–2024."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h3
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


OUTPUT_STEM = "geo_rainfall_2017_2024_average_monthly_choropleth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monthly-summary",
        default=str(Path(config.TABLES_DIR) / "geo_station_month_summary.csv"),
    )
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--minimum-station-months", type=int, default=24)
    parser.add_argument("--h3-resolution", type=int, default=7)
    parser.add_argument("--basemap-config", default=str(DEFAULT_BASEMAP_CONFIG))
    parser.add_argument("--render-png", action="store_true")
    parser.add_argument("--png-width", type=int, default=1600)
    parser.add_argument("--png-height", type=int, default=1000)
    parser.add_argument("--png-wait-seconds", type=float, default=8.0)
    parser.add_argument("--hash-mode", choices=["full", "metadata"], default="metadata")
    parser.add_argument("--publish-legacy", action="store_true")
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("Start year must not exceed end year")
    maximum_months = 12 * (args.end_year - args.start_year + 1)
    if not 1 <= args.minimum_station_months <= maximum_months:
        parser.error(
            f"Minimum station months must lie between 1 and {maximum_months}"
        )
    if not 0 <= args.h3_resolution <= 15:
        parser.error("H3 resolution must be between 0 and 15")
    if args.png_width < 800 or args.png_height < 500 or args.png_wait_seconds < 0:
        parser.error("PNG dimensions must be at least 800x500 and wait must be nonnegative")
    source = Path(args.monthly_summary).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Monthly summary does not exist: {source}")
    carto_key = load_carto_key(args.basemap_config)

    run = RunContext("task1_long_period_average", vars(args), hash_mode=args.hash_mode)
    try:
        run.record_input("monthly_summary", str(source))
        run.record_input("basemap_config", args.basemap_config, mode="metadata")
        tables = Path(run.artifact_path("tables")); tables.mkdir()
        geojson_dir = Path(run.artifact_path("geojson")); geojson_dir.mkdir()
        html_dir = Path(run.artifact_path("interactive")); html_dir.mkdir()
        figures_dir = Path(run.artifact_path("figures")); figures_dir.mkdir()

        monthly = pd.read_csv(source)
        required = {"year", "month", "station_id", "lat", "lon", VALUE}
        missing = required - set(monthly.columns)
        if missing:
            raise ValueError(f"Monthly summary is missing columns: {sorted(missing)}")
        monthly = monthly[
            monthly.year.between(args.start_year, args.end_year)
        ].dropna(subset=["station_id", "lat", "lon", VALUE]).copy()
        if monthly.empty:
            raise ValueError("No station-month rows exist in the requested period")
        if monthly.duplicated(["year", "month", "station_id"]).any():
            raise ValueError("Duplicate station-month rows exist in the requested period")
        monthly["h3_cell"] = [
            h3.latlng_to_cell(float(lat), float(lon), args.h3_resolution)
            for lat, lon in zip(monthly.lat, monthly.lon)
        ]

        station_keys = ["station_id", "lat", "lon", "h3_cell"]
        if "station_name" in monthly.columns:
            station_keys.insert(1, "station_name")
        station_means = monthly.groupby(station_keys, as_index=False).agg(
            average_monthly_rainfall_mm=(VALUE, "mean"),
            qualified_months=(VALUE, "count"),
            represented_years=("year", "nunique"),
        )
        station_means = station_means[
            station_means.qualified_months >= args.minimum_station_months
        ].copy()
        if station_means.empty:
            raise ValueError(
                "No station meets the requested minimum station-month coverage"
            )

        cells = station_means.groupby("h3_cell", as_index=False).agg(
            rainfall_mm=("average_monthly_rainfall_mm", "mean"),
            station_support=("station_id", "nunique"),
            station_months=("qualified_months", "sum"),
            minimum_station_months=("qualified_months", "min"),
        )
        shape = geojson(cells, "rainfall_mm")
        bins = safe_choropleth_bins([shape])

        monthly.to_csv(tables / "station_months_2017_2024.csv", index=False)
        station_means.to_csv(
            tables / "station_average_monthly_rainfall_2017_2024.csv", index=False
        )
        cells.to_csv(
            tables / "h3_average_monthly_rainfall_2017_2024.csv", index=False
        )
        (geojson_dir / f"{OUTPUT_STEM}.geojson").write_text(
            json.dumps(shape, allow_nan=False), encoding="utf-8"
        )

        title = (
            f"Average monthly rainfall geography — {args.start_year}–{args.end_year}"
        )
        map_object = map_page(title, carto_key)
        add_choropleth(
            map_object,
            shape,
            f"Average monthly rainfall, {args.start_year}–{args.end_year}",
            bins,
        )
        save_map(map_object, html_dir / f"{OUTPUT_STEM}.html")
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
                "period": [args.start_year, args.end_year],
                "measure": "mean coverage-normalised monthly rainfall in millimetres",
                "minimum_station_months": args.minimum_station_months,
                "stations_retained": int(station_means.station_id.nunique()),
                "h3_cells": int(cells.h3_cell.nunique()),
                "weighting": "mean over months per station, followed by equal-station mean per H3 cell",
                "spatial_claim": "sensor-supported H3 cells only; no interpolation over unobserved land",
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
                destination = (
                    Path(config.GEOSPATIAL_FIGURES_DIR)
                    if directory.name == "figures"
                    else Path(config.OUTPUT_DIR) / directory.name
                )
                publish_tree(
                    str(directory), str(destination)
                )
        run.finalize()
        print(f"Completed: {run.path}")
    except Exception as exc:
        run.finalize("failed", str(exc))
        raise


if __name__ == "__main__":
    main()
