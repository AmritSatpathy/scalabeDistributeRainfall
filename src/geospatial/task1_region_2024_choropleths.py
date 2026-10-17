"""Official-region rainfall choropleths for August and full-year 2024.

Two separate maps are produced on one shared colour scale:

* August 2024 coverage-normalised monthly rainfall; and
* mean monthly rainfall across the coverage-qualified months of 2024.

Stations are assigned to the five Master Plan 2019 regions from their recorded
longitude and latitude. Stations are equally weighted inside each region-month;
months are equally weighted in the complete-year monthly mean.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import folium
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from run_artifacts import RunContext, publish_tree
from task1_august_geospatial_maps import (
    DEFAULT_BASEMAP_CONFIG,
    VALUE,
    load_carto_key,
    map_page,
    render_pngs,
    save_map,
)
from task1_region_monthly_choropleth import assign_region, colour_bins


DEFAULT_BOUNDARY = (
    Path(config.DATA_DIR)
    / "singapore_map_bounary"
    / "MasterPlan2019RegionBoundaryNoSeaGEOJSON.geojson"
)


def enrich_boundary(boundary: dict, regional: pd.DataFrame, period: str) -> dict:
    """Attach rainfall and coverage fields to a copy of the region boundary."""
    lookup = regional.set_index("region").to_dict("index")
    mapped = deepcopy(boundary)
    for feature in mapped["features"]:
        region_name = feature["properties"]["REGION_N"]
        values = lookup[region_name]
        feature["properties"].update(
            {
                "region": region_name.title(),
                "period": period,
                "average_monthly_rainfall_mm": round(
                    float(values["average_monthly_rainfall_mm"]), 2
                ),
                "represented_months": int(values["represented_months"]),
                "mean_active_stations": round(float(values["mean_active_stations"]), 1),
                "minimum_active_stations": int(values["minimum_active_stations"]),
                "maximum_active_stations": int(values["maximum_active_stations"]),
            }
        )
    return mapped


def add_region_choropleth(
    map_object,
    mapped_boundary: dict,
    regional: pd.DataFrame,
    title: str,
    bins: list[float],
):
    layer = folium.Choropleth(
        geo_data=mapped_boundary,
        data=regional,
        columns=["region", "average_monthly_rainfall_mm"],
        key_on="feature.properties.REGION_N",
        bins=bins,
        fill_color="YlGnBu",
        fill_opacity=0.76,
        line_color="#252525",
        line_weight=1.5,
        line_opacity=0.9,
        nan_fill_color="#d9d9d9",
        nan_fill_opacity=0.65,
        legend_name="Average monthly rainfall (mm)",
        name=title,
        highlight=True,
    ).add_to(map_object)
    folium.GeoJsonTooltip(
        fields=[
            "region",
            "period",
            "average_monthly_rainfall_mm",
            "represented_months",
            "mean_active_stations",
            "minimum_active_stations",
            "maximum_active_stations",
        ],
        aliases=[
            "Planning region",
            "Period",
            "Average monthly rainfall (mm)",
            "Coverage-qualified months",
            "Mean active stations/month",
            "Minimum active stations/month",
            "Maximum active stations/month",
        ],
        localize=True,
        sticky=False,
    ).add_to(layer.geojson)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monthly-summary",
        default=str(Path(config.TABLES_DIR) / "geo_station_month_summary.csv"),
        help="Coverage-normalised station-month rainfall table",
    )
    parser.add_argument("--boundary", default=str(DEFAULT_BOUNDARY))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--focus-month", type=int, default=8)
    parser.add_argument(
        "--minimum-stations-per-region-month",
        type=int,
        default=1,
        help="Exclude region-months supported by fewer gauges than this value",
    )
    parser.add_argument(
        "--minimum-region-months",
        type=int,
        default=12,
        help="Minimum qualified months required for each full-year region estimate",
    )
    parser.add_argument("--basemap-config", default=str(DEFAULT_BASEMAP_CONFIG))
    parser.add_argument("--render-png", action="store_true")
    parser.add_argument("--png-width", type=int, default=1800)
    parser.add_argument("--png-height", type=int, default=1200)
    parser.add_argument("--png-wait-seconds", type=float, default=8.0)
    parser.add_argument("--hash-mode", choices=["full", "metadata"], default="metadata")
    parser.add_argument("--publish-legacy", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.focus_month <= 12:
        parser.error("--focus-month must lie between 1 and 12")
    if args.minimum_stations_per_region_month < 1:
        parser.error("--minimum-stations-per-region-month must be positive")
    if not 1 <= args.minimum_region_months <= 12:
        parser.error("--minimum-region-months must lie between 1 and 12")
    if args.png_width < 800 or args.png_height < 500 or args.png_wait_seconds < 0:
        parser.error("PNG dimensions must be at least 800x500 and wait must be nonnegative")

    monthly_path = Path(args.monthly_summary).expanduser().resolve()
    boundary_path = Path(args.boundary).expanduser().resolve()
    if not monthly_path.is_file():
        parser.error(f"Monthly summary does not exist: {monthly_path}")
    if not boundary_path.is_file():
        parser.error(f"Singapore region boundary does not exist: {boundary_path}")
    carto_key = load_carto_key(args.basemap_config)

    run = RunContext("task1_region_period_choropleths", vars(args), hash_mode=args.hash_mode)
    try:
        run.record_input("monthly_summary", str(monthly_path))
        run.record_input("region_boundary", str(boundary_path))
        run.record_input("basemap_config", args.basemap_config, mode="metadata")

        tables_dir = Path(run.artifact_path("tables")); tables_dir.mkdir()
        geojson_dir = Path(run.artifact_path("geojson")); geojson_dir.mkdir()
        html_dir = Path(run.artifact_path("interactive")); html_dir.mkdir()
        figures_dir = Path(run.artifact_path("figures")); figures_dir.mkdir()

        print("[1/4] Reading 2024 monthly rainfall and region boundaries", flush=True)
        monthly = pd.read_csv(monthly_path)
        required = {"station_id", "lat", "lon", "year", "month", VALUE}
        missing = required - set(monthly.columns)
        if missing:
            raise ValueError(f"Monthly summary is missing columns: {sorted(missing)}")
        monthly = monthly[monthly["year"] == args.year].dropna(
            subset=["station_id", "lat", "lon", VALUE]
        ).copy()
        if monthly.empty:
            raise ValueError(f"No station-month records exist for {args.year}")
        if monthly.duplicated(["station_id", "year", "month"]).any():
            raise ValueError("Duplicate station-year-month rows exist in the input")

        boundary = json.loads(boundary_path.read_text(encoding="utf-8-sig"))
        features = boundary.get("features", [])
        region_names = [f.get("properties", {}).get("REGION_N") for f in features]
        if not features or any(not name for name in region_names):
            raise ValueError("Boundary must contain features with REGION_N properties")
        if len(set(region_names)) != len(region_names):
            raise ValueError("Boundary contains duplicate REGION_N values")

        print("[2/4] Assigning stations to official planning regions", flush=True)
        station_coordinates = monthly[["station_id", "lat", "lon"]].drop_duplicates()
        station_coordinates["region"] = [
            assign_region(float(lon), float(lat), features)
            for lat, lon in zip(station_coordinates["lat"], station_coordinates["lon"])
        ]
        monthly = monthly.merge(
            station_coordinates,
            on=["station_id", "lat", "lon"],
            how="left",
            validate="many_to_one",
        )
        unmatched = monthly[monthly["region"].isna()].copy()
        matched = monthly.dropna(subset=["region"]).copy()
        if matched.empty:
            raise ValueError("No stations fall inside the supplied region boundary")

        print("[3/4] Building August and complete-year region summaries", flush=True)
        region_month = matched.groupby(["region", "month"], as_index=False).agg(
            average_monthly_rainfall_mm=(VALUE, "mean"),
            active_stations=("station_id", "nunique"),
        )
        region_month["coverage_qualified"] = (
            region_month["active_stations"] >= args.minimum_stations_per_region_month
        )
        qualified = region_month[region_month["coverage_qualified"]].copy()

        august = qualified[qualified["month"] == args.focus_month].copy()
        august = august.rename(columns={"active_stations": "mean_active_stations"})
        august["represented_months"] = 1
        august["minimum_active_stations"] = august["mean_active_stations"]
        august["maximum_active_stations"] = august["mean_active_stations"]
        august = august[
            [
                "region",
                "average_monthly_rainfall_mm",
                "represented_months",
                "mean_active_stations",
                "minimum_active_stations",
                "maximum_active_stations",
            ]
        ]

        annual_mean = qualified.groupby("region", as_index=False).agg(
            average_monthly_rainfall_mm=("average_monthly_rainfall_mm", "mean"),
            represented_months=("month", "nunique"),
            mean_active_stations=("active_stations", "mean"),
            minimum_active_stations=("active_stations", "min"),
            maximum_active_stations=("active_stations", "max"),
        )
        annual_mean = annual_mean[
            annual_mean["represented_months"] >= args.minimum_region_months
        ].copy()
        for label, frame in (("focus month", august), ("complete year", annual_mean)):
            missing_regions = sorted(set(region_names) - set(frame["region"]))
            if missing_regions:
                raise ValueError(f"No coverage-qualified {label} data for: {missing_regions}")

        month_name = pd.Timestamp(args.year, args.focus_month, 1).strftime("%B")
        august_boundary = enrich_boundary(
            boundary, august, f"{month_name} {args.year}"
        )
        annual_boundary = enrich_boundary(
            boundary, annual_mean, f"Mean month across {args.year}"
        )
        shared_bins = colour_bins(
            pd.concat(
                [
                    august["average_monthly_rainfall_mm"],
                    annual_mean["average_monthly_rainfall_mm"],
                ],
                ignore_index=True,
            )
        )

        period_suffix = f"{args.year}_{args.focus_month:02d}"
        august_stem = f"geo_region_rainfall_{period_suffix}"
        annual_stem = f"geo_region_average_monthly_rainfall_{args.year}"
        region_month.to_csv(
            tables_dir / f"region_monthly_rainfall_{args.year}.csv", index=False
        )
        august.to_csv(tables_dir / f"region_rainfall_{period_suffix}.csv", index=False)
        annual_mean.to_csv(
            tables_dir / f"region_average_monthly_rainfall_{args.year}.csv", index=False
        )
        station_coordinates.to_csv(
            tables_dir / f"station_region_assignment_{args.year}.csv", index=False
        )
        unmatched.to_csv(
            tables_dir / f"station_months_outside_region_boundary_{args.year}.csv",
            index=False,
        )
        (geojson_dir / f"{august_stem}.geojson").write_text(
            json.dumps(august_boundary, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        (geojson_dir / f"{annual_stem}.geojson").write_text(
            json.dumps(annual_boundary, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

        print("[4/4] Creating two Folium maps and optional PNG files", flush=True)
        maps = (
            (
                august,
                august_boundary,
                f"Regional rainfall — {month_name} {args.year}",
                august_stem,
            ),
            (
                annual_mean,
                annual_boundary,
                f"Average monthly rainfall by region — {args.year}",
                annual_stem,
            ),
        )
        for frame, mapped_boundary, title, stem in maps:
            map_object = map_page(title, carto_key)
            add_region_choropleth(
                map_object, mapped_boundary, frame, title, shared_bins
            )
            save_map(map_object, html_dir / f"{stem}.html")
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
                "year": args.year,
                "focus_month": args.focus_month,
                "official_regions": region_names,
                "focus_month_measure": (
                    "equal-station mean of coverage-normalised monthly rainfall"
                ),
                "complete_year_measure": (
                    "equal-month mean of within-region station monthly rainfall"
                ),
                "minimum_stations_per_region_month": (
                    args.minimum_stations_per_region_month
                ),
                "minimum_region_months": args.minimum_region_months,
                "shared_colour_scale_mm": shared_bins,
                "matched_station_month_rows": int(len(matched)),
                "unmatched_station_month_rows": int(len(unmatched)),
                "boundary": str(boundary_path),
            },
        )
        for label, directory in (
            ("tables", tables_dir),
            ("geojson", geojson_dir),
            ("interactive_maps", html_dir),
            ("figures", figures_dir),
        ):
            run.record_artifact(label, str(directory), mode="full")
        if args.publish_legacy:
            for directory in (tables_dir, geojson_dir, html_dir, figures_dir):
                destination = (
                    Path(config.GEOSPATIAL_FIGURES_DIR)
                    if directory.name == "figures"
                    else Path(config.OUTPUT_DIR) / directory.name
                )
                publish_tree(str(directory), str(destination))
        run.finalize()
        print(f"Completed: {run.path}")
    except Exception as exc:
        run.finalize("failed", str(exc))
        raise


if __name__ == "__main__":
    main()
