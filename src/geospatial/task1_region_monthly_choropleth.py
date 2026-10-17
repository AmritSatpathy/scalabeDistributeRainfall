"""Region-level average monthly rainfall choropleth for Singapore, 2017–2024.

The script uses the official Master Plan 2019 region boundary GeoJSON. Each
station-month is assigned to a region from its longitude and latitude. Gauge
values are averaged within each region-month first, and those regional monthly
values are then averaged across the requested period. This equal-month design
prevents months with more reporting gauges from receiving additional weight.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import folium
import numpy as np
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


DEFAULT_BOUNDARY = (
    Path(config.DATA_DIR)
    / "singapore_map_bounary"
    / "MasterPlan2019RegionBoundaryNoSeaGEOJSON.geojson"
)
def point_in_ring(lon: float, lat: float, ring) -> bool:
    """Return whether a point lies inside or on a GeoJSON linear ring."""
    inside = False
    for first, second in zip(ring, ring[1:] + ring[:1]):
        x1, y1 = float(first[0]), float(first[1])
        x2, y2 = float(second[0]), float(second[1])
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if (
            abs(cross) <= 1e-12
            and min(x1, x2) - 1e-12 <= lon <= max(x1, x2) + 1e-12
            and min(y1, y2) - 1e-12 <= lat <= max(y1, y2) + 1e-12
        ):
            return True
        if (y1 > lat) != (y2 > lat):
            intersection_x = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < intersection_x:
                inside = not inside
    return inside


def point_in_geometry(lon: float, lat: float, geometry: dict) -> bool:
    """Point-in-polygon for GeoJSON Polygon and MultiPolygon geometries."""
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    polygons = [coordinates] if geometry_type == "Polygon" else coordinates
    if geometry_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"Unsupported boundary geometry: {geometry_type}")
    for polygon in polygons:
        if not polygon or not point_in_ring(lon, lat, polygon[0]):
            continue
        # A point belongs to the polygon only when it is outside every hole.
        if not any(point_in_ring(lon, lat, hole) for hole in polygon[1:]):
            return True
    return False


def assign_region(lon: float, lat: float, features: list[dict]) -> str | None:
    matches = [
        feature["properties"]["REGION_N"]
        for feature in features
        if point_in_geometry(lon, lat, feature["geometry"])
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Coordinate ({lat}, {lon}) intersects multiple regions: {matches}"
        )
    return matches[0] if matches else None


def colour_bins(values: pd.Series, count: int = 6) -> list[float]:
    finite = np.asarray(values.dropna(), dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite regional rainfall values are available")
    lower, upper = float(finite.min()), float(finite.max())
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 1.0)
        lower, upper = lower - padding, upper + padding
    else:
        padding = max((upper - lower) * 0.01, 0.001)
        lower, upper = lower - padding, upper + padding
    return np.linspace(lower, upper, count + 1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monthly-summary",
        default=str(Path(config.TABLES_DIR) / "geo_station_month_summary.csv"),
        help="Coverage-normalised station-month rainfall table",
    )
    parser.add_argument("--boundary", default=str(DEFAULT_BOUNDARY))
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--minimum-stations-per-region-month",
        type=int,
        default=1,
        help="Exclude region-months supported by fewer gauges than this value",
    )
    parser.add_argument("--basemap-config", default=str(DEFAULT_BASEMAP_CONFIG))
    parser.add_argument("--render-png", action="store_true")
    parser.add_argument("--png-width", type=int, default=1800)
    parser.add_argument("--png-height", type=int, default=1200)
    parser.add_argument("--png-wait-seconds", type=float, default=8.0)
    parser.add_argument("--hash-mode", choices=["full", "metadata"], default="metadata")
    parser.add_argument("--publish-legacy", action="store_true")
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")
    if args.minimum_stations_per_region_month < 1:
        parser.error("--minimum-stations-per-region-month must be positive")
    if args.png_width < 800 or args.png_height < 500 or args.png_wait_seconds < 0:
        parser.error("PNG dimensions must be at least 800x500 and wait must be nonnegative")

    monthly_path = Path(args.monthly_summary).expanduser().resolve()
    boundary_path = Path(args.boundary).expanduser().resolve()
    if not monthly_path.is_file():
        parser.error(f"Monthly summary does not exist: {monthly_path}")
    if not boundary_path.is_file():
        parser.error(f"Singapore region boundary does not exist: {boundary_path}")
    carto_key = load_carto_key(args.basemap_config)

    run = RunContext(
        "task1_region_monthly_choropleth", vars(args), hash_mode=args.hash_mode
    )
    try:
        run.record_input("monthly_summary", str(monthly_path))
        run.record_input("region_boundary", str(boundary_path))
        run.record_input("basemap_config", args.basemap_config, mode="metadata")

        tables_dir = Path(run.artifact_path("tables")); tables_dir.mkdir()
        geojson_dir = Path(run.artifact_path("geojson")); geojson_dir.mkdir()
        html_dir = Path(run.artifact_path("interactive")); html_dir.mkdir()
        figures_dir = Path(run.artifact_path("figures")); figures_dir.mkdir()

        print("[1/4] Reading monthly rainfall and official region boundaries", flush=True)
        monthly = pd.read_csv(monthly_path)
        required = {"station_id", "lat", "lon", "year", "month", VALUE}
        missing = required - set(monthly.columns)
        if missing:
            raise ValueError(f"Monthly summary is missing columns: {sorted(missing)}")
        monthly = monthly[
            monthly["year"].between(args.start_year, args.end_year)
        ].dropna(subset=["station_id", "lat", "lon", VALUE]).copy()
        if monthly.empty:
            raise ValueError("No station-month records exist in the requested period")
        if monthly.duplicated(["station_id", "year", "month"]).any():
            raise ValueError("Duplicate station-year-month rows exist in the input")

        boundary = json.loads(boundary_path.read_text(encoding="utf-8-sig"))
        features = boundary.get("features", [])
        region_names = [f.get("properties", {}).get("REGION_N") for f in features]
        if not features or any(not name for name in region_names):
            raise ValueError("Boundary must contain features with REGION_N properties")
        if len(set(region_names)) != len(region_names):
            raise ValueError("Boundary contains duplicate REGION_N values")

        print("[2/4] Assigning stations to planning regions", flush=True)
        coordinate_keys = monthly[["station_id", "lat", "lon"]].drop_duplicates()
        coordinate_keys["region"] = [
            assign_region(float(lon), float(lat), features)
            for lat, lon in zip(coordinate_keys["lat"], coordinate_keys["lon"])
        ]
        monthly = monthly.merge(
            coordinate_keys, on=["station_id", "lat", "lon"], how="left", validate="many_to_one"
        )
        unmatched = monthly[monthly["region"].isna()].copy()
        matched = monthly.dropna(subset=["region"]).copy()
        if matched.empty:
            raise ValueError("No rainfall stations fall inside the supplied region boundary")

        print("[3/4] Computing equal-month regional rainfall estimates", flush=True)
        region_month = matched.groupby(
            ["region", "year", "month"], as_index=False
        ).agg(
            average_monthly_rainfall_mm=(VALUE, "mean"),
            active_stations=("station_id", "nunique"),
            station_month_rows=("station_id", "size"),
        )
        region_month["coverage_qualified"] = (
            region_month["active_stations"]
            >= args.minimum_stations_per_region_month
        )
        qualified = region_month[region_month["coverage_qualified"]].copy()
        regional = qualified.groupby("region", as_index=False).agg(
            average_monthly_rainfall_mm=("average_monthly_rainfall_mm", "mean"),
            represented_months=("average_monthly_rainfall_mm", "size"),
            mean_active_stations=("active_stations", "mean"),
            minimum_active_stations=("active_stations", "min"),
            maximum_active_stations=("active_stations", "max"),
        )
        missing_regions = sorted(set(region_names) - set(regional["region"]))
        if missing_regions:
            raise ValueError(f"No coverage-qualified rainfall data for regions: {missing_regions}")

        lookup = regional.set_index("region").to_dict("index")
        mapped_boundary = deepcopy(boundary)
        for feature in mapped_boundary["features"]:
            name = feature["properties"]["REGION_N"]
            values = lookup[name]
            feature["properties"].update(
                {
                    "region": name.title(),
                    "average_monthly_rainfall_mm": round(
                        float(values["average_monthly_rainfall_mm"]), 2
                    ),
                    "represented_months": int(values["represented_months"]),
                    "mean_active_stations": round(float(values["mean_active_stations"]), 1),
                    "minimum_active_stations": int(values["minimum_active_stations"]),
                    "maximum_active_stations": int(values["maximum_active_stations"]),
                }
            )

        period_suffix = f"{args.start_year}_{args.end_year}"
        region_month.to_csv(
            tables_dir / f"region_monthly_rainfall_{period_suffix}.csv", index=False
        )
        regional.to_csv(
            tables_dir / f"region_average_monthly_rainfall_{period_suffix}.csv",
            index=False,
        )
        coordinate_keys.to_csv(tables_dir / "station_region_assignment.csv", index=False)
        unmatched.to_csv(tables_dir / "station_months_outside_region_boundary.csv", index=False)
        output_stem = f"geo_region_average_monthly_rainfall_{period_suffix}"
        geojson_path = geojson_dir / f"{output_stem}.geojson"
        geojson_path.write_text(
            json.dumps(mapped_boundary, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

        print("[4/4] Creating Folium choropleth and optional PNG", flush=True)
        title = (
            f"Average monthly rainfall by Singapore planning region, "
            f"{args.start_year}–{args.end_year}"
        )
        map_object = map_page(title, carto_key)
        layer = folium.Choropleth(
            geo_data=mapped_boundary,
            data=regional,
            columns=["region", "average_monthly_rainfall_mm"],
            key_on="feature.properties.REGION_N",
            bins=colour_bins(regional["average_monthly_rainfall_mm"]),
            fill_color="YlGnBu",
            fill_opacity=0.74,
            line_color="#252525",
            line_weight=1.4,
            line_opacity=0.9,
            nan_fill_color="#d9d9d9",
            nan_fill_opacity=0.65,
            legend_name="Average monthly rainfall (mm)",
            name="Regional average monthly rainfall",
            highlight=True,
        ).add_to(map_object)
        folium.GeoJsonTooltip(
            fields=[
                "region",
                "average_monthly_rainfall_mm",
                "represented_months",
                "mean_active_stations",
                "minimum_active_stations",
                "maximum_active_stations",
            ],
            aliases=[
                "Planning region",
                "Average monthly rainfall (mm)",
                "Coverage-qualified months",
                "Mean active stations/month",
                "Minimum active stations/month",
                "Maximum active stations/month",
            ],
            localize=True,
            sticky=False,
        ).add_to(layer.geojson)
        html_path = html_dir / f"{output_stem}.html"
        save_map(map_object, html_path)
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
                "official_regions": region_names,
                "region_count": len(region_names),
                "measure": "equal-month mean of within-region station monthly rainfall",
                "station_measure": VALUE,
                "minimum_stations_per_region_month": args.minimum_stations_per_region_month,
                "matched_station_month_rows": int(len(matched)),
                "unmatched_station_month_rows": int(len(unmatched)),
                "weighting": (
                    "stations are equally weighted inside each region-month; "
                    "coverage-qualified region-months are equally weighted over time"
                ),
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
