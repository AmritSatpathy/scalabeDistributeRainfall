"""2017/2024 rainfall change: station summaries -> H3 GeoJSON -> Folium.

HTML embeds its GeoJSON and uses credentialled CARTO raster basemaps. Optional
Playwright rendering exports the completed interactive maps as high-resolution
PNG files after map tiles and rainfall polygons have loaded.
"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import quote

import folium
import h3
import numpy as np
import pandas as pd
from branca.element import Element

import config
from run_artifacts import RunContext, latest_artifact, publish_tree

VALUE = "normalised_monthly_rainfall_mm"
BOUNDS = [[1.20, 103.59], [1.49, 104.11]]
DEFAULT_BASEMAP_CONFIG = Path(config.PROJECT_ROOT) / "basemaps.local.json"


def load_carto_key(path):
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Basemap configuration not found: {config_path}. "
            "Create JSON containing carto.api_key."
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    key = str(payload.get("carto", {}).get("api_key", "")).strip()
    if not key:
        raise ValueError(f"Missing carto.api_key in {config_path}")
    return quote(key, safe="_-")


def geojson(frame, value):
    features = []
    for row in frame.to_dict("records"):
        number = row[value]
        if pd.isna(number) or not np.isfinite(number):
            continue
        ring = [[float(lon), float(lat)] for lat, lon in h3.cell_to_boundary(row["h3_cell"])]
        ring.append(ring[0])
        properties = {"h3_cell": row["h3_cell"], "rainfall_mm": round(float(number), 3)}
        if "station_support" in row:
            properties["station_support"] = int(row["station_support"])
        if "observed_days" in row:
            properties["observed_days"] = int(row["observed_days"])
        features.append({"type": "Feature", "properties": properties,
                         "geometry": {"type": "Polygon", "coordinates": [ring]}})
    if not features:
        raise ValueError(f"No finite rainfall values for {value}")
    return {"type": "FeatureCollection", "features": features}


def safe_choropleth_bins(collections, *, symmetric=False, count=8):
    """Return strictly increasing bins that contain the serialized GeoJSON values.

    Folium validates the values against the supplied edges. The GeoJSON values
    are rounded for readable tooltips, so the edges must be calculated from
    those serialized values and padded against floating-point boundary errors.
    """
    values = np.asarray(
        [
            float(feature["properties"]["rainfall_mm"])
            for collection in collections
            for feature in collection["features"]
        ],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot create choropleth bins without finite values")
    if symmetric:
        bound = max(float(np.abs(values).max()), 1.0)
        padding = max(bound * 0.005, 0.001)
        return np.linspace(-(bound + padding), bound + padding, count + 1).tolist()
    lower = min(0.0, float(values.min()))
    upper = float(values.max())
    span = max(upper - lower, 1.0)
    upper += max(span * 0.005, 0.001)
    if lower < 0:
        lower -= max(span * 0.005, 0.001)
    return np.linspace(lower, upper, count + 1).tolist()


def map_page(title, carto_key):
    m = folium.Map(location=[1.35, 103.82], tiles=None, zoom_start=11,
                   width="100%", height="100%", control_scale=True)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="© OpenStreetMap contributors", name="OpenStreetMap streets",
        subdomains="abc", max_zoom=19, show=False).add_to(m)
    folium.TileLayer(
        tiles=("https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
               f"?key={carto_key}"),
        attr="© OpenStreetMap contributors © CARTO", name="CARTO Light",
        max_zoom=19, show=True).add_to(m)
    folium.TileLayer(
        tiles=("https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
               f"?key={carto_key}"),
        attr="© OpenStreetMap contributors © CARTO", name="CARTO Voyager",
        max_zoom=19, show=False).add_to(m)
    m.fit_bounds(BOUNDS)
    m.get_root().header.add_child(Element(
        "<style>html,body{height:100%;margin:0}.folium-map{min-height:650px}"
        ".map-title{position:fixed;top:12px;left:55px;z-index:999;"
        "background:white;padding:10px;max-width:55%;font:16px Georgia}</style>"))
    m.get_root().html.add_child(Element(f'<div class="map-title">{title}</div>'))
    # Explicit messages make failed browser dependencies distinguishable from no data.
    m.get_root().html.add_child(Element(
        '<div id="map-status" style="position:fixed;bottom:28px;left:12px;z-index:9999;'
        'background:white;padding:7px;font:12px Arial">Loading map. If this persists, '
        'check internet access or browser restrictions on JavaScript libraries.</div>'))
    return m


def add_choropleth(m, data, title, bins, difference=False, show=True):
    values = pd.DataFrame([f["properties"] for f in data["features"]])
    layer = folium.Choropleth(
        geo_data=data, data=values, columns=["h3_cell", "rainfall_mm"],
        key_on="feature.properties.h3_cell", name=title, bins=bins,
        fill_color="BrBG" if difference else "YlGnBu", fill_opacity=0.68,
        line_opacity=0.85, line_weight=0.7, line_color="white",
        legend_name=title + " (mm)", show=show, highlight=True,
        nan_fill_opacity=0).add_to(m)
    fields = ["h3_cell", "rainfall_mm"]
    aliases = ["H3 cell", "Change (mm)" if difference else "Estimated monthly rainfall (mm)"]
    if "station_support" in values:
        fields.append("station_support")
        aliases.append("Contributing stations")
    if "observed_days" in values:
        fields.append("observed_days")
        aliases.append("Minimum eligible days in cell")
    folium.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True).add_to(layer.geojson)
    return layer


def add_station_layer(m, frame, title, color, show=True):
    layer = folium.FeatureGroup(name=title, show=show)
    for row in frame.itertuples(index=False):
        station_name = getattr(row, "station_name", "")
        active_months = int(getattr(row, "active_months", 1))
        tooltip = (
            f"Station: {row.station_id}<br>Name: {station_name}<br>"
            f"Coverage-qualified months: {active_months}"
        )
        folium.CircleMarker(
            location=[float(row.lat), float(row.lon)], radius=7,
            color="white", weight=1.4, fill=True, fill_color=color,
            fill_opacity=0.95, tooltip=tooltip,
        ).add_to(layer)
    layer.add_to(m)
    return layer


def save_map(m, path):
    folium.LayerControl(collapsed=False).add_to(m)
    name = m.get_name()
    m.get_root().script.add_child(Element(f"""
    window.addEventListener('load', function() {{
    if (typeof L === 'undefined' || typeof {name} === 'undefined') return;
    document.getElementById('map-status').textContent =
      'Background: OpenStreetMap / CARTO. Use the layer control to change map type.';
    {name}.eachLayer(function(layer) {{
      if (layer instanceof L.TileLayer) {{
        layer.on('tileerror', function() {{
          document.getElementById('map-status').textContent =
            'Background tiles could not load. Select the other map type. Rainfall polygons remain available.';
        }});
      }}
    }});
    setTimeout(function() {{ {name}.invalidateSize(); }}, 200);
    }});
    """))
    m.save(str(path))
    print(f"Saved {path}", flush=True)


class QuietRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return


def render_pngs(html_dir, figures_dir, width, height, wait_seconds):
    """Serve the HTML locally and capture each completed Folium map."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "PNG rendering requires Playwright. Install requirements and run "
            "`python -m playwright install chromium`."
        ) from exc
    figures_dir.mkdir(parents=True, exist_ok=True)
    handler = partial(QuietRequestHandler, directory=str(html_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": width, "height": height}, device_scale_factor=2
            )
            page = context.new_page()
            for html_path in sorted(html_dir.glob("*.html")):
                url = f"http://127.0.0.1:{port}/{html_path.name}"
                print(f"Rendering PNG from {html_path.name}", flush=True)
                page.goto(url, wait_until="domcontentloaded", timeout=120_000)
                page.wait_for_selector(".leaflet-container", state="visible", timeout=30_000)
                page.wait_for_function(
                    "document.querySelectorAll('.leaflet-tile-loaded').length >= 4",
                    timeout=90_000,
                )
                page.wait_for_function(
                    "document.querySelectorAll('.leaflet-overlay-pane path').length >= 1",
                    timeout=30_000,
                )
                page.wait_for_timeout(int(wait_seconds * 1000))
                page.evaluate(
                    "document.getElementById('map-status').style.display = 'none'"
                )
                output = figures_dir / f"{html_path.stem}.png"
                page.locator(".folium-map").screenshot(path=str(output))
                print(f"Saved {output}", flush=True)
            context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def clean_published_geo_maps(output_root, keep_png, keep_html):
    """Remove superseded published map files after replacements exist."""
    targets = [
        (Path(output_root) / "figures", "geo*.png", set(keep_png)),
        (Path(output_root) / "interactive", "geo*.html", set(keep_html)),
    ]
    for directory, pattern, keep in targets:
        if not directory.exists():
            continue
        for path in directory.glob(pattern):
            if path.name not in keep:
                path.unlink()
                print(f"Removed superseded map: {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="Clean Parquet; used only without --monthly-summary")
    ap.add_argument("--monthly-summary", help="Reuse a complete station-month summary CSV; avoids Spark")
    ap.add_argument("--daily-completeness", type=float, default=0.90)
    ap.add_argument("--minimum-eligible-days", type=int, default=20)
    ap.add_argument("--minimum-eligible-months", type=int, default=12)
    ap.add_argument("--h3-resolution", type=int, default=7)
    ap.add_argument("--basemap-config", default=str(DEFAULT_BASEMAP_CONFIG))
    ap.add_argument("--render-png", action="store_true")
    ap.add_argument("--png-width", type=int, default=1600)
    ap.add_argument("--png-height", type=int, default=1000)
    ap.add_argument("--png-wait-seconds", type=float, default=5.0)
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="metadata")
    ap.add_argument("--publish-legacy", action="store_true")
    ap.add_argument("--clean-published-geo", action="store_true")
    args = ap.parse_args()
    if not 0 < args.daily_completeness <= 1 or not 1 <= args.minimum_eligible_days <= 31:
        ap.error("Invalid coverage settings")
    if not 0 <= args.h3_resolution <= 15:
        ap.error("H3 resolution must be between 0 and 15")
    if not 1 <= args.minimum_eligible_months <= 12:
        ap.error("Minimum eligible months must be between 1 and 12")
    if args.png_width < 800 or args.png_height < 500 or args.png_wait_seconds < 0:
        ap.error("PNG dimensions must be at least 800x500 and wait must be nonnegative")
    if args.clean_published_geo and not (args.publish_legacy and args.render_png):
        ap.error("--clean-published-geo requires --publish-legacy and --render-png")
    carto_key = load_carto_key(args.basemap_config)
    source = args.monthly_summary or args.data or latest_artifact("clean", "clean_parquet") or config.CLEAN_PARQUET
    if not Path(source).exists():
        ap.error(f"Missing input: {source}")
    run = RunContext("task1_geo_change", vars(args), hash_mode=args.hash_mode)
    try:
        run.record_input("monthly_summary" if args.monthly_summary else "clean_parquet", source)
        run.record_input("basemap_config", args.basemap_config, mode="metadata")
        if args.monthly_summary:
            print("[1/3] Reusing complete station-month summaries; no Spark job", flush=True)
            monthly = pd.read_csv(source)
        else:
            from spark_utils import get_spark
            from pyspark.sql import functions as F
            from progress_utils import SparkProgressMonitor
            from task1_geospatial_choropleths import station_month_summary
            spark = get_spark("rainfall_change_geojson", shuffle_partitions=64)
            try:
                observations = spark.read.parquet(source).filter(
                    F.col("year").isin(2017, 2024))
                with SparkProgressMonitor(spark, "Aggregate all months for 2017 and 2024"):
                    monthly = station_month_summary(observations, args.daily_completeness,
                                                    args.minimum_eligible_days)
            finally:
                spark.stop()
        required = {"year", "month", "station_id", "lat", "lon", VALUE}
        if not required.issubset(monthly.columns):
            raise ValueError(f"Missing fields: {required - set(monthly.columns)}")
        monthly = monthly[monthly["year"].isin([2017, 2024])].copy()
        monthly = monthly.dropna(subset=["lat", "lon", VALUE])
        if monthly.duplicated(["year", "month", "station_id"]).any():
            raise ValueError("Duplicate station-month rows; resolve coordinates before comparison")
        monthly["h3_cell"] = [h3.latlng_to_cell(float(a), float(b), args.h3_resolution)
                              for a, b in zip(monthly.lat, monthly.lon)]
        tables = Path(run.artifact_path("tables")); tables.mkdir()
        gj_dir = Path(run.artifact_path("geojson")); gj_dir.mkdir()
        html = Path(run.artifact_path("interactive")); html.mkdir()
        figures = Path(run.artifact_path("figures")); figures.mkdir()
        monthly.to_csv(tables / "geo_station_months_2017_2024.csv", index=False)
        print("[2/3] Building August and complete-year matched changes", flush=True)

        station_columns = ["year", "station_id", "lat", "lon", "h3_cell"]
        if "station_name" in monthly.columns:
            station_columns.insert(2, "station_name")
        active = monthly.groupby(station_columns, as_index=False).agg(
            active_months=("month", "nunique")
        )

        august = monthly[monthly.month == 8]
        august_paired = august[august.year == 2017].merge(
            august[august.year == 2024], on=["station_id", "h3_cell"],
            suffixes=("_2017", "_2024"),
        )
        if august_paired.empty:
            raise ValueError("No common station/H3 pairs for the August change map")
        august_paired["difference_mm"] = (
            august_paired[VALUE + "_2024"] - august_paired[VALUE + "_2017"]
        )
        august_delta = august_paired.groupby("h3_cell", as_index=False).agg(
            rainfall_mm=("difference_mm", "mean"),
            station_support=("station_id", "nunique"),
        )
        august_json = geojson(august_delta, "rainfall_mm")

        annual = monthly.groupby(station_columns, as_index=False).agg(
            annual_rainfall_mm=(VALUE, "sum"),
            eligible_months=("month", "nunique"),
        )
        annual = annual[
            annual.eligible_months >= args.minimum_eligible_months
        ].copy()
        annual_paired = annual[annual.year == 2017].merge(
            annual[annual.year == 2024], on=["station_id", "h3_cell"],
            suffixes=("_2017", "_2024"),
        )
        if annual_paired.empty:
            raise ValueError(
                "No common station/H3 pairs satisfy the complete-year coverage rule"
            )
        annual_paired["difference_mm"] = (
            annual_paired["annual_rainfall_mm_2024"]
            - annual_paired["annual_rainfall_mm_2017"]
        )
        annual_frames = {}
        annual_shapes = {}
        for year in (2017, 2024):
            annual_frames[year] = annual[annual.year == year].groupby(
                "h3_cell", as_index=False
            ).agg(
                rainfall_mm=("annual_rainfall_mm", "mean"),
                station_support=("station_id", "nunique"),
            )
            annual_shapes[year] = geojson(annual_frames[year], "rainfall_mm")
            (gj_dir / f"rainfall_complete_{year}.geojson").write_text(
                json.dumps(annual_shapes[year], allow_nan=False)
            )
            annual_frames[year].to_csv(
                tables / f"rainfall_complete_{year}_h3.csv", index=False
            )
        annual_delta = annual_paired.groupby("h3_cell", as_index=False).agg(
            rainfall_mm=("difference_mm", "mean"),
            station_support=("station_id", "nunique"),
        )
        annual_delta_json = geojson(annual_delta, "rainfall_mm")
        (gj_dir / "rainfall_complete_2024_minus_2017.geojson").write_text(
            json.dumps(annual_delta_json, allow_nan=False)
        )
        (gj_dir / "rainfall_august_2024_minus_2017.geojson").write_text(
            json.dumps(august_json, allow_nan=False)
        )
        august_paired.to_csv(tables / "matched_august_station_comparison.csv", index=False)
        annual_paired.to_csv(tables / "matched_annual_station_comparison.csv", index=False)
        active.to_csv(tables / "active_gauges_2017_2024.csv", index=False)

        print("[3/3] Writing annual, change and active-gauge Folium maps", flush=True)
        august_bins = safe_choropleth_bins([august_json], symmetric=True)
        m = map_page("August rainfall change: 2024 minus 2017", carto_key)
        add_choropleth(
            m, august_json, "August rainfall change: 2024 minus 2017",
            august_bins, difference=True,
        )
        save_map(m, html / "geo_rainfall_2024_08_choropleth.html")

        annual_bins = safe_choropleth_bins(
            [annual_shapes[2017], annual_shapes[2024]]
        )
        for year in (2017, 2024):
            m = map_page(f"Complete-year rainfall — {year}", carto_key)
            add_choropleth(
                m, annual_shapes[year], f"Complete-year rainfall — {year}",
                annual_bins,
            )
            save_map(m, html / f"geo_rainfall_{year}_complete_choropleth.html")

        annual_delta_bins = safe_choropleth_bins(
            [annual_delta_json], symmetric=True
        )
        m = map_page("Complete-year rainfall change: 2024 minus 2017", carto_key)
        add_choropleth(
            m, annual_delta_json, "Annual rainfall change: 2024 minus 2017",
            annual_delta_bins, difference=True,
        )
        save_map(m, html / "geo_rainfall_2024_vs_2017_annual_change.html")

        for year, color in ((2017, "#2b8cbe"), (2024, "#d94801")):
            m = map_page(f"Active rainfall gauges — {year}", carto_key)
            add_station_layer(
                m, active[active.year == year], f"Active gauges — {year}", color
            )
            save_map(m, html / f"geo_active_devices_{year}.html")
        if args.render_png:
            render_pngs(
                html, figures, args.png_width, args.png_height, args.png_wait_seconds
            )
        run.write_json("method", {
            "periods": ["August 2017 vs August 2024", "full 2017 vs full 2024"],
            "measure": VALUE,
            "august_interpretation": "Estimated August accumulation: eligible-day mean multiplied by 31",
            "annual_interpretation": "Each annual rainfall map uses gauges with twelve coverage-qualified months; both maps share one colour scale",
            "annual_change_interpretation": "The change map uses the same complete station IDs and H3 cells in both years",
            "annual_minimum_eligible_months": args.minimum_eligible_months,
            "difference": "Mean station-level 2024-minus-2017 change for common station IDs in the same H3 cell",
            "summary_source": source,
            "coverage_settings": "Inherited from summary when --monthly-summary is supplied",
            "geography": "OpenStreetMap / credentialled CARTO Light and Voyager; H3 sensor-supported cells, no spatial interpolation",
            "active_gauge_marker_radius_pixels": 7,
            "png_rendered": args.render_png})
        for label, folder in [("tables", tables), ("geojson", gj_dir),
                              ("interactive_maps", html), ("figures", figures)]:
            run.record_artifact(label, str(folder), mode="full")
        if args.publish_legacy:
            for folder in (tables, gj_dir, html, figures):
                publish_tree(str(folder), str(Path(config.OUTPUT_DIR) / folder.name))
            if args.clean_published_geo:
                clean_published_geo_maps(
                    config.OUTPUT_DIR,
                    keep_png={
                        "geo_rainfall_2024_08_choropleth.png",
                        "geo_rainfall_2017_complete_choropleth.png",
                        "geo_rainfall_2024_complete_choropleth.png",
                        "geo_rainfall_2024_vs_2017_annual_change.png",
                        "geo_active_devices_2017.png",
                        "geo_active_devices_2024.png",
                    },
                    keep_html={
                        "geo_rainfall_2024_08_choropleth.html",
                        "geo_rainfall_2017_complete_choropleth.html",
                        "geo_rainfall_2024_complete_choropleth.html",
                        "geo_rainfall_2024_vs_2017_annual_change.html",
                        "geo_active_devices_2017.html",
                        "geo_active_devices_2024.html",
                    },
                )
        run.finalize()
        print(f"Completed: {run.path}")
    except Exception as exc:
        run.finalize("failed", str(exc))
        raise


if __name__ == "__main__":
    main()
