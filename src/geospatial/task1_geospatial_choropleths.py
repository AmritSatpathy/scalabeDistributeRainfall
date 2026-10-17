"""Focused geospatial EDA for Singapore rainfall.

Outputs are deliberately separated so each PNG can be placed independently in
the report.  Spark performs interval-to-day and day-to-month aggregation; only
small station-month and H3 summaries are collected for plotting.

The five regions in this module are transparent *analytical zones* derived from
station coordinates.  They are Central, North, South, East and West and must not
be described as official URA planning regions.  This distinction matters because
the official five planning regions use North-East rather than South.
"""
from __future__ import annotations

import argparse
import calendar
import math
import os
import sys
from pathlib import Path

import folium
import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from run_artifacts import RunContext, latest_artifact, publish_tree  # noqa: E402
from spark_utils import get_spark  # noqa: E402

from pyspark.sql import functions as F  # noqa: E402


EXPECTED_INTERVALS_PER_DAY = 24 * 60 // config.INTERVAL_MINUTES
SINGAPORE_CENTER = (1.3521, 103.8198)
REGION_ORDER = ["Central", "North", "South", "East", "West"]
REGION_COLORS = {
    "Central": "#7b3294",
    "North": "#008837",
    "South": "#c51b7d",
    "East": "#0571b0",
    "West": "#e66101",
}


def set_academic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def h3_cell(lat: float, lon: float, resolution: int) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(float(lat), float(lon), resolution)
    return h3.geo_to_h3(float(lat), float(lon), resolution)


def h3_boundary(cell: str) -> list[tuple[float, float]]:
    if hasattr(h3, "cell_to_boundary"):
        coordinates = h3.cell_to_boundary(cell)
    else:
        coordinates = h3.h3_to_geo_boundary(cell)
    return [(float(lon), float(lat)) for lat, lon in coordinates]


def h3_center(cell: str) -> tuple[float, float]:
    if hasattr(h3, "cell_to_latlng"):
        lat, lon = h3.cell_to_latlng(cell)
    else:
        lat, lon = h3.h3_to_geo(cell)
    return float(lat), float(lon)


def assign_region(lat: float, lon: float) -> str:
    """Assign a point to a reproducible five-zone analytical geography.

    Longitude is scaled by cos(latitude), then an inner ellipse defines Central.
    Outside it, the dominant displacement from Singapore's geographic centre
    defines North, South, East or West.  The rule is stable across all years.
    """
    dy = float(lat) - SINGAPORE_CENTER[0]
    dx = (float(lon) - SINGAPORE_CENTER[1]) * math.cos(
        math.radians(SINGAPORE_CENTER[0])
    )
    if (dx / 0.036) ** 2 + (dy / 0.027) ** 2 <= 1.0:
        return "Central"
    dx_scaled = dx / 0.036
    dy_scaled = dy / 0.027
    if abs(dx_scaled) >= abs(dy_scaled):
        return "East" if dx > 0 else "West"
    return "North" if dy > 0 else "South"


def add_spatial_keys(frame: pd.DataFrame, resolution: int) -> pd.DataFrame:
    output = frame.dropna(subset=["lat", "lon"]).copy()
    output["h3_cell"] = [
        h3_cell(lat, lon, resolution) for lat, lon in zip(output["lat"], output["lon"])
    ]
    output["analytical_region"] = [
        assign_region(lat, lon) for lat, lon in zip(output["lat"], output["lon"])
    ]
    return output


def station_month_summary(observations, daily_completeness: float, minimum_days: int):
    """Create comparable station-month estimates from complete station-days.

    A month is retained only when it contains at least ``minimum_days`` eligible
    station-days.  Normalised monthly rainfall equals mean eligible-day rainfall
    times calendar days, removing the mechanical bias from unequal day counts.
    """
    daily = (
        observations.groupBy(
            "station_id", "station_name", "lat", "lon", "year", "month", "date"
        )
        .agg(
            F.count("rain_mm").alias("observed_intervals"),
            F.sum("rain_mm").alias("daily_rainfall_mm"),
        )
        .withColumn(
            "daily_coverage",
            F.col("observed_intervals") / F.lit(EXPECTED_INTERVALS_PER_DAY),
        )
        .filter(F.col("daily_coverage") >= F.lit(daily_completeness))
    )
    monthly = (
        daily.groupBy("station_id", "station_name", "lat", "lon", "year", "month")
        .agg(
            F.count("date").alias("eligible_days"),
            F.avg("daily_rainfall_mm").alias("mean_daily_rainfall_mm"),
            F.sum("daily_rainfall_mm").alias("observed_monthly_rainfall_mm"),
        )
        .filter(F.col("eligible_days") >= F.lit(minimum_days))
        .toPandas()
    )
    monthly["calendar_days"] = [
        calendar.monthrange(int(year), int(month))[1]
        for year, month in zip(monthly["year"], monthly["month"])
    ]
    monthly["normalised_monthly_rainfall_mm"] = (
        monthly["mean_daily_rainfall_mm"] * monthly["calendar_days"]
    )
    monthly["period"] = pd.PeriodIndex(
        pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1)),
        freq="M",
    )
    return monthly


def active_devices(observations, target: pd.Period) -> pd.DataFrame:
    """Return station-device locations that actually reported in target month."""
    return (
        observations.filter(
            (F.col("year") == F.lit(target.year))
            & (F.col("month") == F.lit(target.month))
            & F.col("rain_mm").isNotNull()
        )
        .groupBy("station_id", "station_name", "station_device_id", "lat", "lon")
        .agg(
            F.count("rain_mm").alias("reported_intervals"),
            F.countDistinct("date").alias("reporting_days"),
        )
        .orderBy("station_id", "station_device_id")
        .toPandas()
    )


def select_target_month(monthly: pd.DataFrame, requested: str | None) -> pd.Period:
    """Select a wet month with valid previous-month and previous-year evidence."""
    available = set(monthly["period"].unique())
    network = (
        monthly.groupby("period", as_index=False)
        .agg(
            network_normalised_mm=("normalised_monthly_rainfall_mm", "mean"),
            active_stations=("station_id", "nunique"),
        )
        .sort_values("period")
    )
    network["has_previous_month"] = network["period"].map(lambda p: p - 1 in available)
    network["has_previous_year"] = network["period"].map(lambda p: p - 12 in available)
    comparable = network[network["has_previous_month"] & network["has_previous_year"]]
    if requested:
        target = pd.Period(requested, freq="M")
        row = network[network["period"] == target]
        if row.empty:
            raise ValueError(f"Target month {target} is absent after coverage filtering")
        if not bool(row.iloc[0]["has_previous_month"] and row.iloc[0]["has_previous_year"]):
            raise ValueError(
                f"Target month {target} lacks the required previous-month or previous-year baseline"
            )
        return target
    if comparable.empty:
        raise RuntimeError("No month has both previous-month and previous-year baselines")
    return comparable.loc[comparable["network_normalised_mm"].idxmax(), "period"]


def month_frame(monthly: pd.DataFrame, period: pd.Period, resolution: int) -> pd.DataFrame:
    frame = monthly[monthly["period"] == period].copy()
    return add_spatial_keys(frame, resolution)


def aggregate_cells(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Use equal station weights inside each H3 cell."""
    valid = frame.dropna(subset=["h3_cell", value])
    if valid.empty:
        return pd.DataFrame(columns=["h3_cell", value, "station_support"])
    output = (
        valid.groupby("h3_cell", as_index=False)
        .agg(**{value: (value, "mean")}, station_support=("station_id", "nunique"))
    )
    centers = output["h3_cell"].map(h3_center)
    output["lat"] = centers.map(lambda pair: pair[0])
    output["lon"] = centers.map(lambda pair: pair[1])
    output["analytical_region"] = [
        assign_region(lat, lon) for lat, lon in zip(output["lat"], output["lon"])
    ]
    return output


def comparison_cells(
    monthly: pd.DataFrame,
    target: pd.Period,
    baseline: pd.Period,
    resolution: int,
) -> pd.DataFrame:
    value = "normalised_monthly_rainfall_mm"
    current = aggregate_cells(month_frame(monthly, target, resolution), value).rename(
        columns={value: "target_mm", "station_support": "target_station_support"}
    )
    previous = aggregate_cells(month_frame(monthly, baseline, resolution), value).rename(
        columns={value: "baseline_mm", "station_support": "baseline_station_support"}
    )
    merged = current.merge(
        previous[["h3_cell", "baseline_mm", "baseline_station_support"]],
        on="h3_cell",
        how="outer",
    )
    centers = merged["h3_cell"].map(h3_center)
    merged["lat"] = centers.map(lambda pair: pair[0])
    merged["lon"] = centers.map(lambda pair: pair[1])
    merged["analytical_region"] = [
        assign_region(lat, lon) for lat, lon in zip(merged["lat"], merged["lon"])
    ]
    merged["difference_mm"] = merged["target_mm"] - merged["baseline_mm"]
    merged["percent_change"] = np.where(
        merged["baseline_mm"] > 0,
        100.0 * merged["difference_mm"] / merged["baseline_mm"],
        np.nan,
    )
    return merged


def spatial_extent(*frames: pd.DataFrame) -> tuple[float, float, float, float]:
    valid = pd.concat(
        [f[["lon", "lat"]].dropna() for f in frames if {"lon", "lat"}.issubset(f.columns)],
        ignore_index=True,
    )
    return (
        float(valid["lon"].min()) - 0.018,
        float(valid["lon"].max()) + 0.018,
        float(valid["lat"].min()) - 0.014,
        float(valid["lat"].max()) + 0.014,
    )


def format_geo_axis(ax, extent, title: str) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(color="#d9d9d9", linewidth=0.45, alpha=0.65)


def draw_cells(ax, frame, value, cmap, norm, extent, title, *, no_data="#eeeeee"):
    patches, colours = [], []
    for row in frame.itertuples(index=False):
        patches.append(Polygon(h3_boundary(row.h3_cell), closed=True))
        raw = getattr(row, value)
        colours.append(no_data if pd.isna(raw) else mcolors.to_hex(cmap(norm(float(raw)))))
    if patches:
        collection = PatchCollection(
            patches,
            facecolor=colours,
            edgecolor="white",
            linewidth=0.85,
        )
        ax.add_collection(collection)
    format_geo_axis(ax, extent, title)


def add_figure_note(fig, text: str) -> None:
    fig.text(0.01, 0.006, text, ha="left", va="bottom", fontsize=8, color="#444444")


def save_figure(fig, directory: str, name: str) -> str:
    path = os.path.join(directory, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"  wrote figure: {path}")
    return path


def save_table(frame: pd.DataFrame, directory: str, name: str) -> str:
    path = os.path.join(directory, name)
    serial = frame.copy()
    for column in serial.columns:
        if isinstance(serial[column].dtype, pd.PeriodDtype):
            serial[column] = serial[column].astype(str)
    serial.to_csv(path, index=False)
    print(f"  wrote table: {path} ({len(serial):,} rows)")
    return path


def active_device_map(devices, resolution, extent, target, figures_dir, tables_dir):
    devices = add_spatial_keys(devices, resolution)
    save_table(devices, tables_dir, "geo_active_devices.csv")
    cells = devices[["h3_cell"]].drop_duplicates()
    fig, ax = plt.subplots(figsize=(7.25, 5.4), constrained_layout=True)
    for cell in cells["h3_cell"]:
        ax.add_patch(
            Polygon(
                h3_boundary(cell), closed=True, facecolor="#f3f3f3",
                edgecolor="#c8c8c8", linewidth=0.55, zorder=1,
            )
        )
    for region in REGION_ORDER:
        part = devices[devices["analytical_region"] == region]
        ax.scatter(
            part["lon"], part["lat"], s=43, c=REGION_COLORS[region],
            edgecolors="white", linewidths=0.75, label=region, zorder=3,
        )
    duplicate_station = devices.groupby("station_id")["station_device_id"].transform("nunique") > 1
    if duplicate_station.any():
        part = devices[duplicate_station]
        ax.scatter(
            part["lon"], part["lat"], s=78, facecolors="none", edgecolors="#111111",
            linewidths=1.1, zorder=4,
        )
    format_geo_axis(
        ax, extent,
        f"Active rainfall devices across Singapore — {target.strftime('%B %Y')}",
    )
    ax.legend(title="Analytical region", ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    add_figure_note(
        fig,
        f"Each pointer is a reporting station-device pair (n={len(devices)}). "
        "A black ring indicates more than one device ID at the same station during the month.",
    )
    return save_figure(fig, figures_dir, "geo_G1_active_device_locations.png")


def single_month_choropleth(current, extent, target, figures_dir, tables_dir):
    value = "normalised_monthly_rainfall_mm"
    cells = aggregate_cells(current, value)
    save_table(cells, tables_dir, "geo_target_month_h3.csv")
    vmax = max(float(cells[value].quantile(0.98)), 1.0)
    norm = mcolors.Normalize(0, vmax)
    cmap = plt.get_cmap("YlGnBu")
    fig, ax = plt.subplots(figsize=(7.25, 5.35), constrained_layout=True)
    draw_cells(
        ax, cells, value, cmap, norm, extent,
        f"H3 rainfall choropleth — {target.strftime('%B %Y')}",
    )
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.035, pad=0.025)
    cb.set_label("Coverage-normalised monthly rainfall (mm)")
    add_figure_note(
        fig,
        "H3 colour is the equal-weight station mean within each sensor-supported cell; blank land is not imputed.",
    )
    return save_figure(fig, figures_dir, "geo_G2_target_month_h3_choropleth.png"), cells


def comparison_figure(comparison, extent, target, baseline, label, filename, figures_dir, tables_dir):
    table_name = filename.replace(".png", ".csv")
    save_table(comparison, tables_dir, table_name)
    sequential_values = pd.concat(
        [comparison["target_mm"], comparison["baseline_mm"]], ignore_index=True
    ).dropna()
    vmax = max(float(sequential_values.quantile(0.98)), 1.0)
    seq_norm = mcolors.Normalize(0, vmax)
    delta_values = comparison["difference_mm"].dropna()
    bound = max(float(delta_values.abs().quantile(0.98)), 1.0)
    delta_norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.45), constrained_layout=True)
    draw_cells(
        axes[0], comparison, "target_mm", plt.get_cmap("YlGnBu"), seq_norm, extent,
        f"A. {target.strftime('%B %Y')}",
    )
    draw_cells(
        axes[1], comparison, "baseline_mm", plt.get_cmap("YlGnBu"), seq_norm, extent,
        f"B. {baseline.strftime('%B %Y')}",
    )
    draw_cells(
        axes[2], comparison, "difference_mm", plt.get_cmap("BrBG"), delta_norm, extent,
        "C. Change (A − B)",
    )
    cb1 = fig.colorbar(
        plt.cm.ScalarMappable(norm=seq_norm, cmap="YlGnBu"), ax=axes[:2], shrink=0.8, pad=0.02
    )
    cb1.set_label("Coverage-normalised monthly rainfall (mm)")
    cb2 = fig.colorbar(
        plt.cm.ScalarMappable(norm=delta_norm, cmap="BrBG"), ax=axes[2], shrink=0.8, pad=0.02
    )
    cb2.set_label("Rainfall difference (mm); green is wetter")
    fig.suptitle(label, fontweight="bold", fontsize=13)
    add_figure_note(fig, "Only H3 cells observed in at least one comparison month are shown; grey denotes unavailable evidence.")
    return save_figure(fig, figures_dir, filename)


def hotspot_transition(
    monthly, yoy, resolution, target, extent, quantile, figures_dir, tables_dir
):
    value = "normalised_monthly_rainfall_mm"
    history = monthly[(monthly["month"] == target.month) & (monthly["year"] < target.year)]
    historical_cells = []
    for period in sorted(history["period"].unique()):
        cells = aggregate_cells(month_frame(monthly, period, resolution), value)
        cells["period"] = str(period)
        historical_cells.append(cells)
    if not historical_cells:
        raise RuntimeError("No prior same-calendar-month data exists for hotspot thresholding")
    historical = pd.concat(historical_cells, ignore_index=True)
    threshold = float(historical[value].quantile(quantile))
    transitions = yoy.copy()
    complete_comparison = transitions[["target_mm", "baseline_mm"]].notna().all(axis=1)
    transitions["target_hotspot"] = transitions["target_mm"] >= threshold
    transitions["previous_year_hotspot"] = transitions["baseline_mm"] >= threshold
    transitions["hotspot_transition"] = np.select(
        [
            transitions["target_hotspot"] & transitions["previous_year_hotspot"],
            transitions["target_hotspot"] & ~transitions["previous_year_hotspot"],
            ~transitions["target_hotspot"] & transitions["previous_year_hotspot"],
        ],
        ["Persistent hotspot", "Emerging hotspot", "Dissipated hotspot"],
        default="Below threshold in both",
    )
    transitions.loc[~complete_comparison, "hotspot_transition"] = "Insufficient comparison data"
    transition_colors = {
        "Persistent hotspot": "#7f0000",
        "Emerging hotspot": "#ef6548",
        "Dissipated hotspot": "#3182bd",
        "Below threshold in both": "#e5e5e5",
        "Insufficient comparison data": "#969696",
    }
    save_table(historical, tables_dir, "geo_hotspot_historical_reference.csv")
    save_table(transitions, tables_dir, "geo_hotspot_transitions.csv")
    fig, ax = plt.subplots(figsize=(7.25, 5.35), constrained_layout=True)
    for row in transitions.itertuples(index=False):
        ax.add_patch(
            Polygon(
                h3_boundary(row.h3_cell), closed=True,
                facecolor=transition_colors[row.hotspot_transition],
                edgecolor="white", linewidth=0.85,
            )
        )
    format_geo_axis(
        ax,
        extent,
        "Rainfall hotspot persistence — "
        f"{target.strftime('%B %Y')} vs {(target - 12).strftime('%B %Y')}",
    )
    legend = [Patch(facecolor=color, edgecolor="white", label=label) for label, color in transition_colors.items()]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    add_figure_note(
        fig,
        f"Hotspot threshold = historical same-calendar-month H3 {quantile:.0%} quantile "
        f"({threshold:.1f} mm); threshold fixed before classifying both comparison months.",
    )
    path = save_figure(fig, figures_dir, "geo_G5_h3_hotspot_transition.png")
    return path, transitions, threshold


def analytical_region_map(devices, extent, target, figures_dir, tables_dir):
    station_regions = (
        devices.sort_values(["station_id", "reported_intervals"], ascending=[True, False])
        .drop_duplicates("station_id")
        [["station_id", "station_name", "lat", "lon", "analytical_region"]]
    )
    save_table(station_regions, tables_dir, "geo_analytical_region_membership.csv")
    fig, ax = plt.subplots(figsize=(7.25, 5.35), constrained_layout=True)
    for region in REGION_ORDER:
        part = station_regions[station_regions["analytical_region"] == region]
        ax.scatter(
            part["lon"], part["lat"], c=REGION_COLORS[region], s=50,
            edgecolors="white", linewidths=0.7, label=f"{region} (n={len(part)})",
        )
    ax.scatter(
        [SINGAPORE_CENTER[1]], [SINGAPORE_CENTER[0]], marker="*", s=125,
        c="#111111", edgecolors="white", linewidths=0.6, zorder=4,
    )
    format_geo_axis(
        ax,
        extent,
        f"Five-zone analytical geography — station membership in {target.strftime('%B %Y')}",
    )
    ax.legend(title="Analytical zone", loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    add_figure_note(
        fig,
        "Zones use a fixed coordinate rule around Singapore's geographic centre (star). They are analytical, not official URA regions.",
    )
    return save_figure(fig, figures_dir, "geo_G6_five_analytical_regions.png"), station_regions


def region_comparison(monthly, target, figures_dir, tables_dir):
    value = "normalised_monthly_rainfall_mm"
    relevant = monthly[monthly["period"].isin([target, target - 1, target - 12])].copy()
    relevant["analytical_region"] = [
        assign_region(lat, lon) for lat, lon in zip(relevant["lat"], relevant["lon"])
    ]
    regions = (
        relevant.groupby(["period", "analytical_region"], as_index=False)
        .agg(
            rainfall_mm=(value, "mean"),
            station_support=("station_id", "nunique"),
        )
    )
    wide = regions.pivot(index="analytical_region", columns="period", values="rainfall_mm").reindex(REGION_ORDER)
    support = regions.pivot(index="analytical_region", columns="period", values="station_support").reindex(REGION_ORDER)
    def values_for(period: pd.Period, table: pd.DataFrame) -> np.ndarray:
        if period not in table.columns:
            return np.full(len(REGION_ORDER), np.nan)
        return table[period].to_numpy()

    output = pd.DataFrame(
        {
            "analytical_region": REGION_ORDER,
            "target_mm": values_for(target, wide),
            "previous_month_mm": values_for(target - 1, wide),
            "previous_year_mm": values_for(target - 12, wide),
            "target_station_support": values_for(target, support),
            "previous_month_station_support": values_for(target - 1, support),
            "previous_year_station_support": values_for(target - 12, support),
        }
    )
    output["month_on_month_change_mm"] = output["target_mm"] - output["previous_month_mm"]
    output["year_on_year_change_mm"] = output["target_mm"] - output["previous_year_mm"]
    save_table(output, tables_dir, "geo_region_rainfall_comparison.csv")

    x = np.arange(len(REGION_ORDER))
    width = 0.24
    fig, axes = plt.subplots(2, 1, figsize=(7.25, 7.0), constrained_layout=True)
    axes[0].bar(x - width, output["previous_year_mm"], width, label=(target - 12).strftime("%b %Y"), color="#9ecae1")
    axes[0].bar(x, output["previous_month_mm"], width, label=(target - 1).strftime("%b %Y"), color="#fdae6b")
    axes[0].bar(x + width, output["target_mm"], width, label=target.strftime("%b %Y"), color="#2171b5")
    axes[0].set_title("A. Coverage-normalised rainfall by analytical region", loc="left", fontweight="bold")
    axes[0].set_ylabel("Monthly rainfall (mm)")
    axes[0].set_xticks(x, REGION_ORDER)
    axes[0].legend(ncol=3)
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.5)

    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].bar(x - 0.18, output["month_on_month_change_mm"], 0.36, label="Month-on-month", color="#e66101")
    axes[1].bar(x + 0.18, output["year_on_year_change_mm"], 0.36, label="Year-on-year", color="#018571")
    axes[1].set_title("B. Absolute change from temporal baselines", loc="left", fontweight="bold")
    axes[1].set_ylabel("Rainfall change (mm)")
    axes[1].set_xticks(x, REGION_ORDER)
    axes[1].legend(ncol=2)
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.5)
    add_figure_note(fig, "Regional estimates give each eligible station equal weight; station support is retained in the companion CSV.")
    return save_figure(fig, figures_dir, "geo_G7_region_rainfall_comparison.png"), output


def geojson_features(frame, value, layer_label, cmap, norm):
    features = []
    for row in frame.itertuples(index=False):
        raw = getattr(row, value)
        if pd.isna(raw):
            continue
        boundary = h3_boundary(row.h3_cell)
        coordinates = [[lon, lat] for lon, lat in boundary]
        coordinates.append(coordinates[0])
        properties = {
            "H3 cell": row.h3_cell,
            "Layer": layer_label,
            "Rainfall (mm)": round(float(raw), 2),
            "Analytical region": getattr(row, "analytical_region", ""),
            "fill": mcolors.to_hex(cmap(norm(float(raw)))),
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def interactive_map(devices, target_cells, yoy, transitions, target, output_path):
    """Write one optional exploration map; report evidence remains in PNG files."""
    fmap = folium.Map(location=SINGAPORE_CENTER, zoom_start=11, tiles="CartoDB positron", control_scale=True)
    stations_layer = folium.FeatureGroup(
        name=f"Active devices — {target.strftime('%b %Y')}", show=True
    )
    for row in devices.itertuples(index=False):
        folium.CircleMarker(
            location=[row.lat, row.lon], radius=4.5,
            color="#ffffff", weight=1, fill=True,
            fill_color=REGION_COLORS[row.analytical_region], fill_opacity=0.95,
            tooltip=(
                f"Station: {row.station_id}<br>Name: {row.station_name}<br>"
                f"Device: {row.station_device_id}<br>Region: {row.analytical_region}<br>"
                f"Reporting days: {row.reporting_days}"
            ),
        ).add_to(stations_layer)
    stations_layer.add_to(fmap)

    max_rain = max(float(target_cells["normalised_monthly_rainfall_mm"].quantile(0.98)), 1.0)
    rain_norm = mcolors.Normalize(0, max_rain)
    rain_cmap = plt.get_cmap("YlGnBu")
    target_geojson = geojson_features(
        target_cells, "normalised_monthly_rainfall_mm",
        target.strftime("%B %Y"), rain_cmap, rain_norm
    )
    folium.GeoJson(
        target_geojson,
        name=f"Rainfall choropleth — {target.strftime('%b %Y')}",
        style_function=lambda feature: {
            "fillColor": feature["properties"]["fill"], "color": "white",
            "weight": 1, "fillOpacity": 0.78,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["H3 cell", "Rainfall (mm)", "Analytical region"],
            aliases=["H3 cell", "Normalised rainfall (mm)", "Analytical region"],
        ),
        show=True,
    ).add_to(fmap)

    difference = yoy.dropna(subset=["difference_mm"])
    bound = max(float(difference["difference_mm"].abs().quantile(0.98)), 1.0)
    delta_norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    difference_geojson = geojson_features(
        difference, "difference_mm", "Year-on-year difference", plt.get_cmap("BrBG"), delta_norm
    )
    folium.GeoJson(
        difference_geojson,
        name="Year-on-year rainfall change",
        style_function=lambda feature: {
            "fillColor": feature["properties"]["fill"], "color": "white",
            "weight": 1, "fillOpacity": 0.82,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["H3 cell", "Rainfall (mm)", "Analytical region"],
            aliases=["H3 cell", "YoY change (mm)", "Analytical region"],
        ),
        show=False,
    ).add_to(fmap)

    hotspot_layer = folium.FeatureGroup(name="Persistent and emerging hotspots", show=False)
    hotspot_colors = {"Persistent hotspot": "#7f0000", "Emerging hotspot": "#ef6548"}
    for row in transitions[transitions["hotspot_transition"].isin(hotspot_colors)].itertuples(index=False):
        folium.Polygon(
            locations=[(lat, lon) for lon, lat in h3_boundary(row.h3_cell)],
            color="white", weight=1, fill=True,
            fill_color=hotspot_colors[row.hotspot_transition], fill_opacity=0.84,
            tooltip=f"{row.hotspot_transition}<br>H3: {row.h3_cell}<br>Target: {row.target_mm:.1f} mm",
        ).add_to(hotspot_layer)
    hotspot_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(
        folium.Element(
            "<div style='position:fixed;bottom:20px;left:45px;z-index:9999;"
            "background:white;padding:8px;border:1px solid #777;font-size:12px'>"
            "Five regions are analytical zones, not official URA planning regions.</div>"
        )
    )
    fmap.save(output_path)
    print(f"  wrote interactive map: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create focused Singapore rainfall geospatial choropleths"
    )
    parser.add_argument("--data", default=None, help="Clean observation Parquet")
    parser.add_argument("--target-month", default=None, help="Optional YYYY-MM; default selects wettest comparable month")
    parser.add_argument("--daily-completeness", type=float, default=0.90)
    parser.add_argument("--minimum-eligible-days", type=int, default=20)
    parser.add_argument("--h3-resolution", type=int, default=7)
    parser.add_argument("--hotspot-quantile", type=float, default=0.90)
    parser.add_argument("--hash-mode", choices=["full", "metadata"], default="metadata")
    parser.add_argument("--publish-legacy", action="store_true")
    args = parser.parse_args()

    if not 0 < args.daily_completeness <= 1:
        parser.error("--daily-completeness must lie in (0, 1]")
    if not 1 <= args.minimum_eligible_days <= 28:
        parser.error("--minimum-eligible-days must lie between 1 and 28")
    if not 0 <= args.h3_resolution <= 15:
        parser.error("--h3-resolution must lie between 0 and 15")
    if not 0.5 <= args.hotspot_quantile < 1:
        parser.error("--hotspot-quantile must lie in [0.5, 1)")

    config.ensure_dirs()
    args.data = args.data or latest_artifact("clean", "clean_parquet") or config.CLEAN_PARQUET
    if not Path(args.data).exists():
        parser.error(f"Clean data not found: {args.data}")

    run = RunContext("task1_geo_choropleths", vars(args), hash_mode=args.hash_mode)
    run.record_input("clean_parquet", args.data)
    figures_dir = run.artifact_path("figures")
    tables_dir = run.artifact_path("tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    set_academic_style()

    spark = get_spark("task1_geo_choropleths", driver_memory="10g", shuffle_partitions=96)
    try:
        observations = spark.read.parquet(args.data)
        print("[1/8] Building coverage-qualified station-month summaries")
        monthly = station_month_summary(
            observations, args.daily_completeness, args.minimum_eligible_days
        )
        target = select_target_month(monthly, args.target_month)
        save_table(monthly, tables_dir, "geo_station_month_summary.csv")
        print(f"  selected target month: {target} (wettest month with MoM and YoY baselines)")

        print("[2/8] Mapping active station-device locations")
        devices = add_spatial_keys(active_devices(observations, target), args.h3_resolution)
        if devices.empty:
            raise RuntimeError(f"No active devices found for {target}")
        current = month_frame(monthly, target, args.h3_resolution)
        extent = spatial_extent(devices, current)
        active_device_map(devices, args.h3_resolution, extent, target, figures_dir, tables_dir)

        print("[3/8] Building target-month H3 rainfall choropleth")
        _, target_cells = single_month_choropleth(
            current, extent, target, figures_dir, tables_dir
        )

        print("[4/8] Building same-month year-on-year H3 comparison")
        yoy = comparison_cells(monthly, target, target - 12, args.h3_resolution)
        comparison_figure(
            yoy, extent, target, target - 12,
            "Same-calendar-month rainfall comparison",
            "geo_G3_same_month_year_on_year_choropleth.png", figures_dir, tables_dir,
        )

        print("[5/8] Building month-on-month H3 comparison")
        mom = comparison_cells(monthly, target, target - 1, args.h3_resolution)
        comparison_figure(
            mom, extent, target, target - 1,
            "Month-on-month rainfall comparison",
            "geo_G4_month_on_month_choropleth.png", figures_dir, tables_dir,
        )

        print("[6/8] Detecting persistent, emerging and dissipated hotspots")
        _, transitions, threshold = hotspot_transition(
            monthly, yoy, args.h3_resolution, target, extent,
            args.hotspot_quantile, figures_dir, tables_dir,
        )

        print("[7/8] Building analytical-region membership and rainfall comparisons")
        analytical_region_map(devices, extent, target, figures_dir, tables_dir)
        region_comparison(monthly, target, figures_dir, tables_dir)

        print("[8/8] Writing layered Folium inspection map")
        interactive_path = run.artifact_path("interactive/geo_interactive_layers.html")
        os.makedirs(os.path.dirname(interactive_path), exist_ok=True)
        interactive_map(devices, target_cells, yoy, transitions, target, interactive_path)

        run.write_json(
            "geospatial_design",
            {
                "target_month": str(target),
                "target_selection_rule": "wettest coverage-normalised network month with previous-month and previous-year baselines",
                "monthly_measure": "mean rainfall per eligible station-day multiplied by calendar days",
                "daily_completeness": args.daily_completeness,
                "minimum_eligible_days": args.minimum_eligible_days,
                "h3_resolution": args.h3_resolution,
                "hotspot_quantile": args.hotspot_quantile,
                "hotspot_threshold_mm": threshold,
                "regional_weighting": "equal station weight",
                "region_definition": "fixed coordinate-based analytical zones: Central, North, South, East, West",
                "region_caveat": "analytical zones are not official URA planning regions",
            },
        )
    except Exception as exc:
        run.finalize("failed", str(exc))
        raise
    finally:
        spark.stop()

    run.record_artifact("figures", figures_dir)
    run.record_artifact("tables", tables_dir)
    run.record_artifact("interactive_map", interactive_path)
    if args.publish_legacy:
        publish_tree(figures_dir, config.GEOSPATIAL_FIGURES_DIR)
        publish_tree(tables_dir, config.TABLES_DIR)
        publish_tree(os.path.dirname(interactive_path), os.path.join(config.OUTPUT_DIR, "interactive"))
    run.finalize()
    print(f"tracked geospatial choropleth run: {run.run_id} -> {run.path}")


if __name__ == "__main__":
    main()
