"""Coverage-safe geospatial Task 1 analysis with report-ready PNG output.

Heavy grouping remains in Spark. Only station-, month-, H3-, or event-level
summaries are collected to pandas for plotting. The script creates the planned
network, seasonal, anomaly, lag-footprint and antecedent-state figures.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timedelta

import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from run_artifacts import RunContext, latest_artifact, publish_tree  # noqa: E402
from spark_utils import get_spark  # noqa: E402

from pyspark.sql import functions as F  # noqa: E402

FIGURES_DIR = config.FIGURES_DIR
TABLES_DIR = config.TABLES_DIR
EXPECTED_INTERVALS_PER_DAY = 24 * 60 // config.INTERVAL_MINUTES
SEASONS = [
    "Northeast Monsoon\n(Dec–Mar)",
    "First Inter-monsoon\n(Apr–May)",
    "Southwest Monsoon\n(Jun–Sep)",
    "Second Inter-monsoon\n(Oct–Nov)",
]


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _figure_path(name: str) -> str:
    return os.path.join(FIGURES_DIR, name)


def _table_path(name: str) -> str:
    return os.path.join(TABLES_DIR, name)


def _save_table(frame: pd.DataFrame, name: str) -> None:
    path = _table_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, index=False)


def _save_figure(fig, name: str) -> None:
    path = _figure_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _h3_cell(lat: float, lon: float, resolution: int) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(float(lat), float(lon), resolution)
    return h3.geo_to_h3(float(lat), float(lon), resolution)


def _h3_boundary(cell: str) -> list[tuple[float, float]]:
    if hasattr(h3, "cell_to_boundary"):
        points = h3.cell_to_boundary(cell)
    else:
        points = h3.h3_to_geo_boundary(cell)
    return [(float(lon), float(lat)) for lat, lon in points]


def aggregate_h3(
    frame: pd.DataFrame,
    value_column: str,
    resolution: int,
) -> pd.DataFrame:
    """Average station summaries within H3 cells with equal station weight."""
    required = {"lat", "lon", value_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"H3 input lacks columns: {sorted(missing)}")
    valid = frame.dropna(subset=["lat", "lon", value_column]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["h3_cell", value_column, "station_support"])
    valid["h3_cell"] = [
        _h3_cell(lat, lon, resolution) for lat, lon in zip(valid["lat"], valid["lon"])
    ]
    return (
        valid.groupby("h3_cell", as_index=False)
        .agg(**{value_column: (value_column, "mean")}, station_support=(value_column, "size"))
    )


def draw_h3(
    ax,
    frame: pd.DataFrame,
    value_column: str,
    cmap,
    norm,
    title: str,
) -> None:
    patches = []
    values = []
    for row in frame.itertuples(index=False):
        patches.append(Polygon(_h3_boundary(row.h3_cell), closed=True))
        values.append(float(getattr(row, value_column)))
    if patches:
        collection = PatchCollection(
            patches,
            cmap=cmap,
            norm=norm,
            edgecolor="white",
            linewidth=0.7,
        )
        collection.set_array(np.asarray(values))
        ax.add_collection(collection)
        all_points = np.asarray([point for patch in patches for point in patch.get_xy()])
        ax.set_xlim(all_points[:, 0].min() - 0.01, all_points[:, 0].max() + 0.01)
        ax.set_ylim(all_points[:, 1].min() - 0.008, all_points[:, 1].max() + 0.008)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#d9d9d9", linewidth=0.4, alpha=0.5)


def _season_column():
    return (
        F.when(F.col("month").isin(12, 1, 2, 3), SEASONS[0])
        .when(F.col("month").isin(4, 5), SEASONS[1])
        .when(F.col("month").isin(6, 7, 8, 9), SEASONS[2])
        .otherwise(SEASONS[3])
    )


def eligible_station_days(observations, completeness: float):
    return (
        observations.groupBy("station_id", "station_name", "lat", "lon", "year", "month", "date")
        .agg(
            F.count("rain_mm").alias("observed_intervals"),
            F.sum("rain_mm").alias("daily_mm"),
            F.max("rain_mm").alias("daily_max_5m_mm"),
        )
        .withColumn("expected_intervals", F.lit(EXPECTED_INTERVALS_PER_DAY))
        .withColumn("coverage", F.col("observed_intervals") / F.col("expected_intervals"))
        .filter(F.col("coverage") >= F.lit(completeness))
    )


def observation_network_figure(observations) -> None:
    station_life = (
        observations.groupBy("station_id", "station_name", "lat", "lon")
        .agg(
            F.min("year").alias("first_year"),
            F.max("year").alias("last_year"),
            F.countDistinct("year").alias("active_years"),
            F.countDistinct("station_device_id").alias("device_ids"),
            F.countDistinct(F.struct("observed_lat", "observed_lon")).alias("coordinate_pairs"),
            F.count(F.lit(1)).alias("observations"),
        )
        .orderBy("station_id")
        .toPandas()
    )
    monthly = (
        observations.groupBy("year", "month")
        .agg(F.countDistinct("station_id").alias("active_stations"))
        .orderBy("year", "month")
        .toPandas()
    )
    monthly["period"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month"], day=1)
    )
    _save_table(station_life, "geo_station_continuity.csv")
    _save_table(monthly.drop(columns="period"), "geo_active_stations_monthly.csv")

    fig = plt.figure(figsize=(7.2, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.25])
    ax_map = fig.add_subplot(grid[0])
    changed = (station_life["device_ids"] > 1) | (station_life["coordinate_pairs"] > 1)
    scatter = ax_map.scatter(
        station_life["lon"], station_life["lat"],
        c=station_life["first_year"], cmap="viridis", s=35 + 10 * station_life["active_years"],
        edgecolors=np.where(changed, "#d55e00", "#202020"), linewidths=np.where(changed, 1.5, 0.5),
        alpha=0.9,
    )
    ax_map.set_title("A. Rain-gauge network and continuity", loc="left", fontweight="bold")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(color="#dddddd", linewidth=0.5)
    colorbar = fig.colorbar(scatter, ax=ax_map, fraction=0.035, pad=0.02)
    colorbar.set_label("First observed year")
    ax_map.text(
        0.01, 0.01, "Orange outline: device or coordinate change\nPoint size: active years",
        transform=ax_map.transAxes, fontsize=8, va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )

    ax_time = fig.add_subplot(grid[1])
    ax_time.plot(monthly["period"], monthly["active_stations"], color="#1f4e79", linewidth=1.5)
    ax_time.fill_between(monthly["period"], monthly["active_stations"], color="#7aa6c2", alpha=0.3)
    ax_time.set_title("B. Active stations by month", loc="left", fontweight="bold")
    ax_time.set_ylabel("Stations")
    ax_time.set_xlabel("Observation month")
    ax_time.xaxis.set_major_locator(mdates.YearLocator())
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_time.grid(axis="y", color="#dddddd", linewidth=0.5)
    _save_figure(fig, "geo_G1_network_continuity.png")


def seasonal_geography_figure(daily, resolution: int) -> None:
    station_season = (
        daily.withColumn("season", _season_column())
        .groupBy("station_id", "lat", "lon", "season")
        .agg(
            F.avg("daily_mm").alias("mm_per_eligible_day"),
            F.count("date").alias("eligible_days"),
        )
        .toPandas()
    )
    _save_table(station_season, "geo_seasonal_station_summary.csv")
    mapped = {
        season: aggregate_h3(
            station_season[station_season["season"] == season],
            "mm_per_eligible_day",
            resolution,
        )
        for season in SEASONS
    }
    all_values = pd.concat(mapped.values(), ignore_index=True)["mm_per_eligible_day"]
    vmax = float(all_values.quantile(0.98)) if len(all_values) else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=max(vmax, 0.01))
    cmap = plt.get_cmap("cividis")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), constrained_layout=True)
    for label, ax in zip(SEASONS, axes.flat):
        draw_h3(ax, mapped[label], "mm_per_eligible_day", cmap, norm, label)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.8)
    colorbar.set_label("Mean rainfall per eligible station-day (mm)")
    fig.suptitle("Seasonal geography of rainfall", fontweight="bold")
    _save_figure(fig, "geo_G2_seasonal_geography.png")


def station_month_frame(daily) -> pd.DataFrame:
    return (
        daily.groupBy("station_id", "lat", "lon", "year", "month")
        .agg(
            F.avg("daily_mm").alias("mm_per_eligible_day"),
            F.count("date").alias("eligible_days"),
        )
        .filter(F.col("eligible_days") >= F.lit(20))
        .toPandas()
    )


def choose_target_period(monthly: pd.DataFrame) -> pd.Period:
    network = (
        monthly.groupby(["year", "month"], as_index=False)
        .agg(network_mm_per_day=("mm_per_eligible_day", "mean"), active_stations=("station_id", "nunique"))
        .sort_values(["year", "month"])
    )
    z_values = []
    for row in network.itertuples(index=False):
        history = network[(network["month"] == row.month) & (network["year"] < row.year)]
        if len(history) >= 2 and history["network_mm_per_day"].std(ddof=0) > 0:
            z = (row.network_mm_per_day - history["network_mm_per_day"].mean()) / history["network_mm_per_day"].std(ddof=0)
        else:
            z = np.nan
        z_values.append(z)
    network["same_month_prior_z"] = z_values
    eligible = network.dropna(subset=["same_month_prior_z"])
    selected = eligible.loc[eligible["same_month_prior_z"].idxmax()] if not eligible.empty else network.loc[network["network_mm_per_day"].idxmax()]
    _save_table(network, "geo_month_selection_audit.csv")
    return pd.Period(year=int(selected["year"]), month=int(selected["month"]), freq="M")


def anomaly_frame(monthly: pd.DataFrame, target: pd.Period) -> pd.DataFrame:
    data = monthly.copy()
    data["period"] = pd.PeriodIndex(
        pd.to_datetime(dict(year=data["year"], month=data["month"], day=1)), freq="M"
    )
    current = data[data["period"] == target][
        ["station_id", "lat", "lon", "mm_per_eligible_day", "eligible_days"]
    ].rename(columns={"mm_per_eligible_day": "target_mm_per_day"})

    def baseline_for_period(period: pd.Period, name: str) -> pd.DataFrame:
        return data[data["period"] == period][["station_id", "mm_per_eligible_day"]].rename(
            columns={"mm_per_eligible_day": name}
        )

    result = current.merge(baseline_for_period(target - 1, "previous_month"), on="station_id", how="left")
    trailing = (
        data[data["period"].isin([target - 1, target - 2, target - 3])]
        .groupby("station_id", as_index=False)["mm_per_eligible_day"].mean()
        .rename(columns={"mm_per_eligible_day": "trailing_3m"})
    )
    result = result.merge(trailing, on="station_id", how="left")
    result = result.merge(baseline_for_period(target - 12, "previous_year"), on="station_id", how="left")
    historical = (
        data[(data["month"] == target.month) & (data["year"] < target.year)]
        .groupby("station_id", as_index=False)["mm_per_eligible_day"].mean()
        .rename(columns={"mm_per_eligible_day": "prior_same_month_mean"})
    )
    result = result.merge(historical, on="station_id", how="left")
    for baseline in ("previous_month", "trailing_3m", "previous_year", "prior_same_month_mean"):
        result[f"delta_{baseline}"] = result["target_mm_per_day"] - result[baseline]
    return result


def anomaly_dashboard(monthly: pd.DataFrame, resolution: int, requested: str | None) -> pd.Period:
    target = pd.Period(requested, freq="M") if requested else choose_target_period(monthly)
    frame = anomaly_frame(monthly, target)
    _save_table(frame, f"geo_anomaly_{target.strftime('%Y_%m')}.csv")
    columns = [
        ("target_mm_per_day", f"A. {target.strftime('%B %Y')}"),
        ("delta_previous_month", "B. Difference from previous month"),
        ("delta_trailing_3m", "C. Difference from prior three months"),
        ("delta_previous_year", "D. Difference from same month one year earlier"),
        ("delta_prior_same_month_mean", "E. Difference from prior same-month mean"),
    ]
    h3_frames = {name: aggregate_h3(frame, name, resolution) for name, _ in columns}
    target_values = h3_frames["target_mm_per_day"]["target_mm_per_day"]
    target_norm = mcolors.Normalize(0.0, max(float(target_values.quantile(0.98)), 0.01))
    delta_values = pd.concat(
        [h3_frames[name][name] for name, _ in columns[1:]], ignore_index=True
    ).dropna()
    delta_bound = max(float(delta_values.abs().quantile(0.98)), 0.01)
    delta_norm = mcolors.TwoSlopeNorm(vmin=-delta_bound, vcenter=0.0, vmax=delta_bound)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.4), constrained_layout=True)
    for index, (name, title) in enumerate(columns):
        ax = axes.flat[index]
        if index == 0:
            draw_h3(ax, h3_frames[name], name, plt.get_cmap("cividis"), target_norm, title)
        else:
            draw_h3(ax, h3_frames[name], name, plt.get_cmap("RdBu"), delta_norm, title)
    axes.flat[-1].axis("off")
    cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=target_norm, cmap="cividis"), ax=axes[:, 0], shrink=0.75)
    cb1.set_label("Rainfall (mm per eligible day)")
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=delta_norm, cmap="RdBu"), ax=axes[:, 1:], shrink=0.75)
    cb2.set_label("Difference (mm per eligible day); blue is wetter")
    fig.suptitle("Multi-baseline spatial rainfall anomaly", fontweight="bold")
    _save_figure(fig, "geo_G4_multibaseline_anomaly.png")
    return target


def _parse_event_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=None)


def choose_event_origin(features, min_stations: int, requested: str | None) -> datetime:
    if requested:
        return _parse_event_time(requested)
    selected = (
        features.filter(F.col("roll_30m_coverage") >= F.lit(1.0))
        .groupBy("ts")
        .agg(
            F.avg("roll_30m_mm").alias("network_mean_30m_mm"),
            F.countDistinct("station_id").alias("active_stations"),
        )
        .filter(F.col("active_stations") >= F.lit(min_stations))
        .orderBy(F.col("network_mean_30m_mm").desc(), F.col("ts"))
        .first()
    )
    if selected is None:
        raise RuntimeError("No event origin met the station and coverage requirements")
    return selected["ts"]


def event_figures(features, resolution: int, min_stations: int, requested: str | None) -> datetime:
    origin = choose_event_origin(features, min_stations, requested)
    offsets = (-60, -30, -15, 0, 30)
    frame_times = [origin + timedelta(minutes=value) for value in offsets]
    lagged = (
        features.filter(F.col("ts").isin(frame_times))
        .select("station_id", "lat", "lon", "ts", "roll_30m_mm", "roll_30m_coverage")
        .filter(F.col("roll_30m_coverage") >= F.lit(1.0))
        .toPandas()
    )
    _save_table(lagged, "geo_selected_event_lag_frames.csv")
    mapped = {}
    for offset, frame_time in zip(offsets, frame_times):
        part = lagged[pd.to_datetime(lagged["ts"]) == pd.Timestamp(frame_time)]
        mapped[offset] = aggregate_h3(part, "roll_30m_mm", resolution)
    all_values = pd.concat(mapped.values(), ignore_index=True)["roll_30m_mm"]
    norm = mcolors.Normalize(0.0, max(float(all_values.quantile(0.98)), 0.01))
    cmap = plt.get_cmap("cividis")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.3), constrained_layout=True)
    for index, offset in enumerate(offsets):
        role = "forecast origin" if offset == 0 else ("outcome" if offset > 0 else "history")
        title = f"{chr(65 + index)}. t{offset:+d} min ({role})"
        draw_h3(axes.flat[index], mapped[offset], "roll_30m_mm", cmap, norm, title)
    axes.flat[-1].axis("off")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.78)
    colorbar.set_label("Previous 30-minute rainfall (mm)")
    fig.suptitle(f"Lagged storm footprint around {origin:%Y-%m-%d %H:%M} SGT", fontweight="bold")
    _save_figure(fig, "geo_G10_lagged_storm_footprint.png")

    state_columns = [
        ("roll_30m_mm", "A. Previous 30 min"),
        ("roll_180m_mm", "B. Previous 3 h"),
        ("roll_1440m_mm", "C. Previous 24 h"),
        ("intensity_change_30m_mm", "D. Change from preceding 30 min"),
        ("roll_180m_wet_fraction", "E. Wet-slot fraction, previous 3 h"),
        ("roll_30m_max_mm", "F. Peak five-minute rain, previous 30 min"),
    ]
    state = features.filter(F.col("ts") == F.lit(origin)).select(
        "station_id", "lat", "lon", *[name for name, _ in state_columns]
    ).toPandas()
    _save_table(state, "geo_selected_event_feature_state.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.3), constrained_layout=True)
    for index, (name, title) in enumerate(state_columns):
        mapped_state = aggregate_h3(state, name, resolution)
        values = mapped_state[name].dropna()
        if name == "intensity_change_30m_mm":
            bound = max(float(values.abs().quantile(0.98)), 0.01)
            state_norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
            state_cmap = plt.get_cmap("RdBu")
        else:
            state_norm = mcolors.Normalize(0.0, max(float(values.quantile(0.98)), 0.01))
            state_cmap = plt.get_cmap("cividis")
        draw_h3(axes.flat[index], mapped_state, name, state_cmap, state_norm, title)
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=state_norm, cmap=state_cmap),
            ax=axes.flat[index], fraction=0.046, pad=0.02,
        )
        if name.endswith("wet_fraction"):
            cb.set_label("Fraction")
        else:
            cb.set_label("mm")
    fig.suptitle(f"Antecedent rainfall state at {origin:%Y-%m-%d %H:%M} SGT", fontweight="bold")
    _save_figure(fig, "geo_G8_G9_antecedent_intensification.png")
    return origin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Clean observation Parquet")
    parser.add_argument("--features", default=None, help="Spatiotemporal feature Parquet")
    parser.add_argument("--daily-completeness", type=float, default=0.90)
    parser.add_argument("--h3-resolution", type=int, default=7)
    parser.add_argument("--target-month", default=None, help="Optional YYYY-MM override")
    parser.add_argument("--event-time", default=None, help="Optional ISO local-time override")
    parser.add_argument("--minimum-event-stations", type=int, default=20)
    parser.add_argument("--hash-mode", choices=["full", "metadata"], default="full")
    parser.add_argument("--publish-legacy", action="store_true")
    args = parser.parse_args()

    if not 0 < args.daily_completeness <= 1:
        parser.error("--daily-completeness must lie in (0, 1]")
    if not 0 <= args.h3_resolution <= 15:
        parser.error("--h3-resolution must lie between 0 and 15")
    config.ensure_dirs()
    args.data = args.data or latest_artifact("clean", "clean_parquet") or config.CLEAN_PARQUET
    args.features = (
        args.features
        or latest_artifact("spatiotemporal_features", "spatiotemporal_features")
        or os.path.join(config.PROCESSED_DIR, "spatiotemporal_features")
    )
    run = RunContext("task1_geospatial", vars(args), hash_mode=args.hash_mode)
    run.record_input("clean_parquet", args.data)
    run.record_input("spatiotemporal_features", args.features)
    global FIGURES_DIR, TABLES_DIR
    FIGURES_DIR = run.artifact_path("figures")
    TABLES_DIR = run.artifact_path("tables")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    _set_style()

    spark = get_spark("task1_geospatial", driver_memory="12g", shuffle_partitions=192)
    observations = spark.read.parquet(args.data)
    features = spark.read.parquet(args.features)
    daily = eligible_station_days(observations, args.daily_completeness)

    print("[1/5] observation network and continuity")
    observation_network_figure(observations)
    print("[2/5] seasonal H3 geography")
    seasonal_geography_figure(daily, args.h3_resolution)
    print("[3/5] multi-baseline monthly anomaly")
    monthly = station_month_frame(daily)
    selected_month = anomaly_dashboard(monthly, args.h3_resolution, args.target_month)
    print("[4/5] lagged storm footprint")
    print("[5/5] antecedent state and intensification")
    selected_event = event_figures(
        features, args.h3_resolution, args.minimum_event_stations, args.event_time
    )

    selection = {
        "daily_completeness": args.daily_completeness,
        "h3_resolution": args.h3_resolution,
        "target_month": str(selected_month),
        "event_origin_sgt": selected_event.isoformat(sep=" "),
        "target_month_rule": "largest positive same-month prior-year z-score unless overridden",
        "event_rule": "largest network mean complete 30-minute accumulation unless overridden",
    }
    run.write_json("geospatial_selection", selection)
    spark.stop()
    run.record_artifact("figures", FIGURES_DIR)
    run.record_artifact("tables", TABLES_DIR)
    if args.publish_legacy:
        publish_tree(FIGURES_DIR, config.FIGURES_DIR)
        publish_tree(TABLES_DIR, config.TABLES_DIR)
    run.finalize()
    print(f"tracked geospatial Task 1 run: {run.run_id} -> {run.path}")


if __name__ == "__main__":
    main()

