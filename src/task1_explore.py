"""
Task 1 - Explore Singapore rainfall weather patterns (2017-2024).

Reads the cleaned Parquet dataset and produces a set of aggregation tables
(CSV, under output/tables) and figures (PNG, under output/figures) that
describe how rainfall behaves *within* a year (seasonal / diurnal cycles)
and *across* the years 2017-2024 (annual totals and trends), plus the
spatial pattern across the ~70 monitoring stations.

All heavy aggregation is done in Spark; only the small summary tables are
pulled to the driver (pandas) for plotting with matplotlib.

Usage
-----
    python src/task1_explore.py
    python src/task1_explore.py --data data/processed/rainfall_clean
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from spark_utils import get_spark, save_table  # noqa: E402
from run_artifacts import RunContext, latest_artifact, publish_tree  # noqa: E402

from pyspark.sql import functions as F  # noqa: E402

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ACTIVE_FIGURES_DIR = config.FIGURES_DIR
ACTIVE_TABLES_DIR = config.TABLES_DIR


def fig_path(name: str) -> str:
    return os.path.join(ACTIVE_FIGURES_DIR, name)


def tbl_path(name: str) -> str:
    return os.path.join(ACTIVE_TABLES_DIR, name)


# ---------------------------------------------------------------------------
# 1. Annual totals & trend across years
# ---------------------------------------------------------------------------
def annual_summary(df):
    """Mean per-station annual rainfall + data completeness per year."""
    # Total rainfall per station per year, then average across stations so
    # the annual figure is not biased by how many stations exist that year.
    per_station = df.groupBy("year", "station_id").agg(
        F.sum("rain_mm").alias("station_total_mm"),
        F.count(F.lit(1)).alias("n_readings"),
    )
    annual = (
        per_station.groupBy("year")
        .agg(
            F.avg("station_total_mm").alias("mean_station_annual_mm"),
            F.expr("percentile_approx(station_total_mm, 0.5)").alias("median_station_annual_mm"),
            F.countDistinct("station_id").alias("n_stations"),
            F.sum("n_readings").alias("total_readings"),
        )
        .orderBy("year")
    )
    pdf = annual.toPandas()
    save_table(pdf, tbl_path("annual_summary.csv"))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(pdf["year"], pdf["mean_station_annual_mm"], color="#3b7dd8", alpha=0.8)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean per-station annual rainfall (mm)", color="#3b7dd8")
    ax1.tick_params(axis="y", labelcolor="#3b7dd8")
    ax2 = ax1.twinx()
    ax2.plot(pdf["year"], pdf["n_stations"], "o-", color="#d8663b", label="# stations")
    ax2.set_ylabel("Active stations", color="#d8663b")
    ax2.tick_params(axis="y", labelcolor="#d8663b")
    plt.title("Annual rainfall and station coverage, Singapore 2017-2024")
    fig.tight_layout()
    fig.savefig(fig_path("annual_rainfall.png"), dpi=130)
    plt.close(fig)
    return pdf


# ---------------------------------------------------------------------------
# 2. Monthly / seasonal cycle within a year, per year (monsoon signal)
# ---------------------------------------------------------------------------
def monthly_cycle(df):
    per_station_month = df.groupBy("year", "month", "station_id").agg(
        F.sum("rain_mm").alias("station_month_mm")
    )
    monthly = (
        per_station_month.groupBy("year", "month")
        .agg(F.avg("station_month_mm").alias("mean_station_month_mm"))
        .orderBy("year", "month")
    )
    pdf = monthly.toPandas()
    save_table(pdf, tbl_path("monthly_cycle.csv"))

    # Heatmap: year x month
    piv = pdf.pivot(index="year", columns="month", values="mean_station_month_mm")
    piv = piv.reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title("Mean per-station monthly rainfall (mm)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mm")
    fig.tight_layout()
    fig.savefig(fig_path("monthly_heatmap.png"), dpi=130)
    plt.close(fig)

    # Average monthly climatology (mean across years) to show the monsoon shape
    clim = (
        pdf.groupby("month")["mean_station_month_mm"].mean().reindex(range(1, 13))
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(MONTHS, clim.values, color="#2a9d8f")
    ax.set_ylabel("Mean per-station monthly rainfall (mm)")
    ax.set_title("Average monthly rainfall climatology, 2017-2024")
    # Shade monsoon seasons: NE monsoon (Dec-Mar), SW monsoon (Jun-Sep)
    ax.axvspan(-0.5, 2.5, color="#264653", alpha=0.08)
    ax.axvspan(11 - 0.5, 11.5, color="#264653", alpha=0.08)
    ax.axvspan(5 - 0.5, 8.5, color="#e9c46a", alpha=0.15)
    fig.tight_layout()
    fig.savefig(fig_path("monthly_climatology.png"), dpi=130)
    plt.close(fig)
    return pdf


# ---------------------------------------------------------------------------
# 3. Diurnal cycle (hour of day) - when does it rain?
# ---------------------------------------------------------------------------
def diurnal_cycle(df):
    # Average rainfall intensity by hour, and probability a 5-min reading is wet.
    hourly = (
        df.groupBy("hour")
        .agg(
            F.avg("rain_mm").alias("mean_rain_mm"),
            F.avg((F.col("rain_mm") > 0).cast("double")).alias("wet_fraction"),
        )
        .orderBy("hour")
    )
    pdf = hourly.toPandas()
    save_table(pdf, tbl_path("diurnal_cycle.csv"))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(pdf["hour"], pdf["mean_rain_mm"], color="#457b9d", alpha=0.8)
    ax1.set_xlabel("Hour of day (SGT)")
    ax1.set_ylabel("Mean rainfall per 5-min reading (mm)", color="#457b9d")
    ax1.tick_params(axis="y", labelcolor="#457b9d")
    ax2 = ax1.twinx()
    ax2.plot(pdf["hour"], 100 * pdf["wet_fraction"], "o-", color="#e63946")
    ax2.set_ylabel("Wet-reading probability (%)", color="#e63946")
    ax2.tick_params(axis="y", labelcolor="#e63946")
    ax1.set_xticks(range(0, 24, 2))
    plt.title("Diurnal rainfall cycle, Singapore 2017-2024")
    fig.tight_layout()
    fig.savefig(fig_path("diurnal_cycle.png"), dpi=130)
    plt.close(fig)
    return pdf


# ---------------------------------------------------------------------------
# 4. Diurnal cycle by season (NE vs SW monsoon vs inter-monsoon)
# ---------------------------------------------------------------------------
def diurnal_by_season(df):
    season = (
        F.when(F.col("month").isin(12, 1, 2, 3), "NE monsoon (Dec-Mar)")
        .when(F.col("month").isin(6, 7, 8, 9), "SW monsoon (Jun-Sep)")
        .otherwise("Inter-monsoon (Apr,May,Oct,Nov)")
    )
    d = df.withColumn("season", season)
    hourly = (
        d.groupBy("season", "hour")
        .agg(F.avg("rain_mm").alias("mean_rain_mm"))
        .orderBy("season", "hour")
    )
    pdf = hourly.toPandas()
    save_table(pdf, tbl_path("diurnal_by_season.csv"))

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, g in pdf.groupby("season"):
        ax.plot(g["hour"], g["mean_rain_mm"], "o-", label=name, markersize=3)
    ax.set_xlabel("Hour of day (SGT)")
    ax.set_ylabel("Mean rainfall per 5-min reading (mm)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.set_title("Diurnal rainfall cycle by monsoon season")
    fig.tight_layout()
    fig.savefig(fig_path("diurnal_by_season.png"), dpi=130)
    plt.close(fig)
    return pdf


# ---------------------------------------------------------------------------
# 5. Spatial pattern: wettest / driest stations
# ---------------------------------------------------------------------------
def spatial_pattern(df):
    per_station = (
        df.groupBy("station_id", "station_name", "lat", "lon")
        .agg(
            F.sum("rain_mm").alias("total_mm"),
            F.countDistinct("year").alias("n_years"),
            F.avg((F.col("rain_mm") > 0).cast("double")).alias("wet_fraction"),
        )
    )
    # Normalise to a per-year average so stations with fewer years are comparable
    per_station = per_station.withColumn(
        "avg_annual_mm", F.col("total_mm") / F.col("n_years")
    )
    pdf = per_station.orderBy(F.col("avg_annual_mm").desc()).toPandas()
    save_table(pdf, tbl_path("station_rainfall.csv"))

    # Scatter map coloured by average annual rainfall
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        pdf["lon"], pdf["lat"], c=pdf["avg_annual_mm"], cmap="viridis",
        s=60, edgecolor="k", linewidth=0.3,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Average annual rainfall by station (map)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Average annual rainfall (mm)")
    fig.tight_layout()
    fig.savefig(fig_path("station_map.png"), dpi=130)
    plt.close(fig)

    # Top / bottom 10 bar chart
    top = pdf.head(10)
    bot = pdf.tail(10)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(top["station_name"][::-1], top["avg_annual_mm"][::-1], color="#1d3557")
    axes[0].set_title("Top 10 wettest stations")
    axes[0].set_xlabel("Avg annual rainfall (mm)")
    axes[1].barh(bot["station_name"][::-1], bot["avg_annual_mm"][::-1], color="#a8dadc")
    axes[1].set_title("10 driest stations")
    axes[1].set_xlabel("Avg annual rainfall (mm)")
    fig.tight_layout()
    fig.savefig(fig_path("station_top_bottom.png"), dpi=130)
    plt.close(fig)
    return pdf


# ---------------------------------------------------------------------------
# 6. Rain intensity distribution & heavy-rain days
# ---------------------------------------------------------------------------
def intensity_distribution(df):
    wet = df.filter(F.col("rain_mm") > 0)
    # Percentiles of wet 5-min intensities
    q = wet.select(
        F.expr("percentile_approx(rain_mm, array(0.5,0.9,0.95,0.99,0.999))").alias("q")
    ).collect()[0]["q"]
    labels = ["p50", "p90", "p95", "p99", "p99.9"]
    import pandas as pd
    pdf_q = pd.DataFrame({"quantile": labels, "rain_mm_5min": q})
    save_table(pdf_q, tbl_path("wet_intensity_quantiles.csv"))

    # Daily station rainfall -> count "heavy rain" days (>=50mm) per year
    daily = df.groupBy("year", "date", "station_id").agg(
        F.sum("rain_mm").alias("daily_mm")
    )
    heavy = (
        daily.withColumn("heavy_day", (F.col("daily_mm") >= 50).cast("int"))
        .groupBy("year")
        .agg(
            F.sum("heavy_day").alias("station_days_over_50mm"),
            F.max("daily_mm").alias("max_station_daily_mm"),
        )
        .orderBy("year")
    )
    pdf_h = heavy.toPandas()
    save_table(pdf_h, tbl_path("heavy_rain_days.csv"))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(pdf_h["year"], pdf_h["station_days_over_50mm"], color="#6a4c93")
    ax.set_xlabel("Year")
    ax.set_ylabel("Station-days with >= 50 mm")
    ax.set_title("Heavy-rain station-days per year (daily total >= 50 mm)")
    fig.tight_layout()
    fig.savefig(fig_path("heavy_rain_days.png"), dpi=130)
    plt.close(fig)
    return pdf_q, pdf_h


def coverage_normalised_analysis(df, completeness_threshold=0.95,
                                 stable_year_coverage=0.90):
    """Coverage audit, stable cohort, and heavy-rain rates with denominators."""
    expected_per_day = 24 * 60 // config.INTERVAL_MINUTES
    daily = (
        df.groupBy("year", "date", "station_id")
        .agg(F.count("*").alias("observed_intervals"), F.sum("rain_mm").alias("daily_mm"))
        .withColumn("expected_intervals", F.lit(expected_per_day))
        .withColumn("completeness", F.col("observed_intervals") / F.col("expected_intervals"))
        .withColumn("eligible", F.col("completeness") >= F.lit(completeness_threshold))
    )
    annual_coverage = (
        daily.groupBy("year").agg(
            F.count("*").alias("observed_station_days"),
            F.sum(F.col("eligible").cast("long")).alias("eligible_station_days"),
            F.sum("observed_intervals").alias("observed_intervals"),
            F.sum("expected_intervals").alias("expected_intervals"),
            F.countDistinct("station_id").alias("stations_observed"),
        ).withColumn("interval_coverage", F.col("observed_intervals") / F.col("expected_intervals"))
        .orderBy("year")
    )
    station_year = (
        df.groupBy("year", "station_id").agg(
            F.count("*").alias("observed_intervals"),
            F.sum("rain_mm").alias("annual_mm"),
        ).withColumn(
            "days_in_year",
            F.dayofyear(F.to_date(F.concat(F.col("year"), F.lit("-12-31"))))
        ).withColumn("expected_intervals", F.col("days_in_year") * expected_per_day)
        .withColumn("year_coverage", F.col("observed_intervals") / F.col("expected_intervals"))
    )
    stable_ids = (
        station_year.groupBy("station_id").agg(
            F.countDistinct("year").alias("years_present"),
            F.min("year_coverage").alias("min_year_coverage"),
        ).filter(
            (F.col("years_present") == len(config.YEARS))
            & (F.col("min_year_coverage") >= F.lit(stable_year_coverage))
        ).select("station_id")
    )
    all_annual = station_year.groupBy("year").agg(
        F.avg("annual_mm").alias("all_station_mean_mm")
    )
    stable_annual = (
        station_year.join(F.broadcast(stable_ids), "station_id", "inner")
        .groupBy("year").agg(
            F.avg("annual_mm").alias("stable_cohort_mean_mm"),
            F.countDistinct("station_id").alias("stable_station_count"),
        )
    )
    comparison = all_annual.join(stable_annual, "year", "left").orderBy("year")
    eligible_daily = daily.filter("eligible")
    heavy = (
        eligible_daily.groupBy("year").agg(
            F.count("*").alias("eligible_station_days"),
            F.sum((F.col("daily_mm") >= 50).cast("long")).alias("heavy_station_days"),
        ).withColumn(
            "heavy_station_days_per_1000",
            1000.0 * F.col("heavy_station_days") / F.col("eligible_station_days"),
        ).orderBy("year")
    )
    sensitivity = None
    for threshold in (0.90, 0.95, 0.99):
        part = (
            daily.groupBy("year").agg(
                F.sum((F.col("completeness") >= threshold).cast("long")).alias("eligible_station_days"),
                F.sum(((F.col("completeness") >= threshold) & (F.col("daily_mm") >= 50)).cast("long")).alias("heavy_station_days"),
            ).withColumn("completeness_threshold", F.lit(threshold))
            .withColumn("heavy_per_1000", 1000.0 * F.col("heavy_station_days") / F.col("eligible_station_days"))
        )
        sensitivity = part if sensitivity is None else sensitivity.unionByName(part)

    pdf_cov = annual_coverage.toPandas()
    pdf_cmp = comparison.toPandas()
    pdf_heavy = heavy.toPandas()
    save_table(pdf_cov, tbl_path("coverage_by_year.csv"))
    save_table(pdf_cmp, tbl_path("annual_stable_cohort.csv"))
    save_table(pdf_heavy, tbl_path("heavy_rain_rate_normalised.csv"))
    save_table(sensitivity.orderBy("completeness_threshold", "year").toPandas(),
               tbl_path("coverage_sensitivity.csv"))
    save_table(stable_ids.orderBy("station_id").toPandas(), tbl_path("stable_cohort_stations.csv"))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(pdf_cmp["year"], pdf_cmp["all_station_mean_mm"], "o-", label="All observed stations")
    ax.plot(pdf_cmp["year"], pdf_cmp["stable_cohort_mean_mm"], "s--", label="Stable cohort")
    ax.set_xlabel("Year"); ax.set_ylabel("Mean annual rainfall per station (mm)")
    ax.set_title("Annual rainfall: all stations versus stable-coverage cohort")
    ax.legend(); fig.tight_layout()
    fig.savefig(fig_path("annual_rainfall_coverage_adjusted.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(pdf_heavy["year"], pdf_heavy["heavy_station_days_per_1000"], color="#6a4c93")
    ax.set_xlabel("Year"); ax.set_ylabel("Heavy station-days per 1,000 eligible station-days")
    ax.set_title(f"Coverage-normalised heavy rainfall (daily completeness ≥ {completeness_threshold:.0%})")
    fig.tight_layout(); fig.savefig(fig_path("heavy_rain_rate_normalised.png"), dpi=160); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="Clean Parquet. Default: latest tracked clean run, then legacy path.")
    ap.add_argument("--daily-completeness", type=float, default=0.95)
    ap.add_argument("--stable-year-coverage", type=float, default=0.90)
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="full")
    ap.add_argument("--publish-legacy", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    args.data = args.data or latest_artifact("clean", "clean_parquet") or config.CLEAN_PARQUET
    run = RunContext("task1", vars(args), hash_mode=args.hash_mode)
    global ACTIVE_FIGURES_DIR, ACTIVE_TABLES_DIR
    ACTIVE_FIGURES_DIR = run.artifact_path("figures")
    ACTIVE_TABLES_DIR = run.artifact_path("tables")
    os.makedirs(ACTIVE_FIGURES_DIR, exist_ok=True)
    os.makedirs(ACTIVE_TABLES_DIR, exist_ok=True)
    run.record_input("clean_parquet", args.data)
    spark = get_spark("task1_explore", driver_memory="11g")
    df = spark.read.parquet(args.data)
    df = df.cache()
    total = df.count()
    print(f"Loaded {total:,} cleaned readings")

    print("[1/6] annual summary...");    annual_summary(df)
    print("[2/6] monthly cycle...");      monthly_cycle(df)
    print("[3/6] diurnal cycle...");      diurnal_cycle(df)
    print("[4/6] diurnal by season..."); diurnal_by_season(df)
    print("[5/6] spatial pattern...");    spatial_pattern(df)
    print("[6/6] intensity dist...");     intensity_distribution(df)
    print("[7/7] coverage-normalised checks...")
    coverage_normalised_analysis(df, args.daily_completeness, args.stable_year_coverage)

    print(f"Task 1 complete. Immutable artifacts: {run.artifacts_dir}")
    spark.stop()
    run.record_artifact("figures", ACTIVE_FIGURES_DIR)
    run.record_artifact("tables", ACTIVE_TABLES_DIR)
    if args.publish_legacy:
        publish_tree(ACTIVE_FIGURES_DIR, config.FIGURES_DIR)
        publish_tree(ACTIVE_TABLES_DIR, config.TABLES_DIR)
    run.finalize()


if __name__ == "__main__":
    main()
