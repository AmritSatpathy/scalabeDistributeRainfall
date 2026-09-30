# RATIONALE — the "why" behind every decision

This document explains *why* each choice was made in the pipeline: the data
handling, the cleaning rules, each Task 1 visualisation, the feature design,
and the Task 2 model. It complements the `README.md` (which says *how* to run)
and `PROGRESS.md` (which says *what* is done).

---

## 1. Data acquisition

**Why download via the data.gov.sg poll-download API instead of manual CSVs.**
The portal serves each yearly dataset through a two-step signed-URL flow
(`poll-download` → S3 URL). Scripting it (`src/download_data.py`) makes the
pipeline reproducible end-to-end: anyone can regenerate the exact inputs
without hunting for files, and the dataset ids are pinned in `src/config.py`.

**Why only 2017–2024.** The collection also contains a partial 2016 dataset
(coverage starts 2 Dec 2016). The assignment specifies 2017–2024, and including
a two-day 2016 fragment would distort annual comparisons, so 2016 is excluded.

**Why keep the raw CSVs out of git.** ~7.8 GB of text is not appropriate for a
repository and is fully regenerable from the API. `.gitignore` excludes
`data/`, and the README documents how to recreate it.

---

## 2. Cleaning decisions (`src/clean_data.py`)

**Why convert CSV → Parquet.** The source is ~7.8 GB of repetitive text.
Parquet is columnar and compressed, so it is far smaller on disk, far faster to
scan, and preserves types. Partitioning by `year` lets Spark prune whole years
when a query only needs some (partition pruning), which speeds up both tasks.

**Why pin the Spark timezone to `Asia/Singapore`.** All source timestamps are
SGT (ISO-8601 with `+08:00`). If Spark parsed them under a different session
zone, the derived `hour`/`date`/diurnal features would silently shift. Pinning
SGT guarantees every calendar feature and the 30-minute label align with the
real local time the assignment cares about.

**Why cast `reading_value` and validate it.**
- *Cast to double:* it arrives as text; arithmetic and ML need a numeric type.
- *Drop nulls:* a missing rainfall value cannot be used or imputed reliably for
  a 5-minute total.
- *Drop negatives:* rainfall cannot be negative; a negative is a data error.
- *Drop > 60 mm in 5 min:* even Singapore's most intense bursts stay well below
  this; a larger 5-minute total is a sensor/encoding error, not weather.
  Removing it prevents a single bad record from dominating sums and the model.

**Why de-duplicate on `(station_id, timestamp)` keeping the latest
`update_timestamp`.** The feed can re-issue a reading as corrections arrive.
Keeping the most recently updated record uses the best available value and
guarantees exactly one reading per station per 5-minute slot — essential for
correct rolling sums and for the diurnal/seasonal aggregations.

**Why resolve one canonical station name + median coordinates per
`station_id`.** In the raw data a station's `station_name` alternates between
its real name (e.g. "Alexandra Road") and its bare id (e.g. "S77"), while the
coordinates are stable. We pick the real name (preferring non-id-like strings)
so tables/plots are human-readable, and take the **median** lat/lon so a rare
stray coordinate can't move a station on the map. Coordinates are needed for
the spatial analysis and as model features.

**Why compute calendar parts once during cleaning.** `year, month, day, hour,
minute, dayofweek, dayofyear, date` are needed repeatedly by both tasks.
Deriving them once in the clean step avoids recomputing them in every
downstream job and keeps the analysis code simple.

---

## 3. Task 1 — why each visualisation exists

The goal is to show how rainfall changes **within** a year and **across**
2017–2024, plus its spatial pattern. Each figure answers a specific question.

**Per-station normalisation (used throughout).** The station network grew from
~54 stations (2019) to ~80 (2020). Raw totals would rise just because there are
more sensors. Averaging per-station figures removes that bias so year-to-year
comparisons reflect weather, not coverage.

- **`annual_rainfall.png` (annual totals + station count).** Answers "is it
  getting wetter across the years?" The overlaid station count is shown
  precisely so the reader can see totals are normalised and not an artefact of
  network growth.
- **`monthly_heatmap.png` (year × month).** Answers "how does the seasonal
  cycle differ *between* years?" A heatmap makes anomalous months (unusually wet
  or dry) pop out at a glance across all 8 years.
- **`monthly_climatology.png` (avg month, monsoon shading).** Answers "what is
  the typical seasonal shape?" Shading the NE (Dec–Mar) and SW (Jun–Sep)
  monsoon windows ties the statistics to Singapore's known climate drivers.
- **`diurnal_cycle.png` (hour of day).** Answers "when in the day does it
  rain?" Two axes — mean intensity *and* wet-reading probability — separate
  "how hard" from "how often," both of which peak in the afternoon.
- **`diurnal_by_season.png`.** Answers "does the time-of-day pattern change by
  monsoon?" Useful because NE-monsoon rain (often morning/nocturnal) differs
  from the afternoon-convective SW/inter-monsoon pattern.
- **`station_map.png` (lon/lat coloured by rainfall).** Answers "is rainfall
  spatially uneven across the island?" A geographic scatter is the natural way
  to reveal wetter/drier regions.
- **`station_top_bottom.png` (wettest/driest 10).** Answers "which specific
  stations are extreme?" Named bars make the spatial story concrete.
- **`heavy_rain_days.png` (station-days ≥ 50 mm/year).** Answers "are heavy
  events becoming more frequent?" Extreme-event frequency matters more for
  flooding/impact than the annual mean, so it gets its own view.
- **`wet_intensity_quantiles.csv`.** Quantifies the sparsity/skew of 5-minute
  intensities (median wet reading ≈ 0.2 mm, p99.9 ≈ 10.6 mm), which directly
  justifies the class-imbalance handling in Task 2.

**Why aggregate in Spark but plot in matplotlib.** The raw data is ~48 M rows —
too big to plot directly. Spark does the heavy grouping; only the small summary
tables are pulled to the driver for plotting. This keeps memory bounded and the
figures fast to regenerate.

---

## 4. Task 2 — why the problem is framed this way

**Why treat the threshold as a model *feature* rather than training one model
per threshold.** The assignment requires the threshold to be a *user input* at
prediction time. Encoding it as a feature and generating one labelled example
per candidate threshold lets a **single** trained model answer *any* threshold,
including values not in the training set (it interpolates). Training a separate
model per threshold would not generalise to arbitrary user-supplied values and
would multiply maintenance/storage.

**Why a 30-minute forward-sum label.** The task is "more than X mm in the next
30 minutes." With 5-minute readings, that is exactly the sum of the next six
intervals — a direct, unambiguous target with no smoothing assumptions.

**Why require the full future window (`future_n >= 6`).** If some of the next
six readings are missing, the 30-minute accumulation is under-counted and the
label would be wrong (biased toward "no rain"). Dropping incomplete windows
keeps labels honest.

### Feature choices (all leakage-free — computed only from data up to the
prediction time)

- **Recent-rainfall context (`rain_mm`, `lag_5m_mm`, rolling 30/60/180-min
  sums, `mins_since_rain`).** Rain is strongly autocorrelated at short lead
  times: if it is raining now, it very likely rains in the next 30 min. The
  inference sanity-check confirmed this is the dominant signal (dry context →
  ~3% probability, wet context → 70–92%). These are the most predictive
  features and are the core of a nowcast.
- **Cyclical hour/month (`sin`/`cos`).** Time-of-day and season matter (Task 1
  showed a strong afternoon peak and monsoon structure). Sine/cosine encoding
  represents the *cyclic* nature (hour 23 is adjacent to hour 0; December is
  adjacent to January) without a false ordinal jump.
- **Monsoon + weekend flags.** Cheap categorical signals for the seasonal
  regime and any weekday/weekend differences.
- **Station identity + location (`station_idx`, `lat`, `lon`).** Different parts
  of the island have different rainfall climatology (Task 1's map). Location and
  station id let the model learn site-specific behaviour.

**Why materialise features once (`task2_build_features.py`) before training.**
The rolling/forward windows are expensive per-station time-series computations.
If left inside the training script, Spark recomputed them for *every* action
(count, fit, transform, each per-threshold evaluation), which was extremely slow
and caused out-of-memory kills. Writing the base features to Parquet once and
reading them back is the standard "compute-once, reuse-many" Spark pattern.

### Model choice

**Why Gradient-Boosted Trees as the primary model.**
- The signal is **non-linear and interaction-heavy**: e.g. "heavy recent rain
  *and* afternoon *and* SW-monsoon" is far more predictive than any feature
  alone. Trees capture interactions automatically; a linear model cannot without
  manual cross-terms.
- Features are on **very different scales** (mm vs hours vs lat/lon) and rain is
  highly skewed. Trees are scale-invariant and robust to skew/outliers, so no
  careful scaling is required.
- GBT usually **outperforms a single tree or random forest** on tabular data of
  this kind, and Spark MLlib provides a distributed implementation that scales
  to the cluster later.

**Why also a Logistic Regression baseline.** A simple, fast linear model gives a
reference point: if GBT only marginally beats it, the extra complexity is not
justified; if it beats it clearly (expected here), that demonstrates the value
of modelling non-linear interactions. Good practice for the report.

### Why GBT and not an RNN/LSTM (and why the GPU does not change this)

This is a genuinely sequential, autocorrelated problem, so a recurrent model
(LSTM/GRU) is a legitimate — arguably more natural — contender. We are honest
about that: GBT is chosen primarily because it is the **required-library**
choice, not because RNNs are unsuitable.

- **The assignment requires Spark MLlib, and MLlib itself has no RNN/LSTM
  model class.** This is a statement about the *library*, not about Spark in
  general. `pyspark.ml`'s only neural model is a plain feedforward
  `MultilayerPerceptronClassifier`; there is no recurrent layer. LSTMs *can*
  absolutely be used in a Spark-based workflow — either by using Spark to
  clean/engineer/window the data and then training the LSTM in TensorFlow/
  PyTorch (on GPU), or via bridge libraries that run Keras/TF/PyTorch on a Spark
  cluster (Elephas, BigDL/Analytics-Zoo, TensorFlowOnSpark, Horovod-on-Spark,
  `spark-tensorflow-distributor`). The point is only that the LSTM *model* would
  then live outside `pyspark.ml`. GBT keeps the graded model strictly inside
  MLlib as the brief asks.
- **A CUDA GPU does not accelerate Spark MLlib.** MLlib is CPU/JVM-based; GBT and
  Logistic Regression do not use CUDA at all, so an NVIDIA GPU sits idle for the
  MLlib pipeline. Actually using the GPU would require training outside Spark
  (e.g. a PyTorch LSTM) or a GPU-accelerated stack (XGBoost-on-Spark with GPU,
  or the NVIDIA RAPIDS Spark plugin) — all of which are extra infrastructure,
  not MLlib. In short: the GPU is only useful here if we deliberately step
  outside the required library.
- **The threshold-as-feature trick is trivial for GBT** (just another input
  column, so one model serves any threshold) but more awkward for an RNN, which
  would need the threshold fused into the final dense layer or per-threshold
  models.
- **Where an RNN would genuinely help** (and where we recommend it as future
  work): longer prediction horizons where temporal *evolution* matters more than
  the current state; and a **multivariate spatiotemporal** model that ingests all
  stations jointly (e.g. ConvLSTM or graph-temporal networks) to learn how storm
  cells move across the island — this is what operational precipitation nowcasting
  systems use, and it is the setting where deep learning clearly beats tabular
  GBT. With gridded radar/satellite input rather than point gauges, CNN/ConvLSTM
  would dominate outright.
- **Planned comparison.** To test this properly rather than assert it, the
  intended extension is an **optional PyTorch LSTM (`task2_lstm.py`) trained on
  the GPU**, using the same temporal features, the same 30-minute label, the same
  temporal train/test split, and the same AUC-ROC / AUC-PR metrics — a like-for-
  like benchmark against the MLlib GBT. GBT remains the graded MLlib deliverable;
  the LSTM is supplementary evidence for the report's model-comparison and
  future-work discussion.

> **Bottom line.** "LSTMs are not for Spark" would be *wrong* — Spark is
> routinely paired with LSTMs (via TF/PyTorch or bridge libraries), and Spark
> can do the large-scale data prep for them. The precise, correct statement is:
> **Spark's MLlib library has no LSTM model class**, so an LSTM would be a
> non-MLlib model even in a Spark workflow. Given the brief mandates MLlib and
> the GPU cannot accelerate MLlib anyway, GBT is the right *graded* choice; the
> LSTM is the right *comparison/future-work* choice, ideally benchmarked head-to
> -head on the GPU.

**Why *not* one-hot encode the station.** With ~91 stations, one-hot makes the
feature vector wide and forces GBT to densify it, which was memory-heavy.
Keeping station as a single indexed categorical column (with `maxBins ≥ 91` so
the tree can still split on it) — plus explicit `lat`/`lon` — captures site
identity far more cheaply.

**Why `maxBins = 128`.** Spark trees require `maxBins ≥ the number of categories
in any categorical feature`. With ~91 stations, a smaller value errors out. 128
comfortably covers the station count with headroom.

### Evaluation choices

**Why a temporal train/test split (train on earlier years, test on the most
recent year) instead of a random split.** A random split leaks future
information into training and inflates metrics. Forecasting is inherently
temporal — you predict the future from the past — so testing on the held-out
latest year mirrors real deployment and gives an honest estimate.

**Why report AUC-ROC *and* AUC-PR (and per-threshold).** Rain is rare,
especially at high thresholds (positives are a small minority). Accuracy is
misleading here — always predicting "no heavy rain" scores high. **AUC-ROC**
measures ranking/discrimination independent of the cut-off; **AUC-PR** is the
honest metric under heavy class imbalance because it focuses on the rare
positive class. Reporting them **per threshold** shows how difficulty changes
(higher thresholds → rarer positives → lower AUC-PR even as AUC-ROC stays high).

**Why the `--max-train-rows` cap for the local runs.** The threshold-explode
multiplies rows 5×, so the full training set is ~200 M+ rows — beyond a single
machine's memory for GBT. Capping to a random, class-proportional subsample
keeps the local fit feasible while preserving the label distribution in
expectation. The uncapped full run is reserved for the HPC cluster (see
`PROGRESS.md`).

---

## 5. Engineering choices worth noting

- **`local[*]` with a large driver + `MEMORY_AND_DISK` persistence.** A
  single-node setup is simplest for development and reproducible for the grader;
  disk-backed persistence lets Spark spill instead of failing when memory is
  tight. The code is plain PySpark, so it lifts to a cluster with only path and
  master changes.
- **Fixed random seeds.** Sampling and the classifiers use fixed seeds so runs
  are reproducible and reported numbers are stable.
- **Small tables to CSV, big data to Parquet.** Human-readable summaries
  (metrics, station list) go to `output/tables` as CSV for the report; bulk
  intermediate data stays in Parquet for efficiency.
