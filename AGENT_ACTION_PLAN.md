# Agent Action Plan — Rainfall Project Gap Closure

## 1. Purpose

This is the implementation and submission plan for closing the gaps between
`project_info.md`, the current documentation, the code in `src/`, and the
tracked artifacts. It is written as an agent handoff: complete the phases in
order, record evidence for every acceptance criterion, and do not write the
final report around metrics that have not survived the correctness checks in
Phase 1.

The assignment requires:

1. Spark-based cleaning and exploration of Singapore rainfall from 2017–2024.
2. A Spark MLlib model that estimates
   `P(next 30-minute accumulation > user threshold)` from station, prediction
   time, and threshold.
3. Model evaluation, figures/tables, environment details, limitations, AI-use
   disclosure, complete source, a run guide, a trained model, and a PDF report.

## 2. Current-state assessment

### What is genuinely present

- All pipeline scripts exist and pass `python3 -m compileall`.
- Eight Task 1 figures, Task 2 comparison output, summary CSVs, a fitted Spark
  `PipelineModel`, and `models/model_metadata.json` are tracked.
- The metadata records a 2017–2023 train / 2024 test run with 7,997,496 capped
  training rows and 31,872,380 test rows.
- The current GBT result is AUC-ROC 0.9245 and AUC-PR 0.5841 overall, with
  per-threshold results available in the metadata and CSV outputs.
- Data acquisition, cleaning, feature materialisation, training, evaluation,
  plotting, and historical inference paths have all been implemented.

### What is not currently available or complete

- `data/` is absent from this checkout, so full-data behavioural checks and
  corrected reruns must be executed in the documented WSL/data environment.
- `report/` is absent; no PDF-ready report exists.
- Group names and student IDs are not recorded.
- There is no automated test suite, CI check, data-quality report, probability
  calibration analysis, or monotonic-threshold validation.
- The current artifacts predate the correctness improvements below and must be
  treated as provisional until regenerated.

### Documentation contradictions to resolve

- `PROGRESS.md:4-8` says the full local run is done, but `PROGRESS.md:18-21`
  says it is not done.
- `task2_train.py:6-8` still says station identity is one-hot encoded, although
  the implementation uses only `StringIndexer`.
- `task2_train.py:15` names a different model output path from the code.
- The progress file calls Task 1 and Task 2 verified. They are operational,
  but the methodological checks below remain open.

## 3. Priority gap register

| ID | Priority | Area | Gap and impact | Required resolution |
|---|---|---|---|---|
| G-01 | P0 | Feature correctness | Rolling windows use inclusive `[-N minutes, 0]` ranges (`task2_features.py:77-79` and `task2_predict.py:53-55`). On exact five-minute data this contains 7 observations for “30m”, 13 for “60m”, and 37 for “180m”, not 6/12/36. Feature semantics and documentation may therefore be wrong. | Define timestamp semantics from the source, implement exact interval boundaries, add hand-calculated tests, rebuild features, and retrain. |
| G-02 | P0 | Inference correctness | `task2_predict.py:48-49` claims calendar features come from the requested time, but they come from the latest matched reading. A non-exact query can silently use stale time and rain context. | Define exact-slot vs as-of behaviour; parse and validate SGT time; enforce a maximum staleness; derive calendar fields from the actual prediction instant; expose matched timestamp and age. |
| G-03 | P0 | Probability validity | A GBT score is reported as a probability, but only ranking metrics are evaluated. No Brier score, log loss, reliability curve, expected calibration error, or post-hoc calibration is present. | Add a validation year, calibrate without touching 2024, and report discrimination plus calibration by threshold. |
| G-04 | P0 | Threshold contract | Training uses only five thresholds. A tree can be piecewise and non-monotonic for arbitrary thresholds, even though `P(Y > t)` must never increase as `t` increases. Two anecdotal queries are not proof. | State the supported threshold domain; sweep a dense threshold grid over many contexts; measure violation rate/magnitude; adopt a monotonic strategy or restrict inputs to validated thresholds. |
| G-05 | P0 | Task 1 validity | Annual/monthly totals average stations even when stations have partial coverage (`task1_explore.py:50-67`, `90-97`). Missing readings reduce totals and can make years/stations look artificially dry. | Compute expected-vs-observed slots, set completeness rules, report exclusions, and perform a sensitivity analysis. |
| G-06 | P0 | Extreme-event validity | Heavy-rain counts are raw station-day totals (`task1_explore.py:267-277`) and are not normalised by active complete station-days. Network growth can create an artificial trend. The reported 786 mm day is not investigated. | Require daily completeness, report event rate per 1,000 valid station-days, inspect the extreme’s raw time series and neighbouring stations, and flag/remove only with evidence. |
| G-07 | P1 | Evaluation | Accuracy and Spark’s weighted F1 are dominated by the negative class. Pooled metrics combine five correlated copies of each base row. No positive-class precision/recall, confusion matrix, calibration, uncertainty, or naive baseline is shown. | Make per-threshold metrics primary; add prevalence, PR baseline, precision, recall, positive F1, specificity, confusion counts, Brier/log loss, confidence intervals, and persistence/climatology baselines. |
| G-08 | P1 | Sampling | Uniform sampling occurs after threshold explosion, so threshold copies of one context can be sampled independently and rare high-threshold positives receive no explicit protection. | Sample base observation keys first, preserve all thresholds per selected key, log class counts by threshold, and compare stratified/weighted alternatives. |
| G-09 | P1 | Data contract | Cleaning does not verify `reading_type`, `reading_unit`, source-year agreement, five-minute alignment, coordinate validity, or exact duplicate tie-breaking. It uses only `update_timestamp`, not the potentially relevant reading update field. | Add schema/domain assertions, reject/audit tables with reason counts, deterministic deduplication, cadence/duplicate diagnostics, and station-coordinate checks. |
| G-10 | P1 | Label completeness | `future_n >= 6` counts rows in a range but does not explicitly verify all six expected timestamps. | Validate exact expected future slots and expose missing-window statistics by station/year. |
| G-11 | P1 | Spatial analysis | Station “average annual” rainfall is `total / number of represented years` (`task1_explore.py:206-218`), so partial years bias station rankings. The map has no geographic basemap or uncertainty/coverage encoding. | Aggregate only complete station-years or annualise with an explicit defensible rule; add coverage and uncertainty; improve the map context. |
| G-12 | P1 | Reproducibility | LogReg metrics are hard-coded in `task2_plot.py:32-43`; each training run overwrites the common model and metrics. Metadata omits git SHA, data fingerprint, Spark/Java/Python versions, hyperparameters, class counts, and duration. | Use versioned run directories and machine-readable run manifests; plotting must consume artifacts, never source-code constants. |
| G-13 | P1 | Architecture/scale | `spark_utils.py:17-25` forces `local[*]`, driver memory, and a local temp path. This contradicts the claim that cluster execution needs only path changes. | Allow Spark-submit/cluster settings to take precedence; use CLI/env profiles and URI-capable paths; keep local defaults optional. |
| G-14 | P1 | Verification | No unit, integration, regression, or artifact-consistency tests exist. | Add fast local Spark tests using tiny deterministic fixtures and a smoke pipeline target. |
| G-15 | P1 | Submission | The report, PDF, member details, and final submission checklist are missing. | Build and visually inspect the report only after corrected artifacts are frozen. |
| G-16 | P2 | Modelling/design | The feature set has no cross-station storm movement, recency quality flag, trend/rate features, or feature-importance/ablation analysis. | Add only after P0/P1 validity work; benchmark each improvement against the same frozen split. |
| G-17 | P2 | Operations | Downloader has no retry/backoff policy for transient stream failures, checksums, content validation, or data manifest. | Add robust download metadata and integrity checks. |
| G-18 | P1 | MLflow/MLOps | There is no experiment tracker or model registry. Parameters, data lineage, metrics, performance telemetry, plots, and model versions are split across overwritten local files and hard-coded values. | Make MLflow the authoritative run/model index, use explicit Spark-aware logging, add a database-backed registry, and gate aliases on automated validation. |

## 4. Target architecture

Keep the design small enough for the assignment, but separate data contracts,
feature definitions, model runs, and report artifacts.

```text
data.gov.sg
    |
    v
raw CSV + download_manifest.json
    |
    v
validated clean Parquet + rejects/quality tables
    |
    +---------------------> Task 1 coverage-aware summaries/figures
    |
    v
versioned base features + feature manifest
    |
    v
base-key sampling -> threshold expansion -> train/validation/test split
    |
    +--> LogReg / persistence baselines
    +--> GBT candidate -> probability calibration -> threshold validation
    |
    v
MLflow Tracking run (parameters, datasets, metrics, calibration, artifacts)
    |
    +--> native Spark MLflow model + immutable local export
    +--> Model Registry version -> candidate -> champion alias after gates
    |
    v
historical prediction CLI + report inputs + final PDF
```

Recommended repository shape:

```text
src/
  config.py                 # URI/path and execution profiles
  data_contract.py          # schema, domain, cadence, completeness rules
  clean_data.py
  task1_explore.py
  task2_features.py         # one canonical training/inference feature definition
  task2_build_features.py
  task2_train.py
  task2_evaluate.py         # discrimination, calibration, threshold invariants
  task2_predict.py
  mlflow_tracking.py        # shared run, tag, dataset, metric, model logging
  promote_model.py          # validation gates and registry alias changes
  artifact_manifest.py
config/
  mlflow.local.env.example  # non-secret local defaults
  model_validation.yaml     # promotion thresholds and regression tolerances
infra/mlflow/
  README.md                 # local server and optional shared-server setup
requirements/
  base.in                   # direct dependencies
  base.lock                 # fully resolved, hash-checked environment
tests/
  test_clean_data.py
  test_task1_coverage.py
  test_task2_windows.py
  test_task2_inference.py
  test_threshold_monotonicity.py
artifacts/runs/<run_id>/
  model/
  run_manifest.json
  metrics_overall.csv
  metrics_by_threshold.csv
  calibration_by_threshold.csv
  figures/
report/
  report.md
  report.pdf
```

Design rules:

- Training and inference must call the same feature-building primitives.
- Every aggregate must carry its denominator/coverage alongside its value.
- Every model/plot must be traceable to a run ID and immutable manifest.
- MLflow is the searchable system of record for runs and registered models;
  exported manifests/CSVs remain portable submission evidence.
- The 2024 test set is touched once for final evaluation. Use 2023, or a
  time-ordered slice before 2024, for model selection and calibration.
- A claimed probability must be calibrated and threshold-monotonic over the
  supported input domain.

## 5. Execution plan

### Phase 0 — Freeze evidence and align documentation

Tasks:

- Create a baseline run manifest from the currently tracked metadata and CSVs.
- Record the current git SHA, file checksums, model kind, thresholds, data row
  counts, and provisional metrics.
- Correct the contradictions in `PROGRESS.md`, stale training docstrings, model
  path names, and the machine-memory mismatch in `spark_utils.py` comments.
- Mark current model and plots as `baseline_pre_validation`; do not delete them.

Acceptance criteria:

- One baseline manifest reproduces every number quoted in documentation.
- `README.md`, `PROGRESS.md`, code docstrings, and actual paths agree.
- No document labels the current probabilities “verified correct” before
  calibration and invariant tests pass.

### Phase 1 — Correct data and feature semantics

Tasks:

1. Confirm whether a timestamp denotes the start or end of its five-minute
   accumulation using source metadata and sample sequences.
2. Implement exact 30/60/180-minute lookbacks. For a convention where the
   current observation closes at `t`, a 30-minute feature should contain six
   interval totals, not seven.
3. Verify labels contain exactly the six expected future timestamps
   `t+5, ..., t+30`; reject incomplete horizons.
4. Add cleaned-data quality outputs:
   - rows read/accepted/rejected by year and reason;
   - duplicates before/after resolution;
   - invalid units/types/coordinates;
   - off-grid timestamps and cadence gaps;
   - observed/expected slots per station-day/month/year.
5. Make deduplication deterministic and document which update timestamp wins.
6. Share one feature implementation between build and predict. The prediction
   query must use an explicit SGT timestamp, a freshness limit, and either an
   exact reading or clearly documented as-of behaviour.
7. Reject negative thresholds and out-of-domain timestamps/stations with clear
   messages; never silently turn missing features into zero unless zero is the
   defined meaning.

Acceptance criteria:

- Hand-built fixtures prove exact rolling sums, gaps, year boundaries, future
  labels, first-wet behaviour, and SGT parsing.
- Training and inference return identical features for the same historical
  station/timestamp.
- A stale query fails with an actionable error rather than scoring old data.
- Full feature data is rebuilt and row/quality deltas are explained.

### Phase 2 — Make Task 1 statistically defensible

Tasks:

- Define completeness thresholds (proposed: station-day at least 95% of 288
  slots; station-month/year based on valid days) and show sensitivity at 90%,
  95%, and 99%.
- Recompute annual and monthly comparisons using complete station-periods and a
  common-station sensitivity panel.
- Replace raw heavy station-day count as the primary comparison with heavy-day
  rate per 1,000 valid station-days; retain raw count as context.
- Recompute station rankings from complete station-years and add years/coverage
  beside every estimate.
- Investigate the 2022 ~786 mm station-day: plot its five-minute sequence,
  compare nearby stations and update records, and document whether it is a
  credible event or data anomaly.
- Add uncertainty or distribution summaries (median/IQR or bootstrap interval)
  rather than presenting only a mean.
- Improve figures with units, denominators, source note, readable legends, and
  coverage annotations. A Singapore boundary basemap is optional if licensing
  and sourcing are clean.

Acceptance criteria:

- Each cross-year chart is robust to station-network growth and missingness.
- Every table contains coverage/denominator fields.
- The extreme-event decision has an auditable evidence file.
- Conclusions are rewritten if corrected results change the ordering of years
  or stations.

### Phase 3 — Redesign evaluation around the probability requirement

Tasks:

- Split chronologically into train, calibration/validation, and untouched test.
- Add two baselines:
  - threshold-specific climatology by station/season/hour;
  - persistence or recent-rain rule based on rolling rainfall.
- Preserve all threshold copies of a selected base observation during sampling.
- For every threshold, save sample count, positives, prevalence, ROC-AUC,
  PR-AUC, PR baseline, Brier score, log loss, precision, recall, positive-class
  F1, specificity, and confusion counts at a declared operating point.
- Add reliability diagrams and calibration tables. Fit Platt/isotonic
  calibration on the validation period only when it improves held-out Brier/log
  loss without material ranking degradation.
- Add bootstrap confidence intervals using day- or storm-block resampling so
  autocorrelated rows are not treated as independent.
- Evaluate by station group, month/monsoon, hour, wet/dry context, and data
  freshness to reveal failure modes.
- Add ablations for station ID, location, temporal features, and rain-history
  features. Extract feature importance with an explicit caveat that GBT
  importance is not causal.

Acceptance criteria:

- Final tables make class imbalance visible and do not rely on accuracy or
  weighted F1 as headline evidence.
- The selected model beats both baselines on the chosen primary metrics.
- Calibration plots and numerical scores support the word “probability.”
- All metrics are generated from run artifacts, not hard-coded Python values.

### Phase 4 — Resolve arbitrary-threshold behaviour

Run a dense sweep (for example 0–20 mm in 0.2 mm steps) on a large, diverse
sample of contexts and test:

```text
if t1 < t2, predicted P(Y > t1) >= predicted P(Y > t2)
```

Evaluate three options and document the decision:

1. **Lowest-risk submission option:** support only the five trained thresholds,
   validate them, and reject other thresholds in the CLI. This is honest but
   may be narrower than the assignment wording.
2. **Single-model option:** retain threshold as a feature and apply a validated
   monotonic post-processing/calibration layer across a threshold grid.
3. **Distribution option:** predict the conditional 30-minute rainfall
   distribution or ordered exceedance outputs, then derive any exceedance
   probability from one coherent monotone curve. This is architecturally best
   but more work in pure MLlib.

Do not claim that a tree “interpolates” arbitrary thresholds until the dense
sweep supports that claim.

Acceptance criteria:

- Supported threshold range and precision are explicit in README and CLI help.
- Monotonic violation rate is zero after the selected method, or any documented
  tolerance is quantitatively justified.
- Probabilities remain within `[0, 1]` and calibration is reported across the
  supported threshold range.

### Phase 5 — Add MLflow experiment tracking and model governance

#### 5.1 Version and compatibility baseline

Use an exact, tested environment rather than floating dependencies:

| Component | Initial pin/profile | Rule |
|---|---|---|
| Python | `3.12.x` exact patch in the lock/environment manifest | Use the same minor version for driver, workers, training, and model loading. |
| Java | OpenJDK 17 exact distribution/build | Spark 3.5 officially documents Java 8/11/17; replace the current Java 21 runtime for the reproducible profile. |
| PySpark | `3.5.3` | Keep the model’s existing Spark line initially; evaluate an upgrade such as 3.5.9 as a separate compatibility run, never silently. |
| Scala binary line | `2.12` | Any `mlflow-spark` JAR must match Spark’s Scala binary line. |
| MLflow client/server | `3.16.1` | Pin client and server to the same tested patch; upgrade only through a migration + smoke-test PR. |
| System telemetry | pinned `psutil` | Required for MLflow CPU, memory, disk, and network metrics. |

`PySpark 3.5.3` is within MLflow's currently documented datasource-autolog
compatibility range (`3.4.4` through `4.2.0`). Still run a compatibility smoke
test because compatibility statements do not validate this pipeline, Java
runtime, filesystem, or cluster distribution.

Create `requirements/base.in` for direct constraints and generate a fully
resolved `requirements/base.lock` with hashes. Record `pip freeze`, Python,
Java, Spark, Scala, Hadoop, MLflow, OS, git SHA, and dirty-worktree status in
every training run. Keep the current `requirements.txt` as a short bootstrap
file or replace it with a documented lock installation path.

#### 5.2 Tracking and registry topology

- **Local/assignment profile:** run a localhost-only MLflow server with SQLite
  as the backend/registry store and a workspace-local artifact directory. Do
  not use the legacy file backend as the authoritative store.
- **Shared/HPC profile:** use one authenticated tracking server, PostgreSQL for
  metadata/registry, and durable object storage (GCS/S3/DBFS-compatible storage)
  for artifacts. Executors do not need database credentials; the driver logs
  through the tracking server.
- Configure `MLFLOW_TRACKING_URI`, registry URI, experiment name, and artifact
  location through environment/CLI. Never commit credentials, signed URLs,
  tokens, database passwords, or a developer-specific absolute path.
- Back up the SQLite database and artifact directory for local work. For the
  submitted archive, export the selected run manifest, metrics, plots, model,
  and environment files so grading does not require a running MLflow server.

#### 5.3 Experiment and version taxonomy

Use one experiment per stable purpose:

- `rainfall/task2-development` for smoke tests and feature experiments;
- `rainfall/task2-validation` for comparable full candidate runs;
- `rainfall/task1-analysis` for versioned analysis/figure production.

Use a parent run for one dataset/split/feature configuration and child runs for
GBT, LogReg, baselines, hyperparameter candidates, and calibration choices.
Suggested run name:
`<UTC timestamp>_<git-short-sha>_<model>_<execution-profile>`.

Keep these version concepts separate:

- `data_version`: source dataset IDs + raw manifest digest + clean-contract
  version;
- `feature_version`: semantic version of feature/label definitions;
- `model_contract_version`: input/output schema and supported-threshold API;
- `git_sha`: exact implementation;
- MLflow `run_id`: one execution;
- MLflow registered-model `version`: immutable registered artifact version.

Required tags include `project`, `task`, `run_type`, `git_sha`, `git_dirty`,
`data_version`, `feature_version`, `model_contract_version`, `model_kind`,
`execution_profile`, `test_year`, `validation_status`, and `owner`. Parameters
include thresholds, split dates, sampling policy, row cap, seeds, feature list,
classifier/calibrator settings, Spark configuration, and input URIs with secrets
removed.

#### 5.4 Explicit PySpark logging contract

Implement a small `mlflow_tracking.py` adapter and keep business logic free of
scattered MLflow calls. It must:

1. Establish the tracking URI and experiment before Spark reads training data.
2. Open/close runs with failure-safe context managers and tag failed runs with
   the exception type and pipeline stage.
3. Log Spark datasets with `mlflow.data.from_spark(...)` and
   `mlflow.log_input(..., context="training|validation|test")`, including source
   path, target, digest, schema, row count, period, prevalence, and manifest.
4. Log parameters/tags explicitly; fail if required lineage values are absent.
5. Log the fitted `PipelineModel` explicitly with `mlflow.spark.log_model`, a
   declared signature/input example, pinned pip requirements, and metadata.
6. Load the just-logged native Spark flavor and score a fixture before the run
   can become a candidate.
7. Log all generated CSV/JSON/PNG artifacts plus the selected configuration,
   quality report, feature manifest, run manifest, and environment lock.

Use autologging only as a supplement:

- `mlflow.spark.autolog()` captures datasource lineage on a best-effort,
  asynchronous basis and requires the matching `mlflow-spark` JAR; it is not a
  replacement for explicit dataset logging.
- `mlflow.pyspark.ml.autolog()` can capture supported estimator parameters,
  datasets, and post-training metrics, but Spark Pipeline/meta-estimator model
  logging has limitations. Avoid duplicate/ambiguous model artifacts by making
  the explicit `mlflow.spark.log_model` artifact authoritative.
- If datasource autologging is enabled under PySpark 3.2+, set and test the
  documented `PYSPARK_PIN_THREAD=false` requirement and ensure the JAR version
  and Scala suffix match the installed MLflow/Spark build.
- Keep `silent=False` in development so dropped logging is visible. A training
  run must not be marked successful merely because best-effort autologging was
  unavailable.

#### 5.5 Metrics, evaluation, and performance logging

Log metrics with stable names, for example
`test/thr_1p0/auc_pr`, `test/thr_1p0/brier`, and
`validation/thr_5p0/positive_recall`. For every threshold and split, log:

- rows, positives, negatives, prevalence, and PR no-skill baseline;
- ROC-AUC, PR-AUC, Brier score, log loss, calibration error;
- TP/FP/TN/FN, precision, recall, positive F1, specificity, and declared
  operating cutoff;
- calibration-bin table, reliability plot, ROC/PR plots, confidence intervals,
  and subgroup metrics;
- monotonic-threshold violation count, rate, and maximum magnitude.

Compute full-scale metrics in Spark and explicitly log the results. MLflow's
classic `mlflow.models.evaluate()` may be run on a fixed, versioned evaluation
sample for standard plots and cross-checks, but it must not replace distributed
full-test evaluation or cause the whole 2024 prediction frame to be collected
to pandas. Use `mlflow.validate_evaluation_results()` or equivalent explicit
gates to compare a candidate with the current champion.

Log modelling performance separately from predictive quality:

- wall-clock duration for read, feature build, fit, score, per-threshold
  evaluation, model logging, and total run;
- processed rows/second, training/test row counts, partition counts, model size,
  and artifact upload duration;
- driver CPU/memory/disk/network via `log_system_metrics=True` and pinned
  `psutil`;
- executor/task statistics, shuffle read/write, spill, failed/retried tasks,
  peak executor memory, and stage durations from Spark event logs or a compact
  parsed summary artifact.

MLflow driver system metrics do not substitute for Spark executor telemetry.
Keep raw event logs outside git and log a bounded summary plus the event-log URI
and digest.

#### 5.6 Model Registry and promotion

- Register only models whose run completed all correctness, calibration,
  monotonicity, inference-parity, and artifact-load checks.
- Use model version tags such as `validation_status=pending|passed|rejected`,
  `data_version`, `feature_version`, `git_sha`, and the headline test metrics.
- Use aliases, not deprecated model stages. Suggested aliases are `candidate`,
  `champion`, and `baseline`.
- `promote_model.py` must be idempotent and may move `champion` only when all
  absolute gates pass and regression tolerances against the current champion
  pass. It must emit a promotion-decision JSON artifact.
- Inference defaults to an immutable run/model URI for reproducibility; an alias
  may be accepted for operational use. It must log/resurface resolved name,
  version, run ID, contract version, and feature version.
- Never overwrite or delete the previous champion as part of promotion.

MLflow implementation acceptance criteria:

- One command starts the local SQLite-backed server and UI; one environment
  variable points the client at it.
- A fixture GBT and LogReg run appear as child runs with complete lineage,
  parameters, metrics, plots, performance telemetry, and loadable Spark models.
- A full candidate run can be reproduced from its MLflow metadata and exported
  manifest without consulting console history.
- Model registry aliases resolve to immutable versions; promotion fails closed
  when calibration, monotonicity, performance, or artifact checks fail.
- Training still finishes locally if optional datasource autologging fails, but
  it fails if required explicit model/lineage logging cannot be committed.
- No secrets or large raw datasets are copied into MLflow artifacts.

Reference basis checked on 2026-09-30:

- [MLflow Spark model logging and datasource autologging](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.spark.html)
- [MLflow PySpark ML autologging and its limitations](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyspark.ml.html)
- [MLflow Spark dataset tracking](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.data.html)
- [MLflow classic model evaluation and validation gates](https://mlflow.org/docs/latest/ml/evaluation/)
- [MLflow Model Registry aliases and stage deprecation](https://mlflow.org/docs/latest/ml/model-registry/workflow/)
- [MLflow tracking-server/backend/artifact architecture](https://mlflow.org/docs/latest/self-hosting/architecture/overview/)
- [MLflow system metrics](https://mlflow.org/docs/latest/ml/tracking/system-metrics/)
- [Apache Spark 3.5 runtime compatibility](https://spark.apache.org/docs/3.5.9/api/python/getting_started/install.html)
- [MLflow 3.16.1 release](https://github.com/mlflow/mlflow/releases/tag/v3.16.1)

### Phase 6 — Reproducible local and cluster execution

Tasks:

- Add local and cluster execution profiles. Do not force `.master("local[*]")`
  or driver memory when launched by `spark-submit` under a cluster manager.
- Support local paths plus `gs://`, `s3://`, `dbfs:/`, or `hdfs://` inputs via
  CLI/config without editing source.
- Add a single documented pipeline entry point (Makefile or shell commands are
  sufficient) for download, clean, quality, explore, features, train, evaluate,
  plot, report, and smoke-test stages.
- Version outputs under a run ID; never overwrite the only model or baseline.
- Record dependencies with exact versions and capture environment details in
  the run manifest.
- Add download retry/backoff, response/content validation, size/checksum, source
  dataset ID, retrieval time, and final raw-data manifest.

Acceptance criteria:

- A tiny fixture pipeline runs from a clean environment.
- The same code runs locally and under Spark submit without editing Python.
- A model can be traced to data manifest, feature version, code SHA, parameters,
  environment, and metrics.

### Phase 7 — Automated verification

Minimum tests:

- Cleaning: parsing, units/types, boundaries, deterministic duplicates,
  canonical station metadata, reject reasons.
- Coverage: expected slot counts, partial-day/month/year treatment.
- Features: exact rolling interval count/sum, gaps, lag, last-wet time,
  six-slot future label, no future leakage.
- Split: no train/calibration/test time overlap.
- Sampling: selected base keys retain their full threshold grid.
- Inference parity: offline feature row equals online/historical feature row.
- Contract: invalid station/time/threshold and stale observations fail clearly.
- Model artifact: load and score fixture; output is finite and bounded.
- Threshold property: exceedance probabilities are monotonic.
- Artifact consistency: documentation/report metric values match the selected
  run manifest and CSVs.
- MLflow: required tags/params/inputs exist, model round-trip scoring matches the
  directly fitted pipeline, failed runs are visible, and promotion gates fail
  closed.

Acceptance criteria:

- Fast tests run without the 48M-row dataset.
- A smoke test exercises clean -> features -> train -> evaluate -> predict.
- Static compilation and all tests pass before any full rerun is accepted.

### Phase 8 — Optional model improvements

Only begin after Phases 1–7 pass.

- Add rainfall trend features such as last 10/20-minute slope and wet-run
  duration.
- Add neighbouring-station lagged aggregates using a fixed spatial-neighbour
  table to capture storm movement without a deep-learning stack.
- Tune GBT depth, iterations, step size, subsampling, and class weighting on the
  validation period, with a fixed search budget.
- Consider an LSTM only as a supplementary benchmark. It is outside MLlib and
  must not delay the required corrected Spark model, calibrated evaluation, or
  report.
- Treat the uncapped HPC run as an experiment, not automatically “better.” It
  must beat the capped validated model on untouched 2024 metrics and cost/time.

Acceptance criteria:

- Every added feature/model has an ablation result and runtime/resource cost.
- No test-set-driven tuning occurs.
- Optional work does not replace the required MLlib deliverable.

### Phase 9 — Report and submission

Create `report/report.md` and a rendered `report/report.pdf` containing:

1. Group names and student IDs.
2. Problem statement and exact interpretation of timestamps, horizon, and
   supported thresholds.
3. Dataset source, 2017–2024 scope, schema, data manifest, and quality findings.
4. Cleaning and completeness rules with before/after counts.
5. Computing environment and actual local/cluster configuration used for the
   selected run.
6. Task 1 methods, corrected figures/tables, findings, sensitivity analysis,
   and the extreme-event investigation.
7. Task 2 feature and label definitions, split diagram, sampling, baselines,
   model selection, calibration, per-threshold metrics, confidence intervals,
   subgroup results, and inference example.
8. Limitations: gauges are point observations; missingness; no radar/weather
   predictors; short nowcast horizon; threshold domain; distribution shift;
   local training cap if retained.
9. Reproducibility instructions, MLflow experiment/run ID, registered-model
   name/version, aliases, data/feature versions, and promotion evidence.
10. Specific AI-use declaration describing tools, tasks, human verification,
    and responsibility for final content.

Final submission gate:

- Source and README run from a clean checkout with documented prerequisites.
- Selected trained model loads and predicts.
- Tables/figures/report all reference the same immutable run ID.
- The submitted model export matches the registered MLflow model version and
  passes a load-and-score checksum/smoke test without the tracking server.
- PDF is visually inspected page by page; figures and tables are legible.
- Claims match generated artifacts and no provisional metric remains.
- Member details are present.
- Deliverable archive excludes raw/processed data and temporary Spark files but
  includes source, requirements, README, model, report, and required artifacts.

## 6. Recommended work order and ownership

| Sequence | Work package | Dependency | Expected output |
|---|---|---|---|
| 1 | Documentation truth + baseline manifest | None | Consistent status and frozen baseline |
| 2 | Window/label/inference fixes + unit tests | 1 | Correct feature contract |
| 3 | Cleaning quality + completeness framework | 1 | Quality and coverage tables |
| 4 | Task 1 recomputation and extreme audit | 3 | Defensible exploration artifacts |
| 5 | Sampling, split, baselines, metrics, calibration | 2–3 | Valid model comparison |
| 6 | Threshold monotonicity decision | 5 | Honest threshold API |
| 7 | MLflow tracking, lineage, registry, and promotion gates | 2–6 | Reproducible tracked fixture runs |
| 8 | Full corrected local run logged to MLflow | 2–7 | Registered candidate version |
| 9 | Optional cluster/model experiments | 8 | Evidence-backed improvement only |
| 10 | Champion decision and portable artifact export | 8–9 | Immutable selected model/run |
| 11 | Report creation and visual QA | 4, 10 | Submission PDF |
| 12 | Final clean-checkout smoke test | 11 | Submission-ready package |

## 7. Agent operating rules

- Read `project_info.md`, `PROGRESS.md`, `README.md`, `RATIONALE.md`, and this
  file before changing implementation.
- Preserve existing tracked artifacts until a replacement run passes all gates.
- Do not invent metrics or copy console values into source code.
- Do not create untracked training runs: every model comparison must have an
  MLflow run ID, and every report metric must resolve to a logged artifact.
- Never move a registry alias manually around failed or unrecorded validation
  gates; promotion decisions must be generated by the checked-in policy.
- Do not use 2024 for tuning, calibration, feature selection, or threshold
  policy selection.
- For each phase, update `PROGRESS.md` with command, run ID, result, open issue,
  and next action.
- If the full data is unavailable, complete fixture tests and code review, then
  leave the exact rerun command and expected artifacts; do not mark the phase
  complete.
- Stop and investigate when row counts, prevalence, or Task 1 rankings change
  unexpectedly. A large change after a correctness fix is evidence to explain,
  not a reason to restore the old result.

## 8. Definition of done

The project is done when the pipeline is reproducible, coverage-aware, tested,
and produces calibrated, threshold-consistent exceedance probabilities; MLflow
captures complete data/code/environment lineage, evaluation and Spark runtime
performance, and a gate-approved immutable model version; Task 1
claims survive completeness checks; the selected model and every reported
number share one traceable run manifest; and the complete, visually checked PDF
plus member details and trained model are ready for submission.
