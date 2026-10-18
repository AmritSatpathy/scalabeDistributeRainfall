# Spatiotemporal Analysis and Prediction Plan

## Singapore Rainfall Analysis and Prediction

**Module:** Scalable Distributed Computing for Data Science (DSA5208)  
**Planning scope:** Task 1 geospatial exploration and Task 2 probability of rainfall-threshold exceedance in the next 30 minutes  
**Data period:** 2017–2024  
**Status:** Design document only; no pipeline changes or experiments are authorised by this plan

## 1. Recommended formulation

For station \(s\), forecast origin \(t\), and user-supplied rainfall threshold \(q\), estimate

\[
P\left(\sum_{h=1}^{6} R_{s,t+5h} > q \;\middle|\; \mathcal{I}_{t}\right),
\]

where \(R\) is a five-minute rainfall total and \(\mathcal{I}_{t}\) contains only information available at or before \(t\). Six future five-minute readings form the 30-minute target window.

| Design element | Recommended decision | Reason |
|---|---|---|
| Prediction unit | One station–forecast-time–threshold tuple | Matches the assignment input contract |
| Output | Calibrated exceedance probability | Supports different operational risk tolerances |
| Target interval | \((t,t+30\text{ min}]\) | Prevents the reading at the forecast origin from entering the future label |
| Primary model | Calibrated Spark MLlib gradient-boosted tree classifier | Captures nonlinear lag, threshold, season and station interactions while satisfying the Spark MLlib requirement |
| Principal validation | Yearly rolling-origin evaluation with an expanding training window | Respects chronology and tests transfer across complete wet/dry seasonal cycles |
| Locked final test | Calendar year 2024 | Preserves the most recent year for one final, untouched assessment |
| Primary metric | PR-AUC by threshold and fold | More informative than accuracy or ROC-AUC for rare positive rainfall events |

## 2. Preliminary data inspection

Only the first 100 records of `rainfall_2017.csv` were inspected. Headers alone were compared across all eight canonical annual files; no full dataset scan was performed.

| Sample finding | Observed result | Planning implication |
|---|---:|---|
| Rows inspected | 100 | Describes schema and grain only; it cannot estimate annual distributions |
| Columns | 12 | Time, station metadata, coordinates, value, type and unit are available |
| Sample time span | 2017-01-01 08:04:59 to 08:09:59 +08:00 | Two consecutive reporting intervals are represented |
| Distinct timestamps | 2 | The sample is a cross-section of stations at two times |
| Distinct stations | 56 | The data are a station panel rather than one national series |
| Observed interval | 300 seconds | Consistent with five-minute rainfall totals |
| Duplicate station–timestamp keys | 0 | The sample is unique, but full-data deduplication must still be audited |
| Value range | 0.0–0.0 mm | The opening sample is dry and must not be used to infer class balance |
| Reading definition | `TB1 Rainfall 5 Minute Total F` | Each reading appears to represent a five-minute accumulation |
| Schema across 2017–2024 | Identical headers | A common ingestion contract is feasible |
| Timestamp detail | Values end in `:59` | Normalise to canonical five-minute slots only after verifying this convention across the full data |

### 2.1 Fields and proposed roles

| Field | Proposed role | Treatment |
|---|---|---|
| `timestamp` | Event time | Parse as timezone-aware Singapore time; use as the main temporal key |
| `reading_value` | Five-minute rainfall amount | Cast to non-negative numeric millimetres; retain zeroes |
| `station_id` | Station identity | Categorical model input and partition/group key |
| `location_latitude`, `location_longitude` | Spatial context | Validate per-station consistency; use for neighbour features |
| `update_timestamp`, `reading_update_timestamp` | Data revision/availability metadata | Use to resolve revisions and assess whether operational as-of filtering is needed |
| `station_name`, `station_device_id` | Audit metadata | Do not use directly unless identity changes reveal station/device discontinuities |
| `reading_type`, `reading_unit` | Measurement contract | Filter to the expected five-minute rainfall series in millimetres |
| `date` | Source convenience field | Re-derive calendar fields from parsed event time to avoid disagreement |

Before modelling, the full-data audit should confirm timestamp semantics, correction behaviour, impossible negative values, duplicate keys, station relocation, device replacement, missing intervals, and station entry/exit dates. `station_device_id` must not be treated as an independent location: it should be used to identify device changes within a station history.

## 3. Geospatial exploratory analysis plan for Task 1

### 3.1 Analytical framing

This is a spatiotemporal panel: rainfall varies over time, while stations observe different locations and may enter, leave, move or change devices. Task 1 should therefore answer questions about **where**, **when**, and **how consistently** rainfall occurs.

| Research question | Spatial statistic | Temporal comparison | Intended conclusion |
|---|---|---|---|
| Where is rainfall typically greatest? | Coverage-normalised mean or median rainfall per eligible station-day, aggregated to H3 cells | Full period and monsoon season | Persistent spatial rainfall structure |
| Where are heavy events concentrated? | Frequency of station-hours or station-days above a fixed heavy-rain threshold | By year and season | Spatial distribution of extremes |
| How does a selected period differ from its recent context? | Absolute anomaly in mm per eligible day | Target month vs previous month and trailing three months | Short-term change |
| Is that period unusual for the season? | Standardised anomaly against the same calendar month in earlier years | Same month one, two and three years earlier; then historical same-month baseline | Year-on-year and climatological departure |
| How does a storm move across Singapore? | H3/station rainfall intensity at successive timestamps | Six to eight consecutive 30-minute frames | Event evolution and propagation |
| Could apparent differences be caused by the network? | Active stations, expected-slot coverage and device/station continuity | Alongside every comparison | Separates rainfall change from observation change |

### 3.2 Spatial representation

| Representation | Role | Recommendation |
|---|---|---|
| Station points | Primary observational layer | Preserve measured values, station identity and local coverage; show these in coverage and event maps |
| H3 cells | Common analytical grid | Aggregate stations into stable spatial cells for comparable choropleths; test resolutions 8 and 9 and select the coarsest resolution with adequate station support |
| Singapore boundary | Geographic context | Use a clean vector outline and planning-region/coastline layer if locally available |
| Nearest-station graph | Prediction-oriented diagnostic | Connect each station to a small number of geographically nearest stations to show information pathways |
| Folium | Interactive exploratory aid | Use for internal inspection, popups and time sliders; do not rely on browser screenshots as the final report artwork |
| GeoPandas/Matplotlib with H3 | Final static rendering path | Produces repeatable high-resolution PNGs with shared colour scales, fonts and panel layouts |

H3 aggregation should not imply that rainfall was measured continuously over every cell. A cell without sufficient station support must be shown as unavailable, not as zero rainfall. If many cells contain only one station, the figure should disclose that fact or remain a station-point map.

### 3.3 Coverage-safe spatial metrics

Raw rainfall totals are not comparable when stations have different reporting coverage. Metrics should first be calculated per station and eligible time unit, then combined with equal station weight within an H3 cell.

| Metric | Definition concept | Use |
|---|---|---|
| Eligible station-day | Station-day meeting a predeclared fraction of expected five-minute slots | Denominator for daily and monthly spatial summaries |
| Mean rainfall per eligible day | Sum of valid rainfall divided by eligible station-days | Comparison of periods with unequal station availability |
| Wet-period frequency | Fraction of eligible intervals with rainfall above the chosen wet threshold | Persistent wetness independent of total accumulation |
| Heavy-rain frequency | Fraction of eligible station-hours/days above a declared threshold | Spatial extreme-event comparison |
| Standardised anomaly | Difference from same-calendar-month historical mean divided by its historical standard deviation | Shows unusual local conditions on a comparable scale |
| Absolute anomaly | Target-period mm per eligible day minus baseline | Keeps the physical magnitude visible |
| Spatial support | Active-station count, eligible-day count and median coverage | Required companion information for each map |

The eligibility threshold should be selected before inspecting rainfall patterns. A sensitivity check at 80%, 90% and 95% coverage is recommended; 90% is a reasonable primary candidate if sufficient observations remain. Comparisons should use the intersection of eligible stations or cells wherever practical.

### 3.4 Proposed report figures

| Figure | Planned composition | Main encoding | Why it belongs in the report |
|---|---|---|---|
| G1. Observation network and continuity | Singapore map with station locations; point colour for first active year and outline/shape for device change or relocation; side panel with active-station count by month | Location, station lifetime and network stability | Establishes whether later spatial comparisons are observationally comparable |
| G2. Seasonal rainfall geography | Four aligned maps for Northeast Monsoon, first inter-monsoon, Southwest Monsoon and second inter-monsoon using the same scale | H3 or stable-station mean mm per eligible day | Answers how rainfall geography changes within a year |
| G3. Same-month year comparison | Small multiples for one preselected calendar month across 2017–2024, with identical extent and colour scale | Coverage-normalised rainfall | Shows year-to-year spatial variation without mixing seasons |
| G4. Multi-baseline anomaly dashboard | Target month map plus differences from previous month, trailing three-month mean, previous year's same month and historical same-month baseline | Absolute and standardised anomalies with a centred diverging scale | Directly addresses recent, annual and multi-year comparison requested by the project |
| G5. Storm-evolution storyboard | Six or eight chronological panels at 30-minute spacing, selected by a declared event rule | Station/H3 intensity with a shared sequential scale and time labels | Makes the spatiotemporal character visually clear and shows storm movement |
| G6. Heavy-rain hotspot stability | Main hotspot-frequency map plus small rank-change panel across years | Heavy-rain event frequency and rank | Distinguishes persistent hotspots from one exceptional event |
| G7. Spatial support companion | Active stations or eligible station-days by cell for the same mapped period | Neutral greyscale support map | Prevents sparse-data areas from being overinterpreted |

The strongest main-report set is **G1, G2, G4 and G5**. G3, G6 and G7 can appear in the appendix or replace a main figure if they reveal a substantive result. G7 may also be inset beside G2/G4 rather than consume a full page.

### 3.5 Period and event selection rules

The target month and showcase storm must be selected by reproducible criteria rather than visual preference.

| Item | Predeclared rule |
|---|---|
| Target month for G4 | Select the month with the largest positive network-wide standardised rainfall anomaly among months meeting the coverage requirement; disclose the rule |
| Same-month comparison | Compare the chosen calendar month across all available years, then emphasise the preceding one, two and three years |
| Recent baseline | Previous complete calendar month |
| Medium baseline | Immediately preceding three complete calendar months, normalised per eligible day |
| Seasonal baseline | Same calendar month over prior available years only |
| Showcase event | Select the day or six-hour window with the largest coverage-normalised network rainfall among adequately observed periods |
| Animation/storyboard frame | Aggregate to 30-minute windows for legibility while retaining five-minute data for selection and calculation |

For descriptive Task 1 figures covering 2017–2024, all years may be analysed. Any spatial statistic later used as a Task 2 predictive feature must be recomputed from training history only within each temporal fold.

### 3.6 Static PNG design specification

| Element | Specification |
|---|---|
| Output | PNG at 300 dpi, plus vector PDF/SVG working copy when supported |
| Report width | Design at final insertion width: approximately 7.0–7.2 inches for a full-width A4 figure |
| Typography | Times New Roman where available; minimum effective report size 9 pt, panel titles 10–11 pt |
| Sequential rainfall scale | Perceptually uniform blue/teal or `cividis`; zero given a quiet neutral tone |
| Difference/anomaly scale | Colour-blind-aware diverging palette centred exactly at zero, with symmetric limits |
| Comparability | Identical spatial extent and colour limits within every comparison figure |
| Geography | North arrow, compact scale bar, coastline/boundary and a small location inset only when useful |
| Legends | One shared colour bar outside the map panels; units and aggregation period stated explicitly |
| Missing data | Light grey with hatching or clear `insufficient coverage` legend entry |
| Basemap | Prefer a clean offline vector boundary; if map tiles are used, retain readable attribution |
| Annotation | Label only major findings, selected stations and event direction; avoid overlapping station labels |

Folium maps can be visually appealing during exploration, but they depend on browser rendering and often produce inconsistent labels at print scale. The final PNG should therefore be composed explicitly for the report, even if Folium is used to discover the view.

### 3.7 Geospatial EDA experiment sequence

| Order | Analysis | Comparison/control | Output |
|---:|---|---|---|
| 1 | Station/device continuity audit | Station coordinate and device histories | G1 and eligibility table |
| 2 | H3-resolution sensitivity | Resolution 8 vs 9; stations and eligible days per cell | Resolution decision table |
| 3 | Coverage-threshold sensitivity | 80%, 90% and 95% expected-slot eligibility | Robustness table; select primary threshold |
| 4 | Seasonal geography | Four meteorological regimes with shared scale | G2 |
| 5 | Same-month comparison | 2017–2024 stable cohort/shared scale | G3 |
| 6 | Multi-baseline anomaly | Recent, three-month, annual and historical same-month baselines | G4 |
| 7 | Event evolution | Objectively selected high-rain event | G5 |
| 8 | Extreme hotspot stability | Fixed threshold and stable eligible denominator | G6 |
| 9 | Antecedent-state comparison | 30-minute, 3-hour and 24-hour past-only accumulation | G8 |
| 10 | Intensification and persistence | Current vs preceding 30-minute state; wet-slot fraction | G9 |
| 11 | Lagged event footprint | Past lag panels plus a clearly separated future outcome panel | G10 |
| 12 | Spatial predictive structure | Event-level lead–lag, distance decay and neighbour ablation | G13 and supporting statistics |

### 3.8 Connection to Task 2

The EDA should lead to testable predictive features rather than remain decorative.

| EDA finding | Candidate predictive feature | Required leakage control |
|---|---|---|
| Nearby cells become wet before the target station | Distance-weighted neighbour rainfall and wet-neighbour count | Neighbour event/availability time must be at or before forecast origin |
| Rainfall moves along a consistent direction during events | North/south/east/west gradient and recent centroid displacement | Use only preceding frames |
| Stations have stable seasonal differences | Station × hour × month climatology | Fit on training history only and smooth sparse groups |
| Some cells have poor support | Active-neighbour count and spatial-coverage flag | Never impute unsupported cells as dry |
| Heavy-rain hotspots persist | Geographic coordinates, station identity and local climatology | Demonstrate out-of-time benefit through ablation |

### 3.9 Feature-derived spatiotemporal EDA

The same features proposed for prediction can support Task 1. Mapping these derived states is more informative than repeatedly mapping raw five-minute rainfall because it reveals antecedent conditions, storm persistence, intensification and spatial propagation.

| Derived feature | Definition at station and time \(t\) | Scientific question | Recommended visual |
|---|---|---|---|
| Lag profile | Rainfall at 5, 15, 30, 60 and 180 minutes before \(t\) | Where was the rain system previously located? | Five-panel lag map with one shared scale and the forecast origin clearly marked |
| Rolling accumulation | Past-only rainfall sum over 30 minutes, 3 hours and 24 hours | Does the current burst sit inside a longer wet period? | Three aligned maps using scale appropriate to each duration, with units in mm per window |
| Rolling intensity | Rolling sum divided by valid observed duration | Which areas experienced the highest sustained rain rate? | H3 choropleth or station bubbles, accompanied by coverage |
| Rolling maximum | Maximum five-minute amount in the past 1 or 3 hours | Where were the most intense short bursts? | Hotspot map with labelled top stations/cells |
| Wet-slot fraction | Fraction of valid five-minute slots above the declared wet threshold in the past 3 or 24 hours | Was rainfall persistent or intermittent? | Bounded 0–1 choropleth |
| Intensity trend | Difference between the most recent 30-minute mean and the preceding 30-minute mean | Where is rainfall intensifying or weakening? | Diverging map centred at zero |
| Short-term slope | Robust slope fitted to the previous six or twelve five-minute values | Is a local storm building or dissipating? | Direction-coloured station markers; suppress slopes with insufficient coverage |
| Time since last rain | Minutes since the most recent positive reading, capped at a declared limit | Which areas are currently active, recently wet or long dry? | Ordered categorical map rather than a highly skewed continuous scale |
| Wet-spell duration | Consecutive wet slots ending at \(t\) | Where is rainfall persisting? | Station-point or H3 duration map |
| Event cumulative rainfall | Sum from a reproducibly defined event start to time \(t\) | Where has the event deposited the most rain? | Progressive cumulative maps or start/end comparison |
| Neighbour wetness | Distance-weighted current/rolling rainfall among nearby stations | Is rainfall locally isolated or spatially organised? | Target-station map overlaid with neighbour rings or local spatial average |
| Local contrast | Station rainfall minus distance-weighted neighbour rainfall | Which observations are local peaks versus broad regional rain? | Diverging local-anomaly map |
| Historical percentile | Current rolling accumulation ranked against the same station, month and hour using historical data | How unusual is the current local state? | Percentile map with fixed bands such as below 50th, 50th–90th, 90th–99th and above 99th |

For descriptive figures, historical percentiles may use the explicitly stated comparison period. For predictive features and model diagnostics, percentiles and baselines must use only information that predates the forecast origin and must be fitted separately inside each training fold.

### 3.10 Additional high-value figures from engineered features

| Figure | Composition | Main message | Report priority |
|---|---|---|---:|
| G8. Antecedent rainfall state | Three maps at the showcase forecast origin: 30-minute, 3-hour and 24-hour rolling accumulation | Separates immediate rain from accumulated wet conditions | High |
| G9. Storm intensification map | Current 30-minute accumulation plus the change from the preceding 30 minutes | Shows where the event is strengthening and weakening | High |
| G10. Lagged storm footprint | Maps of rainfall at \(t-60\), \(t-30\), \(t-15\), \(t\) and, for descriptive evaluation only, \(t+30\) | Communicates movement and supplies an intuitive bridge to the prediction target | High |
| G11. Event cumulative footprint | Cumulative rainfall at event start +1 h, +3 h, +6 h and event end | Shows spatial exposure over the entire event | Medium |
| G12. Persistence–intensity typology | Two-dimensional plot of wet-slot fraction versus rolling maximum, coloured by season and faceted by spatial region | Distinguishes persistent moderate rain from brief intense bursts | Medium |
| G13. Lead–lag station network | Edges connect nearby stations with a stable positive lagged association; arrows indicate estimated lead direction | Explores whether neighbouring stations provide predictive advance warning | Medium/experimental |
| G14. Feature-season matrix | Heat map of monthly median rolling accumulation, wet-spell duration and heavy-rain frequency by region | Compact summary of temporal and spatial seasonality | Medium |
| G15. Predictive-feature atlas | Small multiples for recent accumulation, trend, wet-spell duration, neighbour wetness and historical percentile at one forecast origin | Explains what the model observes before making a prediction | High if space permits |

G10 must label \(t+30\) as an **outcome panel**, never as an input feature. This makes the figure useful for explaining the forecasting problem without implying access to future rainfall.

### 3.11 Cumulative-feature design

Cumulative rainfall requires an explicit start point. Three definitions serve different purposes and must not be mixed in one comparison.

| Cumulative definition | Start rule | Best use | Main limitation |
|---|---|---|---|
| Fixed window | Sum over the previous 30 minutes, 3 hours or 24 hours | Prediction features and comparable maps | Can split a storm across the window boundary |
| Calendar period | Sum since start of day, week or month | Monthly and weekly descriptive comparison | Sensitive to incomplete reporting and differing period length |
| Event based | Sum from event start, defined by network wetness exceeding a rule after a minimum dry gap | Storm footprint and event severity | Requires a predeclared event-segmentation rule |

Calendar totals should be converted to rainfall per eligible station-day for cross-period comparisons. Event totals should be paired with event duration and spatial coverage so a long weak event is not confused with a short extreme event.

### 3.12 Lead–lag and spatial-dependence analysis

Lead–lag analysis can determine whether the geospatial information has predictive value beyond visual appeal.

| Analysis | Planned method | Interpretation safeguard |
|---|---|---|
| Pairwise lag association | For geographically close station pairs, compare cross-correlation at lags from −60 to +60 minutes | Calculate within individual events or season; common seasonality can create false correlation |
| Directional propagation | Convert stable pairwise lead times into arrows and summarise prevailing event direction | Require minimum event count and agreement; do not interpret correlation as physical causation |
| Spatial autocorrelation | Moran's I or a local spatial statistic on coverage-qualified rolling rainfall | Use a declared neighbour-weight matrix and permutation significance test |
| Neighbour-value test | Compare a model with station history only against one adding past-only neighbour features | Out-of-time ablation is the decisive test of predictive usefulness |
| Distance-decay profile | Plot rainfall similarity or predictive gain against station separation | Bin distances with sample counts and confidence intervals |

A national cross-correlation calculated over the full eight-year series would be misleading because dry periods and annual cycles dominate it. Lead–lag relationships should be estimated within objectively detected rainfall events and then aggregated across events.

### 3.13 Revised figure shortlist

To keep the report focused, the engineered-feature figures should replace weaker descriptive plots rather than continuously expand the document.

| Placement | Selected figure | Purpose |
|---|---|---|
| Main report 1 | G1. Observation network and continuity | Establish data coverage and station geography |
| Main report 2 | G2. Seasonal rainfall geography | Describe within-year spatial change |
| Main report 3 | G4. Multi-baseline anomaly dashboard | Compare recent, annual and climatological conditions |
| Main report 4 | G10. Lagged storm footprint | Show storm movement and define the forecasting challenge |
| Main report 5 | G8/G9 combined. Antecedent state and intensification | Show the engineered temporal state at the forecast origin |
| Main report 6, if space permits | G15. Predictive-feature atlas | Connect Task 1 patterns to Task 2 model inputs |
| Appendix | G3, G5, G6, G7, G11–G14 | Supply supporting comparisons and sensitivity evidence |

The combined G8/G9 figure should use a compact two-row layout: raw/rolling rainfall states on the first row and change/persistence states on the second. This gives the report a richer analytical figure without creating many repetitive maps.

## 4. Target construction

| Step | Rule |
|---:|---|
| 1 | Define a canonical five-minute station–time grid after validating the source timestamp convention. |
| 2 | At forecast origin \(t\), calculate the sum of exactly the next six expected slots for the same station. |
| 3 | Create \(y=1\) when the future sum is strictly greater than \(q\); otherwise create \(y=0\). |
| 4 | Exclude a target when any of its six future slots are unavailable, unless a separately justified missing-data policy is adopted. A missing reading must never be treated as zero by default. |
| 5 | Store `label_end_timestamp = t + 30 minutes` so split-boundary leakage can be detected and purged. |

Recommended development thresholds are 0.2, 1, 2, 5 and 10 mm. Thresholds 0.5, 3 and 7 mm should be reserved for interpolation and monotonicity testing. Every base station–time context should receive the same threshold set so threshold prevalence does not become entangled with temporal sampling.

## 5. Feature engineering

### 5.1 Recommended feature hierarchy

| Group | Candidate features | Temporal interpretation | Priority |
|---|---|---|---:|
| Immediate state | Current five-minute amount; lags at 5, 10, 15, 20, 25 and 30 minutes | Rainfall persistence and short storm bursts | P0 |
| Recent memory | Lags at 45, 60, 90, 120 and 180 minutes | Development and dissipation of convective cells | P0 |
| Rolling accumulation | Past-only sums over 15, 30, 60 and 180 minutes; 6, 12 and 24 hours | Antecedent intensity at several physical time scales | P0 |
| Rolling shape | Mean, maximum, standard deviation, wet-slot count, non-zero fraction, recent slope and change in slope | Separates steady rain, isolated bursts and intensifying events | P0/P1 |
| Wet-spell state | Minutes since last positive reading; duration of current wet spell; consecutive dry slots | Encodes event persistence | P1 |
| Calendar cycle | Hour sine/cosine, day-of-year sine/cosine, month, day of week | Diurnal and annual seasonality without artificial December–January discontinuity | P0 |
| Seasonal regime | Northeast monsoon, inter-monsoon and Southwest monsoon indicators | Singapore-specific broad seasonal structure | P1 |
| Long-term state | Past-only trailing 7-, 30- and 90-day wet frequency and rainfall statistics | Slow changes in seasonal background | P1 |
| Historical climatology | Training-only station × hour × month wet probability and amount quantiles, smoothed toward network values | Expected local rainfall by place and season | P1 |
| Spatial context | Current and recent rain among nearest stations; distance-weighted sums; wet-neighbour count; north/east rainfall gradients | Movement and spatial extent of rain systems | P1 |
| Threshold context | \(q\), `log1p(q)`, threshold × recent-accumulation interactions | Lets one model answer different exceedance queries | P0 |
| Data quality | Availability flags, rolling coverage ratios, reading age, active-neighbour count | Prevents missingness from silently behaving like dry weather | P0 |

All rolling windows must be left-closed at the forecast origin: they may include observations at \(t\) and earlier, but never a reading after \(t\).

### 5.2 Specific treatment of recency, seasonality and trend

| Proposed idea | Recommendation | Rationale |
|---|---|---|
| Last 30 minutes | Include detailed lags and aggregate statistics | Closest representation of immediate storm state |
| Last hour and last three hours | Include sums, maxima, wet counts and trend | Captures persistence and whether rainfall is strengthening |
| Last 24 hours | Include aggregate amount and coverage, not 288 raw lags | Provides daily context without excessive dimensionality |
| Last 50 hours | Do not make it a primary window; test only in an ablation | It has no clear physical or calendar interpretation compared with 24, 48 or 72 hours |
| Same time yesterday / last week | Optional exact lag plus availability flag | Can capture repeated diurnal patterns, but is weaker than recent storm evidence |
| Same time six months ago | Prefer seasonal encodings and historical climatology | A single distant observation is noisy and six months usually represents a different monsoon regime |
| Same time one year ago | Treat as an optional ablation, not a core feature | Exact annual rainfall does not repeat reliably; training-only month/hour climatology is more stable |
| Three-month or annual trend | Use trailing, past-only distributions and wet frequencies | Describes background regime without looking into future records |

### 5.3 First feature set and extended feature set

| Set | Included features | Purpose |
|---|---|---|
| Minimal defensible set | Station, coordinates, threshold, immediate lags through 60 minutes, 15/30/60/180-minute sums and maxima, coverage, hour/day-of-year cycles | Establish a strong, interpretable benchmark with modest computation |
| Extended set | Minimal set plus 6/12/24-hour summaries, wet-spell features, monsoon regime, 7/30/90-day state, training-only climatology and neighbour summaries | Test whether long-range and spatial context add stable predictive value |
| Research ablation set | Exact 24-hour, 7-day, six-month and one-year lags with availability indicators | Quantify whether intuitive distant lags help after climatology is present |

## 6. Temporal training, validation and testing

### 6.1 Primary rolling-origin protocol

The training window expands as calendar time advances. Validation and test windows move forward by one year and remain fixed at one year so that each evaluation covers Singapore's full seasonal cycle.

| Fold | Training period | Validation period | Test period | Use |
|---:|---|---|---|---|
| 1 | 2017–2019 | 2020 | 2021 | Early development comparison |
| 2 | 2017–2020 | 2021 | 2022 | Stability under additional history |
| 3 | 2017–2021 | 2022 | 2023 | Final model and feature selection |
| Locked final | 2017–2022 | 2023 | 2024 | One final estimate of recent out-of-time performance |

The test set does not itself grow. Keeping each test period at one year makes folds comparable, while the growing training set implements the requested expanding-history design. A growing test period would repeatedly count earlier observations and make fold comparisons difficult to interpret.

### 6.2 Boundary and fitting rules

| Risk | Required control |
|---|---|
| Future labels cross a split boundary | Purge forecast origins whose `label_end_timestamp` extends beyond their partition |
| Preprocessing learns future distributions | Fit imputers, encoders, scalers, station eligibility rules and climatology only on the fold's training period |
| Validation is reused as training | Select hyperparameters, calibration and decision cutoff with validation; report test once per fold |
| Rows from one storm appear on both sides through random splitting | Split strictly by event time; never use random row splitting |
| Threshold-expanded copies leak between partitions | Split base station–time contexts first, then expand thresholds inside each partition |
| Spatial features access later reports | Include neighbour measurements only when their event/availability time is at or before the forecast origin |
| Historical features use incomplete windows | Attach coverage and availability flags; define a minimum coverage policy before training |

If hyperparameter search needs more temporal samples, use quarterly rolling origins *inside the training years only*. The yearly test folds remain the primary evidence because they cover seasonality comprehensively.

## 7. Model strategy

| Model | Role | Strength | Limitation / control |
|---|---|---|---|
| Recent-window empirical prevalence | Naive baseline | Establishes whether learning adds value beyond base rates | Must be estimated from training data only |
| Persistence/rule baseline | Naive baseline | Tests whether recent rain alone is sufficient | Produces coarse probabilities |
| Spark MLlib logistic regression | Statistical baseline | Interpretable, scalable and easy to calibrate | Cannot naturally capture complex nonlinear interactions |
| Spark MLlib gradient-boosted trees | **Recommended primary model** | Handles nonlinear lags, thresholds, station effects and feature interactions | Requires probability calibration and careful tuning for imbalance |
| Random forest | Secondary tree benchmark | Robust and parallelisable | Often less accurate and less probability-efficient than boosting |
| TCN/LSTM sequence network | Optional supplementary experiment | Learns temporal patterns directly | Adds substantial complexity, leaves the required MLlib path, and does not automatically solve leakage or calibration |

The recommended primary result is a Spark MLlib GBT classifier trained on the engineered tabular sequence features, followed by isotonic calibration learned from validation predictions. Logistic regression should remain as the principal scientific baseline. A sequence neural network is justified only after the engineered GBT is stable and if compute time permits.

For a single threshold-conditioned model, verify that predicted exceedance probability does not increase as \(q\) increases for the same station and time. Report the monotonicity-violation rate. If violations are material, compare against separate per-threshold models or apply a validation-fitted monotonic projection before presenting the model as a threshold-query system.

## 8. Sampling and scalable computation

| Concern | Planned treatment |
|---|---|
| Severe class imbalance | Retain all positive base contexts; sample negatives by time block/station if necessary and record sampling probabilities |
| Probability distortion from sampling | Apply sample weights where supported and calibrate on an unsampled validation distribution |
| Repeated threshold rows | Create the base temporal feature table once, then cross-join the small threshold table |
| Rolling computation | Partition by station and order by canonical event time; cache the reusable feature table |
| Spatial computation | Precompute a station-neighbour lookup with distance and rank, then join contemporaneous past-only summaries |
| Large annual files | Read all years with an explicit schema, prune columns early, and persist intermediate data in a partitioned columnar format |
| Experiment comparability | Fix data version, split contract, sample seed, feature version and threshold set; store fold-level artefacts and timings |

Sampling must occur at the base station–time level before threshold expansion. Sampling individual expanded rows could give one rainfall context an inconsistent set of thresholds and corrupt monotonicity evaluation.

## 9. Evaluation framework

| Dimension | Required outputs |
|---|---|
| Ranking | PR-AUC as primary; ROC-AUC as secondary, both by threshold and temporal fold |
| Probability quality | Brier score, log loss, reliability diagram, expected calibration error and calibration slope/intercept |
| Positive class | Precision, recall, F1 and confusion matrix at a cutoff chosen only on validation data |
| Threshold behaviour | Metrics for seen and unseen thresholds; monotonicity-violation rate for ordered thresholds |
| Temporal robustness | Fold median, minimum, maximum and year-by-year table rather than only one pooled score |
| Spatial robustness | Performance by station and by station activity/coverage group |
| Operational relevance | Recall at fixed false-alarm or precision levels, where feasible |
| Scalability | Data volume, executor/worker configuration, wall-clock time, throughput and speed-up under controlled resources |

Aggregate metrics must be accompanied by positive prevalence and sample count. Otherwise, apparent changes across thresholds or years may only reflect changing class balance.

## 10. Prediction experiment sequence

| Order | Experiment | Comparison | Decision enabled |
|---:|---|---|---|
| 1 | Target and cadence audit | Manual examples around dry, wet and missing intervals | Confirms label correctness before fitting models |
| 2 | Naive baselines | Prevalence and persistence | Establishes the minimum useful performance |
| 3 | Minimal feature models | Logistic regression vs GBT | Selects the primary model family |
| 4 | Recency ablation | Add 3-hour and 24-hour summaries incrementally | Measures useful memory length |
| 5 | Seasonal ablation | Add cyclic calendar, monsoon and training-only climatology | Tests seasonal contribution |
| 6 | Spatial ablation | Add nearest-station summaries | Tests storm-propagation information |
| 7 | Distant-lag ablation | Add 24-hour, 7-day, six-month and annual exact lags | Tests the user's proposed recurrence features directly |
| 8 | Calibration | Raw vs isotonic-calibrated probability | Supports the required probability interpretation |
| 9 | Threshold generalisation | Seen vs held-out thresholds and monotonicity | Validates the threshold-input design |
| 10 | Rolling-origin assessment | Folds 1–3 | Selects the final specification based on temporal stability |
| 11 | Locked assessment | One run on 2024 | Produces the final unbiased performance table |

Only one feature family should be added at a time. This makes the report academically stronger because each performance change can be attributed to a stated hypothesis.

## 11. Recommended final prediction experiment matrix

| Experiment ID | Model | Feature set | Calibration | Expected purpose |
|---|---|---|---|---|
| E0 | Historical prevalence | Calendar/station/threshold groups | Native | Base-rate reference |
| E1 | Persistence rule | Recent rainfall only | Validation mapping | Immediate-state reference |
| E2 | Logistic regression | Minimal | Platt or isotonic | Interpretable MLlib baseline |
| E3 | GBT | Minimal | Isotonic | Primary low-complexity candidate |
| E4 | GBT | Extended without spatial | Isotonic | Tests long-term and seasonal context |
| E5 | GBT | Full extended | Isotonic | Tests spatial contribution; proposed final candidate |
| E6 | GBT | Full plus distant exact lags | Isotonic | Research ablation for annual/six-month recurrence |

## 12. Acceptance criteria before implementation

| Checkpoint | Evidence required |
|---|---|
| Timestamp contract | Documented mapping from raw `:59` timestamps to canonical five-minute slots, verified across years |
| Label contract | Hand-checked station/time examples showing the six future readings and resulting threshold labels |
| Missingness contract | Explicit distinction between zero rainfall and unavailable readings |
| Split contract | Counts plus minimum/maximum forecast and label-end timestamps in every fold |
| Feature contract | Automated check that every feature source time is at or before forecast time |
| Online/offline parity | Identical features for selected station/time queries under batch and prediction-time construction |
| Threshold contract | Every retained base context has the expected threshold rows in its own partition |

## 13. Planning decision

Task 1 should first establish the observation network, seasonal geography and multi-baseline anomalies through coverage-safe station and H3 maps. It should then use lag, rolling accumulation, intensity trend, wet-spell and neighbour features to explain storm state and movement. The strongest report sequence is G1, G2, G4, G10 and a combined G8/G9 figure, produced as consistent, high-resolution static figures.

The best first Task 2 implementation is a calibrated Spark MLlib GBT using a compact set of past-only lag, rolling, calendar, station, threshold and coverage features. It should be evaluated with yearly expanding-window rolling origins and a locked 2024 test. Spatial summaries and training-only climatology are the highest-value extensions. Exact observations from six months or one year earlier should be tested as ablations; they should not replace cyclic seasonality and historical climatology because isolated distant readings are sparse and meteorologically unstable.

