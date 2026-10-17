# Singapore Rainfall Analysis and Prediction — post-revision gap analysis

**Review date:** 16 October 2026  
**Reviewed artifact:** [Historical_Rainfall_Singapore_Report.docx](Historical_Rainfall_Singapore_Report.docx)  
**Review basis:** [DSA5208 project requirements](../project_assignment_guidelines.md)  
**Current verdict:** academically presentable and substantially complete as a report; predictive-validity and controlled-systems evidence remain the main scientific gaps.

## 1. Current rating

The score below is a reviewer-defined diagnostic, not an official NUS rubric or predicted grade.

| Assessment dimension | Weight | Current score | Evidence-based assessment |
|---|---:|---:|---|
| Assignment coverage | 15 | 14 | Both tasks, environment, results, limitations, AI use and group identities are explicit |
| Data cleaning and Task 1 | 15 | 12 | All major descriptive views are embedded; rule-level cleaning losses and coverage-normalised checks remain absent |
| Prediction methodology and evaluation | 20 | 14 | Target, features, parameters, temporal holdout and canonical run are documented; calibration and positive-class evidence remain absent |
| Distributed architecture explanation | 15 | 14 | Logical pipeline, local, Standalone, Docker and model architectures are clearly separated and honestly scoped to one host |
| Systems experiment quality | 15 | 7 | Deployment observations are reported accurately, but resources, samples and repetitions are not controlled |
| Result traceability | 10 | 8 | Canonical saved-model and comparison experiments are separated; immutable end-to-end run identifiers are unavailable |
| Presentation and standalone readiness | 10 | 9 | Formal A4 layout, contents, captions, readable figures, references and consistent academic typography are present |
| **Total** | **100** | **78** | **Strong report presentation; experimental validation remains the limiting factor** |

## 2. Requirement-to-evidence status

| DSA5208 requirement | Current evidence in report | Status | Remaining action |
|---|---|---|---|
| Use annual rainfall data from 2017–2024 | Dataset profile, annual summaries and figures | Met | Add immutable dataset/version identifier when available |
| Explain cleaning | Sequential cleaning rules and screening boundary | Partly met | Requires measured removal counts, cadence gaps and outlier sensitivity |
| Analyse changes within a year | Monthly and diurnal figures with discussion | Met | Optional completeness-stratified sensitivity |
| Analyse changes across years | Annual rainfall and heavy station-day figures | Partly met | Requires stable-cohort and eligible station-day denominators |
| Predict 30-minute exceedance probability | Target contract, features and GBT pipeline | Met methodologically | Correctness checks and probability calibration remain open |
| Use and evaluate Spark MLlib | GBT parameters, canonical metrics and MLlib baselines | Met | Add positive-class and calibrated-probability evaluation |
| Describe computing environment | Host, WSL2, JVM, Spark, heaps, cores and partitions | Met | Runtime allocation evidence would strengthen it |
| Present and discuss both tasks | Figures, tables, measured findings and limitations | Met | No document-only gap remains |
| Declare AI use | Dedicated declaration and reference entry | Met | Keep wording aligned with actual final usage |
| List group names and student IDs | Cover lists both authors and IDs | Met | None |
| Discuss additional work and limitations | Threats, systems protocol and proposed extension | Met | Execute selected studies only if stronger claims are desired |

## 3. Document-only improvements completed

| Improvement | Implementation in the current DOCX | Result |
|---|---|---|
| Academic navigation | Added a page-referenced contents page with internal links | Major sections and appendices are easy to locate |
| Research framing | Converted contribution bullets into three explicit research questions plus one contribution statement | Abstract now reads as a research report rather than a project log |
| Table conventions | Added numbered captions above all ten evidence tables | Data, parameters, results and proposals are clearly distinguished |
| Result provenance | Captions identify canonical, comparison, deployment and proposed-experiment contexts | Reduces the risk of treating unlike runs as one evaluation |
| Probability wording | Replaced “still required” instructions with a formal probability-evidence boundary | Unsupported operational claims are excluded without pretending missing tests were performed |
| Data-quality wording | Recast requested audits as explicit limitations of retained evidence | Removes drafting language and preserves scientific honesty |
| Systems wording | Labelled controlled benchmarks as future protocols rather than completed results | Observations and proposals cannot be confused |
| Validity wording | Reframed review notes as threats to validity and rerun conditions | Section fits an academic report structure |
| Architecture and presentation | Retained the completed architecture atlas, readable equations/logos, NUS cover treatment and Times New Roman body | No further architecture redesign is required for submission readiness |

## 4. Implemented experiment designs awaiting execution

The code paths for these items are now implemented in the corrected pipeline and documented in [VALIDATED_EXPERIMENT_RUNBOOK.md](../VALIDATED_EXPERIMENT_RUNBOOK.md). The report score remains unchanged until the experiments are executed and their tracked artifacts are reviewed; implementation alone is not scientific evidence.

| Priority | Remaining gap | Why it matters | Evidence needed to close it |
|---|---|---|---|
| P0 | Rolling-window interval semantics | An inclusive 30-minute range can include seven five-minute grid points | Known-sequence unit case and regenerated feature/model evidence if values change |
| P0 | Exact cadence for lag and future labels | Previous-row logic does not guarantee an observation exactly five minutes earlier | Offset validation, gap policy and counts of excluded/incomplete windows |
| P0 | Temporal split boundary | Late-2023 labels may reach into 2024 unless target-window end is purged | Boundary audit using prediction and label-end timestamps |
| P0 | Training/inference feature parity | Last-wet context differs when training can search 24 hours but inference loads six hours | Offline/online feature comparison for identical station-time-threshold queries |
| P0 | Immutable run lineage | Model, samples, metrics and figures need one unambiguous identifier | Run manifest with dataset checksum, feature version, row membership and artifact paths |
| P1 | Rule-level cleaning audit | The effect of null, duplicate, negative and high-reading rules is unknown | Before/after counts for every rule and year |
| P1 | Coverage-aware Task 1 checks | Changing station participation can influence annual and heavy-event comparisons | Stable-station cohort, completeness threshold and eligible station-day denominator |
| P1 | Extreme-day audit | The reported 786 mm station-day requires trace-level confirmation | Five-minute trace, duplication/unit checks and retained/excluded sensitivity |
| P1 | Probability calibration | ROC-AUC and PR-AUC evaluate ranking, not probability accuracy | Brier or log loss and reliability bins with sample counts |
| P1 | Rare-event decision quality | Weighted F1 is dominated by the majority class | Prevalence, confusion matrix and positive precision/recall/F1 at a validation-selected cutoff |
| P1 | Threshold generalisation | Five trained/tested thresholds do not prove interpolation or monotonicity | Tests at unseen thresholds and monotonicity-violation rate across ordered q |
| P1 | Baseline skill | Model value beyond persistence and prevalence is not quantified | Persistence, prevalence-only and feature-ablation results on identical test contexts |
| P1 | Controlled deployment comparison | Existing configurations vary cores, heaps, partitions and samples | Frozen sample, matched resources, warm-up plus at least three measured repetitions |
| P1 | Scaling and stage evidence | Single elapsed times do not explain bottlenecks or scalability | Speedup, efficiency, throughput, shuffle, spill, GC and task-duration distributions |
| P2 | Multi-machine scalability | All current executors share one laptop | Independent nodes plus shared storage and strong/weak scaling measurements |

## 5. Architecture assessment

| Question | Current assessment |
|---|---|
| Is the architecture meaningful for DSA5208? | Yes. It shows distributed Spark data preparation, temporal features, MLlib fitting and deployment boundaries |
| Is local versus Standalone versus Docker represented honestly? | Yes. The report explicitly distinguishes process-level distribution from physical scale-out |
| Are additional architecture changes required now? | No. A1–A7 and M1–M8 cover logical, physical, evaluation and model views at readable report scale |
| What architecture work remains optional? | A future multi-node deployment view only becomes useful if an actual multi-machine experiment is executed |
| What currently limits the systems contribution? | Experimental controls and runtime measurements, not diagram coverage |

## 6. Recommended experiment order

| Order | Experiment or repair | Minimum acceptable output | Academic value |
|---:|---|---|---|
| 1 | Validate window, cadence, split and inference contracts | Boundary examples, mismatch counts and one corrected canonical run if required | Establishes trust in every predictive result |
| 2 | Produce cleaning and coverage audit | Rule counts, stable cohort and normalised heavy-event rate | Strengthens Task 1 with defensible denominators |
| 3 | Add probability and rare-event evaluation | Prevalence, Brier/log loss, reliability and positive-class metrics | Directly supports the assignment’s probability objective |
| 4 | Run a matched-resource deployment experiment | Local[4], 2×2 Standalone and 2×2 Docker on identical rows/resources with repeats | Converts descriptive timing into a defensible systems result |
| 5 | Measure feature-reuse break-even | Recompute versus materialise/reuse for 1, 3 and 6 fits | Strong, feasible distributed-computing contribution |
| 6 | Attempt station-month halo partitioning | Exact output equality before runtime comparison | Distinctive extension if core validity is already complete |

## 7. Claims that should remain bounded

| Current evidence supports | Current evidence does not support |
|---|---|
| GBT ranks events better than the reported logistic baseline at the five tested thresholds | Calibrated operational rainfall probabilities |
| 2021–2022 are wetter and 2019 is drier in the reported station-normalised summaries | A coverage-invariant national climatology |
| Afternoon wet-reading frequency is higher in the reported data | A causal meteorological explanation |
| Local execution is fastest among the reported single-host configurations | General Docker, Standalone or multi-node Spark performance |
| Spark executes windows, aggregation and MLlib through distributed operators | Linear scale-out with additional machines |
| GPU models are supplementary implementations | A controlled CPU-versus-GPU algorithm comparison |

## 8. Submission readiness

| Gate | Status | Note |
|---|---|---|
| Formal report structure | Pass | Cover, contents, abstract, methods, results, limitations, references and appendices present |
| Authors and IDs | Pass | Both members listed on the cover |
| A4 layout and readable visuals | Pass | Final document rendered and inspected at report scale |
| Architecture honesty | Pass | Single-host and supplementary-GPU limits are explicit |
| AI declaration | Pass | Dedicated section plus reference statement |
| Predictive validity | Conditional | Outstanding P0 checks prevent stronger operational claims |
| Controlled systems evidence | Conditional | Current timing comparison remains descriptive |
| Assignment deliverables | Confirm with instructor | Written guidelines request source code, README, trained model and a PDF report; document-only submission differs from that text |

## 9. Evidence index

| Evidence | Local artifact |
|---|---|
| Assignment requirements | [project_assignment_guidelines.md](../project_assignment_guidelines.md) |
| Revised report | [Historical_Rainfall_Singapore_Report.docx](Historical_Rainfall_Singapore_Report.docx) |
| Architecture specification | [Architecture_Diagram_Specification.md](Architecture_Diagram_Specification.md) |
| Main model benchmark | [benchmark_results.csv](tables/benchmark_results.csv) |
| Deployment benchmark | [benchmark_compute.csv](tables/benchmark_compute.csv) |
| Saved-model record | [model_metadata.json](../models/model_metadata.json) |
| Annual summary | [annual_summary.csv](tables/annual_summary.csv) |
