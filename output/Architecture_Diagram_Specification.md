# Architecture Diagram Specification

**Project:** Singapore Rainfall Analysis and Prediction  
**Course:** Scalable Distributed Computing for Data Science DSA5208

## Purpose and visual language

This diagram set documents the rainfall study as an academic systems-and-model architecture. It uses a restrained IEEE-style visual language: black typography, thin rules, neutral grey boundaries and low-saturation fills. NUS Orange and NUS Blue are reserved for the report cover rather than imposed on research figures. Solid arrows denote the main data or prediction path. Dashed arrows denote control, scheduling, fitting or resource-allocation paths. Dotted boundaries denote deployment isolation such as a JVM, host process or Docker container. Product logos identify technologies only; they do not imply sponsorship.

## Figure production rules

| Element | Implemented rule |
|---|---|
| Typography | Times New Roman throughout, with bold titles and restrained italic subtitles or arrow annotations. |
| Figure header | A three-pixel black rule replaces the former thick orange band. |
| Containers | Text is wrapped, vertically centred and reduced dynamically when required; no label may cross a container boundary. |
| Arrow labels | Labels sit beside the path without opaque white panels, so the line and arrowhead remain visible. |
| Long routes | Orthogonal waypoints route long connections around components and reduce crossings. |
| Logos | Transparent-background Apache Spark, Apache Parquet, Docker and PyTorch marks are scaled prominently while preserving whitespace for labels. |
| Density | Each canvas uses a compact information-bearing region; sparse architectures are reduced to a few strong components rather than stretched across empty space. |
| Meaning | Solid paths describe data, task or prediction flow; dashed paths describe control, fitting, registration or allocation. |
| Claim boundary | Deployment diagrams label the shared physical host so process or container distribution is not mistaken for multi-machine scale-out. |

## Diagram inventory

| ID | Diagram | Research purpose | Main report location |
|---|---|---|---|
| A1 | End-to-end research workflow | Establishes the complete path from annual rainfall files to Task 1 evidence, Task 2 models, and systems experiments. | Computing architecture |
| A2 | Distributed data-engineering pipeline | Makes partitioning, shuffle boundaries, materialised features, and small-result collection explicit. | Data and methods |
| A3 | Local Spark topology | Shows why `local[*]` is multithreaded Spark execution in one JVM rather than a physical cluster. | Execution environments |
| A4 | Spark Standalone topology | Shows the driver, master, executors, shared host resources, RPC, task dispatch, and shuffle. | Execution environments |
| A5 | Docker Spark topology | Shows the master and workers as containers on a Docker network with a shared bind-mounted project volume. | Execution environments |
| A6 | Common Spark MLlib pipeline | Separates common indexing/assembly from optional scaling, estimator fitting, scoring, and evaluation. | Predictive methodology |
| M1 | Gradient-Boosted Trees | Documents the primary non-linear additive tree ensemble and its probability output. | Model architecture atlas |
| M2 | Random Forest | Documents parallel bagged trees, feature subsampling, and probability averaging. | Model architecture atlas |
| M3 | Logistic Regression | Documents scaled features, a regularised linear logit, and native probability output. | Model architecture atlas |
| M4 | Linear SVC | Documents scaled features, the maximum-margin classifier, and the absence of native probability output. | Model architecture atlas |
| M5 | Spark MLlib MLP | Documents the implemented feed-forward network `[input, 32, 16, 2]`. | Model architecture atlas |
| M6 | Factorization Machines | Documents the linear term plus learned rank-8 pairwise interactions. | Model architecture atlas |
| M7 | LSTM sequence model | Documents the recent-rain sequence encoder, static-context fusion, and PyTorch classification head. | Model architecture atlas |
| M8 | Temporal CNN | Documents the standard and dilated temporal convolutions, global pooling, context fusion, and PyTorch head. | Model architecture atlas |
| A7 | Controlled experiment design | Shows how a frozen dataset and resource contract should feed matched local, Standalone, and Docker runs. | Systems evaluation |

## Detailed diagram contracts

| ID | Inputs | Processing and boundaries | Outputs and labels that must be visible |
|---|---|---|---|
| A1 | Eight annual CSV collections; station metadata | Ingest and validate in Spark; persist cleaned Parquet; branch to descriptive aggregation and temporal feature construction; fit MLlib models; run bounded sequence experiments | Report figures, saved predictive pipeline, benchmark metrics; 2017–2023 training and 2024 test boundary |
| A2 | Raw station-time rainfall records | Parse, type-check and deduplicate; year partition; station/time windows; threshold expansion; explicit shuffle at repartition/window stages; collect only compact summaries | Clean Parquet, materialised features, Task 1 summaries, Task 2 train/test matrices |
| A3 | Driver program and local data | One JVM contains Spark driver and executor threads; memory, CPU and shuffle files share one host | `local[2/4/8/*]`; no networked worker nodes; local scheduling path |
| A4 | Driver, Standalone master, shared local filesystem | Master registers workers; driver requests executors; master allocates resources; driver dispatches tasks; executors exchange shuffle blocks | 2×2 and 4×2 layouts; requested cores and heaps; same physical-host boundary |
| A5 | Job submitted in master container; bind-mounted project/data | Docker bridge network; Spark master service; scalable worker containers; executors read the same bind mount and exchange shuffle traffic | `docker-2w` and `docker-3w`; worker CPU/memory advertisement; container isolation and shared-host limitation |
| A6 | Station ID, numeric weather/context features, threshold and label | `StringIndexer`; `VectorAssembler`; optional `MinMaxScaler`; estimator; transformation; evaluator. Dashed fit path from label to estimator distinguishes training from scoring | Probability or margin, prediction, ROC-AUC and PR-AUC; explicit warning for Linear SVC probability absence |
| M1 | Unscaled assembled features | Sequential shallow trees fit residuals; learning rate 0.1; 40 rounds; depth 5; 70% subsampling | Summed margin transformed to probability; primary model role |
| M2 | Unscaled assembled features | Sixty depth-8 trees trained on subsamples; per-tree predictions run in parallel | Averaged class probability; bagging comparison role |
| M3 | Min-max scaled features | Weighted linear combination plus intercept; L2 regularisation; sigmoid link | Native probability and class prediction; interpretable linear baseline |
| M4 | Min-max scaled features | Linear score and maximum-margin optimisation with regularisation | Decision margin and class; no calibrated probability |
| M5 | Min-max scaled feature vector | Dense input layer; 32-unit hidden layer; 16-unit hidden layer; two-node output; 100 iterations | Binary class distribution and prediction |
| M6 | Min-max scaled feature vector | Linear term plus pairwise latent-factor interactions with factor size 8 | Sigmoid probability and prediction; interaction-aware linear-family model |
| M7 | Recent rainfall sequence `[B,L,1]`; static context and threshold | One-layer LSTM with hidden size 32; final state concatenated with context; dense 32-unit ReLU/dropout head | Sigmoid probability; single-device PyTorch/GPU supplement |
| M8 | Recent rainfall sequence `[B,L,1]`; static context and threshold | 32-channel kernel-3 convolution; dilation-2 convolution; adaptive average pooling; context concatenation; dense head | Sigmoid probability; single-device PyTorch/GPU supplement |
| A7 | Frozen row IDs, feature version, estimator settings and evaluation set | Run matched resource envelopes; separate warm-up from measured repetitions; collect stage/task and model metrics | Median/range runtime, speed-up, efficiency, shuffle/spill/GC, ROC-AUC and PR-AUC |

## Academic use and claim boundaries

| Boundary | Required wording in the report |
|---|---|
| Standalone and Docker | These are separate executor processes or containers on one physical laptop. They demonstrate Spark deployment and orchestration, not multi-machine scale-out. |
| GPU models | LSTM and TCN are single-device PyTorch supplements. They are not distributed Spark training and use a bounded sequence dataset. |
| MLlib logo | MLlib is represented as a labelled Apache Spark component because it has no independent official project logo. |
| Virtual machine | The host/WSL or VM boundary uses a neutral server symbol rather than a vendor logo because no specific VM product is part of the reported design. |
| Model metrics | Model diagrams show computation, not evidence of superiority. Performance claims must remain tied to the reported evaluation tables. |
| Trademark use | Apache Spark and the Spark logo are trademarks of the Apache Software Foundation. Docker marks belong to Docker, Inc. PyTorch and its logo are trademarks of the Linux Foundation. |

## Asset sources

| Asset | Source |
|---|---|
| Apache Spark logo | <https://spark.apache.org/images/spark-logo.png> |
| Apache Parquet logo | <https://spark.apache.org/images/Apache_Parquet_logo.png> |
| Docker mark | <https://www.docker.com/company/newsroom/media-resources/> |
| PyTorch mark | <https://pytorch-fdn.github.io/artwork/> |
| NUS logo | Project-supplied `assets/NUS_logo_full-horizontal.jpg` |
