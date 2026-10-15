# EXPERIMENT MATRIX — compute environments × models

> **STATUS — COMPLETE.** All models (M0–M7: GBT, RF, LogReg, LinearSVC, MLP, FM,
> LSTM, TCN) and compute tiers E0–E2 (`local[*]`, Spark Standalone 2×2/4×2,
> Docker 2w/3w) plus the GPU tier (E4) are built, run, and benchmarked; figures
> + report are done — all on a single machine. A multi-node HPC tier was
> considered but **dropped (no cluster access)** and is out of scope. The "TODO"
> markers below reflect the *original* plan and are retained for traceability;
> treat them as done. See `PROGRESS.md` and `report/REPORT.md` for final results.

Master plan for the expanded project. The goal is to move from "we ran one
PySpark job" to a proper **distributed-computing + model-comparison study**:
run the rainfall prediction (Task 2) across several **compute environments** and
several **models**, benchmark them, and discuss the trade-offs.

Two axes:
- **Compute environment (how/where Spark or the model runs)** — the
  distributed-systems axis.
- **Model** — the machine-learning axis.

Everything is anchored to what is *actually feasible* on the verified hardware
(see "Feasibility facts" at the bottom).

---

## Axis 1 — Compute environments (tiers)

| ID | Environment | Spark master | Distributed? | Uses GPU? | Status | What it demonstrates |
|----|-------------|--------------|--------------|-----------|--------|----------------------|
| **E0** | Single-process PySpark | `local[*]` | No (1 JVM, threads) | No | **DONE** | Baseline; simplest setup |
| **E1** | Local Standalone cluster | `spark://` (1 master + N workers, same host) | Yes (real executor JVMs, network shuffle) | No | TODO | Deploy/submit to a cluster; read Spark UI; scale-out vs scale-up |
| **E2** | Dockerized Spark cluster | `spark://` (master + worker **containers**) | Yes (containers on a Docker net) | No | TODO | Reproducible containerized cluster; portability; bridge to K8s/cloud |
| **E4** | GPU single-node (deep learning) | n/a (PyTorch, not Spark) | No (1 GPU) | **Yes** | TODO | GPU training for the LSTM; Spark used only for data prep |

_(A multi-node HPC/cluster tier was considered but dropped — no cluster access.)_

Notes / honest caveats:
- **E1/E2 on one physical machine will likely be *slower* than E0** (serialization
  + network + container overhead on the same 16 cores / 24 GB). That is a valid,
  reportable finding: cluster mode is about scaling *out*, not speeding up one box.
- Real speed-up (and uncapped full-data training) would need multiple physical
  nodes, which were not available — so this is noted as a limitation, not run.
- **E4** is bounded by the **4 GB VRAM** RTX 3050 Ti — modest LSTM, careful batch/
  sequence sizing.

---

## Axis 2 — Models

| ID | Model | Library / paradigm | In MLlib? | Distributed on Spark? | GPU? | Effort | Status |
|----|-------|--------------------|-----------|-----------------------|------|--------|--------|
| **M0** | Gradient-Boosted Trees | Spark MLlib | Yes | Yes | No | — | **DONE** (graded model) |
| **M1** | Logistic Regression | Spark MLlib | Yes | Yes | No | — | **DONE** (baseline) |
| **M2** | Random Forest | Spark MLlib | Yes | Yes | No | Low | TODO |
| **M3** | Linear SVC (SVM) | Spark MLlib | Yes | Yes | No | Low | TODO |
| **M4** | Multilayer Perceptron | Spark MLlib | Yes (only NN in MLlib) | Yes | No | Low | TODO |
| **M5** | Factorization Machines | Spark MLlib | Yes | Yes | No | Low-Med | Optional |
| **M6** | XGBoost / LightGBM on Spark | XGBoost4J / SynapseML | No | Yes | **Optional** | Med-High | Stretch |
| **M7** | LSTM / GRU | PyTorch | No | No (single GPU) | **Yes** | Med | TODO (key comparison) |
| **M8** | 1D-CNN / Temporal Conv | PyTorch | No | No | Yes | Med | Optional |
| **M9** | Distributed DL | Horovod / Elephas | No | Yes | Yes | High | Skip (overkill) |

All MLlib models (M0–M5) share the **same** feature pipeline, threshold-as-
feature design, temporal split, and AUC-ROC/AUC-PR evaluation, so they slot into
the existing `--model` switch with minimal code.

---

## The combined matrix (environment × model)

Legend: ✅ done · ▶ planned (core) · ◇ stretch · ✖ not applicable

| Model \\ Env | E0 local[*] | E1 Standalone | E2 Docker | E4 GPU |
|--------------|-------------|---------------|-----------|--------|
| **M0 GBT**        | ✅ | ✅ | ✅ | ✖ (CPU model) |
| **M1 LogReg**     | ✅ | ✅ | ✅ | ✖ |
| **M2 RandomForest** | ✅ | ✅ | ✅ | ✖ |
| **M3 LinearSVC**  | ✅ | ✅ | ✅ | ✖ |
| **M4 MLP (MLlib)** | ✅ | ◇ | ◇ | ✖ |
| **M5 FM**         | ✅ | ◇ | ◇ | ✖ |
| **M6 XGBoost/Spark** | ◇ | ◇ | ◇ | ◇ (GPU) |
| **M7 LSTM**       | ✖ (not Spark) | ✖ | ✖ | ✅ |
| **M8 1D-CNN (TCN)** | ✖ | ✖ | ✖ | ✅ |

Reading the matrix:
- **Distributed-Spark study** = the MLlib rows across E0→E1→E2.
- **Model-quality study** = all models on their best single environment
  (MLlib on E0, LSTM/TCN on E4), compared on AUC.
- **Scaling study** = one fixed model (M0 GBT) across E0→E1→E2 with varying
  executor/worker counts, plotting runtime (overhead, not speed-up, on one box).

---

## Cross-cutting deliverables (enable the matrix)

| ID | Deliverable | Why | Status |
|----|-------------|-----|--------|
| **D1** | Cluster-ready config: parameterize `--master`, executor cores/memory, shuffle partitions via CLI/env in `spark_utils`/`config` | One codebase runs on E0–E2 unchanged | TODO |
| **D2** | `--model` extended to M2–M5 (+ optional M6) in `task2_train.py` | Model-comparison table | TODO |
| **D3** | Benchmark harness `src/benchmark.py` — times clean/feature-build/train per (env, model, #executors), logs to CSV | Scaling & comparison data | TODO |
| **D4** | Benchmark plots `src/benchmark_plot.py` — runtime vs #executors, speed-up curves, model AUC bars | Report figures | TODO |
| **D5** | Local Standalone launch scripts (`cluster/standalone_*.sh`) | Run E1 | TODO |
| **D6** | Docker `docker/docker-compose.yml` + `submit.sh` (configurable worker replicas) | Run E2 | TODO |
| **D8** | `src/task2_lstm.py` (PyTorch GPU) — same features/label/split/metrics | Run E4 / M7 | TODO |
| **D9** | Report sections: cluster deployment, scaling study, model comparison; update RATIONALE/PROGRESS/README | Marks | TODO |

---

## Suggested execution order (value-per-effort)

1. **D1** cluster-ready config (unlocks E1/E2 with one code change).
2. **D2** add MLlib models M2–M4 (cheap, immediate comparison table).
3. **D5 + run E1** local standalone; capture baseline vs cluster timing.
4. **D6 + run E2** Docker cluster; capture timing.
5. **D3 + D4** benchmark harness + plots across E0/E1/E2 × {M0,M2,M4}.
6. **D8** LSTM on GPU (E4); add to model-comparison.
7. **D9** fold everything into the report.

---

## Feasibility facts (verified on this machine)

- **GPU:** NVIDIA RTX 3050 Ti Laptop, **4 GB VRAM**, driver 610.88, visible in
  WSL. CUDA-capable → LSTM feasible with modest model/batch sizes. PyTorch
  **not yet installed** (CUDA wheel ≈ 2.5 GB download).
- **Docker:** Docker 29.7.2 + Compose v5.5.0 present in WSL → E2 feasible.
- **Host/WSL:** 32 GB host, **WSL capped at 24 GB** RAM + 8 GB swap, 16 cores.
  E1/E2 containers/executors share this, so expect overhead, not speed-up,
  locally.
- **Data:** cleaned Parquet `data/processed/rainfall_clean` (48.5M rows, 91
  stations); base features `data/processed/features_base` (47.6M rows) already
  materialised.
- **No HPC/multi-node cluster** was available, so that tier is out of scope.

---

## Open decisions (resolved)

1. Scope — all MLlib models + LSTM/TCN across local/Standalone/Docker. ✅
2. PyTorch-CUDA installed for the LSTM/TCN. ✅
3. Group member names + student IDs — still to be filled into the report.
