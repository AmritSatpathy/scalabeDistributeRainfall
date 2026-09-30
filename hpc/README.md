# HPC / cluster runs (compute tier E3) — PREPARED, not yet executed

Scripts to run the **full, uncapped** rainfall training on a real multi-node
cluster. On the local machine we always cap training at `--max-train-rows` so
the fit fits in memory; the point of the HPC run is to drop that cap
(`--max-train-rows 0`) and train on the full ~240M-row (48.5M base × 5
thresholds) dataset.

These are prepared in advance — we do not have HPC access yet — so they will
need small site-specific edits (module names, partitions, storage paths).

## Option A — Spark-on-Slurm (self-managed cluster)

`submit_spark_job.slurm` allocates N nodes, starts an ephemeral Spark
Standalone cluster across them (master on node 0, one worker per node via
`srun`), submits the training driver with `--compute hpc`, then tears the
cluster down. Resource sizing (executor cores/memory/instances,
shuffle partitions) is derived from the Slurm allocation and exported as
`SPARK_*` env vars, which `spark_utils.get_spark` picks up.

```bash
# edit the module load lines + SBATCH partition/limits for your site first
sbatch --nodes=4 --ntasks-per-node=1 --cpus-per-task=8 --mem=32G \
       --time=02:00:00 hpc/submit_spark_job.slurm gbt

# results land in output/tables/benchmark_hpc.csv; logs in hpc/logs/
```

Scale the study by resubmitting with `--nodes=1,2,4,8` to get a true
multi-node speed-up curve (the local runs only vary parallelism on one box).

## Option B — Managed Spark (Dataproc / Databricks / EMR)

The code is plain PySpark with the master/resources driven by env vars, so it
runs unchanged on a managed cluster. Typical Dataproc example:

```bash
# stage cleaned data + code to cloud storage
gsutil -m cp -r data/processed/rainfall_clean gs://<bucket>/rainfall/clean
gsutil -m cp -r src gs://<bucket>/rainfall/src

# submit (cluster already created with N workers)
gcloud dataproc jobs submit pyspark gs://<bucket>/rainfall/src/task2_train.py \
    --cluster=<cluster> --region=<region> \
    -- --model gbt --compute hpc --max-train-rows 0 \
       --features gs://<bucket>/rainfall/features_base
```

On Databricks: upload `src/` to the workspace/repo, put the Parquet on DBFS,
and run `task2_train.py` from a notebook/job with the cluster's Spark session
(remove the local `master()` — env-driven config already supports this).

## Data staging checklist

1. **Do NOT upload the raw CSVs** (~7.8 GB). Upload the cleaned Parquet
   (`data/processed/rainfall_clean`, far smaller) — or regenerate on the
   cluster with `download_data.py` + `clean_data.py`.
2. For Task 2, also stage `data/processed/features_base` (materialised
   features) to skip recomputing the windows, or run
   `task2_build_features.py` on the cluster once.
3. Point paths at the cluster filesystem (`hdfs://`, `gs://`, `dbfs:/`). The
   single place to change defaults is `src/config.py` (`PROCESSED_DIR` /
   `CLEAN_PARQUET`), or pass `--features <url>` explicitly.
4. Ensure the cluster Python is 3.10+ (our code uses modern typing) and has
   `pyspark`, `pandas`, `numpy`. Match `PYSPARK_PYTHON` on driver + executors.
5. Keep `spark.sql.session.timeZone=Asia/Singapore` (already set in
   `get_spark`) so timestamp features match SGT.

## Expected outcome vs local

- Removing the 8M cap lets the model see all positives → likely better AUC-PR
  on the rare high thresholds than the capped local runs.
- With multiple physical nodes, wall-clock training should **decrease** as
  nodes increase (real scale-out) — unlike the single-machine clusters where
  distribution only added overhead.
- Record the cluster's node/VM spec in the report's environment section.
