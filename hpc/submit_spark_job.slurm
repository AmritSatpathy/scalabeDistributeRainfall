#!/bin/bash
# =====================================================================
# Slurm batch script: run the rainfall Task-2 training on an HPC Spark
# cluster (compute tier E3). PREPARED for a multi-node run; not yet
# executed (pending HPC access).
#
# It brings up an ephemeral Spark Standalone cluster ACROSS the Slurm-
# allocated nodes (one master + one worker per node), then submits the
# training driver against it. This is the standard "Spark-on-Slurm"
# pattern when the site does not provide a managed Spark/YARN service.
#
# Submit with, e.g.:
#   sbatch --nodes=4 --ntasks-per-node=1 --cpus-per-task=8 \
#          --mem=32G --time=02:00:00 hpc/submit_spark_job.slurm gbt
#
# Adjust the SBATCH defaults below to your site's partitions/limits.
# =====================================================================
#SBATCH --job-name=rainfall-spark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail

MODEL="${1:-gbt}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
mkdir -p hpc/logs

# ---- environment -----------------------------------------------------
# Load site modules (EDIT for your cluster) and activate the venv.
# module load Java/17 Python/3.12 2>/dev/null || true
source .venv/bin/activate

PYSPARK_DIR="$(python -c 'import os,pyspark;print(os.path.dirname(pyspark.__file__))')"
SPARK_CLASS="${PYSPARK_DIR}/bin/spark-class"

# ---- derive cluster layout from the Slurm allocation -----------------
NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
MASTER_NODE="${NODES[0]}"
MASTER_HOST="$(hostname -f)"          # this script runs on the first node
MASTER_PORT=7077
MASTER_URL="spark://${MASTER_HOST}:${MASTER_PORT}"
CORES_PER_WORKER="${SLURM_CPUS_PER_TASK:-8}"
# leave ~2 GB headroom for the OS/driver per node
WORKER_MEM="$(( ${SLURM_MEM_PER_NODE:-32768} / 1024 - 2 ))g"

echo "[hpc] nodes: ${NODES[*]}"
echo "[hpc] master: ${MASTER_URL}  cores/worker=${CORES_PER_WORKER}  mem/worker=${WORKER_MEM}"

# ---- start master on this node --------------------------------------
"${SPARK_CLASS}" org.apache.spark.deploy.master.Master \
    --host "${MASTER_HOST}" --port "${MASTER_PORT}" --webui-port 8080 \
    > "hpc/logs/master-${SLURM_JOB_ID}.log" 2>&1 &
sleep 10

# ---- start one worker per allocated node via srun -------------------
srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
    "${SPARK_CLASS}" org.apache.spark.deploy.worker.Worker \
        --cores "${CORES_PER_WORKER}" --memory "${WORKER_MEM}" \
        "${MASTER_URL}" \
    > "hpc/logs/workers-${SLURM_JOB_ID}.log" 2>&1 &
sleep 15

# ---- submit the training driver against the cluster -----------------
# Full, UNCAPPED dataset (--max-train-rows 0) - the point of the HPC run.
# Executors: one per node, using that node's cores/mem.
export SPARK_MASTER="${MASTER_URL}"
export SPARK_DRIVER_MEMORY="8g"
export SPARK_EXECUTOR_MEMORY="${WORKER_MEM}"
export SPARK_EXECUTOR_CORES="${CORES_PER_WORKER}"
export SPARK_EXECUTOR_INSTANCES="${SLURM_JOB_NUM_NODES}"
export SPARK_SHUFFLE_PARTITIONS=$(( SLURM_JOB_NUM_NODES * CORES_PER_WORKER * 4 ))

python src/task2_train.py \
    --model "${MODEL}" \
    --compute hpc \
    --max-train-rows 0 \
    --benchmark-csv output/tables/benchmark_hpc.csv

# ---- tear down -------------------------------------------------------
pkill -f 'org.apache.spark.deploy.worker.Worker' 2>/dev/null || true
pkill -f 'org.apache.spark.deploy.master.Master' 2>/dev/null || true
echo "[hpc] done."
