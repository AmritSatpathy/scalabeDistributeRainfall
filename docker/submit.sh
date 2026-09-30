#!/usr/bin/env bash
# Submit a Task-2 training run to the Dockerized Spark cluster (tier E2).
#
# Runs the driver INSIDE the master container (where spark://spark-master:7077
# resolves), using the container's Python + bundled pyspark 3.5.3. The project
# is bind-mounted at /project, so src/ and the cleaned Parquet are available.
#
# Usage:
#   docker/submit.sh <model> <compute-config> [extra task2_train.py args...]
# Example:
#   docker/submit.sh gbt docker-2w --sample-fraction 0.15 --max-train-rows 1000000
#
# The cluster must already be up:
#   docker compose -f docker/docker-compose.yml up -d --build --scale spark-worker=2
set -euo pipefail

MODEL="${1:?usage: submit.sh <model> <compute-config> [extra args]}"
COMPUTE="${2:?usage: submit.sh <model> <compute-config> [extra args]}"
shift 2
EXTRA_ARGS=("$@")

COMPOSE="docker compose -f docker/docker-compose.yml"
MASTER_CONTAINER="rainfall-spark-master"

echo "[docker] submitting model=${MODEL} compute=${COMPUTE} to ${MASTER_CONTAINER}"
# Run inside the master container. The container's `python` already has pyspark
# 3.5.3 + pandas/numpy (see Dockerfile). We write the benchmark row to the
# compute CSV so it joins the standalone/local results.
# SPARK_DRIVER_HOST=spark-master lets executors in the worker containers
# connect back to the driver (which runs here in the master container).
${COMPOSE} exec -T \
    -e SPARK_DRIVER_HOST=spark-master \
    -e SPARK_DRIVER_MEMORY=2g \
    -e SPARK_LOCAL_DIR=/tmp/spark-local \
    spark-master \
    python /project/src/task2_train.py \
        --model "${MODEL}" \
        --compute "${COMPUTE}" \
        --no-save \
        --benchmark-csv /project/output/tables/benchmark_compute.csv \
        "${EXTRA_ARGS[@]}"
