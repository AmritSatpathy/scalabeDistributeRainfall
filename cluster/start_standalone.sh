#!/usr/bin/env bash
# Start a local Spark Standalone cluster using the pyspark-bundled `spark-class`
# (the pip pyspark wheel ships spark-class + the deploy jars, but NOT the
# start-master.sh/start-worker.sh sbin scripts, so we launch the daemons
# directly via their Java classes).
#
# Usage:
#   cluster/start_standalone.sh [NUM_WORKERS] [CORES_PER_WORKER] [MEM_PER_WORKER]
# Example:
#   cluster/start_standalone.sh 4 2 3g
#
# Prints the master URL (spark://HOST:7077). Logs go to cluster/logs/.
set -euo pipefail

# Defaults chosen so per-worker memory (3g) exceeds the 2g executor memory the
# standalone-*x2 compute configs request (avoids the "requires more resource
# than any Worker" scheduling deadlock).
NUM_WORKERS="${1:-2}"
CORES_PER_WORKER="${2:-2}"
MEM_PER_WORKER="${3:-3g}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
PYSPARK_DIR="$(${VENV_PY} -c 'import os,pyspark;print(os.path.dirname(pyspark.__file__))')"
SPARK_CLASS="${PYSPARK_DIR}/bin/spark-class"

LOG_DIR="${PROJECT_ROOT}/cluster/logs"
PID_DIR="${PROJECT_ROOT}/cluster/pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

MASTER_HOST="localhost"
MASTER_PORT=7077
MASTER_WEBUI=8080
MASTER_URL="spark://${MASTER_HOST}:${MASTER_PORT}"

echo "[cluster] starting master at ${MASTER_URL} (web UI :${MASTER_WEBUI})"
nohup "${SPARK_CLASS}" org.apache.spark.deploy.master.Master \
  --host "${MASTER_HOST}" --port "${MASTER_PORT}" --webui-port "${MASTER_WEBUI}" \
  > "${LOG_DIR}/master.log" 2>&1 &
echo $! > "${PID_DIR}/master.pid"

# give the master a moment to bind
sleep 5

for i in $(seq 1 "${NUM_WORKERS}"); do
  WEBUI=$((8080 + i))
  echo "[cluster] starting worker ${i} (${CORES_PER_WORKER} cores, ${MEM_PER_WORKER})"
  nohup "${SPARK_CLASS}" org.apache.spark.deploy.worker.Worker \
    --cores "${CORES_PER_WORKER}" --memory "${MEM_PER_WORKER}" \
    --webui-port "${WEBUI}" "${MASTER_URL}" \
    > "${LOG_DIR}/worker-${i}.log" 2>&1 &
  echo $! > "${PID_DIR}/worker-${i}.pid"
done

sleep 4
echo "[cluster] up. Master URL: ${MASTER_URL}"
echo "[cluster] export it for the training scripts:"
echo "    export SPARK_MASTER_URL=${MASTER_URL}"
echo "[cluster] stop with: cluster/stop_standalone.sh"
