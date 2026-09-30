#!/usr/bin/env bash
# Stop the local Spark Standalone cluster started by start_standalone.sh.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="${PROJECT_ROOT}/cluster/pids"

if [ ! -d "${PID_DIR}" ]; then
  echo "[cluster] no pid dir; nothing to stop"
  exit 0
fi

# stop workers first, then master
for pidfile in "${PID_DIR}"/worker-*.pid "${PID_DIR}"/master.pid; do
  [ -e "${pidfile}" ] || continue
  pid="$(cat "${pidfile}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "[cluster] stopping pid ${pid} ($(basename "${pidfile}"))"
    kill "${pid}" 2>/dev/null || true
  fi
  rm -f "${pidfile}"
done

# fallback: kill any lingering Spark deploy daemons for this project
pkill -f 'org.apache.spark.deploy.master.Master' 2>/dev/null || true
pkill -f 'org.apache.spark.deploy.worker.Worker' 2>/dev/null || true
echo "[cluster] stopped."
