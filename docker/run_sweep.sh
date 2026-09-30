#!/usr/bin/env bash
# Run the 4 MLlib models against the Docker cluster for a given compute config.
# Usage: docker/run_sweep.sh <compute-config>   e.g. docker/run_sweep.sh docker-2w
# The cluster must already be up with the matching worker count.
set -u  # NOT pipefail: the per-model grep returns non-zero when it filters all
        # lines, which must not abort the merge step at the end.

COMPUTE="${1:?usage: run_sweep.sh <docker-2w|docker-3w>}"
COMPOSE="docker compose -f docker/docker-compose.yml"
HOST_CSV="output/tables/benchmark_compute.csv"
# Write inside the container to a NON-bind-mounted path (WSL2 bind-mount
# write-back from container is unreliable), then read it out via `exec cat` and
# merge. This guarantees the benchmark rows persist.
CONT_CSV="/tmp/benchmark_docker.csv"
# start fresh so we only merge this sweep's rows
${COMPOSE} exec -T spark-master rm -f "${CONT_CSV}" 2>/dev/null || true

for M in gbt rf logreg linsvc; do
  echo "=== ${M} ${COMPUTE} ==="
  ${COMPOSE} exec -T \
    -e SPARK_DRIVER_HOST=spark-master \
    -e SPARK_DRIVER_MEMORY=2g \
    -e SPARK_LOCAL_DIR=/tmp/spark-local \
    spark-master \
    python /project/src/task2_train.py \
      --model "${M}" --compute "${COMPUTE}" --no-save \
      --benchmark-csv "${CONT_CSV}" \
      --sample-fraction 0.15 --max-train-rows 1000000 \
    2>&1 | grep -aE 'train time|predict time|OVERALL|appended benchmark|requires more|accepted any resources'
done

# Read the container-accumulated CSV via `exec cat` (more reliable than
# docker cp / bind-mount write-back on WSL2) and append its data rows to the
# host CSV.
${COMPOSE} exec -T spark-master cat "${CONT_CSV}" > /tmp/benchmark_docker_host.csv 2>/dev/null
if [ -s /tmp/benchmark_docker_host.csv ]; then
  tail -n +2 /tmp/benchmark_docker_host.csv >> "${HOST_CSV}"
  echo "merged $(tail -n +2 /tmp/benchmark_docker_host.csv | wc -l) docker rows into ${HOST_CSV}"
else
  echo "WARN: container CSV ${CONT_CSV} was empty/missing"
fi
