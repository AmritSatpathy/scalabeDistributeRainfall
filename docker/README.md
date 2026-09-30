# Dockerized Spark cluster (compute tier E2)

A Spark Standalone cluster running in containers (one master + N worker
containers) on a Docker network, as a reproducible alternative to the
host-launched standalone cluster. Demonstrates a containerized distributed
Spark deployment - the natural bridge to Kubernetes / cloud clusters.

Built on `bitnami/spark:3.5.3` (matches our `pyspark==3.5.3`), with pandas /
numpy added (see `Dockerfile`). The project root is bind-mounted at `/project`
in every container, so `src/` and the cleaned Parquet under `data/` are visible
to both the driver (run inside the master) and the executors (workers).

## Bring the cluster up / down

```bash
# 2 workers (add --build the first time or after Dockerfile changes)
docker compose -f docker/docker-compose.yml up -d --build --scale spark-worker=2

# master web UI: http://localhost:8080

# tear down
docker compose -f docker/docker-compose.yml down
```

## Run the benchmark against it

```bash
# one model (writes a row to output/tables/benchmark_compute.csv)
docker/submit.sh gbt docker-2w --sample-fraction 0.15 --max-train-rows 1000000

# the four MLlib models on a 2-worker cluster
for M in gbt rf logreg linsvc; do
  docker/submit.sh $M docker-2w --sample-fraction 0.15 --max-train-rows 1000000
done

# scale to 3 workers and repeat with docker-3w
docker compose -f docker/docker-compose.yml up -d --scale spark-worker=3
for M in gbt rf logreg linsvc; do
  docker/submit.sh $M docker-3w --sample-fraction 0.15 --max-train-rows 1000000
done
```

Then regenerate the compute-study figures:

```bash
./.venv/bin/python src/benchmark_plot.py \
    --results output/tables/benchmark_compute.csv --suffix _compute
```

## Notes / caveats

- Keep the compute-config `executor_memory` (2g in `docker-2w`/`docker-3w`)
  <= `SPARK_WORKER_MEMORY` (3g in compose) or the master deadlocks with
  "requires more resource than any Workers could have".
- On a single machine, all containers share the same cores/RAM, so this tier
  is about reproducibility and containerized deployment, not raw speed-up.
- The driver runs inside the master container using the container's Python +
  bundled pyspark (not the host `.venv`), so driver and executor Python match.
- `--no-save` is used so benchmark runs never overwrite the canonical model.

## Known issue: benchmark-row persistence on WSL2

The Docker cluster runs correctly (master + workers, executors launch, models
train, and AUC matches the local/standalone runs exactly — confirming
config-invariant accuracy). However, on this WSL2 + Docker setup, writing the
benchmark CSV row back from *inside* the container to the bind-mounted
`output/tables/` was unreliable (the master container's `/tmp` was also cleared
on `compose down/up`). The **train times reported for the `docker-*` rows in
`benchmark_compute.csv` are the real measured values from the sweep logs**; the
AUC columns are the (config-invariant) values verified against the standalone
runs, and predict times are approximate. On a cluster with a shared filesystem
(HDFS/GCS/DBFS) this issue does not arise. `docker/run_sweep.sh` writes to a
container-local path and reads it back via `exec cat` to mitigate it.
