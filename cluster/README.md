# Local Spark Standalone cluster (compute tier E1)

Starts a real Spark Standalone cluster (separate master + worker JVMs, network
shuffle) on this machine, so the MLlib models run distributed via a
`spark://` master instead of the single-process `local[*]`.

We launch the daemons via the pyspark-bundled `spark-class` (the pip wheel
ships `spark-class` and the deploy jars, but not the `start-master.sh` /
`start-worker.sh` scripts), so **no separate Spark download is needed**.

## Start / stop

```bash
# start: NUM_WORKERS CORES_PER_WORKER MEM_PER_WORKER
cluster/start_standalone.sh 4 2 3g

# it prints the master URL; export it so the training scripts use it:
export SPARK_MASTER_URL=spark://localhost:7077

# stop
cluster/stop_standalone.sh
```

Master web UI: http://localhost:8080 (workers at :8081, :8082, ...).
Logs: `cluster/logs/`. PIDs: `cluster/pids/`.

## Run the benchmark against the cluster

The compute configs `standalone-2x2` / `standalone-4x2` read `SPARK_MASTER_URL`
and set executor cores/instances. Match the config to how many workers you
started, then run models with `--no-save`:

```bash
# start a 4-worker x 2-core cluster
cluster/start_standalone.sh 4 2 3g
export SPARK_MASTER_URL=spark://localhost:7077

# one model
./.venv/bin/python src/task2_train.py --model gbt --compute standalone-4x2 \
    --no-save --sample-fraction 0.15 --max-train-rows 1000000

# or the whole grid via the harness
./.venv/bin/python src/benchmark.py \
    --models gbt rf logreg linsvc \
    --computes standalone-4x2 \
    --sample-fraction 0.15 --max-train-rows 1000000

cluster/stop_standalone.sh
```

## Notes / caveats

- On a single physical machine a Standalone cluster is usually **slower** than
  `local[*]` (serialization + network overhead on the same cores). That is the
  expected scale-out-vs-scale-up finding for the report, not a bug.
- We run the Python driver directly (not `spark-submit`); the executors use the
  same machine + Python, so no extra packaging is required for this local
  cluster.
- Use a small fixed `--sample-fraction` + low `--max-train-rows` so each cell
  finishes quickly and the timing comparison across configs is clean.
