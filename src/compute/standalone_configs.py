"""Spark Standalone cluster compute configurations (E1).

These target a Standalone master started on the local host (see
``cluster/`` launch scripts). The master URL is read from the SPARK_MASTER_URL
env var if set (the launch script exports it), otherwise defaults to the
conventional local master. Executor cores/instances model different
worker layouts, e.g. 2 workers x 2 cores vs 4 workers x 2 cores.
"""
from __future__ import annotations

import os

from .base import ComputeConfig

_MASTER = os.environ.get("SPARK_MASTER_URL", "spark://localhost:7077")

# NOTE: executor_memory MUST be <= the per-worker memory used when starting the
# cluster (cluster/start_standalone.sh <workers> <cores> <mem>), otherwise the
# Standalone master reports "requires more resource than any of Workers could
# have" and the app hangs. We use 2g executors and start workers with 3g.
CONFIGS = [
    ComputeConfig(
        name="standalone-2x2",
        master=_MASTER,
        description="Standalone: 2 executors x 2 cores",
        driver_memory="4g",
        executor_memory="2g",
        executor_cores=2,
        executor_instances=2,
        shuffle_partitions=32,
        launch_hint="cluster/start_standalone.sh 2 2 3g   # workers cores mem",
    ),
    ComputeConfig(
        name="standalone-4x2",
        master=_MASTER,
        description="Standalone: 4 executors x 2 cores",
        driver_memory="4g",
        executor_memory="2g",
        executor_cores=2,
        executor_instances=4,
        shuffle_partitions=64,
        launch_hint="cluster/start_standalone.sh 4 2 3g",
    ),
]
