"""Dockerized Spark cluster compute configurations (E2).

These target the Spark master container defined in ``docker/docker-compose.yml``
(service name ``spark-master``, port 7077). Worker count is controlled by the
compose ``--scale`` flag; the configs here just record the intended layout and
the resource knobs. Jobs are submitted from inside the master container (see
``docker/submit.sh``), where ``spark://spark-master:7077`` resolves.
"""
from __future__ import annotations

from .base import ComputeConfig

_MASTER = "spark://spark-master:7077"

CONFIGS = [
    ComputeConfig(
        name="docker-2x2-matched",
        master=_MASTER,
        description="Matched benchmark: 2 containers x 2 cores, 2 GB each, 32 shuffle partitions",
        driver_memory="4g", executor_memory="2g", executor_cores=2,
        executor_instances=2, shuffle_partitions=32,
        launch_hint="docker compose -f docker/docker-compose.yml up -d --scale spark-worker=2",
    ),
    ComputeConfig(
        name="docker-2w",
        master=_MASTER,
        description="Docker cluster: 2 worker containers",
        driver_memory="2g",
        executor_memory="1g",   # MUST be well under worker mem (3g) + overhead
        executor_cores=2,
        executor_instances=2,
        shuffle_partitions=32,
        launch_hint="docker compose -f docker/docker-compose.yml up -d --scale spark-worker=2",
    ),
    ComputeConfig(
        name="docker-3w",
        master=_MASTER,
        description="Docker cluster: 3 worker containers",
        driver_memory="2g",
        executor_memory="1g",
        executor_cores=2,
        executor_instances=3,
        shuffle_partitions=48,
        launch_hint="docker compose -f docker/docker-compose.yml up -d --scale spark-worker=3",
    ),
]
