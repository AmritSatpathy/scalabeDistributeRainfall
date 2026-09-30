"""Compute-configuration abstraction - the *architecture* axis of the study.

A ``ComputeConfig`` captures everything that distinguishes one Spark
deployment/resource setup from another: the master URL and the executor/driver
resource knobs. It produces a dict of environment-variable overrides that
``spark_utils.get_spark`` already understands (SPARK_MASTER, SPARK_DRIVER_MEMORY,
SPARK_EXECUTOR_*, SPARK_SHUFFLE_PARTITIONS), so selecting a compute config never
requires touching the model or training code.

Standalone / Docker configs may additionally describe how to *launch* the
cluster (documented via ``launch_hint``); actually bringing them up is handled
by the scripts under ``cluster/`` and ``docker/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ComputeConfig:
    name: str                      # registry key / CLI value, e.g. "local8"
    master: str                    # spark master URL
    description: str = ""
    driver_memory: Optional[str] = None
    executor_memory: Optional[str] = None
    executor_cores: Optional[int] = None
    executor_instances: Optional[int] = None
    shuffle_partitions: Optional[int] = None
    launch_hint: str = ""          # how to start the cluster (standalone/docker)
    extra: dict = field(default_factory=dict)

    def env(self) -> dict[str, str]:
        """Environment overrides consumed by spark_utils.get_spark."""
        e: dict[str, str] = {"SPARK_MASTER": self.master}
        if self.driver_memory:
            e["SPARK_DRIVER_MEMORY"] = str(self.driver_memory)
        if self.executor_memory:
            e["SPARK_EXECUTOR_MEMORY"] = str(self.executor_memory)
        if self.executor_cores:
            e["SPARK_EXECUTOR_CORES"] = str(self.executor_cores)
        if self.executor_instances:
            e["SPARK_EXECUTOR_INSTANCES"] = str(self.executor_instances)
        if self.shuffle_partitions:
            e["SPARK_SHUFFLE_PARTITIONS"] = str(self.shuffle_partitions)
        return e

    def apply(self) -> None:
        """Export this config's settings into the current process environment."""
        import os

        os.environ.update(self.env())
