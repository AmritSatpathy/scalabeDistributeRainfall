"""Local (single-JVM) compute configurations.

E0-style runs: no cluster, Spark uses N worker threads in one process.
Varying N (`local[*]`, `local[2/4/8]`) lets us study how the same job scales
with available parallelism on one machine - the simplest slice of the
compute-config axis.
"""
from __future__ import annotations

from .base import ComputeConfig

CONFIGS = [
    ComputeConfig(
        name="local-4-matched",
        master="local[4]",
        description="Matched benchmark: 4 compute cores, 4 GB driver, 32 shuffle partitions",
        driver_memory="4g",
        shuffle_partitions=32,
    ),
    ComputeConfig(
        name="local-all",
        master="local[*]",
        description="Single process, all cores (16)",
        driver_memory="20g",
        shuffle_partitions=96,
    ),
    ComputeConfig(
        name="local-2",
        master="local[2]",
        description="Single process, 2 cores",
        driver_memory="20g",
        shuffle_partitions=16,
    ),
    ComputeConfig(
        name="local-4",
        master="local[4]",
        description="Single process, 4 cores",
        driver_memory="20g",
        shuffle_partitions=32,
    ),
    ComputeConfig(
        name="local-8",
        master="local[8]",
        description="Single process, 8 cores",
        driver_memory="20g",
        shuffle_partitions=64,
    ),
]
