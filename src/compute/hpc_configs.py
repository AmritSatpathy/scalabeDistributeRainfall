"""HPC / multi-node cluster compute configuration (tier E3).

Unlike the local/standalone/docker configs, the HPC config does NOT hard-code a
master or resource sizing: those are derived from the Slurm allocation at run
time and exported as SPARK_* environment variables by
``hpc/submit_spark_job.slurm``. This config is therefore a thin marker that
lets ``task2_train.py --compute hpc`` participate in the registry while
deferring every knob to the environment (env wins in spark_utils.get_spark).

PREPARED for a future multi-node run; not yet executed (pending HPC access).
"""
from __future__ import annotations

import os

from .base import ComputeConfig

# Master + executor sizing come from the environment (set by the Slurm script).
# We fall back to a placeholder master only so the object is well-formed; the
# real value is SPARK_MASTER exported at submit time.
_MASTER = os.environ.get("SPARK_MASTER", "spark://HPC-MASTER:7077")

CONFIGS = [
    ComputeConfig(
        name="hpc",
        master=_MASTER,
        description="HPC multi-node Spark Standalone (sizing from Slurm/env)",
        # Intentionally leave memory/cores/instances unset here so the
        # environment (exported by the Slurm script) fully controls them.
        launch_hint="sbatch --nodes=4 --cpus-per-task=8 --mem=32G "
                    "hpc/submit_spark_job.slurm gbt",
    ),
]
