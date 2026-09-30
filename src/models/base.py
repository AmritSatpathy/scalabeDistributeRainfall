"""Model abstraction shared by every model in the registry.

A ``ModelSpec`` describes one model in a framework-agnostic way so that the
orchestrator (``task2_train.py``) and the benchmark harness never need to know
the details of any particular algorithm. MLlib models implement
``build_pipeline``; deep-learning models (added later, e.g. a PyTorch LSTM) can
set ``framework = "torch"`` and implement their own train/predict hooks without
disturbing the registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

# Imported lazily by MLlib specs; kept here as a type hint only.
try:  # pragma: no cover - import guard for environments without pyspark on path
    from pyspark.ml import Pipeline
except Exception:  # pragma: no cover
    Pipeline = object  # type: ignore


@dataclass
class ModelSpec:
    """Everything the orchestrator needs to run one model.

    Attributes
    ----------
    name:
        Registry key / CLI value, e.g. ``"gbt"``.
    framework:
        ``"mllib"`` (default) or ``"torch"``. Controls which code path the
        orchestrator takes.
    description:
        One-line human description for tables/reports.
    needs_scaling:
        MLlib only: whether numeric features should be MinMax-scaled before the
        classifier (linear / margin / neural models benefit; trees do not).
    build_pipeline:
        MLlib only: ``(feature_cols, station_id_col) -> pyspark.ml.Pipeline``.
    build_torch:
        Torch only: factory that returns an object exposing ``fit`` / ``predict``
        (wired up when the deep-learning models are added).
    """

    name: str
    framework: str = "mllib"
    description: str = ""
    needs_scaling: bool = False
    build_pipeline: Optional[Callable] = None
    build_torch: Optional[Callable] = None
    extra: dict = field(default_factory=dict)
