"""Compute-config registry - the single control point for *which architecture*
(compute environment) a run uses.

Add a new environment by defining a ``ComputeConfig`` in one of the
``*_configs.py`` modules (or a new one) and registering it here. The training
orchestrator and benchmark harness select configs purely by name.
"""
from __future__ import annotations

from .base import ComputeConfig
from . import local_configs, standalone_configs, docker_configs

_REGISTRY: dict[str, ComputeConfig] = {}
for _mod in (local_configs, standalone_configs, docker_configs):
    for _cfg in _mod.CONFIGS:
        _REGISTRY[_cfg.name] = _cfg


def available_computes() -> list[str]:
    return list(_REGISTRY.keys())


def get_compute(name: str) -> ComputeConfig:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown compute config {name!r}; available: {', '.join(_REGISTRY)}"
        )
    return _REGISTRY[name]


def describe() -> list[tuple[str, str, str]]:
    """(name, master, description) for every config - handy for the report."""
    return [(c.name, c.master, c.description) for c in _REGISTRY.values()]
