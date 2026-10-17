"""Model registry - the single control point for *which model* is used.

Add a new model by creating ``models/<name>.py`` that exposes a ``SPEC``
(a ``ModelSpec``) and importing it here. Everything else (the training
orchestrator, the benchmark harness) selects models purely by name via
``get_model`` / ``available_models`` and never imports a concrete model file.
"""
from __future__ import annotations

from importlib import import_module

from .base import ModelSpec
from . import gbt, random_forest, logreg, linear_svc, mlp

_DEEP_MODELS = {
    "lstm": "LSTM over recent rainfall sequence (GPU)",
    "tcn": "1D-CNN / temporal conv over rainfall sequence (GPU)",
}

# name -> ModelSpec
_REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        gbt.SPEC,
        random_forest.SPEC,
        logreg.SPEC,
        linear_svc.SPEC,
        mlp.SPEC,
    )
}


def available_models() -> list[str]:
    """List of registered model names (CLI choices)."""
    return [*_REGISTRY.keys(), *_DEEP_MODELS.keys()]


def get_model(name: str) -> ModelSpec:
    if name in _DEEP_MODELS and name not in _REGISTRY:
        # Import torch only for an explicitly requested deep-learning run.
        module = import_module(f"{__name__}.{name}")
        _REGISTRY[name] = module.SPEC
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model {name!r}; available: {', '.join(available_models())}"
        )
    return _REGISTRY[name]


def mllib_models() -> list[str]:
    return [n for n, s in _REGISTRY.items() if s.framework == "mllib"]


def describe() -> list[tuple[str, str, str]]:
    """(name, framework, description) for every model - handy for the report."""
    rows = [(s.name, s.framework, s.description) for s in _REGISTRY.values()]
    rows.extend((name, "torch", description) for name, description in _DEEP_MODELS.items()
                if name not in _REGISTRY)
    return rows
