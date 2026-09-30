"""Model registry - the single control point for *which model* is used.

Add a new model by creating ``models/<name>.py`` that exposes a ``SPEC``
(a ``ModelSpec``) and importing it here. Everything else (the training
orchestrator, the benchmark harness) selects models purely by name via
``get_model`` / ``available_models`` and never imports a concrete model file.
"""
from __future__ import annotations

from .base import ModelSpec
from . import gbt, random_forest, logreg, linear_svc, mlp, fm

# Deep-learning specs are imported lazily: importing them pulls in torch, which
# we don't want to require for pure-MLlib runs. They register only if torch is
# importable.
_DEEP_SPECS = []
try:
    from . import lstm, tcn  # noqa: F401
    _DEEP_SPECS = [lstm.SPEC, tcn.SPEC]
except Exception:  # torch not installed / unavailable
    _DEEP_SPECS = []

# name -> ModelSpec
_REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        gbt.SPEC,
        random_forest.SPEC,
        logreg.SPEC,
        linear_svc.SPEC,
        mlp.SPEC,
        fm.SPEC,
        *_DEEP_SPECS,
    )
}


def available_models() -> list[str]:
    """List of registered model names (CLI choices)."""
    return list(_REGISTRY.keys())


def get_model(name: str) -> ModelSpec:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model {name!r}; available: {', '.join(_REGISTRY)}"
        )
    return _REGISTRY[name]


def mllib_models() -> list[str]:
    return [n for n, s in _REGISTRY.items() if s.framework == "mllib"]


def describe() -> list[tuple[str, str, str]]:
    """(name, framework, description) for every model - handy for the report."""
    return [(s.name, s.framework, s.description) for s in _REGISTRY.values()]
