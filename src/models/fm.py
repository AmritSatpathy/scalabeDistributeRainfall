"""Factorization Machines classifier (Spark MLlib) - models pairwise
feature interactions explicitly; a middle ground between linear and trees."""
from __future__ import annotations

from pyspark.ml.classification import FMClassifier

from .base import ModelSpec
from ._mllib_common import make_pipeline

NEEDS_SCALING = True  # FM is sensitive to feature scale


def _build_pipeline(feature_cols, station_id_col="station_id"):
    def clf(features_col):
        return FMClassifier(
            labelCol="label", featuresCol=features_col,
            factorSize=8, stepSize=0.01, maxIter=50, seed=42,
        )

    return make_pipeline(feature_cols, clf, NEEDS_SCALING)


SPEC = ModelSpec(
    name="fm",
    framework="mllib",
    description="Factorization Machines (pairwise interactions)",
    needs_scaling=NEEDS_SCALING,
    build_pipeline=_build_pipeline,
)
