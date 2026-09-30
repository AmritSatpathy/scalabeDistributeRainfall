"""Gradient-Boosted Trees (Spark MLlib) - the primary graded model."""
from __future__ import annotations

from pyspark.ml.classification import GBTClassifier

from .base import ModelSpec
from ._mllib_common import make_pipeline

NEEDS_SCALING = False


def _build_pipeline(feature_cols, station_id_col="station_id"):
    def clf(features_col):
        # maxBins must be >= the number of station categories (~91) so the tree
        # can split on station identity as a categorical feature.
        return GBTClassifier(
            labelCol="label", featuresCol=features_col,
            maxIter=40, maxDepth=5, stepSize=0.1, maxBins=128,
            subsamplingRate=0.7, cacheNodeIds=True, seed=42,
        )

    return make_pipeline(feature_cols, clf, NEEDS_SCALING)


SPEC = ModelSpec(
    name="gbt",
    framework="mllib",
    description="Gradient-Boosted Trees (non-linear, interaction-aware)",
    needs_scaling=NEEDS_SCALING,
    build_pipeline=_build_pipeline,
)
