"""Random Forest classifier (Spark MLlib) - bagged-tree comparison to GBT."""
from __future__ import annotations

from pyspark.ml.classification import RandomForestClassifier

from .base import ModelSpec
from ._mllib_common import make_pipeline

NEEDS_SCALING = False


def _build_pipeline(feature_cols, station_id_col="station_id"):
    def clf(features_col):
        return RandomForestClassifier(
            labelCol="label", featuresCol=features_col,
            numTrees=60, maxDepth=8, maxBins=128,
            subsamplingRate=0.7, seed=42,
        )

    return make_pipeline(feature_cols, clf, NEEDS_SCALING)


SPEC = ModelSpec(
    name="rf",
    framework="mllib",
    description="Random Forest (bagged trees)",
    needs_scaling=NEEDS_SCALING,
    build_pipeline=_build_pipeline,
)
