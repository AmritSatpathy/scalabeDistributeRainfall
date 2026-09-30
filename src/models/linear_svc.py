"""Linear SVC (Spark MLlib) - linear max-margin baseline.

Note: LinearSVC has no native probability output, but its ``rawPrediction``
margin is still usable for AUC ranking, and it produces ``prediction`` for
accuracy/F1.
"""
from __future__ import annotations

from pyspark.ml.classification import LinearSVC

from .base import ModelSpec
from ._mllib_common import make_pipeline

NEEDS_SCALING = True


def _build_pipeline(feature_cols, station_id_col="station_id"):
    def clf(features_col):
        return LinearSVC(
            labelCol="label", featuresCol=features_col,
            maxIter=50, regParam=0.01,
        )

    return make_pipeline(feature_cols, clf, NEEDS_SCALING)


SPEC = ModelSpec(
    name="linsvc",
    framework="mllib",
    description="Linear SVC (linear max-margin; no native probability)",
    needs_scaling=NEEDS_SCALING,
    build_pipeline=_build_pipeline,
)
