"""Logistic Regression (Spark MLlib) - linear baseline with calibrated probs."""
from __future__ import annotations

from pyspark.ml.classification import LogisticRegression

from .base import ModelSpec
from ._mllib_common import make_pipeline

NEEDS_SCALING = True  # linear model benefits from feature scaling


def _build_pipeline(feature_cols, station_id_col="station_id"):
    def clf(features_col):
        return LogisticRegression(
            labelCol="label", featuresCol=features_col,
            maxIter=50, regParam=0.01, elasticNetParam=0.0,
        )

    return make_pipeline(feature_cols, clf, NEEDS_SCALING)


SPEC = ModelSpec(
    name="logreg",
    framework="mllib",
    description="Logistic Regression (linear baseline)",
    needs_scaling=NEEDS_SCALING,
    build_pipeline=_build_pipeline,
)
