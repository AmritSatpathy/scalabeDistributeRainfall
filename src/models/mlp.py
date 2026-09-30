"""Multilayer Perceptron (Spark MLlib) - the only neural model inside MLlib.

The input layer width must equal the number of assembled features. We compute
it from the feature list at build time so the spec stays correct if the feature
set changes.
"""
from __future__ import annotations

from pyspark.ml.classification import MultilayerPerceptronClassifier

from .base import ModelSpec
from ._mllib_common import make_pipeline

NEEDS_SCALING = True


def _build_pipeline(feature_cols, station_id_col="station_id"):
    n_features = len(feature_cols)

    def clf(features_col):
        # Feedforward net: input -> 32 -> 16 -> 2 (binary output layer).
        return MultilayerPerceptronClassifier(
            labelCol="label", featuresCol=features_col,
            layers=[n_features, 32, 16, 2], maxIter=100, blockSize=128, seed=42,
        )

    return make_pipeline(feature_cols, clf, NEEDS_SCALING)


SPEC = ModelSpec(
    name="mlp",
    framework="mllib",
    description="Multilayer Perceptron (feedforward NN, MLlib)",
    needs_scaling=NEEDS_SCALING,
    build_pipeline=_build_pipeline,
)
