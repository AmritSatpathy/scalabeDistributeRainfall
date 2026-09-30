"""Shared MLlib pipeline assembly used by every MLlib model spec.

Each model file only supplies its classifier stage plus whether it wants scaled
features; this module handles the common preprocessing so the feature handling
is defined in exactly one place:

    StringIndexer(station_id -> station_idx)
    VectorAssembler(base features + threshold + station_idx -> features_raw)
    [optional] MinMaxScaler(features_raw -> features)   # linear/neural/fm
    classifier(featuresCol = features | features_raw)
"""
from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, StringIndexer, VectorAssembler


def make_pipeline(feature_cols, classifier_factory, needs_scaling: bool) -> Pipeline:
    """Build a full classification Pipeline.

    Parameters
    ----------
    feature_cols:
        Numeric feature columns to assemble (already includes ``threshold`` and
        ``station_idx``).
    classifier_factory:
        ``featuresCol -> Estimator``. Called with the correct features column
        name depending on whether scaling was applied.
    needs_scaling:
        If True, MinMax-scale the assembled vector before the classifier.
    """
    idx = StringIndexer(
        inputCol="station_id", outputCol="station_idx", handleInvalid="keep"
    )
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    stages = [idx, assembler]

    if needs_scaling:
        stages.append(MinMaxScaler(inputCol="features_raw", outputCol="features"))
        features_col = "features"
    else:
        features_col = "features_raw"

    stages.append(classifier_factory(features_col))
    return Pipeline(stages=stages)
