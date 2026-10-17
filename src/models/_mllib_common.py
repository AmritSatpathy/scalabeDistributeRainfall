"""Shared MLlib pipeline assembly used by every MLlib model spec.

Each model file only supplies its classifier stage plus whether it wants scaled
features; this module handles the common preprocessing so the feature handling
is defined in exactly one place:

    StringIndexer(station_id -> station_idx)
    VectorAssembler(base features + threshold + station_idx -> features_raw)
    [optional] MinMaxScaler(features_raw -> features)   # linear/neural/fm
    classifier(featuresCol = features | features_raw)

IMPORTANT - station_idx is NOT encoded the same way for every model:

  * Tree models (GBT, RandomForest; needs_scaling=False) consume `features_raw`
    directly. StringIndexer attaches nominal (categorical) ML attribute
    metadata to station_idx, and VectorAssembler propagates it, so the trees
    can split on station identity CATEGORICALLY provided maxBins is >= the
    station cardinality (the GBT uses maxBins=128 >= ~91 stations).

  * Scaled models (LogisticRegression, LinearSVC, MLP, FMClassifier;
    needs_scaling=True) pass the assembled vector - including station_idx -
    through MinMaxScaler. Scaling drops the categorical metadata and maps the
    integer index into [0, 1], so these models treat station identity as a
    NUMERIC ORDINAL feature (an arbitrary ordering of stations), not a true
    category.

So the common claim "the pipeline treats station as categorical" is only true
for the tree models. This difference is a known property of the comparison, not
a defect; it does not affect the canonical saved GBT model.
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
