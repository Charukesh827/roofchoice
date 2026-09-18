"""Builds and loads labeled feature datasets used for training loop-cost prediction models."""

import pandas as pd

FEATURE_COLUMNS = ["oi", "bound", "trip_count", "working_set_bytes", "has_loop_carried_dep"]
LABEL_COLUMN = "profitable_label"

# The Step 7 sweep logs low-level Numba compilation toggles, not a column literally named
# "vectorize" -- fastmath/parallel/fastmath_parallel are all empirical stand-ins for "let the
# compiler auto-vectorize/parallelize", so they're pooled into the "vectorize" bucket to match
# loopcost.heuristic.decide's four canonical transform categories.
TRANSFORM_TYPE_SOURCES = {
    "vectorize": ("fastmath", "parallel", "fastmath_parallel"),
    "unroll": ("unroll",),
    "tile": ("tile",),
    "fuse": ("fuse",),
}


def _encode_features(subset):
    """Selects and numerically encodes the static feature columns (drops kernel/size identifiers)."""
    features = subset[FEATURE_COLUMNS].copy()
    features["bound"] = (features["bound"] == "compute-bound").astype(int)
    features["has_loop_carried_dep"] = features["has_loop_carried_dep"].astype(bool).astype(int)
    return features.reset_index(drop=True)


def load_dataset(path="data/training_data.csv"):
    """Loads training_data.csv and splits it into one (X, y) pair per transform type.

    Returns (X_by_transform, y_by_transform): dicts keyed by "tile"/"vectorize"/"unroll"/"fuse".
    Each X is a DataFrame of the numeric static features (kernel name and size are dropped as
    identifiers; "transform_applied" and "speedup" are dropped too -- the latter is what the
    label is derived from, so keeping it would leak the label). Each y is the boolean
    profitable_label for the rows where that transform type was actually applied.
    """
    df = pd.read_csv(path)

    X_by_transform = {}
    y_by_transform = {}
    for transform_type, source_labels in TRANSFORM_TYPE_SOURCES.items():
        subset = df[df["transform_type"].isin(source_labels)]
        X_by_transform[transform_type] = _encode_features(subset)
        y_by_transform[transform_type] = subset[LABEL_COLUMN].astype(bool).reset_index(drop=True)

    return X_by_transform, y_by_transform
