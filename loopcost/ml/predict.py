"""Loads a trained model and produces transformation-profitability predictions for new loops."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from loopcost.ml.dataset import FEATURE_COLUMNS

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _encode(features):
    row = []
    for col in FEATURE_COLUMNS:
        value = features[col]
        if col == "bound":
            value = 1 if value == "compute-bound" else 0
        elif col == "has_loop_carried_dep":
            value = int(bool(value))
        row.append(value)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _decision_path_explanation(model, x, feature_names):
    """Renders the sequence of feature-threshold tests the tree took for x, as human-readable text."""
    tree = model.tree_
    node_indicator = model.decision_path(x)
    leaf_id = model.apply(x)[0]
    node_index = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]
    x_values = np.asarray(x)[0]

    steps = []
    for node_id in node_index:
        if node_id == leaf_id:
            predicted = model.classes_[np.argmax(tree.value[node_id])]
            steps.append(f"leaf: predict {bool(predicted)}")
            continue
        feature = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        value = x_values[tree.feature[node_id]]
        direction = "<=" if value <= threshold else ">"
        steps.append(f"{feature} = {value:g} {direction} {threshold:g}")

    return "; ".join(steps)


def predict_profitability(features, transform_type):
    """Predicts whether `transform_type` is profitable for `features`, with an explanation.

    `features` is a dict with keys matching loopcost.ml.dataset.FEATURE_COLUMNS
    (oi, bound, trip_count, working_set_bytes, has_loop_carried_dep).

    Returns (profitable: bool, confidence: float, decision_path_explanation: str).
    """
    model = joblib.load(MODELS_DIR / f"{transform_type}_model.pkl")
    x = _encode(features)

    proba = model.predict_proba(x)[0]
    predicted_class_idx = int(np.argmax(proba))
    profitable = bool(model.classes_[predicted_class_idx])
    confidence = float(proba[predicted_class_idx])

    explanation = _decision_path_explanation(model, x, FEATURE_COLUMNS)

    return profitable, confidence, explanation
