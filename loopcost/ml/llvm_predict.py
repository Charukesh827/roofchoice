"""Loads the trained LLVM-choice model (models/llvm_choice_model.pkl, from llvm_train.py) and
predicts the best LLVM optimization variant for a given kernel function + njit flags.
"""

from pathlib import Path

import joblib
import pandas as pd

from loopcost.ml.llvm_dataset import extract_static_features
from loopcost.ml.llvm_train import MODEL_PATH


def load_model(model_path=None):
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"no trained model at {path}; run loopcost.ml.llvm_train.train_best_model() first"
        )
    return joblib.load(path)


def predict_llvm_variant(fn, args, fastmath=False, boundscheck=False, model_path=None):
    """Predicts the best LLVM variant for fn(*args) under the given njit flags, using the
    trained model saved by llvm_train.train_best_model().

    Returns (predicted_variant, feature_dict) -- feature_dict is the exact static feature
    vector fed to the model, useful for inspecting why it predicted what it did.
    """
    bundle = load_model(model_path)
    model, feature_columns = bundle["model"], bundle["feature_columns"]

    feats = extract_static_features(fn, args)
    if feats is None:
        raise ValueError(f"{fn.__name__}: no explicit loop nest found; cannot extract static features")
    feats = dict(feats)
    feats["fastmath"] = int(fastmath)
    feats["boundscheck"] = int(boundscheck)

    X = pd.DataFrame([feats], columns=feature_columns)
    prediction = model.predict(X)[0]
    return prediction, feats
