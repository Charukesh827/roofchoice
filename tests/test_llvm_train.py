"""Tests for the LLVM-choice ML model comparison/training (real sweep data required)."""
from pathlib import Path

import pytest

from loopcost.ml import llvm_train as train
from loopcost.ml.llvm_dataset import load_dataset

_SWEEP_CSV = (
    Path(__file__).resolve().parents[2] / "experiments" / "sweep" / "results" / "sweep_combined.csv"
)


def test_heuristic_table_classifier_matches_the_hand_derived_table():
    import pandas as pd

    from loopcost.heuristic.llvm_choice import _LLVM_CHOICE_TABLE

    clf = train.HeuristicTableClassifier().fit(None)
    X = pd.DataFrame(
        [
            {"has_transcendental": 1, "float_ops": 5, "boundscheck": 0},
            {"has_transcendental": 0, "float_ops": 0, "boundscheck": 1},
            {"has_transcendental": 0, "float_ops": 3, "boundscheck": 1},
        ]
    )
    preds = clf.predict(X)
    assert preds[0] == _LLVM_CHOICE_TABLE[("transcendental", False)][0]
    assert preds[1] == _LLVM_CHOICE_TABLE[("non_fp", True)][0]
    assert preds[2] == _LLVM_CHOICE_TABLE[("vectorizable", True)][0]


@pytest.mark.skipif(not _SWEEP_CSV.exists(), reason="sibling experiments/sweep results not present")
def test_pct_of_peak_is_one_when_predicting_the_true_best_variant():
    import numpy as np

    X, y, meta = load_dataset()
    fracs = train._pct_of_peak(meta, y, y)  # predicting the true label must always hit peak
    assert np.allclose(fracs, 1.0)


@pytest.mark.skipif(not _SWEEP_CSV.exists(), reason="sibling experiments/sweep results not present")
def test_evaluate_candidates_produces_a_sorted_comparison_table():
    X, y, meta = load_dataset()
    df = train.evaluate_candidates(X, y, meta)

    assert set(df["model"]) == set(train.CANDIDATES)
    assert list(df.columns) == ["model", "stratified_acc", "loko_acc", "loko_pct_of_peak"]
    # sorted descending by the primary (loko_pct_of_peak) metric
    assert (df["loko_pct_of_peak"].diff().dropna() <= 1e-9).all()
    for col in ("stratified_acc", "loko_acc", "loko_pct_of_peak"):
        assert ((df[col] >= 0) & (df[col] <= 1)).all()


@pytest.mark.skipif(not _SWEEP_CSV.exists(), reason="sibling experiments/sweep results not present")
def test_train_best_model_saves_a_model_and_report():
    df, best_name, best_model = train.train_best_model()

    assert best_name in train.CANDIDATES
    assert best_name != "heuristic_table"  # the report picks an actually-trained model
    assert train.MODEL_PATH.exists()
    assert train.REPORT_TXT.exists() and train.REPORT_TXT.stat().st_size > 0

    import joblib

    bundle = joblib.load(train.MODEL_PATH)
    assert "model" in bundle and "feature_columns" in bundle
    assert hasattr(bundle["model"], "predict")
