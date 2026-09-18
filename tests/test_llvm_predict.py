"""Tests for predicting the best LLVM variant from a trained model (real sweep data required
to have a model to load -- the fixture trains one if it's missing)."""
from pathlib import Path

import pytest

from loopcost.benchmarks import financial_kernels as fk
from loopcost.heuristic.llvm_choice import LLVM_VARIANT_SETTINGS
from loopcost.ml import llvm_predict as predict
from loopcost.ml import llvm_train as train

_SWEEP_CSV = (
    Path(__file__).resolve().parents[2] / "experiments" / "sweep" / "results" / "sweep_combined.csv"
)


@pytest.fixture(scope="module")
def trained_model_path():
    if not _SWEEP_CSV.exists():
        pytest.skip("sibling experiments/sweep results not present")
    train.train_best_model()
    return train.MODEL_PATH


def test_load_model_raises_a_clear_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict.load_model(tmp_path / "does_not_exist.pkl")


def test_predict_llvm_variant_returns_a_valid_variant_and_features(trained_model_path):
    variant, feats = predict.predict_llvm_variant(fk.garch_11, (100,), boundscheck=True)
    assert variant in LLVM_VARIANT_SETTINGS
    assert feats["boundscheck"] == 1
    assert feats["has_transcendental"] == 1  # garch_11 calls sqrt


def test_predict_llvm_variant_raises_for_kernel_with_no_explicit_loop(trained_model_path):
    with pytest.raises(ValueError):
        predict.predict_llvm_variant(fk.historical_var, (100, 5))
