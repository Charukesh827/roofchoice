"""Tests for the static-feature dataset used to train the LLVM-choice ML model."""
from pathlib import Path

import pytest

from loopcost.benchmarks import financial_kernels as fk
from loopcost.ml.llvm_dataset import FEATURE_COLUMNS, extract_static_features, load_dataset

_SWEEP_CSV = (
    Path(__file__).resolve().parents[2] / "experiments" / "sweep" / "results" / "sweep_combined.csv"
)


def test_extract_static_features_returns_none_for_kernel_with_no_explicit_loop():
    # historical_var is pure vectorized numpy/BLAS -- no explicit scalar loop for the static
    # analyzer to find (same case documented in classify_kernel_style)
    assert extract_static_features(fk.historical_var, (100, 5)) is None


def test_extract_static_features_returns_all_expected_keys():
    feats = extract_static_features(fk.garch_11, (100,))
    assert feats is not None
    expected_keys = set(FEATURE_COLUMNS) - {"fastmath", "boundscheck"}
    assert expected_keys.issubset(feats.keys())


def test_extract_static_features_flags_transcendental_kernel():
    feats = extract_static_features(fk.garch_11, (100,))  # calls sqrt
    assert feats["has_transcendental"] == 1


def test_extract_static_features_concretizes_shape_driven_trip_counts():
    # yield_curve_bootstrap's loop bound is derived from an array's .shape, not a literal --
    # without concretizing against the real call args, static_ai/float_ops silently read 0
    feats = extract_static_features(fk.yield_curve_bootstrap, (20,))
    assert feats["float_ops"] > 0
    assert feats["log_trip_count"] > 0


@pytest.mark.skipif(not _SWEEP_CSV.exists(), reason="sibling experiments/sweep results not present")
def test_load_dataset_produces_a_row_per_kernel_njit_variant_combo():
    X, y, meta = load_dataset()
    assert len(X) == len(y) == len(meta)
    assert len(X) > 0
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(meta.columns) == {"kernel", "njit_variant", "suite"}
    # contour_integral is known to fail static IR extraction and must be excluded, not silently
    # dropped without accounting -- every OTHER kernel's 4 njit variants should be present
    assert "contour_integral" not in meta["kernel"].values
    assert (meta["kernel"].value_counts() == 4).all()
