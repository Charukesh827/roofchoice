"""Tests for the LLVM-variant choice heuristic and its validation against the real PAPI sweep."""
from pathlib import Path

import pytest

from loopcost.benchmarks import financial_kernels as fk
from loopcost.heuristic.llvm_choice import (
    LLVM_VARIANT_SETTINGS,
    _LLVM_CHOICE_TABLE,
    choose_llvm_variant,
    classify_kernel_style,
    validate_against_sweep,
)

ALL_STYLES = ("non_fp", "transcendental", "vectorizable")
ALL_LLVM_VARIANTS = set(LLVM_VARIANT_SETTINGS)


def test_transcendental_kernels_are_classified_correctly():
    # black_scholes_grid calls math.erf; garch_11 calls sqrt -- both should be flagged
    assert classify_kernel_style(fk.black_scholes_grid, (64,)) == "transcendental"
    assert classify_kernel_style(fk.garch_11, (100,)) == "transcendental"


def test_non_transcendental_arithmetic_kernel_is_vectorizable():
    # yield_curve_bootstrap: only add/mul, no transcendental calls, no explicit array
    # vectorization blockers -- should fall into the general "vectorizable" bucket
    assert classify_kernel_style(fk.yield_curve_bootstrap, (20,)) == "vectorizable"


def test_kernel_with_no_loop_nest_is_non_fp():
    # historical_var is pure vectorized numpy/BLAS -- no explicit scalar loop for the static
    # analyzer to find, so it falls back to "non_fp" (no arithmetic detected at all)
    assert classify_kernel_style(fk.historical_var, (100, 5)) == "non_fp"


@pytest.mark.parametrize("style", ALL_STYLES)
@pytest.mark.parametrize("boundscheck", [True, False])
def test_choice_table_covers_every_style_and_boundscheck_combo(style, boundscheck):
    llvm_variant, avg_pct = _LLVM_CHOICE_TABLE[(style, boundscheck)]
    assert llvm_variant in ALL_LLVM_VARIANTS
    assert 0.0 <= avg_pct <= 100.0


def test_choose_llvm_variant_returns_a_valid_variant_and_explanation():
    llvm_variant, reason = choose_llvm_variant(fk.garch_11, (100,), boundscheck=True)
    assert llvm_variant in ALL_LLVM_VARIANTS
    assert isinstance(reason, str) and reason.strip() != ""
    assert llvm_variant in reason
    assert "boundscheck=True" in reason


def test_choose_llvm_variant_never_recommends_o0():
    # O0 measured 6-35% of peak in every bucket -- the heuristic must never pick it
    for style in ALL_STYLES:
        for boundscheck in (True, False):
            llvm_variant, _ = _LLVM_CHOICE_TABLE[(style, boundscheck)]
            assert llvm_variant != "llvm_O0"


def test_llvm_variant_settings_are_actionable():
    # every table entry must resolve to concrete env vars / set_options a caller can apply
    for llvm_variant in ALL_LLVM_VARIANTS:
        settings = LLVM_VARIANT_SETTINGS[llvm_variant]
        assert "env" in settings and "set_options" in settings
        assert isinstance(settings["env"], dict) and settings["env"]
        assert isinstance(settings["set_options"], list)


def test_numbas_own_default_needs_no_special_configuration():
    settings = LLVM_VARIANT_SETTINGS["llvm_O3_loopvec"]
    assert settings["set_options"] == []
    assert settings["env"] == {"NUMBA_OPT": "3", "NUMBA_LOOP_VECTORIZE": "1", "NUMBA_SLP_VECTORIZE": "0"}


_SWEEP_CSV = (
    Path(__file__).resolve().parents[2] / "experiments" / "sweep" / "results" / "sweep_combined.csv"
)


@pytest.mark.skipif(not _SWEEP_CSV.exists(), reason="sibling experiments/sweep results not present")
def test_heuristic_beats_always_using_numbas_default_on_the_real_sweep():
    result = validate_against_sweep()
    assert result is not None
    assert result["n"] > 0
    # the heuristic must be at least as good on average as blindly using numba's own default
    assert result["heuristic_avg"] >= result["baseline_avg"]
    # and it should be solidly good in absolute terms, not just "less bad"
    assert result["heuristic_avg"] > 0.90
