"""Tests that loopcost.jit produces bit-correct results and that its analysis pass actually ran."""
import numpy as np
import pytest
from numba import njit

import loopcost
from loopcost import pipeline
from loopcost.benchmarks import financial_kernels as fk

STEP6_CASES = [
    (fk.binomial_tree_price, (30,)),
    (fk.black_scholes_grid, (32,)),
    (fk.garch_11, (200,)),
    (fk.yield_curve_bootstrap, (15,)),
]


@pytest.fixture(autouse=True)
def _clear_log():
    pipeline.clear_decision_log()
    yield
    pipeline.clear_decision_log()


@pytest.mark.parametrize("fn, args", STEP6_CASES, ids=[fn.__name__ for fn, _ in STEP6_CASES])
def test_loopcost_jit_matches_plain_numba_njit(fn, args):
    wrapped_result = loopcost.jit(fn)(*args)
    plain_result = njit(fn)(*args)
    assert np.array_equal(np.asarray(wrapped_result), np.asarray(plain_result))


@pytest.mark.parametrize("fn, args", STEP6_CASES, ids=[fn.__name__ for fn, _ in STEP6_CASES])
def test_loopcost_jit_records_a_decision_log_entry(fn, args):
    assert len(pipeline.DECISION_LOG) == 0
    loopcost.jit(fn)(*args)

    assert len(pipeline.DECISION_LOG) == 1
    entry = pipeline.DECISION_LOG[0]
    assert entry["func_name"] == fn.__name__
    assert set(entry["heuristic_decision"]) == set(pipeline.TRANSFORM_TYPES)
    assert set(entry["final_decision"]) == set(pipeline.TRANSFORM_TYPES)
    assert entry["applied_transform"] in pipeline.TRANSFORM_TYPES + (None,)
    assert isinstance(entry["transform_succeeded"], bool)
    assert isinstance(entry["features"]["oi"], float)


def test_repeated_calls_reuse_the_cached_dispatcher_and_log_only_once():
    wrapped = loopcost.jit(fk.yield_curve_bootstrap)
    wrapped(10)
    wrapped(10)
    wrapped(20)  # a different argument value, but the same int64 type -> still cached

    assert len(pipeline.DECISION_LOG) == 1


def test_ml_override_is_logged_when_it_disagrees_with_the_heuristic():
    # tile/unroll/fuse's models were trained on tiny, single-class data (Step 8), so they
    # predict "profitable" almost unconditionally -- this deliberately exercises that, and
    # confirms disagreements get recorded rather than silently swallowed.
    loopcost.jit(fk.binomial_tree_price)(30)
    entry = pipeline.DECISION_LOG[0]
    assert entry["disagreements"], "expected at least one heuristic/ML disagreement to be logged"
    for transform_type in entry["disagreements"]:
        assert entry["ml_decision"][transform_type] is not None
        assert entry["ml_decision"][transform_type][0] != entry["heuristic_decision"][transform_type][0]


def test_tiling_rewrite_actually_applies_for_a_literal_bounded_loop():
    def literal_loop(c):
        total = 0.0
        for i in range(2_000_000):
            total += c[i % c.shape[0]]
        return total

    c = np.arange(1000.0)
    wrapped_result = loopcost.jit(literal_loop)(c)
    plain_result = literal_loop(c)
    assert np.isclose(wrapped_result, plain_result)

    entry = pipeline.DECISION_LOG[0]
    assert entry["applied_transform"] == "tile"
    assert entry["transform_succeeded"] is True


def test_vectorize_pipeline_class_is_real_and_produces_correct_results():
    from loopcost.transforms import vectorize

    @njit(pipeline_class=vectorize.make_pipeline_class())
    def add_arrays(a, b, c):
        for i in range(a.shape[0]):
            c[i] = a[i] + b[i]

    a = np.arange(10.0)
    b = np.arange(10.0) * 2
    out = np.zeros(10)
    add_arrays(a, b, out)
    assert np.array_equal(out, a + b)
