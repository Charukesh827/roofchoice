"""Smoke tests: every financial kernel runs at a small size, plain and @njit'd, with finite output."""
import numpy as np
import pytest
from numba import njit

from loopcost.benchmarks import financial_kernels as fk

CASES = [
    (fk.monte_carlo_option_price, (200, 20)),
    (fk.binomial_tree_price, (30,)),
    (fk.black_scholes_grid, (10,)),
    (fk.historical_var, (500, 5)),
    (fk.portfolio_covariance_var, (6,)),
    (fk.rolling_volatility, (50, 5)),
    (fk.garch_11, (50,)),
    (fk.yield_curve_bootstrap, (10,)),
    (fk.markowitz_optimize, (6,)),
    (fk.correlated_asset_paths, (4, 10, 8)),
    (fk.black_scholes_pde_crank_nicolson, (40, 40)),
]


@pytest.mark.parametrize("fn, args", CASES, ids=[fn.__name__ for fn, _ in CASES])
def test_kernel_runs_plain_and_returns_finite_output(fn, args):
    result = fn(*args)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("fn, args", CASES, ids=[fn.__name__ for fn, _ in CASES])
def test_kernel_runs_under_njit_unchanged_and_returns_finite_output(fn, args):
    jitted = njit(fn)
    result = jitted(*args)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("fn, args", CASES, ids=[fn.__name__ for fn, _ in CASES])
def test_plain_and_njit_results_match(fn, args):
    plain_result = fn(*args)
    jit_result = njit(fn)(*args)
    assert np.allclose(plain_result, jit_result)
