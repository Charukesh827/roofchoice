"""Defines representative financial computation kernels used as benchmark loop workloads."""

import math

import numpy as np

_S0 = 100.0
_K = 100.0
_R = 0.05
_SIGMA = 0.2
_T = 1.0
_SEED = 42


def monte_carlo_option_price(n_paths, n_steps):
    """Prices a European call by Monte Carlo simulation of GBM paths."""
    dt = _T / n_steps
    nudt = (_R - 0.5 * _SIGMA * _SIGMA) * dt
    sigsdt = _SIGMA * np.sqrt(dt)

    np.random.seed(_SEED)
    log_paths = np.full(n_paths, np.log(_S0))
    for _ in range(n_steps):
        z = np.random.standard_normal(n_paths)
        log_paths = log_paths + nudt + sigsdt * z

    payoffs = np.maximum(np.exp(log_paths) - _K, 0.0)
    return np.exp(-_R * _T) * np.mean(payoffs)


def binomial_tree_price(n_steps):
    """Prices an American call via backward induction on a binomial tree."""
    dt = _T / n_steps
    u = np.exp(_SIGMA * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(_R * dt) - d) / (u - d)
    disc = np.exp(-_R * dt)

    values = np.empty(n_steps + 1)
    for i in range(n_steps + 1):
        s_final = _S0 * (u ** (n_steps - i)) * (d ** i)
        values[i] = max(s_final - _K, 0.0)

    for step in range(n_steps - 1, -1, -1):
        for i in range(step + 1):
            continuation = disc * (p * values[i] + (1.0 - p) * values[i + 1])
            s_node = _S0 * (u ** (step - i)) * (d ** i)
            intrinsic = max(s_node - _K, 0.0)
            values[i] = max(continuation, intrinsic)

    return values[0]


def black_scholes_grid(n_options):
    """Prices a grid of European calls over a range of strikes/expiries via closed-form Black-Scholes."""
    strikes = np.linspace(50.0, 150.0, n_options)
    expiries = np.linspace(0.1, 2.0, n_options)
    prices = np.empty(n_options)

    for i in range(n_options):
        k = strikes[i]
        t = expiries[i]
        d1 = (np.log(_S0 / k) + (_R + 0.5 * _SIGMA * _SIGMA) * t) / (_SIGMA * np.sqrt(t))
        d2 = d1 - _SIGMA * np.sqrt(t)
        cdf_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        cdf_d2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
        prices[i] = _S0 * cdf_d1 - k * np.exp(-_R * t) * cdf_d2

    return prices


def black_scholes_grid_unrolled(n_options):
    """Manually unrolled (factor 4) variant of black_scholes_grid: 4 grid points per loop iteration."""
    strikes = np.linspace(50.0, 150.0, n_options)
    expiries = np.linspace(0.1, 2.0, n_options)
    prices = np.empty(n_options)

    unrolled_end = (n_options // 4) * 4
    i = 0
    while i < unrolled_end:
        k0, t0 = strikes[i], expiries[i]
        k1, t1 = strikes[i + 1], expiries[i + 1]
        k2, t2 = strikes[i + 2], expiries[i + 2]
        k3, t3 = strikes[i + 3], expiries[i + 3]

        d1_0 = (np.log(_S0 / k0) + (_R + 0.5 * _SIGMA * _SIGMA) * t0) / (_SIGMA * np.sqrt(t0))
        d1_1 = (np.log(_S0 / k1) + (_R + 0.5 * _SIGMA * _SIGMA) * t1) / (_SIGMA * np.sqrt(t1))
        d1_2 = (np.log(_S0 / k2) + (_R + 0.5 * _SIGMA * _SIGMA) * t2) / (_SIGMA * np.sqrt(t2))
        d1_3 = (np.log(_S0 / k3) + (_R + 0.5 * _SIGMA * _SIGMA) * t3) / (_SIGMA * np.sqrt(t3))

        d2_0 = d1_0 - _SIGMA * np.sqrt(t0)
        d2_1 = d1_1 - _SIGMA * np.sqrt(t1)
        d2_2 = d1_2 - _SIGMA * np.sqrt(t2)
        d2_3 = d1_3 - _SIGMA * np.sqrt(t3)

        prices[i] = _S0 * (0.5 * (1.0 + math.erf(d1_0 / math.sqrt(2.0)))) - k0 * np.exp(-_R * t0) * (
            0.5 * (1.0 + math.erf(d2_0 / math.sqrt(2.0)))
        )
        prices[i + 1] = _S0 * (0.5 * (1.0 + math.erf(d1_1 / math.sqrt(2.0)))) - k1 * np.exp(-_R * t1) * (
            0.5 * (1.0 + math.erf(d2_1 / math.sqrt(2.0)))
        )
        prices[i + 2] = _S0 * (0.5 * (1.0 + math.erf(d1_2 / math.sqrt(2.0)))) - k2 * np.exp(-_R * t2) * (
            0.5 * (1.0 + math.erf(d2_2 / math.sqrt(2.0)))
        )
        prices[i + 3] = _S0 * (0.5 * (1.0 + math.erf(d1_3 / math.sqrt(2.0)))) - k3 * np.exp(-_R * t3) * (
            0.5 * (1.0 + math.erf(d2_3 / math.sqrt(2.0)))
        )
        i += 4

    for idx in range(unrolled_end, n_options):
        k = strikes[idx]
        t = expiries[idx]
        d1 = (np.log(_S0 / k) + (_R + 0.5 * _SIGMA * _SIGMA) * t) / (_SIGMA * np.sqrt(t))
        d2 = d1 - _SIGMA * np.sqrt(t)
        cdf_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        cdf_d2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
        prices[idx] = _S0 * cdf_d1 - k * np.exp(-_R * t) * cdf_d2

    return prices


def historical_var(n_scenarios, n_assets):
    """Computes 95% historical VaR by applying a scenario return matrix to an equal-weight portfolio."""
    np.random.seed(_SEED)
    returns = np.random.normal(0.0, 0.01, (n_scenarios, n_assets))
    weights = np.full(n_assets, 1.0 / n_assets)

    portfolio_returns = returns @ weights
    sorted_returns = np.sort(portfolio_returns)

    idx = int(0.05 * n_scenarios)
    return -sorted_returns[idx]


def portfolio_covariance_var(n_assets):
    """Computes portfolio return std-dev via w @ Sigma @ w for a synthetic covariance matrix."""
    np.random.seed(_SEED)
    a = np.random.normal(0.0, 1.0, (n_assets, n_assets))
    sigma = a @ a.T / n_assets + np.eye(n_assets) * 0.01

    weights = np.full(n_assets, 1.0 / n_assets)
    variance = weights @ sigma @ weights
    return np.sqrt(variance)


def rolling_volatility(n_periods, window):
    """Computes the rolling (trailing-window) standard deviation of a synthetic return series."""
    np.random.seed(_SEED)
    returns = np.random.normal(0.0, 0.01, n_periods)
    vol = np.empty(n_periods)

    for i in range(n_periods):
        start = max(0, i - window + 1)
        vol[i] = np.std(returns[start : i + 1])

    return vol


def garch_11(n_periods):
    """Simulates a GARCH(1,1) variance recursion: a genuine loop-carried dependency across periods."""
    omega = 1e-6
    alpha = 0.08
    beta = 0.9

    np.random.seed(_SEED)
    returns = np.empty(n_periods)
    variance = np.empty(n_periods)

    variance[0] = omega / (1.0 - alpha - beta)
    returns[0] = np.sqrt(variance[0]) * np.random.standard_normal()

    for t in range(1, n_periods):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]
        returns[t] = np.sqrt(variance[t]) * np.random.standard_normal()

    return variance


def yield_curve_bootstrap(n_tenors):
    """Bootstraps zero-coupon discount factors sequentially from a synthetic par-swap-rate curve."""
    par_rates = 0.02 + 0.001 * np.arange(n_tenors)
    discount_factors = np.empty(n_tenors)

    for i in range(n_tenors):
        coupon = par_rates[i]
        coupon_sum = 0.0
        for k in range(i):
            coupon_sum += coupon * discount_factors[k]
        discount_factors[i] = (1.0 - coupon_sum) / (1.0 + coupon)

    return discount_factors


def markowitz_optimize(n_assets):
    """Solves for Sigma^-1 mu via a Cholesky-factor-based linear solve (unconstrained tangency weights)."""
    np.random.seed(_SEED)
    a = np.random.normal(0.0, 1.0, (n_assets, n_assets))
    sigma = a @ a.T / n_assets + np.eye(n_assets) * 0.1
    mu = 0.05 + 0.01 * np.random.standard_normal(n_assets)

    l = np.linalg.cholesky(sigma)
    y = np.linalg.solve(l, mu)
    return np.linalg.solve(l.T, y)


def correlated_asset_paths(n_assets, n_paths, n_steps):
    """Simulates correlated GBM asset paths by applying a Cholesky factor to independent draws each step."""
    corr = np.full((n_assets, n_assets), 0.3)
    for i in range(n_assets):
        corr[i, i] = 1.0
    l = np.linalg.cholesky(corr)

    dt = _T / n_steps
    nudt = (_R - 0.5 * _SIGMA * _SIGMA) * dt
    sigsdt = _SIGMA * np.sqrt(dt)

    np.random.seed(_SEED)
    log_paths = np.full((n_paths, n_assets), np.log(_S0))
    for _ in range(n_steps):
        z = np.random.standard_normal((n_paths, n_assets))
        correlated_z = z @ l.T
        log_paths = log_paths + nudt + sigsdt * correlated_z

    return np.exp(log_paths)


def correlated_asset_paths_tiled(n_assets, n_paths, n_steps, path_tile=32):
    """Manually tiled variant of correlated_asset_paths: applies the Cholesky factor via an explicit,
    path-blocked triple-nested loop instead of a single whole-array matrix multiply per step."""
    corr = np.full((n_assets, n_assets), 0.3)
    for i in range(n_assets):
        corr[i, i] = 1.0
    l = np.linalg.cholesky(corr)

    dt = _T / n_steps
    nudt = (_R - 0.5 * _SIGMA * _SIGMA) * dt
    sigsdt = _SIGMA * np.sqrt(dt)

    np.random.seed(_SEED)
    log_paths = np.full((n_paths, n_assets), np.log(_S0))
    for _ in range(n_steps):
        z = np.random.standard_normal((n_paths, n_assets))
        for path_start in range(0, n_paths, path_tile):
            path_end = min(path_start + path_tile, n_paths)
            for p in range(path_start, path_end):
                for a in range(n_assets):
                    acc = 0.0
                    for k in range(n_assets):
                        acc += z[p, k] * l[a, k]
                    log_paths[p, a] = log_paths[p, a] + nudt + sigsdt * acc

    return np.exp(log_paths)


def black_scholes_pde_crank_nicolson(n_space, n_time):
    """Prices a European call by Crank-Nicolson finite differences with a tridiagonal solve per step."""
    s_max = 3.0 * _K
    d_s = s_max / n_space
    dt = _T / n_time

    v = np.empty(n_space + 1)
    for i in range(n_space + 1):
        v[i] = max(i * d_s - _K, 0.0)

    n_interior = n_space - 1
    a_lower = np.empty(n_interior)
    a_diag = np.empty(n_interior)
    a_upper = np.empty(n_interior)
    b_lower = np.empty(n_interior)
    b_diag = np.empty(n_interior)
    b_upper = np.empty(n_interior)

    for k in range(n_interior):
        i = k + 1
        sig2i2 = _SIGMA * _SIGMA * i * i
        alpha = 0.25 * dt * (sig2i2 - _R * i)
        beta = -0.5 * dt * (sig2i2 + _R)
        gamma = 0.25 * dt * (sig2i2 + _R * i)
        a_lower[k] = -alpha
        a_diag[k] = 1.0 - beta
        a_upper[k] = -gamma
        b_lower[k] = alpha
        b_diag[k] = 1.0 + beta
        b_upper[k] = gamma

    tau = 0.0
    for _ in range(n_time):
        tau += dt
        rhs = np.empty(n_interior)
        for k in range(n_interior):
            val = b_diag[k] * v[k + 1]
            if k > 0:
                val += b_lower[k] * v[k]
            if k < n_interior - 1:
                val += b_upper[k] * v[k + 2]
            rhs[k] = val

        v0_new = 0.0
        vn_new = s_max - _K * np.exp(-_R * tau)
        rhs[0] -= a_lower[0] * v0_new
        rhs[n_interior - 1] -= a_upper[n_interior - 1] * vn_new

        # Thomas algorithm: tridiagonal solve of a_lower/a_diag/a_upper against rhs
        c_prime = np.empty(n_interior)
        d_prime = np.empty(n_interior)
        c_prime[0] = a_upper[0] / a_diag[0]
        d_prime[0] = rhs[0] / a_diag[0]
        for k in range(1, n_interior):
            denom = a_diag[k] - a_lower[k] * c_prime[k - 1]
            c_prime[k] = a_upper[k] / denom
            d_prime[k] = (rhs[k] - a_lower[k] * d_prime[k - 1]) / denom

        v[n_space - 1] = d_prime[n_interior - 1]
        for k in range(n_interior - 2, -1, -1):
            v[k + 1] = d_prime[k] - c_prime[k] * v[k + 2]

        v[0] = v0_new
        v[n_space] = vn_new

    idx = _S0 / d_s
    i_lo = int(idx)
    if i_lo >= n_space:
        i_lo = n_space - 1
    frac = idx - i_lo
    return v[i_lo] * (1.0 - frac) + v[i_lo + 1] * frac


def black_scholes_pde_crank_nicolson_fused(n_space, n_time):
    """Fused variant of black_scholes_pde_crank_nicolson: merges the RHS-build loop into the forward-
    elimination loop (computing each rhs[k] inline right where the Thomas algorithm needs it) instead
    of running them as two separate passes over the interior grid."""
    s_max = 3.0 * _K
    d_s = s_max / n_space
    dt = _T / n_time

    v = np.empty(n_space + 1)
    for i in range(n_space + 1):
        v[i] = max(i * d_s - _K, 0.0)

    n_interior = n_space - 1
    a_lower = np.empty(n_interior)
    a_diag = np.empty(n_interior)
    a_upper = np.empty(n_interior)
    b_lower = np.empty(n_interior)
    b_diag = np.empty(n_interior)
    b_upper = np.empty(n_interior)

    for k in range(n_interior):
        i = k + 1
        sig2i2 = _SIGMA * _SIGMA * i * i
        alpha = 0.25 * dt * (sig2i2 - _R * i)
        beta = -0.5 * dt * (sig2i2 + _R)
        gamma = 0.25 * dt * (sig2i2 + _R * i)
        a_lower[k] = -alpha
        a_diag[k] = 1.0 - beta
        a_upper[k] = -gamma
        b_lower[k] = alpha
        b_diag[k] = 1.0 + beta
        b_upper[k] = gamma

    tau = 0.0
    for _ in range(n_time):
        tau += dt
        v0_new = 0.0
        vn_new = s_max - _K * np.exp(-_R * tau)

        c_prime = np.empty(n_interior)
        d_prime = np.empty(n_interior)

        rhs0 = b_diag[0] * v[1]
        if n_interior > 1:
            rhs0 += b_upper[0] * v[2]
        rhs0 -= a_lower[0] * v0_new
        c_prime[0] = a_upper[0] / a_diag[0]
        d_prime[0] = rhs0 / a_diag[0]

        for k in range(1, n_interior):
            val = b_diag[k] * v[k + 1] + b_lower[k] * v[k]
            if k < n_interior - 1:
                val += b_upper[k] * v[k + 2]
            if k == n_interior - 1:
                val -= a_upper[k] * vn_new
            denom = a_diag[k] - a_lower[k] * c_prime[k - 1]
            c_prime[k] = a_upper[k] / denom
            d_prime[k] = (val - a_lower[k] * d_prime[k - 1]) / denom

        v[n_space - 1] = d_prime[n_interior - 1]
        for k in range(n_interior - 2, -1, -1):
            v[k + 1] = d_prime[k] - c_prime[k] * v[k + 2]

        v[0] = v0_new
        v[n_space] = vn_new

    idx = _S0 / d_s
    i_lo = int(idx)
    if i_lo >= n_space:
        i_lo = n_space - 1
    frac = idx - i_lo
    return v[i_lo] * (1.0 - frac) + v[i_lo + 1] * frac
