"""Pairwise asset-return correlation (Gram) matrix -- dense nested dot products with heavy row
reuse, 'vectorizable' style, and compute-bound (each row is read O(n_assets) times)."""
import numpy as np


def correlation_matrix(returns, out):
    n_assets, n_obs = returns.shape
    for a in range(n_assets):
        for b in range(n_assets):
            acc = 0.0
            for t in range(n_obs):
                acc += returns[a, t] * returns[b, t]
            out[a, b] = acc


KERNEL = correlation_matrix
ARGS = (np.random.rand(32, 256), np.zeros((32, 32)))
