"""Per-element Taylor-series price approximation (high-order polynomial, Horner's method) --
many FLOPs per byte of memory traffic, 'vectorizable' style, and genuinely compute-bound
(arithmetic intensity scales with polynomial degree while memory traffic per element is fixed)."""
import numpy as np


def taylor_price_approx(x, coeffs, out):
    for i in range(x.shape[0]):
        acc = coeffs[0]
        for k in range(1, coeffs.shape[0]):
            acc = acc * x[i] + coeffs[k]
        out[i] = acc


KERNEL = taylor_price_approx
ARGS = (np.random.rand(4096), np.random.rand(24), np.zeros(4096))
