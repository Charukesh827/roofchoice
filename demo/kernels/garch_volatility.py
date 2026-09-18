"""GARCH(1,1) conditional volatility recursion -- sqrt-heavy, 'transcendental' style."""
import math

import numpy as np


def garch_volatility(returns, omega, alpha, beta, out):
    var = omega / (1.0 - alpha - beta)
    for i in range(returns.shape[0]):
        var = omega + alpha * returns[i] ** 2 + beta * var
        out[i] = math.sqrt(var)


KERNEL = garch_volatility
ARGS = (np.random.normal(0, 0.01, 2048), 1e-6, 0.08, 0.9, np.zeros(2048))
