"""Black-Scholes call price -- exp/log/sqrt/erf per option, 'transcendental' style."""
import math

import numpy as np


def black_scholes_call(spot, strike, rate, vol, t, out):
    for i in range(spot.shape[0]):
        d1 = (math.log(spot[i] / strike[i]) + (rate + 0.5 * vol[i] ** 2) * t[i]) / (
            vol[i] * math.sqrt(t[i])
        )
        d2 = d1 - vol[i] * math.sqrt(t[i])
        nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
        out[i] = spot[i] * nd1 - strike[i] * math.exp(-rate * t[i]) * nd2


KERNEL = black_scholes_call
ARGS = (
    np.random.uniform(50, 150, 1024),
    np.random.uniform(50, 150, 1024),
    0.03,
    np.random.uniform(0.1, 0.6, 1024),
    np.random.uniform(0.1, 2.0, 1024),
    np.zeros(1024),
)
