"""Newton-iteration implied volatility solve -- exp/log/sqrt-heavy, 'transcendental' style."""
import math

import numpy as np


def implied_vol_newton(price, spot, strike, rate, t, out):
    for i in range(price.shape[0]):
        vol = 0.2
        for _ in range(5):
            d1 = (math.log(spot[i] / strike[i]) + (rate + 0.5 * vol * vol) * t[i]) / (
                vol * math.sqrt(t[i])
            )
            d2 = d1 - vol * math.sqrt(t[i])
            nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
            nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
            model_price = spot[i] * nd1 - strike[i] * math.exp(-rate * t[i]) * nd2
            vega = spot[i] * math.sqrt(t[i]) * math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
            vol = vol - (model_price - price[i]) / (vega + 1e-8)
        out[i] = vol


KERNEL = implied_vol_newton
ARGS = (
    np.random.uniform(1, 20, 512),
    np.random.uniform(50, 150, 512),
    np.random.uniform(50, 150, 512),
    0.03,
    np.random.uniform(0.1, 2.0, 512),
    np.zeros(512),
)
