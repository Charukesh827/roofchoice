"""Simple moving average of a price series -- running-sum float arithmetic, 'vectorizable' style."""
import numpy as np


def moving_average(prices, window, out):
    n = prices.shape[0]
    for i in range(window, n):
        total = 0.0
        for j in range(window):
            total += prices[i - j]
        out[i] = total / window


KERNEL = moving_average
ARGS = (np.random.rand(2048), 16, np.zeros(2048))
