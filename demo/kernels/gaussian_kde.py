"""Gaussian kernel density estimate -- exp-heavy, 'transcendental' style."""
import math

import numpy as np


def gaussian_kde(samples, grid, bandwidth, out):
    n = samples.shape[0]
    norm = 1.0 / (n * bandwidth * math.sqrt(2.0 * math.pi))
    for i in range(grid.shape[0]):
        total = 0.0
        for j in range(n):
            z = (grid[i] - samples[j]) / bandwidth
            total += math.exp(-0.5 * z * z)
        out[i] = total * norm


KERNEL = gaussian_kde
ARGS = (np.random.normal(0, 1, 256), np.linspace(-4, 4, 256), 0.3, np.zeros(256))
