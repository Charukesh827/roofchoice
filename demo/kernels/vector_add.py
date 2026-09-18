"""Elementwise vector addition -- dense float arithmetic, no transcendentals: 'vectorizable' style."""
import numpy as np


def vector_add(a, b, out):
    for i in range(a.shape[0]):
        out[i] = a[i] + b[i]


KERNEL = vector_add
ARGS = (np.random.rand(4096), np.random.rand(4096), np.zeros(4096))
