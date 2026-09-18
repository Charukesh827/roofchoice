"""SAXPY (out = alpha*x + y) -- dense float multiply-add, 'vectorizable' style."""
import numpy as np


def saxpy(alpha, x, y, out):
    for i in range(x.shape[0]):
        out[i] = alpha * x[i] + y[i]


KERNEL = saxpy
ARGS = (2.5, np.random.rand(4096), np.random.rand(4096), np.zeros(4096))
