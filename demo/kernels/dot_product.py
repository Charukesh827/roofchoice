"""Dense dot product -- multiply-accumulate reduction, 'vectorizable' style."""
import numpy as np


def dot_product(a, b):
    total = 0.0
    for i in range(a.shape[0]):
        total += a[i] * b[i]
    return total


KERNEL = dot_product
ARGS = (np.random.rand(4096), np.random.rand(4096))
