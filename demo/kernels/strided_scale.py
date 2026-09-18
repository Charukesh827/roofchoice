"""Column-major scaling of a row-major 2D array -- non-unit-stride float arithmetic, 'vectorizable' style."""
import numpy as np


def strided_scale(matrix, factor, out):
    rows, cols = matrix.shape
    for j in range(cols):
        for i in range(rows):
            out[i, j] = matrix[i, j] * factor


KERNEL = strided_scale
ARGS = (np.random.rand(64, 64), 1.5, np.zeros((64, 64)))
