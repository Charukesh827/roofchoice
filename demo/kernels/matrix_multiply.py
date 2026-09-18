"""Dense matrix multiply -- nested-loop multiply-accumulate, 'vectorizable' style."""
import numpy as np


def matrix_multiply(A, B, C):
    n = A.shape[0]
    k = A.shape[1]
    m = B.shape[1]
    for i in range(n):
        for j in range(m):
            acc = 0.0
            for p in range(k):
                acc += A[i, p] * B[p, j]
            C[i, j] = acc


KERNEL = matrix_multiply
ARGS = (np.random.rand(48, 48), np.random.rand(48, 48), np.zeros((48, 48)))
