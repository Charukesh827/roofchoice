"""Population count (number of set bits) per element -- integer bitwise-only: 'non_fp' style."""
import numpy as np


def popcount(values, out):
    for i in range(values.shape[0]):
        v = values[i]
        count = 0
        while v:
            count += v & 1
            v >>= 1
        out[i] = count


KERNEL = popcount
ARGS = (np.random.randint(0, 2**16, 4096).astype(np.int64), np.zeros(4096, dtype=np.int64))
