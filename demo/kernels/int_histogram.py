"""Integer histogram bucketing -- integer comparisons/increments only: 'non_fp' style."""
import numpy as np


def int_histogram(values, bin_width, counts):
    n_bins = counts.shape[0]
    for i in range(values.shape[0]):
        b = values[i] // bin_width
        if b >= n_bins:
            b = n_bins - 1
        counts[b] += 1


KERNEL = int_histogram
ARGS = (np.random.randint(0, 1000, 8192), 10, np.zeros(100, dtype=np.int64))
