"""Count elements above a threshold -- integer comparisons only, no arithmetic ops: 'non_fp' style."""
import numpy as np


def count_matches(values, threshold):
    count = 0
    for i in range(values.shape[0]):
        if values[i] > threshold:
            count += 1
    return count


KERNEL = count_matches
ARGS = (np.random.randint(0, 100, 8192), 50)
