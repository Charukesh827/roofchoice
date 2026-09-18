"""Softmax normalization -- exp-heavy, 'transcendental' style."""
import math

import numpy as np


def softmax_normalize(scores, out):
    n = scores.shape[0]
    max_score = scores[0]
    for i in range(n):
        if scores[i] > max_score:
            max_score = scores[i]
    total = 0.0
    for i in range(n):
        e = math.exp(scores[i] - max_score)
        out[i] = e
        total += e
    for i in range(n):
        out[i] = out[i] / total


KERNEL = softmax_normalize
ARGS = (np.random.rand(2048), np.zeros(2048))
