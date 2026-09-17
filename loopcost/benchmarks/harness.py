"""Provides timing and measurement utilities for running benchmark kernels under different transformations."""

import time

import numpy as np


def time_kernel(fn, args, repeats=20, reference=None, rtol=1e-6, atol=1e-8):
    """Times fn(*args) with perf_counter, after warmup and an optional correctness cross-check.

    `fn` is called twice before timing starts (warmup: triggers compilation and fills any
    caches) -- neither warmup call is timed. If `reference` is given, the first warmup
    call's result is checked against it with np.allclose *before* any timing happens, so a
    functionally-wrong variant fails loudly with a clear error instead of silently reporting
    a meaningless speedup.

    Returns a dict of {"min", "median", "mean", "max"} wall-clock seconds across `repeats`
    timed calls.
    """
    warmup_result = fn(*args)
    fn(*args)  # second warmup call, in case the first one does extra one-time setup

    if reference is not None and not np.allclose(warmup_result, reference, rtol=rtol, atol=atol):
        raise ValueError(
            f"correctness check failed for {getattr(fn, '__name__', fn)}: result does not "
            f"match the reference (got {np.asarray(warmup_result).ravel()[:3]}, "
            f"expected {np.asarray(reference).ravel()[:3]})"
        )

    times = np.empty(repeats)
    for i in range(repeats):
        start = time.perf_counter()
        fn(*args)
        times[i] = time.perf_counter() - start

    return {
        "min": float(np.min(times)),
        "median": float(np.median(times)),
        "mean": float(np.mean(times)),
        "max": float(np.max(times)),
    }
