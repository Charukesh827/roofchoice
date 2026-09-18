"""Provides timing and measurement utilities for running benchmark kernels under different transformations."""

import os
import statistics
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


class PapiHarness:
    """Measures real hardware performance counters (PAPI) around a compiled kernel call.

    Pins the process to one physical core and reads PAPI_DP_OPS (double-precision floating-
    point ops -- hardware ground truth, unlike a static IR flop count) plus the DRAM memory
    controller's UNC_M_CAS_COUNT (true DRAM bytes moved) around each call, giving a genuinely
    measured (arithmetic intensity, GFLOP/s) roofline point instead of a static estimate.
    Ported from the sibling experiments/ project's PAPI harness (same machine, same technique).

    Not for use inside the routine test suite: creating PAPI EventSets and pinning process
    CPU affinity are real OS-level, hardware-specific, process-wide side effects, unsuitable
    for a fast, isolated unit test.
    """

    def __init__(self, core=0):
        from loopcost.benchmarks.papi_ctypes import CounterSet, discover_uncore_imc_events, init_library

        init_library()
        os.sched_setaffinity(0, {core})
        self.core_id = core
        self.core_counters = CounterSet(["PAPI_TOT_CYC", "PAPI_DP_OPS"])
        self.uncore_counters = CounterSet(discover_uncore_imc_events(), cpu=core)

    def close(self):
        self.core_counters.close()
        self.uncore_counters.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def measure(self, fn, args, reps=5, reference=None):
        """Warms up (uncounted, with an optional correctness check), then takes `reps`
        measured repetitions. Returns the medians: dp_ops, dram_bytes, time_seconds,
        gflops_per_sec (dp_ops / time_seconds), ai_dram (dp_ops / dram_bytes, or +inf when
        dram_bytes is 0 and dp_ops > 0 -- the same "no measured bytes but real compute"
        convention as cache_model.operational_intensity).
        """
        warmup_result = fn(*args)
        if reference is not None and not np.allclose(warmup_result, reference):
            raise ValueError(
                f"correctness check failed for {getattr(fn, '__name__', fn)}: "
                f"result does not match the reference"
            )

        samples = []
        for _ in range(reps):
            self.core_counters.start()
            self.uncore_counters.start()
            t0 = time.perf_counter()
            fn(*args)
            t1 = time.perf_counter()
            core_vals = self.core_counters.stop()
            uncore_vals = self.uncore_counters.stop()
            samples.append(
                {
                    "cycles": core_vals["PAPI_TOT_CYC"],
                    "dp_ops": core_vals["PAPI_DP_OPS"],
                    "dram_bytes": sum(uncore_vals.values()) * 64,
                    "time_seconds": t1 - t0,
                }
            )

        med = {k: statistics.median(s[k] for s in samples) for k in samples[0]}
        dp_ops = med["dp_ops"]
        dram_bytes = med["dram_bytes"]
        time_seconds = med["time_seconds"]
        return {
            "dp_ops": dp_ops,
            "dram_bytes": dram_bytes,
            "time_seconds": time_seconds,
            "gflops_per_sec": dp_ops / time_seconds / 1e9 if time_seconds > 0 else 0.0,
            "ai_dram": (float("inf") if dp_ops > 0 else 0.0) if dram_bytes <= 0 else dp_ops / dram_bytes,
        }
