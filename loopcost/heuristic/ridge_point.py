"""Computes the arithmetic-intensity ridge point separating memory-bound from compute-bound loops."""

import csv
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from numba import njit
import llvmlite.binding as llvm

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_CALIBRATION_FILE = _DATA_DIR / "calibration.json"

_CARM_RESULTS_DIR_ENV = "CARM_ROOFLINE_RESULTS_DIR"
_CARM_CASE_DATA_TYPE = "f64"
_CARM_CASE_NUM_THREADS = "1"
_CARM_CASE_OPERATION = "fma"  # matches this file's own FMA micro-benchmark below
_CARM_CASE_CACHE_LEVEL = "DRAM"  # the classical Roofline memory ceiling


def _get_cpu_model():
    """Returns a string identifying the CPU model, used as the calibration cache key."""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _get_simd_width(dtype_bytes=8):
    """Returns the number of dtype-sized elements that fit in a SIMD register on the host CPU."""
    features = llvm.get_host_cpu_features()
    reg_bits = 128
    if features.get("avx512f"):
        reg_bits = 512
    elif features.get("avx2") or features.get("avx"):
        reg_bits = 256
    elif features.get("sse2"):
        reg_bits = 128
    return max(1, reg_bits // (dtype_bytes * 8))


def _preferred_carm_isa():
    """Picks the carm-roofline ISA tag matching the host's best available vector extension."""
    features = llvm.get_host_cpu_features()
    if features.get("avx512f"):
        return "x86_avx512"
    if features.get("avx2"):
        return "x86_avx2"
    if features.get("sse2"):
        return "x86_sse"
    return "x86"


def _carm_results_dir():
    env_dir = os.environ.get(_CARM_RESULTS_DIR_ENV)
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parents[3] / "carm-roofline" / "results"


def _load_carm_summary_rows():
    """Loads carm-roofline's measured summary.csv (a sibling project), or None if unavailable."""
    csv_path = _carm_results_dir() / "summary.csv"
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return None


def _carm_peak_gflops(rows, isa):
    """Finds the single-thread f64 FMA arithmetic-ceiling row for the given ISA in carm's summary."""
    for row in rows:
        if (
            row.get("type") == "arithmetic"
            and row.get("isa") == isa
            and row.get("data_type") == _CARM_CASE_DATA_TYPE
            and row.get("operation") == _CARM_CASE_OPERATION
            and row.get("num_threads") == _CARM_CASE_NUM_THREADS
        ):
            return float(row["performance_gops"])
    return None


def _carm_peak_bandwidth_gbps(rows, isa):
    """Finds the single-thread f64 DRAM-level bandwidth-ceiling row for the given ISA in carm's summary."""
    for row in rows:
        if (
            row.get("type") == "memory"
            and row.get("isa") == isa
            and row.get("data_type") == _CARM_CASE_DATA_TYPE
            and row.get("cache_level") == _CARM_CASE_CACHE_LEVEL
            and row.get("num_threads") == _CARM_CASE_NUM_THREADS
        ):
            return float(row["bandwidth_gbps"])
    return None


@njit(cache=True, fastmath=True)
def _flops_kernel(a, b, reps):
    n = a.shape[0]
    x = 1.0000001
    for _ in range(reps):
        for i in range(n):
            a[i] = a[i] * x + b[i]
    return a[0]


@njit(cache=True, fastmath=True)
def _triad_kernel(a, b, c, scalar):
    n = a.shape[0]
    for i in range(n):
        c[i] = a[i] + scalar * b[i]


def measure_peak_flops(duration_s=0.5):
    """Returns single-thread f64 FMA peak GFLOP/s for this machine's best available ISA.

    Sourced from carm-roofline's measured summary.csv (a rigorous, published Cache-Aware
    Roofline Model benchmark for this exact CPU) when that sibling project's results are
    available; falls back to this module's own Numba FMA micro-benchmark otherwise.
    """
    carm_rows = _load_carm_summary_rows()
    if carm_rows is not None:
        value = _carm_peak_gflops(carm_rows, _preferred_carm_isa())
        if value is not None:
            return value
    return _measure_peak_flops_numba(duration_s)


def _measure_peak_flops_numba(duration_s=0.5):
    """Times a tight Numba FMA loop, sized to the host's SIMD width, and returns achieved GFLOP/s."""
    simd_width = _get_simd_width(dtype_bytes=8)
    n = simd_width * 4096

    a = np.random.rand(n)
    b = np.random.rand(n)

    _flops_kernel(a.copy(), b, 1)  # trigger JIT compilation before timing

    reps = 1
    while True:
        a_run = a.copy()
        start = time.perf_counter()
        _flops_kernel(a_run, b, reps)
        elapsed = time.perf_counter() - start
        if elapsed >= duration_s or reps > 1 << 20:
            break
        reps *= 2

    total_flops = 2.0 * n * reps  # one multiply + one add per element per rep
    return total_flops / elapsed / 1e9


def measure_peak_bandwidth():
    """Returns single-thread f64 DRAM-level peak GB/s for this machine's best available ISA.

    Sourced from carm-roofline's measured summary.csv when available (see measure_peak_flops);
    falls back to this module's own STREAM-triad Numba micro-benchmark otherwise.
    """
    carm_rows = _load_carm_summary_rows()
    if carm_rows is not None:
        value = _carm_peak_bandwidth_gbps(carm_rows, _preferred_carm_isa())
        if value is not None:
            return value
    return _measure_peak_bandwidth_numba()


def _measure_peak_bandwidth_numba():
    """Times a STREAM-triad-style Numba kernel over an array larger than LLC and returns achieved GB/s."""
    n = 32 * 1024 * 1024  # 256MB per float64 array, comfortably larger than typical LLC

    a = np.random.rand(n)
    b = np.random.rand(n)
    c = np.empty(n)
    scalar = 3.14159

    _triad_kernel(a[:1024].copy(), b[:1024].copy(), c[:1024].copy(), scalar)  # trigger JIT compilation

    start = time.perf_counter()
    _triad_kernel(a, b, c, scalar)
    elapsed = time.perf_counter() - start

    bytes_moved = 3.0 * n * 8  # read a, read b, write c, 8 bytes each (float64)
    return bytes_moved / elapsed / 1e9


def _load_cache():
    if _CALIBRATION_FILE.exists():
        try:
            with open(_CALIBRATION_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache):
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CALIBRATION_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def get_ridge_point(force_recalibrate=False):
    """Returns (peak_gflops, peak_bandwidth_GBps, ridge_point), caching measurements per CPU model."""
    cpu_model = _get_cpu_model()
    cache = _load_cache()
    entry = cache.get(cpu_model)

    if entry is not None and not force_recalibrate:
        peak_gflops = entry["peak_gflops"]
        peak_bandwidth_gbps = entry["peak_bandwidth_GBps"]
    else:
        peak_gflops = measure_peak_flops()
        peak_bandwidth_gbps = measure_peak_bandwidth()
        entry = cache.get(cpu_model, {})
        entry["peak_gflops"] = peak_gflops
        entry["peak_bandwidth_GBps"] = peak_bandwidth_gbps
        cache[cpu_model] = entry
        _save_cache(cache)

    i_ridge = peak_gflops / peak_bandwidth_gbps
    return peak_gflops, peak_bandwidth_gbps, i_ridge


if __name__ == "__main__":
    peak_gflops, peak_bandwidth_gbps, i_ridge = get_ridge_point()
    print(f"peak_gflops: {peak_gflops:.3f}")
    print(f"peak_bandwidth_GBps: {peak_bandwidth_gbps:.3f}")
    print(f"I_ridge: {i_ridge:.3f}")
