"""Builds the static-feature dataset for predicting the best LLVM optimization variant.

One row per (kernel, njit_variant) in the real, PAPI-measured sweep (experiments/sweep/
results/sweep_combined.csv): static features extracted from the kernel's own IR (Steps 3-5)
plus a roofline-derived "static performance" figure and the njit optimization flags, labeled
with whichever of the 6 LLVM variants actually measured the highest GFLOP/s for that row.
"""

import csv
import inspect
import math
import sys
from pathlib import Path

import pandas as pd

from loopcost.benchmarks import sweep
from loopcost.heuristic.classify import classify_bound, has_loop_carried_dependency
from loopcost.heuristic.ridge_point import get_ridge_point
from loopcost.ir_features.access_pattern import classify_accesses
from loopcost.ir_features.cache_model import estimate_bytes_moved, operational_intensity, working_set_bytes
from loopcost.ir_features.flops import count_flops, count_intops, total_ops
from loopcost.ir_features.loop_info import find_loop_nests

SWEEP_ROOT = Path(__file__).resolve().parents[3] / "experiments" / "sweep"
SWEEP_CSV = SWEEP_ROOT / "results" / "sweep_combined.csv"

FEATURE_COLUMNS = [
    "static_ai",                  # static operational intensity: (FLOPs + IntOps) / bytes accessed
    "static_performance_gflops",  # roofline-predicted ceiling at static_ai: min(peak, ai * bandwidth)
    "is_compute_bound",           # classify_bound(static_ai, ridge_point) == "compute-bound"
    "has_transcendental",         # hottest loop calls exp/log/sqrt/erf/etc.
    "float_ops",                  # per-iteration float add+mul+div+transcendental count
    "int_ops",                    # per-iteration integer add+mul+floordiv/mod count (index arithmetic)
    "log_working_set_bytes",      # log1p(static working-set-bytes estimate)
    "log_trip_count",             # log1p(best-effort static trip count)
    "n_contiguous",                # count of unit-stride array accesses in the hottest loop
    "n_strided",                   # count of constant-strided (non-unit) array accesses
    "n_irregular",                 # count of data-dependent/non-affine array accesses
    "has_loop_carried_dep",       # a running accumulator/recurrence -- blocks vectorization
    "itemsize",                    # bytes per element (4 or 8)
    "fastmath",                    # njit optimization type, axis 1
    "boundscheck",                 # njit optimization type, axis 2
]
LABEL_COLUMN = "best_llvm_variant"


def extract_static_features(fn, args):
    """Extracts the static feature vector for fn(*args)'s most work-heavy loop nest.

    Returns a dict of the kernel-level features (everything except fastmath/boundscheck,
    which are per-njit-variant, not per-kernel), or None if no explicit loop nest is found.
    """
    func_ir, typemap = sweep._capture_ir(fn, args)
    loop_nests = find_loop_nests(func_ir)
    representative = sweep._select_representative(loop_nests, func_ir, typemap)
    if representative is None:
        return None

    # Bounds like "tmp.shape[0]" are symbolic in the IR -- substitute the real call args so
    # total_ops()/estimate_bytes_moved() see concrete trip counts (same pattern as
    # sweep.extract_features; without it, every shape-driven kernel silently reports 0 ops).
    param_names = list(inspect.signature(fn).parameters)
    arg_values = dict(zip(param_names, args))
    concrete = sweep._concretize(representative, arg_values)

    peak_gflops, peak_bandwidth_gbps, ridge_point = get_ridge_point()

    accesses = classify_accesses(representative, func_ir, typemap)
    accesses = sweep._concretize_access_shapes(accesses, arg_values)
    flop_count = count_flops(representative, func_ir, typemap)
    intop_count = count_intops(representative, func_ir, typemap)
    total_flop_ops = total_ops(concrete, func_ir, typemap)
    bytes_estimate = estimate_bytes_moved(concrete, accesses)
    static_ai = operational_intensity(total_flop_ops, bytes_estimate.bytes_moved)

    if static_ai == float("inf"):
        static_performance_gflops = peak_gflops
        static_ai_feature = 10.0 * ridge_point  # cap: "well past the ridge" without an actual inf
    else:
        static_performance_gflops = min(peak_gflops, static_ai * peak_bandwidth_gbps)
        static_ai_feature = static_ai

    bound = classify_bound(static_ai, ridge_point)
    ws_bytes = working_set_bytes(concrete, accesses)
    trip_count = sweep._best_effort_trip_count(concrete)

    return {
        "static_ai": static_ai_feature,
        "static_ai_raw": static_ai,  # unclipped (may be float("inf")) -- for display, not training
        "bytes_moved": bytes_estimate.bytes_moved,
        "static_performance_gflops": static_performance_gflops,
        "is_compute_bound": int(bound == "compute-bound"),
        "bound_label": bound,
        "has_transcendental": int(flop_count.per_iteration.get("transcendental", 0) > 0),
        "float_ops": sum(flop_count.per_iteration.values()),
        "int_ops": sum(intop_count.per_iteration.values()),
        "working_set_bytes": ws_bytes,
        "trip_count": trip_count,
        "log_working_set_bytes": math.log1p(ws_bytes) if isinstance(ws_bytes, (int, float)) else 0.0,
        "log_trip_count": math.log1p(trip_count) if isinstance(trip_count, (int, float)) else 0.0,
        "n_contiguous": sum(1 for a in accesses if a.classification == "contiguous"),
        "n_strided": sum(1 for a in accesses if a.classification == "strided-constant"),
        "n_irregular": sum(1 for a in accesses if a.classification == "irregular"),
        "has_loop_carried_dep": int(has_loop_carried_dependency(representative, func_ir)),
        "itemsize": next((a.itemsize for a in accesses if a.itemsize), 8),
    }


def load_dataset(csv_path=None):
    """Builds (X, y, meta) from the real sweep: one row per (kernel, njit_variant).

    X: DataFrame of FEATURE_COLUMNS. y: Series of the empirically-best LLVM variant name
    (whichever of the 6 measured the highest gflops_per_sec for that row). meta: DataFrame
    with kernel/njit_variant/suite, for readable error analysis (not a feature).
    """
    sweep_path = Path(csv_path) if csv_path else SWEEP_CSV
    if not sweep_path.exists():
        raise FileNotFoundError(f"sweep results not found at {sweep_path}")

    sys.path.insert(0, str(SWEEP_ROOT))
    sys.path.insert(0, str(SWEEP_ROOT.parent))
    from kernel_registry import iter_kernels

    static_features = {}
    for _suite, name, fn, args, _flops in iter_kernels("small"):
        try:
            feats = extract_static_features(fn, args)
        except Exception as e:
            print(f"  [{name}] static feature extraction failed ({type(e).__name__}: {e}); excluded")
            feats = None
        if feats is not None:
            static_features[name] = feats

    with open(sweep_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["gflops_per_sec"] = float(r["gflops_per_sec"])
        r["fastmath"] = r["fastmath"] == "True"
        r["boundscheck"] = r["boundscheck"] == "True"

    groups = {}
    for r in rows:
        groups.setdefault((r["kernel"], r["njit_variant"]), []).append(r)

    X_rows, y_rows, meta_rows = [], [], []
    for (kernel, njit_variant), grp in sorted(groups.items()):
        feats = static_features.get(kernel)
        if feats is None:
            continue
        best = max(grp, key=lambda r: r["gflops_per_sec"])

        row = dict(feats)
        row["fastmath"] = int(grp[0]["fastmath"])
        row["boundscheck"] = int(grp[0]["boundscheck"])
        X_rows.append(row)
        y_rows.append(best["llvm_variant"])
        meta_rows.append({"kernel": kernel, "njit_variant": njit_variant, "suite": grp[0]["suite"]})

    X = pd.DataFrame(X_rows, columns=FEATURE_COLUMNS)
    y = pd.Series(y_rows, name=LABEL_COLUMN)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta
