"""Runs parameter sweeps across benchmark kernels and transformation configurations to collect training data."""

import csv
import dataclasses
import inspect
import math
import warnings
from pathlib import Path

from numba import njit
from numba.core import errors as numba_errors
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
import numba.core.typed_passes as typed_passes

from loopcost.benchmarks import financial_kernels as fk
from loopcost.benchmarks.harness import time_kernel
from loopcost.heuristic.classify import classify_bound, has_loop_carried_dependency
from loopcost.heuristic.ridge_point import get_ridge_point
from loopcost.ir_features.access_pattern import classify_accesses
from loopcost.ir_features.cache_model import estimate_bytes_moved, operational_intensity, working_set_bytes
from loopcost.ir_features.flops import count_flops, count_intops, total_ops
from loopcost.ir_features.loop_info import _int_trip_count, find_loop_nests

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TRAINING_DATA_CSV = DATA_DIR / "training_data.csv"

PROFITABLE_THRESHOLD = 1.02
CSV_FIELDS = [
    "kernel", "size", "oi", "bound", "trip_count", "working_set_bytes",
    "has_loop_carried_dep", "transform_type", "transform_applied", "speedup", "profitable_label",
]

# kernel registry: name -> baseline fn, {SMALL/MEDIUM/LARGE: args}, and any manually-written
# transform variant (fn, transform_type) applicable to that kernel's loop structure.
KERNELS = {
    "monte_carlo_option_price": {
        "fn": fk.monte_carlo_option_price,
        "sizes": {"SMALL": (200, 20), "MEDIUM": (2000, 50), "LARGE": (8000, 80)},
    },
    "binomial_tree_price": {
        "fn": fk.binomial_tree_price,
        "sizes": {"SMALL": (50,), "MEDIUM": (150,), "LARGE": (300,)},
    },
    "black_scholes_grid": {
        "fn": fk.black_scholes_grid,
        "sizes": {"SMALL": (64,), "MEDIUM": (512,), "LARGE": (4096,)},
        "variant": (fk.black_scholes_grid_unrolled, "unroll"),
    },
    "historical_var": {
        "fn": fk.historical_var,
        "sizes": {"SMALL": (500, 5), "MEDIUM": (5000, 20), "LARGE": (20000, 40)},
    },
    "portfolio_covariance_var": {
        "fn": fk.portfolio_covariance_var,
        "sizes": {"SMALL": (10,), "MEDIUM": (50,), "LARGE": (150,)},
    },
    "rolling_volatility": {
        "fn": fk.rolling_volatility,
        "sizes": {"SMALL": (200, 10), "MEDIUM": (2000, 20), "LARGE": (10000, 40)},
    },
    "garch_11": {
        "fn": fk.garch_11,
        "sizes": {"SMALL": (500,), "MEDIUM": (5000,), "LARGE": (30000,)},
    },
    "yield_curve_bootstrap": {
        "fn": fk.yield_curve_bootstrap,
        "sizes": {"SMALL": (20,), "MEDIUM": (100,), "LARGE": (300,)},
    },
    "markowitz_optimize": {
        "fn": fk.markowitz_optimize,
        "sizes": {"SMALL": (10,), "MEDIUM": (50,), "LARGE": (120,)},
    },
    "correlated_asset_paths": {
        "fn": fk.correlated_asset_paths,
        "sizes": {"SMALL": (5, 100, 20), "MEDIUM": (8, 400, 40), "LARGE": (12, 1000, 60)},
        "variant": (fk.correlated_asset_paths_tiled, "tile"),
    },
    "black_scholes_pde_crank_nicolson": {
        "fn": fk.black_scholes_pde_crank_nicolson,
        "sizes": {"SMALL": (50, 50), "MEDIUM": (150, 150), "LARGE": (300, 300)},
        "variant": (fk.black_scholes_pde_crank_nicolson_fused, "fuse"),
    },
}

TOGGLE_COMBOS = [
    ("fastmath", dict(fastmath=True, parallel=False)),
    ("parallel", dict(fastmath=False, parallel=True)),
    ("fastmath_parallel", dict(fastmath=True, parallel=True)),
]

_captured = {}


@register_pass(mutates_CFG=False, analysis_only=False)
class _CaptureTypedState(FunctionPass):
    """Stashes a snapshot of func_ir/typemap right after nopython type inference, for feature extraction."""

    _name = "sweep_capture_typed_state"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        _captured["func_ir"] = state.func_ir.copy()
        _captured["typemap"] = dict(state.typemap)
        return False


class _CaptureAfterTypeInference(CompilerBase):
    """Custom pipeline that inserts _CaptureTypedState right after NopythonTypeInference."""

    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        pm.add_pass_after(_CaptureTypedState, typed_passes.NopythonTypeInference)
        pm.finalize()
        return [pm]


def _capture_ir(fn, args):
    _captured.clear()
    njit(pipeline_class=_CaptureAfterTypeInference)(fn)(*args)
    return _captured["func_ir"], _captured["typemap"]


def _resolve_symbolic(value, arg_values):
    """Best-effort: substitutes known function-argument values into a symbolic bound expression."""
    if isinstance(value, int) or not isinstance(value, str):
        return value
    try:
        result = eval(value, {"__builtins__": {}, "ceil": math.ceil}, dict(arg_values))
        return int(result) if isinstance(result, float) and result.is_integer() else result
    except Exception:
        return value  # references another loop's induction variable, or otherwise unresolvable


def _concretize(loop_nest, arg_values):
    """Returns a copy of loop_nest with every bound/trip_count resolved against arg_values where possible."""
    new_bounds = [
        (
            var,
            _resolve_symbolic(start, arg_values),
            _resolve_symbolic(stop, arg_values),
            _resolve_symbolic(step, arg_values),
        )
        for var, start, stop, step in loop_nest.bounds
    ]
    new_trip_count = _resolve_symbolic(loop_nest.trip_count, arg_values)
    return dataclasses.replace(loop_nest, bounds=new_bounds, trip_count=new_trip_count)


def _concretize_access_shapes(access_records, arg_values):
    """Returns copies of access_records with array_shape filled in from the real call args'
    actual array shapes, wherever the accessed array is (or aliases) a top-level argument.

    Numba array TYPES never carry a concrete shape (only ndim), so classify_accesses() always
    leaves this field None -- which makes estimate_bytes_moved()'s fallback path (any
    non-affine or non-unit-stride access, e.g. any 2D array indexed as arr[i, j] with i from
    an outer loop) assume every iteration touches a brand-new element instead of the array's
    real, reused footprint, systematically inflating bytes moved and so misclassifying
    cache-resident compute-bound kernels (e.g. small dense matmuls) as memory-bound.
    """
    resolved = []
    for access in access_records:
        if access.array_shape is None:
            value = arg_values.get(access.array)
            if value is not None and hasattr(value, "shape"):
                access = dataclasses.replace(access, array_shape=tuple(value.shape))
        resolved.append(access)
    return resolved


def _best_effort_trip_count(loop_nest):
    """Numeric trip-count proxy: the concrete value if resolvable, else the product of whichever
    bounds in the chain ARE concrete (skipping any that depend on another loop's live index, e.g.
    a triangular inner bound), else 0 if nothing at all resolved."""
    if isinstance(loop_nest.trip_count, int):
        return loop_nest.trip_count
    total = 1
    resolved_any = False
    for _, start, stop, step in loop_nest.bounds:
        if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
            total *= _int_trip_count(start, stop, step)
            resolved_any = True
    return total if resolved_any else 0


def _select_representative(loop_nests, func_ir, typemap):
    """Picks the loop_nest with the most statically-detected work (ops + accesses), or None."""
    if not loop_nests:
        return None

    def score(loop_nest):
        flops = count_flops(loop_nest, func_ir, typemap)
        intops = count_intops(loop_nest, func_ir, typemap)
        n_accesses = len(classify_accesses(loop_nest, func_ir, typemap))
        per_iteration_ops = sum(flops.per_iteration.values()) + sum(intops.per_iteration.values())
        return (per_iteration_ops + n_accesses, loop_nest.depth)

    return max(loop_nests, key=score)


def extract_features(fn, args):
    """Extracts the Step 3-5 static feature vector for fn(*args)'s most work-heavy loop nest.

    Returns a dict with oi, bound, trip_count, working_set_bytes, has_loop_carried_dep. When
    no explicit for-loop is found in the IR at all (e.g. a kernel expressed purely as whole-
    array/matrix NumPy operations, which Numba doesn't lower to a scalar loop at the point we
    inspect it), falls back to a zero/"memory-bound" default -- an honest reflection of this
    tool's scope (explicit scalar loops), not a bug.
    """
    func_ir, typemap = _capture_ir(fn, args)
    loop_nests = find_loop_nests(func_ir)
    representative = _select_representative(loop_nests, func_ir, typemap)

    _, _, ridge_point = get_ridge_point()

    if representative is None:
        return {
            "oi": 0.0,
            "bound": classify_bound(0.0, ridge_point),
            "trip_count": 0,
            "working_set_bytes": 0,
            "has_loop_carried_dep": False,
        }

    param_names = list(inspect.signature(fn).parameters)
    arg_values = dict(zip(param_names, args))
    concrete = _concretize(representative, arg_values)

    accesses = classify_accesses(representative, func_ir, typemap)
    accesses = _concretize_access_shapes(accesses, arg_values)
    # AI = (FLOPs + IntOps) / bytes accessed -- total_ops() already weights by the full
    # enclosing-loop-chain trip count, not just this level's own (see its docstring).
    total_flops = total_ops(concrete, func_ir, typemap)

    bytes_estimate = estimate_bytes_moved(concrete, accesses)
    oi = operational_intensity(total_flops, bytes_estimate.bytes_moved)

    return {
        "oi": oi,
        "bound": classify_bound(oi, ridge_point),
        "trip_count": _best_effort_trip_count(concrete),
        "working_set_bytes": working_set_bytes(concrete, accesses),
        "has_loop_carried_dep": has_loop_carried_dependency(representative, func_ir),
    }


def _time_variant(fn, args, reference, **njit_kwargs):
    jitted = njit(**njit_kwargs)(fn)
    stats = time_kernel(jitted, args, reference=reference)
    return stats["median"]


def run_sweep():
    """Runs every kernel x size x transform-toggle (+ manual variant, where written) and logs rows."""
    warnings.filterwarnings("ignore", category=numba_errors.NumbaPerformanceWarning)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    skipped = []

    for kernel_name, spec in KERNELS.items():
        baseline_fn = spec["fn"]
        variant_fn, variant_type = spec.get("variant", (None, None))

        for size_label, args in spec["sizes"].items():
            print(f"[{kernel_name}] {size_label} args={args}", flush=True)

            features = extract_features(baseline_fn, args)
            reference = baseline_fn(*args)  # plain-NumPy ground truth for every variant's correctness check

            try:
                baseline_time = _time_variant(baseline_fn, args, reference, fastmath=False, parallel=False)
            except Exception as e:
                skipped.append((kernel_name, size_label, "baseline", str(e)))
                continue
            rows.append(_row(kernel_name, size_label, features, "baseline", False, 1.0))

            for transform_type, njit_kwargs in TOGGLE_COMBOS:
                try:
                    variant_time = _time_variant(baseline_fn, args, reference, **njit_kwargs)
                except Exception as e:
                    # e.g. Numba's parallel=True uses per-thread RNG streams, so kernels that
                    # draw random numbers can legitimately diverge from the reference under
                    # parallel=True -- a real semantic difference, not a bug in the check.
                    skipped.append((kernel_name, size_label, transform_type, str(e)))
                    continue
                speedup = baseline_time / variant_time
                rows.append(_row(kernel_name, size_label, features, transform_type, True, speedup))

            if variant_fn is not None:
                try:
                    variant_time = _time_variant(variant_fn, args, reference, fastmath=False, parallel=False)
                except Exception as e:
                    skipped.append((kernel_name, size_label, variant_type, str(e)))
                    continue
                speedup = baseline_time / variant_time
                rows.append(_row(kernel_name, size_label, features, variant_type, True, speedup))

    if skipped:
        print(f"\n{len(skipped)} (kernel, size, transform) combinations were skipped:")
        for kernel_name, size_label, transform_type, reason in skipped:
            print(f"  [{kernel_name}] {size_label} {transform_type}: {reason}")

    with open(TRAINING_DATA_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {TRAINING_DATA_CSV}")
    return rows


def _row(kernel_name, size_label, features, transform_type, transform_applied, speedup):
    return {
        "kernel": kernel_name,
        "size": size_label,
        "oi": features["oi"],
        "bound": features["bound"],
        "trip_count": features["trip_count"],
        "working_set_bytes": features["working_set_bytes"],
        "has_loop_carried_dep": features["has_loop_carried_dep"],
        "transform_type": transform_type,
        "transform_applied": transform_applied,
        "speedup": speedup,
        "profitable_label": speedup > PROFITABLE_THRESHOLD,
    }


if __name__ == "__main__":
    run_sweep()
