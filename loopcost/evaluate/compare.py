"""Compares heuristic and ML-based transformation decisions against measured benchmark outcomes."""

import csv
import inspect
import statistics
from math import isqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

import loopcost
from loopcost import pipeline
from loopcost.benchmarks import sweep
from loopcost.benchmarks.harness import time_kernel
from loopcost.benchmarks.sweep import KERNELS
from loopcost.heuristic.classify import classify_bound, has_loop_carried_dependency
from loopcost.heuristic.decide import get_l2_cache_bytes
from loopcost.heuristic.ridge_point import get_ridge_point
from loopcost.ir_features.access_pattern import classify_accesses
from loopcost.ir_features.cache_model import estimate_bytes_moved, operational_intensity, working_set_bytes
from loopcost.ir_features.flops import total_ops
from loopcost.ir_features.loop_info import find_loop_nests
from loopcost.transforms import tiling, unroll, vectorize

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REPORT_CSV = DATA_DIR / "evaluation_report.csv"
REPORT_PNG = DATA_DIR / "evaluation_report.png"
BOUND_DIRECTED_CSV = DATA_DIR / "bound_directed_report.csv"
BOUND_DIRECTED_COMPARISON_PNG = DATA_DIR / "bound_directed_comparison.png"
ROOFLINE_PNG = DATA_DIR / "roofline_plot.png"
ALL_BENCHMARKS_PNG = DATA_DIR / "all_benchmarks_naive_vs_optimized.png"
PAPI_ROOFLINE_CSV = DATA_DIR / "papi_roofline_report.csv"

DEFAULT_REPEATS = 10
DEFAULT_PAPI_REPS = 5

BOUND_DIRECTED_CSV_FIELDS = [
    "kernel", "size", "bound", "oi", "working_set_bytes", "has_loop_carried_dep",
    "action", "tile_size", "vector_length", "reason",
    "time_plain", "time_optimized", "speedup", "achieved_gflops",
]

PAPI_ROOFLINE_CSV_FIELDS = [
    "kernel", "size", "variant", "action",
    "dp_ops", "dram_bytes", "time_seconds", "gflops_per_sec", "ai_dram",
]

CSV_FIELDS = [
    "kernel", "size",
    "time_plain", "time_tier1_heuristic", "time_tier2_ml",
    "speedup_tier1", "speedup_tier2",
    "tier1_applied_transform", "tier2_applied_transform",
    "tier1_reason", "tier2_reason",
    "agree", "tier2_recommended",
]

# dataviz reference palette, categorical slots 1 and 2 (fixed order, CVD-validated defaults)
TIER1_COLOR = "#2a78d6"  # blue
TIER2_COLOR = "#eb6834"  # orange


def _reason_for(entry):
    transform_type = entry["applied_transform"]
    if transform_type is None:
        return "no transform judged profitable"
    return entry["final_decision"][transform_type][2]


def _time_variant(fn, args, reference, repeats):
    return time_kernel(fn, args, reference=reference, repeats=repeats)["median"]


def _compare_one(kernel_name, size_label, fn, args, repeats):
    """Times variants (a) plain njit, (b) loopcost.jit tier-1, (c) loopcost.jit tier-2 for one case."""
    reference = fn(*args)

    time_plain = _time_variant(njit(fn), args, reference, repeats)

    tier1_fn = loopcost.jit(fn, use_ml=False)
    time_tier1 = _time_variant(tier1_fn, args, reference, repeats)
    entry_tier1 = pipeline.DECISION_LOG[-1]

    tier2_fn = loopcost.jit(fn, use_ml=True)
    time_tier2 = _time_variant(tier2_fn, args, reference, repeats)
    entry_tier2 = pipeline.DECISION_LOG[-1]

    return {
        "kernel": kernel_name,
        "size": size_label,
        "time_plain": time_plain,
        "time_tier1_heuristic": time_tier1,
        "time_tier2_ml": time_tier2,
        "speedup_tier1": time_plain / time_tier1,
        "speedup_tier2": time_plain / time_tier2,
        "tier1_applied_transform": entry_tier1["applied_transform"] or "",
        "tier2_applied_transform": entry_tier2["applied_transform"] or "",
        "tier1_reason": _reason_for(entry_tier1),
        "tier2_reason": _reason_for(entry_tier2),
        "agree": entry_tier1["applied_transform"] == entry_tier2["applied_transform"],
        "tier2_recommended": entry_tier2["applied_transform"] is not None,
    }


def run_comparison(kernels=None, repeats=DEFAULT_REPEATS, output_dir=None):
    """Times plain numba.njit vs. loopcost.jit tier-1 (heuristic) vs. tier-2 (ML) for every
    kernel x size in `kernels` (default: the full Step 6 suite from loopcost.benchmarks.sweep).

    Writes evaluation_report.csv, a printed summary table, and a per-kernel speedup bar chart
    (evaluation_report.png) into `output_dir` (default: data/ -- pass a scratch directory,
    e.g. pytest's tmp_path, to avoid overwriting the real report). Returns the list of
    per-(kernel, size) row dicts.
    """
    if kernels is None:
        kernels = KERNELS
    output_dir = Path(output_dir) if output_dir is not None else DATA_DIR
    pipeline.clear_decision_log()

    rows = []
    skipped = []
    for kernel_name, spec in kernels.items():
        fn = spec["fn"]
        for size_label, args in spec["sizes"].items():
            print(f"[{kernel_name}] {size_label} args={args}", flush=True)
            try:
                rows.append(_compare_one(kernel_name, size_label, fn, args, repeats))
            except Exception as e:
                skipped.append((kernel_name, size_label, str(e)))

    if skipped:
        print(f"\n{len(skipped)} (kernel, size) combinations were skipped:")
        for kernel_name, size_label, reason in skipped:
            print(f"  [{kernel_name}] {size_label}: {reason}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "evaluation_report.csv")
    _write_chart(rows, output_dir / "evaluation_report.png")
    _print_summary(rows)
    return rows


def _write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _per_kernel_medians(rows):
    per_kernel = {}
    for row in rows:
        bucket = per_kernel.setdefault(row["kernel"], {"tier1": [], "tier2": []})
        bucket["tier1"].append(row["speedup_tier1"])
        bucket["tier2"].append(row["speedup_tier2"])
    kernels_sorted = sorted(per_kernel)
    tier1_medians = [statistics.median(per_kernel[k]["tier1"]) for k in kernels_sorted]
    tier2_medians = [statistics.median(per_kernel[k]["tier2"]) for k in kernels_sorted]
    return kernels_sorted, tier1_medians, tier2_medians


def _write_chart(rows, path):
    if not rows:
        return
    kernels_sorted, tier1_medians, tier2_medians = _per_kernel_medians(rows)

    x = range(len(kernels_sorted))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(kernels_sorted) * 1.1), 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.bar([i - width / 2 for i in x], tier1_medians, width, label="tier-1 (heuristic)", color=TIER1_COLOR)
    ax.bar([i + width / 2 for i in x], tier2_medians, width, label="tier-2 (ML)", color=TIER2_COLOR)
    ax.axhline(1.0, color="#52514e", linewidth=1, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(kernels_sorted, rotation=30, ha="right")
    ax.set_ylabel("Speedup over plain numba.njit (median across sizes)")
    ax.set_title("loopcost.jit speedup by kernel: tier-1 heuristic vs. tier-2 ML")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def _print_summary(rows):
    if not rows:
        print("no rows to summarize")
        return

    kernels_sorted, tier1_medians, tier2_medians = _per_kernel_medians(rows)
    print("\n=== per-kernel speedup over plain numba.njit (median across sizes) ===")
    print(f"{'kernel':35s} {'tier1':>8s} {'tier2':>8s}")
    for kernel_name, t1, t2 in zip(kernels_sorted, tier1_medians, tier2_medians):
        print(f"{kernel_name:35s} {t1:8.3f} {t2:8.3f}")

    n_agree = sum(1 for r in rows if r["agree"])
    print(f"\ntier-1 and tier-2 agree on {n_agree}/{len(rows)} ({n_agree / len(rows):.1%}) cases")

    recommended = [r for r in rows if r["tier2_recommended"]]
    if recommended:
        speedups = [r["speedup_tier2"] for r in recommended]
        print(
            f"\nHEADLINE: restricted to the {len(recommended)}/{len(rows)} cases where tier-2 "
            f"recommended applying a transform, measured speedup over plain njit: "
            f"min={min(speedups):.3f} median={statistics.median(speedups):.3f} "
            f"max={max(speedups):.3f} mean={statistics.mean(speedups):.3f}"
        )
    else:
        print("\nHEADLINE: tier-2 never recommended applying a transform in this run")


# --- bound-directed comparison: no ML, no tier-1/tier-2 heuristic-vs-ML split -- just a
# direct "classify, then apply the matching optimization" pipeline: tiling for memory-bound
# kernels, vector-length-matched unrolling + noalias for compute-bound ones. ---


def _decide_bound_directed(fn, args):
    """Classifies fn(*args)'s dominant loop using concrete argument values (unlike
    pipeline.py's live in-compiler analysis, which only ever sees types) and picks the
    matching optimization: tiling when memory-bound, vector-length-matched unrolling when
    compute-bound and safe to (no loop-carried dependency). Returns a decision dict.
    """
    func_ir, typemap = sweep._capture_ir(fn, args)
    loop_nests = find_loop_nests(func_ir)
    representative = sweep._select_representative(loop_nests, func_ir, typemap)
    _, _, ridge_point = get_ridge_point()

    if representative is None:
        return {
            "bound": classify_bound(0.0, ridge_point),
            "oi": 0.0,
            "working_set_bytes": 0,
            "has_loop_carried_dep": False,
            "total_flops": 0,
            "var_name": None,
            "action": "none",
            "tile_size": None,
            "vector_length": None,
            "reason": "no explicit scalar loop nest found in this kernel's IR",
        }

    param_names = list(inspect.signature(fn).parameters)
    arg_values = dict(zip(param_names, args))
    concrete = sweep._concretize(representative, arg_values)

    accesses = classify_accesses(representative, func_ir, typemap)
    # AI = (FLOPs + IntOps) / bytes accessed, weighted by the full enclosing-loop-chain trip
    # count (total_ops() handles both -- see its docstring in flops.py).
    total_flops = total_ops(concrete, func_ir, typemap)
    bytes_estimate = estimate_bytes_moved(concrete, accesses)
    oi = operational_intensity(total_flops, bytes_estimate.bytes_moved)
    bound = classify_bound(oi, ridge_point)
    ws_bytes = working_set_bytes(concrete, accesses)
    has_dep = has_loop_carried_dependency(representative, func_ir)
    itemsize = next((a.itemsize for a in accesses if a.itemsize), 8)
    num_arrays = len({a.array for a in accesses}) or 1
    var_name = representative.induction_vars[-1] if representative.induction_vars else None

    decision = {
        "bound": bound,
        "oi": oi,
        "working_set_bytes": ws_bytes,
        "has_loop_carried_dep": has_dep,
        "total_flops": total_flops,
        "var_name": var_name,
        "tile_size": None,
        "vector_length": None,
    }

    if bound == "memory-bound":
        l2_bytes = get_l2_cache_bytes()
        tile_size = max(1, isqrt((l2_bytes // 2) // (itemsize * num_arrays)))
        decision["action"] = "tile"
        decision["tile_size"] = tile_size
        decision["reason"] = (
            f"memory-bound (OI={oi:.3g}); tiling to {tile_size} elements per dimension "
            f"(working_set_bytes={ws_bytes}, L2={l2_bytes})"
        )
    elif has_dep:
        decision["action"] = "none"
        decision["reason"] = f"compute-bound (OI={oi:.3g}) but has a loop-carried dependency; skipping for safety"
    else:
        vector_length = vectorize.choose_vector_length(itemsize)
        decision["action"] = "vectorize"
        decision["vector_length"] = vector_length
        decision["reason"] = (
            f"compute-bound (OI={oi:.3g}); unrolling by this CPU's vector length "
            f"({vector_length} x {itemsize}-byte elements) plus a noalias hint"
        )

    return decision


def _apply_bound_directed(fn, decision):
    """Compiles fn with the transform named by `decision["action"]`, falling back to a plain
    njit compile whenever the source rewrite the action needs isn't safely applicable.
    """
    action = decision["action"]

    if action == "tile" and decision["var_name"] and decision["tile_size"]:
        rewritten = tiling.rewrite_source(fn, decision["var_name"], decision["tile_size"])
        if rewritten is not None:
            return njit(rewritten)

    if action == "vectorize" and decision["var_name"] and decision["vector_length"]:
        rewritten = unroll.rewrite_source(fn, decision["var_name"], decision["vector_length"])
        target = rewritten if rewritten is not None else fn
        return njit(pipeline_class=vectorize.make_pipeline_class())(target)

    return njit(fn)


def run_bound_directed_comparison(kernels=None, repeats=DEFAULT_REPEATS, output_dir=None):
    """Times plain numba.njit vs. a bound-directed optimization (tiling if memory-bound,
    vector-length-matched unrolling if compute-bound) for every kernel x size in `kernels`
    (default: the full Step 6 suite). No ML involved -- classification uses only Steps 3-5's
    static features plus the ridge point, computed with concrete argument values.

    Writes bound_directed_report.csv, a printed summary, and a per-kernel speedup comparison
    plot (bound_directed_comparison.png) plus the naive-vs-optimized small-multiples plot
    (all_benchmarks_naive_vs_optimized.png) into `output_dir` (default: data/ -- pass a
    scratch directory, e.g. pytest's tmp_path, to avoid overwriting the real report).

    Does NOT produce the roofline plot -- that's hardware-measured via PAPI, see
    run_papi_roofline(), a separate, heavier operation not meant to run inside this loop.
    Returns the list of per-(kernel, size) row dicts.
    """
    if kernels is None:
        kernels = KERNELS
    output_dir = Path(output_dir) if output_dir is not None else DATA_DIR

    rows = []
    skipped = []
    for kernel_name, spec in kernels.items():
        fn = spec["fn"]
        for size_label, args in spec["sizes"].items():
            print(f"[{kernel_name}] {size_label} args={args}", flush=True)
            try:
                reference = fn(*args)
                time_plain = _time_variant(njit(fn), args, reference, repeats)

                decision = _decide_bound_directed(fn, args)
                optimized_fn = _apply_bound_directed(fn, decision)
                time_optimized = _time_variant(optimized_fn, args, reference, repeats)

                achieved_gflops = decision["total_flops"] / time_optimized / 1e9 if decision["total_flops"] else 0.0
                rows.append(
                    {
                        "kernel": kernel_name,
                        "size": size_label,
                        "bound": decision["bound"],
                        "oi": decision["oi"],
                        "working_set_bytes": decision["working_set_bytes"],
                        "has_loop_carried_dep": decision["has_loop_carried_dep"],
                        "action": decision["action"],
                        "tile_size": decision["tile_size"] or "",
                        "vector_length": decision["vector_length"] or "",
                        "reason": decision["reason"],
                        "time_plain": time_plain,
                        "time_optimized": time_optimized,
                        "speedup": time_plain / time_optimized,
                        "achieved_gflops": achieved_gflops,
                    }
                )
            except Exception as e:
                skipped.append((kernel_name, size_label, str(e)))

    if skipped:
        print(f"\n{len(skipped)} (kernel, size) combinations were skipped:")
        for kernel_name, size_label, reason in skipped:
            print(f"  [{kernel_name}] {size_label}: {reason}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_bound_directed_csv(rows, output_dir / "bound_directed_report.csv")
    _write_bound_directed_comparison_plot(rows, output_dir / "bound_directed_comparison.png")
    _write_all_benchmarks_plot(rows, output_dir / "all_benchmarks_naive_vs_optimized.png")
    _print_bound_directed_summary(rows)
    return rows


def _write_bound_directed_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BOUND_DIRECTED_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


KERNEL_PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]  # dataviz reference palette, all 8 categorical slots, fixed order
KERNEL_MARKERS = ["o", "s", "^", "D", "P", "X"]  # cycle shape every 8 kernels so none repeat


def _kernel_styles(kernel_names):
    """Assigns each kernel a distinct (color, marker) combo: cycles the 8-color categorical
    palette, changing marker shape every 8 kernels so no two kernels ever share both."""
    styles = {}
    for i, name in enumerate(kernel_names):
        color = KERNEL_PALETTE[i % len(KERNEL_PALETTE)]
        marker = KERNEL_MARKERS[i // len(KERNEL_PALETTE)]
        styles[name] = (color, marker)
    return styles


NONE_COLOR = "#1baf7a"  # dataviz reference palette, categorical slot 3 (aqua)
ACTION_COLORS = {"tile": TIER1_COLOR, "vectorize": TIER2_COLOR, "none": NONE_COLOR}
ACTION_LABELS = {"tile": "tiling applied", "vectorize": "vectorize applied", "none": "no loop nest found"}


def _write_bound_directed_comparison_plot(rows, path):
    if not rows:
        return
    per_kernel = {}
    kernel_action = {}
    for row in rows:
        per_kernel.setdefault(row["kernel"], []).append(row["speedup"])
        kernel_action[row["kernel"]] = row["action"]  # same action for every size of a kernel

    kernels_sorted = sorted(per_kernel)
    medians = [statistics.median(per_kernel[k]) for k in kernels_sorted]
    colors = [ACTION_COLORS[kernel_action[k]] for k in kernels_sorted]

    fig, ax = plt.subplots(figsize=(max(8, len(kernels_sorted) * 0.9), 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.bar(range(len(kernels_sorted)), medians, color=colors)
    ax.axhline(1.0, color="#52514e", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(kernels_sorted)))
    ax.set_xticklabels(kernels_sorted, rotation=30, ha="right")
    ax.set_ylabel("Speedup over plain numba.njit (median across sizes)")
    ax.set_title("Bound-directed optimization speedup by kernel")

    present_actions = [a for a in ("tile", "vectorize", "none") if a in kernel_action.values()]
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACTION_COLORS[a]) for a in present_actions]
    ax.legend(handles, [ACTION_LABELS[a] for a in present_actions], loc="best", fontsize=9)

    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def _write_all_benchmarks_plot(rows, path):
    """Small-multiples plot: every benchmark's naive-njit vs. optimized wall-clock time,
    at every size, side by side -- the absolute numbers behind the speedup ratios shown in
    _write_bound_directed_comparison_plot.
    """
    if not rows:
        return

    per_kernel = {}
    for row in rows:
        per_kernel.setdefault(row["kernel"], {})[row["size"]] = row
    kernels_sorted = sorted(per_kernel)
    size_order = ["SMALL", "MEDIUM", "LARGE"]

    ncols = 4
    nrows = -(-len(kernels_sorted) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.6, nrows * 3.2), facecolor="#fcfcfb", squeeze=False
    )
    axes = axes.flatten()

    for ax, kernel_name in zip(axes, kernels_sorted):
        ax.set_facecolor("#fcfcfb")
        sizes_present = [s for s in size_order if s in per_kernel[kernel_name]]
        x = range(len(sizes_present))
        width = 0.35
        plain_times = [per_kernel[kernel_name][s]["time_plain"] for s in sizes_present]
        opt_times = [per_kernel[kernel_name][s]["time_optimized"] for s in sizes_present]

        ax.bar([i - width / 2 for i in x], plain_times, width, color=TIER1_COLOR, label="naive njit")
        ax.bar([i + width / 2 for i in x], opt_times, width, color=TIER2_COLOR, label="njit + optimization")
        ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels(sizes_present, fontsize=8)
        ax.set_title(kernel_name, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.2)

    for ax in axes[len(kernels_sorted) :]:
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.90))

    fig.suptitle(
        "All benchmarks: naive njit vs. njit + bound-directed optimization (wall-clock, log scale)",
        fontsize=12,
        y=0.99,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TIER1_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=TIER2_COLOR),
    ]
    fig.legend(
        handles, ["naive njit", "njit + optimization"],
        loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.955),
    )
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def _print_bound_directed_summary(rows):
    if not rows:
        print("no rows to summarize")
        return

    print("\n=== bound classification and action chosen, per kernel ===")
    seen = set()
    for row in rows:
        if row["kernel"] in seen:
            continue
        seen.add(row["kernel"])
        print(f"{row['kernel']:35s} bound={row['bound']:14s} action={row['action']}")

    n_by_action = {}
    for row in rows:
        n_by_action[row["action"]] = n_by_action.get(row["action"], 0) + 1
    print(f"\naction counts across {len(rows)} (kernel, size) cases: {n_by_action}")

    print("\n=== per-kernel speedup over plain numba.njit (median across sizes) ===")
    per_kernel = {}
    for row in rows:
        per_kernel.setdefault(row["kernel"], []).append(row["speedup"])
    for kernel_name in sorted(per_kernel):
        print(f"{kernel_name:35s} {statistics.median(per_kernel[kernel_name]):8.3f}")

    applied = [r for r in rows if r["action"] != "none"]
    if applied:
        speedups = [r["speedup"] for r in applied]
        print(
            f"\nHEADLINE: restricted to the {len(applied)}/{len(rows)} cases where an "
            f"optimization was actually attempted, measured speedup over plain njit: "
            f"min={min(speedups):.3f} median={statistics.median(speedups):.3f} "
            f"max={max(speedups):.3f} mean={statistics.mean(speedups):.3f}"
        )
    else:
        print("\nHEADLINE: no optimization was attempted for any case in this run")


# --- PAPI roofline: real hardware-measured (FLOP/s, arithmetic intensity), not a static
# estimate. Exactly two points per kernel -- naive njit and njit + the bound-directed
# optimization -- at one representative size, so the plot shows real measured movement. ---


def _decide_and_apply(fn, args):
    """Runs the same bound-directed decide+apply used by run_bound_directed_comparison."""
    decision = _decide_bound_directed(fn, args)
    return decision, _apply_bound_directed(fn, decision)


def run_papi_roofline(kernels=None, size_label="LARGE", reps=DEFAULT_PAPI_REPS, core=0, output_dir=None):
    """Measures real hardware (PAPI) FLOPs and DRAM bytes for naive njit vs. the bound-
    directed optimization, for every kernel in `kernels` (default: the full Step 6 suite), at
    one representative size (default LARGE) -- exactly two points per kernel, both genuinely
    measured (PAPI_DP_OPS / DRAM UNC_M_CAS_COUNT), not derived from the static IR analysis.

    A real, hardware-level upside over the static approach: PAPI counts actual executed
    floating-point instructions machine-wide, including inside BLAS/LAPACK calls (np.linalg.
    solve, cholesky, @) that Steps 3-5's IR analysis can't see into at all (no explicit Python
    loop) -- so kernels the static roofline had to plot as "no arithmetic detected" can show
    real data here.

    This is NOT run as part of the routine test suite and is not wired into
    run_bound_directed_comparison: creating PAPI EventSets and pinning the process's CPU
    affinity are real OS-level, hardware-specific, process-wide operations, unsuitable for a
    fast, isolated unit test or for interleaving with the plain wall-clock comparison loop.

    Writes papi_roofline_report.csv and roofline_plot.png into `output_dir` (default: data/).
    Returns the list of row dicts (one per kernel per variant).
    """
    from loopcost.benchmarks.harness import PapiHarness

    if kernels is None:
        kernels = KERNELS
    output_dir = Path(output_dir) if output_dir is not None else DATA_DIR

    rows = []
    skipped = []
    with PapiHarness(core=core) as papi:
        for kernel_name, spec in kernels.items():
            if size_label not in spec["sizes"]:
                continue
            fn = spec["fn"]
            args = spec["sizes"][size_label]
            print(f"[papi] {kernel_name} {size_label} args={args}", flush=True)
            try:
                reference = fn(*args)
                decision, optimized_fn = _decide_and_apply(fn, args)

                naive = papi.measure(njit(fn), args, reps=reps, reference=reference)
                optimized = papi.measure(optimized_fn, args, reps=reps, reference=reference)

                for variant, measurement in (("naive_njit", naive), ("njit_optimized", optimized)):
                    rows.append(
                        {
                            "kernel": kernel_name,
                            "size": size_label,
                            "variant": variant,
                            "action": decision["action"] if variant == "njit_optimized" else "none",
                            **measurement,
                        }
                    )
            except Exception as e:
                skipped.append((kernel_name, str(e)))

    if skipped:
        print(f"\n{len(skipped)} kernels were skipped:")
        for kernel_name, reason in skipped:
            print(f"  [{kernel_name}]: {reason}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_papi_roofline_csv(rows, output_dir / "papi_roofline_report.csv")
    _write_papi_roofline_plot(rows, output_dir / "roofline_plot.png")
    _print_papi_roofline_summary(rows)
    return rows


def _write_papi_roofline_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAPI_ROOFLINE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_papi_roofline_plot(rows, path):
    peak_gflops, peak_bandwidth_gbps, ridge_point = get_ridge_point()

    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ai_range = [2**e for e in range(-10, 8)]
    mem_ceiling = [min(peak_gflops * 1.3, ai * peak_bandwidth_gbps) for ai in ai_range]
    ax.plot(ai_range, mem_ceiling, linestyle="--", linewidth=1.2, color="#52514e", label="memory ceiling", zorder=1)
    ax.axhline(peak_gflops, color="#0b0b0b", linewidth=1, linestyle=":", label="compute ceiling (peak FMA)", zorder=1)
    ax.axvline(
        ridge_point, color="#4a3aa7", linewidth=1, linestyle=":", alpha=0.7,
        label=f"ridge point ({ridge_point:.2g})", zorder=1,
    )

    kernels_sorted = sorted({row["kernel"] for row in rows})
    styles = _kernel_styles(kernels_sorted)

    def _point(row):
        ai = row["ai_dram"]
        gflops = row["gflops_per_sec"]
        if ai == float("inf"):
            ai = ai_range[-1]  # draw "infinite AI" (real compute, zero DRAM traffic) at the right edge
        return ai, gflops

    for kernel_name in kernels_sorted:
        color, marker = styles[kernel_name]
        by_variant = {r["variant"]: r for r in rows if r["kernel"] == kernel_name}
        naive, optimized = by_variant.get("naive_njit"), by_variant.get("njit_optimized")

        if naive and optimized:
            nx, ny = _point(naive)
            ox, oy = _point(optimized)
            if (nx, ny) != (ox, oy):
                ax.annotate(
                    "", xy=(ox, oy), xytext=(nx, ny),
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.55, lw=1.1), zorder=4,
                )

        if naive:
            x, y = _point(naive)
            ax.scatter(x, y, marker=marker, s=75, facecolor="none", edgecolor=color, linewidth=1.6, zorder=5)
        if optimized:
            x, y = _point(optimized)
            ax.scatter(x, y, marker=marker, s=75, facecolor=color, edgecolor="#0b0b0b", linewidth=0.6, zorder=6)

    kernel_handles = [
        plt.Line2D([0], [0], marker=styles[k][1], color=styles[k][0], linestyle="",
                   markerfacecolor=styles[k][0], markersize=7)
        for k in kernels_sorted
    ]
    kernel_legend = ax.legend(
        kernel_handles, kernels_sorted, loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=8, title="kernel", frameon=True,
    )
    ax.add_artist(kernel_legend)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Arithmetic Intensity: PAPI DP_OPS / DRAM bytes (hardware-measured)")
    ax.set_ylabel("Performance: PAPI DP_OPS / wall time (hardware-measured GFLOP/s)")
    ax.set_title(
        f"Roofline (hardware-measured via PAPI): naive njit vs. njit + optimization, "
        f"{len(kernels_sorted)} benchmarks"
    )
    ax.legend(loc="lower right", fontsize=8)  # the ceiling-line legend, separate from the kernel one
    ax.grid(True, which="both", alpha=0.25)
    caption = fig.text(
        0.01, 0.01,
        "hollow marker = naive njit  •  filled marker = njit + optimization  •  "
        "arrow = measured movement",
        fontsize=8, color="#52514e",
    )
    fig.savefig(
        path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight",
        bbox_extra_artists=[kernel_legend, caption],
    )
    plt.close(fig)


def _print_papi_roofline_summary(rows):
    if not rows:
        print("no rows to summarize")
        return

    print("\n=== PAPI-measured GFLOP/s and arithmetic intensity, naive vs. optimized ===")
    print(f"{'kernel':35s} {'naive GFLOP/s':>14s} {'opt GFLOP/s':>14s} {'naive AI':>10s} {'opt AI':>10s}")
    kernels_sorted = sorted({row["kernel"] for row in rows})
    for kernel_name in kernels_sorted:
        by_variant = {r["variant"]: r for r in rows if r["kernel"] == kernel_name}
        naive = by_variant.get("naive_njit", {})
        optimized = by_variant.get("njit_optimized", {})
        print(
            f"{kernel_name:35s} "
            f"{naive.get('gflops_per_sec', 0):14.4f} {optimized.get('gflops_per_sec', 0):14.4f} "
            f"{naive.get('ai_dram', 0):10.4g} {optimized.get('ai_dram', 0):10.4g}"
        )


if __name__ == "__main__":
    run_bound_directed_comparison()
    run_papi_roofline()
