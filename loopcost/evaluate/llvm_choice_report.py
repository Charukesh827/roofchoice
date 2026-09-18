"""Runs the LLVM-choice heuristic (loopcost.heuristic.llvm_choice) over the real, PAPI-measured
sweep and reports its improvement over "just @njit" -- numba's own default LLVM pipeline, with
no NUMBA_*/llvmlite tuning at all -- across every kernel and every njit optimization type
(fastmath x boundscheck).

No new hardware measurement is needed: the heuristic only ever picks among the 6 LLVM variants
the sweep already PAPI-measured, so the "what would the suggestion perform like" question is
answered entirely from experiments/sweep/results/sweep_combined.csv's real numbers.
"""

import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loopcost.heuristic.llvm_choice import _LLVM_CHOICE_TABLE, classify_kernel_style

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REPORT_CSV = DATA_DIR / "llvm_choice_improvements.csv"
IMPROVEMENT_PLOT_PNG = DATA_DIR / "llvm_choice_improvements.png"
SUMMARY_PLOT_PNG = DATA_DIR / "llvm_choice_summary.png"

SWEEP_ROOT = Path(__file__).resolve().parents[3] / "experiments" / "sweep"
SWEEP_CSV = SWEEP_ROOT / "results" / "sweep_combined.csv"

BASELINE_LLVM = "llvm_O3_loopvec"  # numba's own default LLVM pipeline -- "just @njit"

CSV_FIELDS = [
    "suite", "kernel", "njit_variant", "fastmath", "boundscheck", "style",
    "baseline_llvm", "baseline_gflops",
    "suggested_llvm", "suggested_gflops", "reason",
    "speedup",
]

# dataviz reference palette, categorical slots 1/2/3 -- one color per static kernel style
STYLE_COLORS = {"vectorizable": "#2a78d6", "transcendental": "#eb6834", "non_fp": "#1baf7a"}
BASELINE_COLOR = "#9a9a95"

NJIT_ORDER = ["njit_default", "njit_fastmath", "njit_boundscheck", "njit_fastmath_boundscheck"]
NJIT_LABELS = {
    "njit_default": "default", "njit_fastmath": "fastmath",
    "njit_boundscheck": "bndchk", "njit_fastmath_boundscheck": "fm+bndchk",
}


def run_llvm_choice_experiment(csv_path=None):
    """For every (kernel, njit_variant) in the real sweep: compares numba's own default LLVM
    pipeline (llvm_O3_loopvec) against loopcost.heuristic.llvm_choice's suggested LLVM
    variant, using the sweep's real PAPI-measured GFLOP/s for both.

    Writes llvm_choice_improvements.csv, two plots, and a printed summary into `data/`.
    Returns the list of per-(kernel, njit_variant) row dicts.
    """
    sweep_path = Path(csv_path) if csv_path else SWEEP_CSV
    if not sweep_path.exists():
        raise FileNotFoundError(
            f"sweep results not found at {sweep_path}; run experiments/sweep/run_sweep.py first"
        )

    sys.path.insert(0, str(SWEEP_ROOT))
    sys.path.insert(0, str(SWEEP_ROOT.parent))
    from kernel_registry import iter_kernels

    print("classifying each kernel's static style (Steps 3-5 IR analysis)...", flush=True)
    styles = {}
    for _suite, name, fn, args, _flops in iter_kernels("small"):
        try:
            styles[name] = classify_kernel_style(fn, args)
        except Exception as e:
            print(f"  [{name}] style classification failed ({type(e).__name__}: {e}); excluded")

    with open(sweep_path) as f:
        sweep_rows = list(csv.DictReader(f))
    for r in sweep_rows:
        r["gflops_per_sec"] = float(r["gflops_per_sec"])
        r["fastmath"] = r["fastmath"] == "True"
        r["boundscheck"] = r["boundscheck"] == "True"

    groups = {}
    for r in sweep_rows:
        groups.setdefault((r["suite"], r["kernel"], r["njit_variant"]), []).append(r)

    rows = []
    for (suite, kernel, njit_variant), grp in sorted(groups.items()):
        style = styles.get(kernel)
        if style is None:
            continue
        fastmath, boundscheck = grp[0]["fastmath"], grp[0]["boundscheck"]
        by_llvm = {r["llvm_variant"]: r["gflops_per_sec"] for r in grp}

        baseline_gflops = by_llvm[BASELINE_LLVM]
        suggested_llvm, avg_pct = _LLVM_CHOICE_TABLE[(style, boundscheck)]
        suggested_gflops = by_llvm[suggested_llvm]
        speedup = (suggested_gflops / baseline_gflops) if baseline_gflops > 0 else float("nan")
        reason = (
            f"style={style}, boundscheck={boundscheck} -> {suggested_llvm} "
            f"(measured avg {avg_pct:.1f}% of peak across the sweep)"
        )

        rows.append(
            {
                "suite": suite,
                "kernel": kernel,
                "njit_variant": njit_variant,
                "fastmath": fastmath,
                "boundscheck": boundscheck,
                "style": style,
                "baseline_llvm": BASELINE_LLVM,
                "baseline_gflops": baseline_gflops,
                "suggested_llvm": suggested_llvm,
                "suggested_gflops": suggested_gflops,
                "reason": reason,
                "speedup": speedup,
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(rows)
    _write_improvement_plot(rows)
    _write_summary_plot(rows)
    _print_summary(rows)
    return rows


def _write_csv(rows):
    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_improvement_plot(rows):
    """Small multiples: one subplot per kernel, njit optimization type on the x-axis, njit's
    default LLVM pipeline vs. the heuristic's suggested LLVM variant as paired bars.
    """
    if not rows:
        return
    per_kernel = {}
    for row in rows:
        per_kernel.setdefault(row["kernel"], {})[row["njit_variant"]] = row
    kernels_sorted = sorted(per_kernel)

    ncols = 4
    nrows = -(-len(kernels_sorted) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.8, nrows * 3.2), facecolor="#fcfcfb", squeeze=False
    )
    axes = axes.flatten()

    for ax, kernel in zip(axes, kernels_sorted):
        ax.set_facecolor("#fcfcfb")
        present = [n for n in NJIT_ORDER if n in per_kernel[kernel]]
        x = range(len(present))
        width = 0.35
        baseline_vals = [per_kernel[kernel][n]["baseline_gflops"] for n in present]
        suggested_vals = [per_kernel[kernel][n]["suggested_gflops"] for n in present]
        style = per_kernel[kernel][present[0]]["style"]
        color = STYLE_COLORS[style]

        ax.bar([i - width / 2 for i in x], baseline_vals, width, color=BASELINE_COLOR)
        ax.bar([i + width / 2 for i in x], suggested_vals, width, color=color)
        ax.set_xticks(list(x))
        ax.set_xticklabels([NJIT_LABELS[n] for n in present], fontsize=6.5, rotation=20)
        ax.set_title(f"{kernel}\n({style})", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", alpha=0.2)

    for ax in axes[len(kernels_sorted):]:
        ax.axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BASELINE_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=STYLE_COLORS["vectorizable"]),
    ]
    fig.legend(
        handles, ["njit default LLVM pipeline (just @njit)", "heuristic-suggested LLVM pipeline"],
        loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.975),
    )
    fig.suptitle(
        "LLVM-choice heuristic vs. njit's default LLVM pipeline (measured GFLOP/s, per kernel x njit type)",
        fontsize=12, y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(IMPROVEMENT_PLOT_PNG, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def _write_summary_plot(rows):
    """One bar per kernel: median speedup (heuristic-suggested / njit-default-LLVM) across
    its 4 njit optimization types, color-coded by static style.
    """
    if not rows:
        return
    per_kernel = {}
    style_by_kernel = {}
    for row in rows:
        per_kernel.setdefault(row["kernel"], []).append(row["speedup"])
        style_by_kernel[row["kernel"]] = row["style"]

    kernels_sorted = sorted(per_kernel)
    medians = [statistics.median(per_kernel[k]) for k in kernels_sorted]
    colors = [STYLE_COLORS[style_by_kernel[k]] for k in kernels_sorted]

    fig, ax = plt.subplots(figsize=(max(9, len(kernels_sorted) * 0.55), 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.bar(range(len(kernels_sorted)), medians, color=colors)
    ax.axhline(1.0, color="#52514e", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(kernels_sorted)))
    ax.set_xticklabels(kernels_sorted, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Speedup: heuristic LLVM / njit-default LLVM (median across njit types)")
    ax.set_title("LLVM-choice heuristic speedup by kernel, vs. just using @njit")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STYLE_COLORS.values()]
    ax.legend(handles, list(STYLE_COLORS.keys()), title="style", loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SUMMARY_PLOT_PNG, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def _print_summary(rows):
    if not rows:
        print("no rows to summarize")
        return

    print(f"\n=== LLVM-choice heuristic vs. njit's default LLVM pipeline ({len(rows)} combinations) ===")
    speedups = [r["speedup"] for r in rows if r["speedup"] == r["speedup"]]
    print(
        f"speedup: min={min(speedups):.3f} median={statistics.median(speedups):.3f} "
        f"max={max(speedups):.3f} mean={statistics.mean(speedups):.3f}"
    )

    changed = [r for r in rows if r["suggested_llvm"] != r["baseline_llvm"]]
    print(f"\n{len(changed)}/{len(rows)} combinations get a DIFFERENT LLVM setting than njit's default")
    if changed:
        changed_speedups = [r["speedup"] for r in changed if r["speedup"] == r["speedup"]]
        print(
            f"  among those: min={min(changed_speedups):.3f} median={statistics.median(changed_speedups):.3f} "
            f"max={max(changed_speedups):.3f} mean={statistics.mean(changed_speedups):.3f}"
        )

    print("\nper-kernel median speedup:")
    per_kernel = {}
    style_by_kernel = {}
    for r in rows:
        per_kernel.setdefault(r["kernel"], []).append(r["speedup"])
        style_by_kernel[r["kernel"]] = r["style"]
    for kernel in sorted(per_kernel):
        med = statistics.median(per_kernel[kernel])
        print(f"  {kernel:28s} style={style_by_kernel[kernel]:14s} median_speedup={med:.3f}")


if __name__ == "__main__":
    run_llvm_choice_experiment()
