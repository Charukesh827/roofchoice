"""Validates the static compute-bound/memory-bound (CB/BB) classification against the real,
PAPI-measured dynamic classification from the sweep.

The static classification (loopcost.benchmarks.sweep.extract_features) is derived entirely
from the kernel's IR and its actual call arguments, with no execution at all: static AI =
(FLOPs + IntOps) / bytes accessed (islpy-exact where possible), compared against the machine's
calibrated roofline ridge point.

The dynamic classification uses the sweep's real PAPI hardware counters: ai_dram = measured
double-precision FLOPs / measured DRAM bytes moved (actual cache and prefetcher behavior, not
a static estimate), for the njit_default + llvm_O3_loopvec row -- i.e. "just @njit", numba's
own default pipeline, with no heuristic-driven tuning applied -- so this is an apples-to-apples
check of the static model against an unmodified, naturally-compiled run.
"""

import csv
import sys
from pathlib import Path

from loopcost.benchmarks import sweep
from loopcost.heuristic.classify import classify_bound
from loopcost.heuristic.ridge_point import get_ridge_point

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REPORT_CSV = DATA_DIR / "bound_validation.csv"

SWEEP_ROOT = Path(__file__).resolve().parents[3] / "experiments" / "sweep"
SWEEP_CSV = SWEEP_ROOT / "results" / "sweep_combined.csv"

REFERENCE_NJIT_VARIANT = "njit_default"
REFERENCE_LLVM_VARIANT = "llvm_O3_loopvec"  # numba's own default pipeline -- "just @njit"

CSV_FIELDS = [
    "suite", "kernel", "static_ai", "static_bound",
    "dynamic_ai_dram", "dynamic_bound", "match",
]


def run_bound_validation(csv_path=None):
    """For every sweep kernel: compares the static CB/BB classification against the dynamic
    (real PAPI-measured) classification at the njit_default/llvm_O3_loopvec reference point.

    Writes data/bound_validation.csv and returns the list of per-kernel row dicts.
    """
    sweep_path = Path(csv_path) if csv_path else SWEEP_CSV
    if not sweep_path.exists():
        raise FileNotFoundError(f"sweep results not found at {sweep_path}")

    sys.path.insert(0, str(SWEEP_ROOT))
    sys.path.insert(0, str(SWEEP_ROOT.parent))
    from kernel_registry import iter_kernels

    _, _, ridge_point = get_ridge_point()

    with open(sweep_path) as f:
        sweep_rows = list(csv.DictReader(f))
    reference_by_kernel = {
        r["kernel"]: float(r["ai_dram"])
        for r in sweep_rows
        if r["njit_variant"] == REFERENCE_NJIT_VARIANT and r["llvm_variant"] == REFERENCE_LLVM_VARIANT
    }

    rows = []
    for suite, name, fn, args, _flops in iter_kernels("small"):
        if name not in reference_by_kernel:
            continue
        try:
            static = sweep.extract_features(fn, args)
        except Exception as e:
            print(f"  [{name}] static feature extraction failed ({type(e).__name__}: {e}); excluded")
            continue

        dynamic_ai = reference_by_kernel[name]
        dynamic_bound = classify_bound(dynamic_ai, ridge_point)

        rows.append(
            {
                "suite": suite,
                "kernel": name,
                "static_ai": static["oi"],
                "static_bound": static["bound"],
                "dynamic_ai_dram": dynamic_ai,
                "dynamic_bound": dynamic_bound,
                "match": static["bound"] == dynamic_bound,
            }
        )

    rows.sort(key=lambda r: r["kernel"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(rows)
    _print_summary(rows)
    return rows


def _write_csv(rows):
    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows):
    if not rows:
        print("no rows to summarize")
        return

    name_w = max(len(r["kernel"]) for r in rows) + 2
    header = (
        f"{'kernel':<{name_w}}{'static AI':<12}{'static bound':<15}"
        f"{'dynamic AI':<12}{'dynamic bound':<15}{'match':<6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        static_ai_str = "inf" if r["static_ai"] == float("inf") else f"{r['static_ai']:.2f}"
        mark = "yes" if r["match"] else "NO"
        print(
            f"{r['kernel']:<{name_w}}{static_ai_str:<12}{r['static_bound']:<15}"
            f"{r['dynamic_ai_dram']:<12.2f}{r['dynamic_bound']:<15}{mark:<6}"
        )

    n = len(rows)
    matches = sum(1 for r in rows if r["match"])
    print(f"\n{matches}/{n} kernels ({100 * matches / n:.1f}%) agree between static and dynamic CB/BB classification")


if __name__ == "__main__":
    run_bound_validation()
