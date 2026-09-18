"""Hackathon demo: for each kernel in demo/kernels/, classifies its static arithmetic style
(loopcost's Step 3-5 IR analysis) and looks up loopcost's hand-derived LLVM-choice heuristic
table (loopcost.heuristic.llvm_choice, calibrated against a real 504-row PAPI-measured sweep)
to suggest the best LLVM optimization pipeline for every njit optimization type.

No ML model, no new hardware measurement -- just the static heuristic, run live against 15
kernels the model has never seen, to show the recommendation in action.
"""

import importlib
import pkgutil
from pathlib import Path

from loopcost.heuristic.llvm_choice import _LLVM_CHOICE_TABLE, classify_kernel_style
from loopcost.ml.llvm_dataset import extract_static_features

KERNELS_DIR = Path(__file__).resolve().parent / "kernels"

NJIT_VARIANTS = [
    ("default", False, False),
    ("fastmath", True, False),
    ("boundscheck", False, True),
    ("fastmath+boundscheck", True, True),
]

# short display codes for the 6 LLVM variants (keeps the table narrow)
VARIANT_CODES = {
    "llvm_O0": "O0",
    "llvm_O3_novec": "O3 novec",
    "llvm_O3_loopvec": "O3 loop-vec",
    "llvm_O3_loopvec_slpvec": "O3 loop+SLP",
    "llvm_O3_vec_forcewidth2": "O3 vec w=2",
    "llvm_O3_vec_forcewidth8": "O3 vec w=8",
}


def iter_demo_kernels():
    """Yields (name, fn, args) for every kernel module in demo/kernels/."""
    for info in sorted(pkgutil.iter_modules([str(KERNELS_DIR)]), key=lambda i: i.name):
        module = importlib.import_module(f"kernels.{info.name}")
        yield info.name, module.KERNEL, module.ARGS, (module.__doc__ or "").strip()


def run_demo():
    import sys

    sys.path.insert(0, str(KERNELS_DIR.parent))  # so "import kernels.<name>" resolves

    rows = []
    print("classifying static arithmetic style + suggesting LLVM variant per kernel...\n")
    for name, fn, args, doc in iter_demo_kernels():
        style = classify_kernel_style(fn, args)
        suggestions = {}
        for njit_name, fastmath, boundscheck in NJIT_VARIANTS:
            variant, avg_pct = _LLVM_CHOICE_TABLE[(style, boundscheck)]
            suggestions[njit_name] = (variant, avg_pct)
        static = extract_static_features(fn, args)
        rows.append(
            {"name": name, "style": style, "doc": doc, "suggestions": suggestions, "static": static}
        )
        print(f"  [{name:24s}] done")

    _print_static_table(rows)
    _print_table(rows)
    _print_summary(rows)
    return rows


def _fmt_ai(static):
    if static is None:
        return "n/a"
    raw = static["static_ai_raw"]
    return "inf" if raw == float("inf") else f"{raw:.2f}"


def _print_static_table(rows):
    name_w = max(len(r["name"]) for r in rows) + 2
    cols = [
        ("static AI", 11),
        ("static perf", 14),
        ("bound", 15),
        ("loop-dep", 10),
    ]
    header = f"{'kernel':<{name_w}}" + "".join(f"{c:<{w}}" for c, w in cols)
    print("\n" + "=" * len(header))
    print("Static IR features (loopcost Step 3-5 analysis, per kernel -- independent of njit flags)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        s = r["static"]
        if s is None:
            print(f"{r['name']:<{name_w}}(no explicit loop nest found -- excluded from static analysis)")
            continue
        perf = f"{s['static_performance_gflops']:.2f} GFLOP/s"
        cells = (
            f"{_fmt_ai(s):<11}"
            f"{perf:<14}"
            f"{s['bound_label']:<15}"
            f"{'yes' if s['has_loop_carried_dep'] else 'no':<10}"
        )
        print(f"{r['name']:<{name_w}}{cells}")
    print("=" * len(header))
    print("(AI = (FLOPs+IntOps)/bytes accessed; static perf = min(peak GFLOP/s, AI x peak bandwidth) -- the roofline ceiling at that AI)")


def _print_table(rows):
    name_w = max(len(r["name"]) for r in rows) + 2
    style_w = max(len(r["style"]) for r in rows) + 2
    njit_labels = [n for n, _, _ in NJIT_VARIANTS]
    col_w = max(len(c) for c in VARIANT_CODES.values()) + 3
    col_w = max(col_w, max(len(n) for n in njit_labels) + 2)

    header = f"{'kernel':<{name_w}}{'style':<{style_w}}" + "".join(
        f"{n:<{col_w}}" for n in njit_labels
    )
    print("\n" + "=" * len(header))
    print("LLVM-choice heuristic suggestions (per kernel x njit optimization type)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = "".join(
            f"{VARIANT_CODES[r['suggestions'][n][0]]:<{col_w}}" for n in njit_labels
        )
        print(f"{r['name']:<{name_w}}{r['style']:<{style_w}}{cells}")
    print("=" * len(header))


def _print_summary(rows):
    from collections import Counter

    style_counts = Counter(r["style"] for r in rows)
    print(f"\n{len(rows)} kernels classified: " + ", ".join(f"{s}={n}" for s, n in style_counts.items()))

    variant_counts = Counter()
    for r in rows:
        for variant, _ in r["suggestions"].values():
            variant_counts[variant] += 1
    print("LLVM variants recommended across all (kernel, njit type) combinations:")
    for variant, n in sorted(variant_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {VARIANT_CODES[variant]:<14s} x{n}")


if __name__ == "__main__":
    run_demo()
