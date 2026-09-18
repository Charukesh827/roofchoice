"""Chooses an LLVM vectorization variant for a kernel, given its njit optimization flags.

A pure heuristic lookup (no trained model): a kernel's dominant arithmetic style is classified
statically via Steps 3-5's IR analysis (classify_kernel_style), then the LLVM variant is read
off a small table (style x boundscheck -> variant), calibrated against a real 21-kernel x
24-variant (4 njit flag combos x 6 LLVM settings, 504 rows) PAPI-measured sweep across the
polybench/npbench/financial suites at small size.

What the sweep showed (see validate_against_sweep() to reproduce): boundscheck collapses
vectorized performance far more than any LLVM setting does for kernels that are otherwise
vectorizable (dense linear algebra, elementwise array ops) -- a 78-96% average-%-of-peak
spread across LLVM choices, depending on boundscheck. Transcendental-heavy kernels (exp/log/
sqrt/etc. -- LLVM's vectorizer can't touch these without a vector math library) and kernels
with no meaningful floating-point arithmetic at all barely respond to *any* LLVM setting
(96-99.9% of peak with almost any real O3 variant) -- for those, the choice mostly doesn't
matter, so the table just picks the empirical best rather than introducing a needless branch.
"""

import csv
import sys
from pathlib import Path

from loopcost.benchmarks import sweep
from loopcost.ir_features.flops import count_flops
from loopcost.ir_features.loop_info import find_loop_nests

# The LLVM-level knobs each named variant corresponds to (ported from the sweep's own
# variants.py, so a caller acting on this module's recommendation knows exactly what to set).
# NUMBA_* env vars are read once at numba's own import time -- they must be set before numba
# is imported at all (typically in a fresh subprocess). `set_options` are llvmlite's global
# LLVM cl::opt flags, applied via llvmlite.binding.set_option at any point before compilation.
LLVM_VARIANT_SETTINGS = {
    "llvm_O0": {
        "env": {"NUMBA_OPT": "0", "NUMBA_LOOP_VECTORIZE": "0", "NUMBA_SLP_VECTORIZE": "0"},
        "set_options": [],
    },
    "llvm_O3_novec": {
        "env": {"NUMBA_OPT": "3", "NUMBA_LOOP_VECTORIZE": "0", "NUMBA_SLP_VECTORIZE": "0"},
        "set_options": [],
    },
    "llvm_O3_loopvec": {  # numba's own default -- no environment changes needed
        "env": {"NUMBA_OPT": "3", "NUMBA_LOOP_VECTORIZE": "1", "NUMBA_SLP_VECTORIZE": "0"},
        "set_options": [],
    },
    "llvm_O3_loopvec_slpvec": {
        "env": {"NUMBA_OPT": "3", "NUMBA_LOOP_VECTORIZE": "1", "NUMBA_SLP_VECTORIZE": "1"},
        "set_options": [],
    },
    "llvm_O3_vec_forcewidth2": {
        "env": {"NUMBA_OPT": "3", "NUMBA_LOOP_VECTORIZE": "1", "NUMBA_SLP_VECTORIZE": "1"},
        "set_options": ["-force-vector-width=2"],
    },
    "llvm_O3_vec_forcewidth8": {
        "env": {"NUMBA_OPT": "3", "NUMBA_LOOP_VECTORIZE": "1", "NUMBA_SLP_VECTORIZE": "1"},
        "set_options": ["-force-vector-width=8"],
    },
}

# style x boundscheck -> (best LLVM variant, measured avg %-of-peak GFLOP/s across the sweep).
# Selected to MAXIMIZE that average -- not "most frequent exact winner" -- since a heuristic
# that reliably lands within a few % of peak is more useful than one that's occasionally
# exactly right and badly wrong the rest of the time. See validate_against_sweep().
_LLVM_CHOICE_TABLE = {
    ("non_fp", False): ("llvm_O3_loopvec", 99.1),
    ("non_fp", True): ("llvm_O3_loopvec", 99.1),
    ("transcendental", False): ("llvm_O3_novec", 99.7),
    ("transcendental", True): ("llvm_O3_novec", 99.7),
    ("vectorizable", False): ("llvm_O3_loopvec", 95.5),
    ("vectorizable", True): ("llvm_O3_vec_forcewidth2", 96.3),
}


def classify_kernel_style(fn, args):
    """Classifies a kernel's dominant arithmetic style from Steps 3-5's static IR analysis.

    'transcendental': its hottest loop calls exp/log/sqrt/etc.
    'non_fp': no meaningful floating-point arithmetic detected in its hottest loop (e.g. an
        integer/bitwise-only kernel like a CRC), or no explicit loop found at all.
    'vectorizable': the default/majority case -- real float arithmetic, no transcendentals
        (dense linear algebra, elementwise array ops). This is the only category where the
        LLVM setting choice meaningfully moves the needle.
    """
    func_ir, typemap = sweep._capture_ir(fn, args)
    loop_nests = find_loop_nests(func_ir)
    representative = sweep._select_representative(loop_nests, func_ir, typemap)
    if representative is None:
        return "non_fp"

    flop_count = count_flops(representative, func_ir, typemap)
    if flop_count.per_iteration.get("transcendental", 0) > 0:
        return "transcendental"
    if sum(flop_count.per_iteration.values()) == 0:
        return "non_fp"
    return "vectorizable"


def choose_llvm_variant(fn, args, boundscheck=False, fastmath=False):
    """Recommends an LLVM vectorization variant for fn(*args), given its njit optimization
    type (fastmath/boundscheck).

    `fastmath` doesn't change the recommendation: splitting the sweep further by fastmath
    showed no consistent pattern beyond sample-size noise (13 rows/cell) for the
    transcendental/non_fp styles, which were already >99% of peak regardless -- only
    boundscheck's effect was robust enough across the full-size buckets to act on.

    Returns (llvm_variant_name, reason). Look up `LLVM_VARIANT_SETTINGS[llvm_variant_name]`
    for the concrete env vars / llvmlite set_options needed to apply it.
    """
    style = classify_kernel_style(fn, args)
    llvm_variant, avg_pct_of_peak = _LLVM_CHOICE_TABLE[(style, boundscheck)]
    reason = (
        f"style={style}, boundscheck={boundscheck} -> {llvm_variant} "
        f"(measured avg {avg_pct_of_peak:.1f}% of peak GFLOP/s across the sweep)"
    )
    return llvm_variant, reason


def validate_against_sweep(csv_path=None):
    """Re-derives each sweep kernel's style via classify_kernel_style() (not a hardcoded
    lookup) and reports how well choose_llvm_variant()'s recommendation performs against the
    real, measured sweep: average %-of-peak GFLOP/s achieved, compared to always using
    numba's own default (llvm_O3_loopvec) as a baseline.

    Requires the sibling experiments/ project's sweep results and kernel registry (same
    sibling-project pattern as ridge_point.py's carm-roofline integration); returns None
    (printing why) if they aren't available.
    """
    sweep_root = Path(__file__).resolve().parents[3] / "experiments" / "sweep"
    csv_path = Path(csv_path) if csv_path else sweep_root / "results" / "sweep_combined.csv"
    if not csv_path.exists():
        print(f"sweep results not found at {csv_path}; skipping validation")
        return None

    sys.path.insert(0, str(sweep_root))
    sys.path.insert(0, str(sweep_root.parent))
    from kernel_registry import iter_kernels

    styles = {}
    for _, name, fn, args, _flops in iter_kernels("small"):
        try:
            styles[name] = classify_kernel_style(fn, args)
        except Exception as e:
            print(f"  [{name}] style classification failed ({type(e).__name__}: {e}); excluded")

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["gflops_per_sec"] = float(r["gflops_per_sec"])
        r["boundscheck"] = r["boundscheck"] == "True"

    groups = {}
    for r in rows:
        groups.setdefault((r["kernel"], r["njit_variant"]), []).append(r)

    heuristic_fracs, baseline_fracs, per_style = [], [], {}
    for (kernel, _njit), grp in groups.items():
        style = styles.get(kernel)
        if style is None:
            continue
        bc = grp[0]["boundscheck"]
        peak = max(r["gflops_per_sec"] for r in grp)
        by_llvm = {r["llvm_variant"]: r["gflops_per_sec"] for r in grp}
        chosen, _ = _LLVM_CHOICE_TABLE[(style, bc)]

        h_frac = (by_llvm[chosen] / peak) if peak > 0 else 1.0
        b_frac = (by_llvm["llvm_O3_loopvec"] / peak) if peak > 0 else 1.0
        heuristic_fracs.append(h_frac)
        baseline_fracs.append(b_frac)
        per_style.setdefault(style, []).append(h_frac)

    n = len(heuristic_fracs)
    if n == 0:
        print("no (kernel, njit_variant) combinations could be validated")
        return None

    print(f"validated against {n} (kernel, njit_variant) combinations")
    print(f"  heuristic avg %-of-peak:            {100 * sum(heuristic_fracs) / n:.1f}%")
    print(f"  always-numba-default avg %-of-peak: {100 * sum(baseline_fracs) / n:.1f}%")
    for style, fracs in sorted(per_style.items()):
        print(f"    style={style:14s} n={len(fracs):3d} avg %-of-peak={100 * sum(fracs) / len(fracs):.1f}%")

    return {
        "n": n,
        "heuristic_avg": sum(heuristic_fracs) / n,
        "baseline_avg": sum(baseline_fracs) / n,
    }


if __name__ == "__main__":
    validate_against_sweep()
