"""Smoke test: runs the tier-1/tier-2/plain comparison on a few small kernels and checks the report.

All output-writing calls pass `output_dir=tmp_path` so tests never touch (or get clobbered
by, or clobber) the real reports under data/ -- see run_comparison()/run_bound_directed_
comparison()'s `output_dir` parameter. The PAPI-based roofline (run_papi_roofline) is
deliberately NOT exercised here: it pins process CPU affinity and creates PAPI EventSets,
real OS-level/hardware side effects unsuitable for the fast test suite.
"""
import csv
import importlib.util
import tempfile
from pathlib import Path

from loopcost.evaluate import compare
from loopcost.benchmarks import financial_kernels as fk

SMALL_KERNELS = {
    "binomial_tree_price": {"fn": fk.binomial_tree_price, "sizes": {"SMALL": (20,)}},
    "black_scholes_grid": {"fn": fk.black_scholes_grid, "sizes": {"SMALL": (16,)}},
    "garch_11": {"fn": fk.garch_11, "sizes": {"SMALL": (50,)}},
}


def test_run_comparison_produces_expected_rows(tmp_path):
    rows = compare.run_comparison(kernels=SMALL_KERNELS, repeats=3, output_dir=tmp_path)

    assert len(rows) == len(SMALL_KERNELS)
    for row in rows:
        assert row["kernel"] in SMALL_KERNELS
        assert row["size"] == "SMALL"
        for time_col in ("time_plain", "time_tier1_heuristic", "time_tier2_ml"):
            assert row[time_col] > 0
        for speedup_col in ("speedup_tier1", "speedup_tier2"):
            assert row[speedup_col] > 0
        assert isinstance(row["agree"], bool)
        assert isinstance(row["tier2_recommended"], bool)
        assert row["tier1_reason"].strip() != ""
        assert row["tier2_reason"].strip() != ""


def test_report_csv_is_written_with_expected_columns(tmp_path):
    compare.run_comparison(kernels=SMALL_KERNELS, repeats=3, output_dir=tmp_path)

    report_csv = tmp_path / "evaluation_report.csv"
    assert report_csv.exists()
    with open(report_csv) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == compare.CSV_FIELDS
        rows = list(reader)
    assert len(rows) == len(SMALL_KERNELS)


def test_report_png_is_written(tmp_path):
    compare.run_comparison(kernels=SMALL_KERNELS, repeats=3, output_dir=tmp_path)

    report_png = tmp_path / "evaluation_report.png"
    assert report_png.exists()
    assert report_png.stat().st_size > 0


def test_headline_speedup_is_computed_only_over_recommended_cases(tmp_path):
    rows = compare.run_comparison(kernels=SMALL_KERNELS, repeats=3, output_dir=tmp_path)

    recommended = [r for r in rows if r["tier2_recommended"]]
    not_recommended = [r for r in rows if not r["tier2_recommended"]]
    assert len(recommended) + len(not_recommended) == len(rows)
    # sanity: the speedup figures used for the headline number come straight from the rows
    for row in recommended:
        assert row["speedup_tier2"] == row["time_plain"] / row["time_tier2_ml"]


# --- bound-directed comparison (no ML): classify, then tile (memory-bound) or vectorize (compute-bound) ---


def test_run_bound_directed_comparison_produces_expected_rows(tmp_path):
    rows = compare.run_bound_directed_comparison(kernels=SMALL_KERNELS, repeats=3, output_dir=tmp_path)

    assert len(rows) == len(SMALL_KERNELS)
    for row in rows:
        assert row["kernel"] in SMALL_KERNELS
        assert row["bound"] in ("memory-bound", "compute-bound")
        assert row["action"] in ("tile", "vectorize", "none")
        assert row["time_plain"] > 0
        assert row["time_optimized"] > 0
        assert row["speedup"] > 0
        assert row["reason"].strip() != ""


def test_bound_directed_report_csv_is_written_with_expected_columns(tmp_path):
    compare.run_bound_directed_comparison(kernels=SMALL_KERNELS, repeats=3, output_dir=tmp_path)

    report_csv = tmp_path / "bound_directed_report.csv"
    assert report_csv.exists()
    with open(report_csv) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == compare.BOUND_DIRECTED_CSV_FIELDS
        rows = list(reader)
    assert len(rows) == len(SMALL_KERNELS)


def test_bound_directed_plots_are_written(tmp_path):
    compare.run_bound_directed_comparison(kernels=SMALL_KERNELS, repeats=3, output_dir=tmp_path)

    comparison_png = tmp_path / "bound_directed_comparison.png"
    all_benchmarks_png = tmp_path / "all_benchmarks_naive_vs_optimized.png"
    assert comparison_png.exists() and comparison_png.stat().st_size > 0
    assert all_benchmarks_png.exists() and all_benchmarks_png.stat().st_size > 0
    # the roofline plot is hardware-measured (run_papi_roofline) and NOT produced here
    assert not (tmp_path / "roofline_plot.png").exists()


def test_memory_bound_kernels_get_tile_action_when_a_loop_nest_is_found(tmp_path):
    rows = compare.run_bound_directed_comparison(kernels=SMALL_KERNELS, repeats=3, output_dir=tmp_path)
    for row in rows:
        if row["bound"] == "memory-bound" and row["action"] != "none":
            assert row["action"] == "tile"
            assert row["tile_size"] != ""


def _make_high_ai_demo():
    # a synthetic, deliberately compute-bound kernel (independent iterations, high arithmetic
    # intensity via many chained FMAs per point -- 64 * 2 flops / 16 bytes touched = 8
    # FLOP/byte, comfortably above this machine's measured ridge point of ~5.1): validates the
    # vectorize branch end-to-end, since none of the real Step 6 kernels are compute-bound
    # under this machine's calibration.
    #
    # Written to a real file and imported as a module (rather than exec()'d from a string):
    # tiling.py/unroll.py's source-to-source rewrites need inspect.getsource(func) to work,
    # which requires a function backed by an actual file on disk.
    lines = [
        "import numpy as np",
        "",
        "def high_ai_demo(n_points):",
        "    x = np.linspace(0.01, 1.0, n_points)",
        "    out = np.empty(n_points)",
        "    c = 1.0000001",
        "    for i in range(n_points):",
        "        acc = x[i]",
    ]
    lines.extend(["        acc = acc * c + c"] * 64)
    lines.append("        out[i] = acc")
    lines.append("    return out")

    tmp_dir = tempfile.mkdtemp()
    module_path = Path(tmp_dir) / "loopcost_test_high_ai_demo.py"
    module_path.write_text("\n".join(lines))

    spec = importlib.util.spec_from_file_location("loopcost_test_high_ai_demo", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.high_ai_demo


def test_compute_bound_independent_loop_gets_vectorize_with_chosen_vector_length():
    high_ai_demo = _make_high_ai_demo()

    decision = compare._decide_bound_directed(high_ai_demo, (500,))
    assert decision["bound"] == "compute-bound"
    assert decision["action"] == "vectorize"
    assert decision["vector_length"] and decision["vector_length"] > 1

    optimized = compare._apply_bound_directed(high_ai_demo, decision)
    import numpy as np

    assert np.array_equal(optimized(500), high_ai_demo(500))
