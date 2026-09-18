"""Smoke test for the LLVM-choice-vs-njit-default experiment/report (real sweep data required)."""
from pathlib import Path

import pytest

from loopcost.evaluate import llvm_choice_report as report

_SWEEP_CSV = report.SWEEP_CSV


@pytest.mark.skipif(not _SWEEP_CSV.exists(), reason="sibling experiments/sweep results not present")
def test_run_llvm_choice_experiment_produces_expected_rows_and_artifacts():
    rows = report.run_llvm_choice_experiment()

    assert len(rows) > 0
    for row in rows:
        assert row["baseline_llvm"] == "llvm_O3_loopvec"
        assert row["style"] in ("vectorizable", "transcendental", "non_fp")
        assert row["baseline_gflops"] >= 0
        assert row["suggested_gflops"] >= 0
        assert row["reason"].strip() != ""
        # speedup==1.0 exactly whenever the heuristic agrees with the baseline LLVM setting
        if row["suggested_llvm"] == row["baseline_llvm"]:
            assert row["speedup"] == pytest.approx(1.0)

    assert report.REPORT_CSV.exists()
    assert report.IMPROVEMENT_PLOT_PNG.exists() and report.IMPROVEMENT_PLOT_PNG.stat().st_size > 0
    assert report.SUMMARY_PLOT_PNG.exists() and report.SUMMARY_PLOT_PNG.stat().st_size > 0


@pytest.mark.skipif(not _SWEEP_CSV.exists(), reason="sibling experiments/sweep results not present")
def test_report_csv_matches_declared_fields():
    import csv

    report.run_llvm_choice_experiment()
    with open(report.REPORT_CSV) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == report.CSV_FIELDS
        rows = list(reader)
    assert len(rows) > 0
