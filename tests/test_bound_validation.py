"""Tests for validating the static CB/BB classification against the real PAPI-measured sweep."""
from pathlib import Path

import pytest

from loopcost.evaluate import bound_validation as bv

_SWEEP_CSV = bv.SWEEP_CSV


@pytest.mark.skipif(not _SWEEP_CSV.exists(), reason="sibling experiments/sweep results not present")
def test_run_bound_validation_produces_expected_rows_and_csv():
    rows = bv.run_bound_validation()

    assert len(rows) > 0
    for row in rows:
        assert row["static_bound"] in ("compute-bound", "memory-bound")
        assert row["dynamic_bound"] in ("compute-bound", "memory-bound")
        assert row["dynamic_ai_dram"] >= 0
        assert row["match"] == (row["static_bound"] == row["dynamic_bound"])

    assert bv.REPORT_CSV.exists()


@pytest.mark.skipif(not _SWEEP_CSV.exists(), reason="sibling experiments/sweep results not present")
def test_report_csv_matches_declared_fields():
    import csv

    bv.run_bound_validation()
    with open(bv.REPORT_CSV) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == bv.CSV_FIELDS
        rows = list(reader)
    assert len(rows) > 0
