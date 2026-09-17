"""Tests that hardware calibration is cached to disk and reused instead of re-measured."""
import pytest

from loopcost.heuristic import ridge_point


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ridge_point, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(ridge_point, "_CALIBRATION_FILE", tmp_path / "calibration.json")


def _fake_measurements(monkeypatch, calls):
    def fake_flops(*args, **kwargs):
        calls["flops"] += 1
        return 100.0

    def fake_bandwidth(*args, **kwargs):
        calls["bandwidth"] += 1
        return 20.0

    monkeypatch.setattr(ridge_point, "measure_peak_flops", fake_flops)
    monkeypatch.setattr(ridge_point, "measure_peak_bandwidth", fake_bandwidth)


def test_calibration_creates_cache_and_is_reused(monkeypatch):
    calls = {"flops": 0, "bandwidth": 0}
    _fake_measurements(monkeypatch, calls)

    peak_gflops1, peak_bw1, ridge1 = ridge_point.get_ridge_point()
    assert calls == {"flops": 1, "bandwidth": 1}
    assert ridge_point._CALIBRATION_FILE.exists()
    assert ridge1 == pytest.approx(100.0 / 20.0)

    peak_gflops2, peak_bw2, ridge2 = ridge_point.get_ridge_point()
    assert calls == {"flops": 1, "bandwidth": 1}  # measurement functions not called again
    assert (peak_gflops1, peak_bw1, ridge1) == (peak_gflops2, peak_bw2, ridge2)


def test_force_recalibrate_bypasses_cache(monkeypatch):
    calls = {"flops": 0, "bandwidth": 0}
    _fake_measurements(monkeypatch, calls)

    ridge_point.get_ridge_point()
    ridge_point.get_ridge_point(force_recalibrate=True)

    assert calls == {"flops": 2, "bandwidth": 2}


# --- carm-roofline integration: prefer the sibling project's measured data over self-measurement ---


def test_carm_peak_gflops_selects_matching_row_only():
    rows = [
        {"type": "arithmetic", "isa": "x86_sse", "data_type": "f64", "operation": "fma",
         "num_threads": "1", "performance_gops": "27.9"},
        {"type": "arithmetic", "isa": "x86_avx2", "data_type": "f64", "operation": "add",
         "num_threads": "1", "performance_gops": "13.9"},  # wrong operation
        {"type": "arithmetic", "isa": "x86_avx2", "data_type": "f32", "operation": "fma",
         "num_threads": "1", "performance_gops": "111.8"},  # wrong data type
        {"type": "arithmetic", "isa": "x86_avx2", "data_type": "f64", "operation": "fma",
         "num_threads": "6", "performance_gops": "333.6"},  # wrong thread count
        {"type": "arithmetic", "isa": "x86_avx2", "data_type": "f64", "operation": "fma",
         "num_threads": "1", "performance_gops": "55.9"},  # the match
    ]
    assert ridge_point._carm_peak_gflops(rows, "x86_avx2") == 55.9
    assert ridge_point._carm_peak_gflops(rows, "x86_avx512") is None


def test_carm_peak_bandwidth_selects_matching_row_only():
    rows = [
        {"type": "memory", "isa": "x86_avx2", "data_type": "f64", "cache_level": "L2",
         "num_threads": "1", "bandwidth_gbps": "0.37"},  # wrong cache level
        {"type": "memory", "isa": "x86_sse", "data_type": "f64", "cache_level": "DRAM",
         "num_threads": "1", "bandwidth_gbps": "0.17"},  # wrong isa
        {"type": "memory", "isa": "x86_avx2", "data_type": "f64", "cache_level": "DRAM",
         "num_threads": "1", "bandwidth_gbps": "10.88"},  # the match
    ]
    assert ridge_point._carm_peak_bandwidth_gbps(rows, "x86_avx2") == 10.88
    assert ridge_point._carm_peak_bandwidth_gbps(rows, "x86_avx512") is None


def test_measure_functions_fall_back_to_numba_when_carm_results_are_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(ridge_point, "_carm_results_dir", lambda: tmp_path / "does-not-exist")

    calls = {"flops": 0, "bandwidth": 0}

    def fake_numba_flops(*args, **kwargs):
        calls["flops"] += 1
        return 42.0

    def fake_numba_bandwidth(*args, **kwargs):
        calls["bandwidth"] += 1
        return 7.0

    monkeypatch.setattr(ridge_point, "_measure_peak_flops_numba", fake_numba_flops)
    monkeypatch.setattr(ridge_point, "_measure_peak_bandwidth_numba", fake_numba_bandwidth)

    assert ridge_point.measure_peak_flops() == 42.0
    assert ridge_point.measure_peak_bandwidth() == 7.0
    assert calls == {"flops": 1, "bandwidth": 1}


def test_measure_functions_use_carm_results_when_available():
    carm_dir = ridge_point._carm_results_dir()
    if not (carm_dir / "summary.csv").exists():
        pytest.skip("carm-roofline results not present on this machine")

    rows = ridge_point._load_carm_summary_rows()
    isa = ridge_point._preferred_carm_isa()
    expected_gflops = ridge_point._carm_peak_gflops(rows, isa)
    expected_bandwidth = ridge_point._carm_peak_bandwidth_gbps(rows, isa)

    assert ridge_point.measure_peak_flops() == expected_gflops
    assert ridge_point.measure_peak_bandwidth() == expected_bandwidth
