"""Tests for cache_model's exact islpy path, conservative fallback path, and OI div-by-zero guard."""
import islpy as isl

from loopcost.ir_features.access_pattern import ArrayAccess
from loopcost.ir_features.cache_model import (
    build_access_map,
    estimate_bytes_moved,
    operational_intensity,
    working_set_bytes,
)
from loopcost.ir_features.loop_info import LoopNest

CACHE_LINE_BYTES = 64
ITEMSIZE = 8  # float64


def _contiguous_loop_nest(trip_count=1024):
    return LoopNest(
        header=0,
        depth=1,
        induction_vars=["i"],
        trip_count=trip_count,
        own_blocks=[0],
        body_blocks=[0],
        bounds=[("i", 0, trip_count, 1)],
    )


def _contiguous_access(array_name, kind):
    return ArrayAccess(
        array=array_name,
        kind=kind,
        is_affine=True,
        coefficients={"dim0": 1},
        classification="contiguous",
        dim_coeffs=[{"i": 1}],
        unit_dim=0,
        itemsize=ITEMSIZE,
    )


# --- (1) contiguous-access kernel: exact islpy path ---


def test_build_access_map_contiguous_produces_a_valid_basic_map():
    loop_nest = _contiguous_loop_nest(1024)
    access = _contiguous_access("a", "read")
    access_map = build_access_map(loop_nest, access)
    assert isinstance(access_map, isl.BasicMap)
    # 8 elements/line * 128 lines = 1024 elements, offsets should be exactly {0, 8, 16, ..., 8184}
    offsets = isl.Set.from_basic_set(access_map.range())
    assert int(str(offsets.count_val())) == 1024


def test_build_access_map_returns_none_for_irregular_access():
    loop_nest = _contiguous_loop_nest(1024)
    access = ArrayAccess(
        array="a", kind="read", is_affine=False, coefficients=None, classification="irregular", itemsize=8
    )
    assert build_access_map(loop_nest, access) is None


def test_estimate_bytes_moved_contiguous_kernel_uses_exact_path():
    loop_nest = _contiguous_loop_nest(1024)
    accesses = [_contiguous_access("a", "read"), _contiguous_access("c", "write")]

    result = estimate_bytes_moved(loop_nest, accesses)

    # hand computation: 1024 float64 elements = 8192 bytes, 8 elements exactly fill each
    # 64-byte cache line with no waste -> 8192 / 64 = 128 distinct lines per array, times 2 arrays
    assert result.path == "exact"
    assert result.bytes_moved == 2 * 128 * CACHE_LINE_BYTES == 16384


def test_estimate_bytes_moved_contiguous_kernel_accounts_for_cache_line_reuse():
    # 10 elements of 8 bytes each only span ceil(80/64) = 2 cache lines, not 10 separate ones
    loop_nest = _contiguous_loop_nest(10)
    accesses = [_contiguous_access("a", "read")]

    result = estimate_bytes_moved(loop_nest, accesses)

    assert result.path == "exact"
    assert result.bytes_moved == 2 * CACHE_LINE_BYTES == 128


def test_estimate_bytes_moved_respects_time_budget_guard():
    # an impossible (already-exceeded) budget must force the fallback path even though
    # this access would otherwise take the exact path
    loop_nest = _contiguous_loop_nest(1024)
    accesses = [_contiguous_access("a", "read")]

    result = estimate_bytes_moved(loop_nest, accesses, time_budget_s=-1.0)

    assert result.path == "fallback"
    assert "time budget" in result.reason


# --- (2) random-index kernel: fallback path forced ---


def test_estimate_bytes_moved_falls_back_for_irregular_access():
    loop_nest = _contiguous_loop_nest(8)
    accesses = [
        # 'a' is the gather source: index is data-dependent, so this is irregular.
        # Its full declared shape (20 elements) is known even though the access isn't affine.
        ArrayAccess(
            array="a",
            kind="read",
            is_affine=False,
            coefficients=None,
            classification="irregular",
            itemsize=ITEMSIZE,
            array_shape=(20,),
        ),
        # 'c' is written contiguously, but its shape isn't known here, so the fallback
        # must estimate it conservatively from the loop's trip count instead.
        _contiguous_access("c", "write"),
    ]

    result = estimate_bytes_moved(loop_nest, accesses)

    # hand computation: a = 20 elements * 8 bytes = 160 (from known shape);
    # c = 8 iterations * 8 bytes = 64 (from trip count, shape unknown) -> total 224
    assert result.path == "fallback"
    assert result.bytes_moved == 160 + 64 == 224
    assert "irregular" in result.reason or "a" in result.reason


def test_working_set_bytes_uses_shape_when_known_and_trip_count_otherwise():
    loop_nest = _contiguous_loop_nest(8)
    accesses = [
        ArrayAccess(
            array="a", kind="read", is_affine=False, coefficients=None, classification="irregular",
            itemsize=ITEMSIZE, array_shape=(20,),
        ),
        _contiguous_access("c", "write"),
    ]
    assert working_set_bytes(loop_nest, accesses) == 224


def test_working_set_bytes_deduplicates_repeated_array_references():
    loop_nest = _contiguous_loop_nest(8)
    accesses = [_contiguous_access("a", "read"), _contiguous_access("a", "read")]
    # 'a' touched twice but must only be counted once: 8 iterations * 8 bytes = 64
    assert working_set_bytes(loop_nest, accesses) == 64


# --- (3) operational_intensity divide-by-zero guard ---


def test_operational_intensity_normal_case():
    assert operational_intensity(500, 2000) == 0.25


def test_operational_intensity_is_infinite_for_zero_bytes_with_real_compute():
    # zero measured data movement with real flops present is the most compute-bound case
    # there is (unboundedly so) -- must not be reported as 0.0, which would read as memory-bound
    assert operational_intensity(100, 0) == float("inf")
    assert operational_intensity(100, -5) == float("inf")


def test_operational_intensity_is_zero_when_uninformative():
    # neither bytes nor flops measured: genuinely no information, not a compute-bound claim
    assert operational_intensity(0, 0) == 0.0
    assert operational_intensity(None, 0) == 0.0
