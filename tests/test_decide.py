"""Tests for decide(): synthetic feature dicts covering all four transforms, profitable and not."""
from loopcost.heuristic.decide import decide, get_l2_cache_bytes
from loopcost.ir_features.loop_info import LoopNest


def _loop_nest(trip_count=1000):
    return LoopNest(
        header=0,
        depth=1,
        induction_vars=["i"],
        trip_count=trip_count,
        own_blocks=[0],
        body_blocks=[0],
        bounds=[("i", 0, trip_count, 1)],
    )


def _assert_reason(reason, *values):
    assert isinstance(reason, str) and reason.strip() != ""
    for value in values:
        assert str(value) in reason, f"expected {value!r} referenced in reason: {reason!r}"


# --- vectorize / unroll: profitable only if compute-bound AND no loop-carried dependency ---


def test_vectorize_and_unroll_profitable_when_compute_bound_and_independent():
    features = {
        "oi": 12.5,
        "ridge_point": 4.0,
        "has_loop_carried_dependency": False,
        "working_set_bytes": 1000,
        "itemsize": 8,
        "trip_count": 1000,
    }
    result = decide(_loop_nest(), features)

    for transform in ("vectorize", "unroll"):
        profitable, params, reason = result[transform]
        assert profitable is True
        assert isinstance(params, dict)
        _assert_reason(reason, "12.5", "4")


def test_vectorize_and_unroll_not_profitable_when_memory_bound():
    features = {
        "oi": 1.2,
        "ridge_point": 4.0,
        "has_loop_carried_dependency": False,
        "working_set_bytes": 1000,
        "itemsize": 8,
        "trip_count": 1000,
    }
    result = decide(_loop_nest(), features)

    for transform in ("vectorize", "unroll"):
        profitable, params, reason = result[transform]
        assert profitable is False
        assert params == {}
        _assert_reason(reason, "1.2", "4")


def test_vectorize_and_unroll_not_profitable_when_loop_carried_dependency():
    # compute-bound (OI above ridge point) but a running accumulator blocks both transforms
    features = {
        "oi": 12.5,
        "ridge_point": 4.0,
        "has_loop_carried_dependency": True,
        "working_set_bytes": 1000,
        "itemsize": 8,
        "trip_count": 1000,
    }
    result = decide(_loop_nest(), features)

    for transform in ("vectorize", "unroll"):
        profitable, params, reason = result[transform]
        assert profitable is False
        assert params == {}
        _assert_reason(reason, "12.5", "4")
        assert "loop-carried dependency" in reason


# --- tile: profitable only if memory-bound AND working_set_bytes > L2 cache size ---


def test_tile_profitable_when_memory_bound_and_working_set_exceeds_l2():
    l2_bytes = get_l2_cache_bytes()
    working_set = l2_bytes * 4  # comfortably larger than L2
    features = {
        "oi": 0.5,
        "ridge_point": 4.0,
        "has_loop_carried_dependency": False,
        "working_set_bytes": working_set,
        "itemsize": 8,
        "num_arrays": 2,
        "trip_count": 1000,
    }
    profitable, params, reason = decide(_loop_nest(), features)["tile"]

    assert profitable is True
    assert isinstance(params.get("tile_size"), int) and params["tile_size"] > 0
    assert params["l2_cache_bytes"] == l2_bytes
    _assert_reason(reason, working_set, l2_bytes)


def test_tile_not_profitable_when_compute_bound():
    l2_bytes = get_l2_cache_bytes()
    features = {
        "oi": 12.5,
        "ridge_point": 4.0,
        "has_loop_carried_dependency": False,
        "working_set_bytes": l2_bytes * 4,
        "itemsize": 8,
        "trip_count": 1000,
    }
    profitable, params, reason = decide(_loop_nest(), features)["tile"]

    assert profitable is False
    assert params == {}
    _assert_reason(reason, "12.5", "4")


def test_tile_not_profitable_when_working_set_fits_in_l2():
    l2_bytes = get_l2_cache_bytes()
    small_working_set = l2_bytes // 4  # comfortably smaller than L2
    features = {
        "oi": 0.5,
        "ridge_point": 4.0,
        "has_loop_carried_dependency": False,
        "working_set_bytes": small_working_set,
        "itemsize": 8,
        "trip_count": 1000,
    }
    profitable, params, reason = decide(_loop_nest(), features)["tile"]

    assert profitable is False
    assert params == {}
    _assert_reason(reason, small_working_set, l2_bytes)


# --- fuse: profitable only for an adjacent loop nest sharing an array with matching bounds ---


def test_fuse_profitable_with_matching_adjacent_loop_nest():
    features = {
        "oi": 1.0,
        "ridge_point": 4.0,
        "arrays": ["a", "b"],
        "adjacent_loop_nests": [{"arrays": ["b", "c"], "trip_count": 1000, "header": 99}],
    }
    profitable, params, reason = decide(_loop_nest(1000), features)["fuse"]

    assert profitable is True
    assert params["shared_arrays"] == ["b"]
    assert params["fused_with_header"] == 99
    _assert_reason(reason, "b", 1000)


def test_fuse_not_profitable_without_adjacent_candidate():
    # no adjacent-loop-nest information at all: must stub conservatively, not guess
    features = {"oi": 1.0, "ridge_point": 4.0}
    profitable, params, reason = decide(_loop_nest(), features)["fuse"]

    assert profitable is False
    assert params == {}
    assert reason == "fusion candidate detection not yet implemented"


def test_fuse_not_profitable_when_adjacent_nest_shares_no_array():
    features = {
        "oi": 1.0,
        "ridge_point": 4.0,
        "arrays": ["a"],
        "adjacent_loop_nests": [{"arrays": ["b", "c"], "trip_count": 1000, "header": 99}],
    }
    profitable, params, reason = decide(_loop_nest(1000), features)["fuse"]

    assert profitable is False
    assert params == {}
    assert reason == "fusion candidate detection not yet implemented"


def test_fuse_not_profitable_when_bounds_differ():
    features = {
        "oi": 1.0,
        "ridge_point": 4.0,
        "arrays": ["a", "b"],
        "adjacent_loop_nests": [{"arrays": ["b"], "trip_count": 2000, "header": 99}],
    }
    profitable, params, reason = decide(_loop_nest(1000), features)["fuse"]

    assert profitable is False
    assert params == {}
    assert reason == "fusion candidate detection not yet implemented"


# --- reasons must always be non-empty strings that reference the actual feature values ---


def test_all_reasons_are_non_empty_strings_for_every_transform():
    scenarios = [
        {"oi": 12.5, "ridge_point": 4.0, "has_loop_carried_dependency": False,
         "working_set_bytes": 1000, "itemsize": 8, "trip_count": 1000},
        {"oi": 0.5, "ridge_point": 4.0, "has_loop_carried_dependency": False,
         "working_set_bytes": get_l2_cache_bytes() * 4, "itemsize": 8, "trip_count": 1000},
    ]
    for features in scenarios:
        for profitable, params, reason in decide(_loop_nest(), features).values():
            assert isinstance(profitable, bool)
            assert isinstance(params, dict)
            assert isinstance(reason, str) and reason.strip() != ""
