"""Applies heuristic rules to select which loop transformations to recommend."""

from math import isqrt
from pathlib import Path

from loopcost.heuristic import ridge_point
from loopcost.heuristic.classify import classify_bound

_SYSFS_CACHE_ROOT = Path("/sys/devices/system/cpu/cpu0/cache")
_DEFAULT_L2_CACHE_BYTES = 256 * 1024  # used only if sysfs is unavailable (e.g. non-Linux, sandboxed)


def _parse_cache_size(text):
    text = text.strip()
    units = {"K": 1024, "M": 1024**2, "G": 1024**3}
    if text and text[-1] in units:
        return int(text[:-1]) * units[text[-1]]
    return int(text)


def _read_l2_cache_bytes_from_sysfs():
    if not _SYSFS_CACHE_ROOT.exists():
        return None
    for index_dir in sorted(_SYSFS_CACHE_ROOT.glob("index*")):
        try:
            level = (index_dir / "level").read_text().strip()
            if level != "2":
                continue
            return _parse_cache_size((index_dir / "size").read_text())
        except (OSError, ValueError):
            continue
    return None


def get_l2_cache_bytes(force_recalibrate=False):
    """Returns this CPU's L2 cache size in bytes, read from sysfs once and cached like the ridge point."""
    cpu_model = ridge_point._get_cpu_model()
    cache = ridge_point._load_cache()
    entry = cache.get(cpu_model, {})

    if "l2_cache_bytes" in entry and not force_recalibrate:
        return entry["l2_cache_bytes"]

    l2_bytes = _read_l2_cache_bytes_from_sysfs()
    if l2_bytes is None:
        l2_bytes = _DEFAULT_L2_CACHE_BYTES

    entry["l2_cache_bytes"] = l2_bytes
    cache[cpu_model] = entry
    ridge_point._save_cache(cache)
    return l2_bytes


def _simd_unroll_decision(transform, features):
    oi = features["oi"]
    r = features["ridge_point"]
    bound = classify_bound(oi, r)

    if bound == "memory-bound":
        reason = f"OI={oi:.3g} < ridge_point={r:.3g} ({bound}); {transform} needs a compute-bound loop to pay off"
        return False, {}, reason

    if features.get("has_loop_carried_dependency"):
        reason = (
            f"OI={oi:.3g} >= ridge_point={r:.3g} ({bound}), but a loop-carried dependency "
            f"prevents {transform}"
        )
        return False, {}, reason

    reason = f"OI={oi:.3g} >= ridge_point={r:.3g} ({bound}) with no loop-carried dependency"
    if transform == "vectorize":
        params = {"simd_width": features["simd_width"]} if "simd_width" in features else {}
    else:
        trip_count = features.get("trip_count")
        unroll_factor = min(8, trip_count) if isinstance(trip_count, int) and trip_count > 1 else 4
        params = {"unroll_factor": unroll_factor}
    return True, params, reason


def _tile_decision(loop_nest, features):
    oi = features["oi"]
    r = features["ridge_point"]
    bound = classify_bound(oi, r)
    working_set = features.get("working_set_bytes")
    l2_bytes = get_l2_cache_bytes()

    if bound != "memory-bound":
        reason = f"OI={oi:.3g} >= ridge_point={r:.3g} ({bound}); tiling targets memory-bound loops"
        return False, {}, reason

    if working_set is None or working_set <= l2_bytes:
        reason = f"working_set_bytes={working_set} fits within L2 cache ({l2_bytes} bytes); tiling unnecessary"
        return False, {}, reason

    itemsize = features.get("itemsize", 8)
    num_arrays = max(1, features.get("num_arrays", 1))
    usable_bytes = l2_bytes // 2  # leave headroom for other data resident in L2
    tile_size = max(1, isqrt(usable_bytes // (itemsize * num_arrays)))

    params = {"tile_size": tile_size, "l2_cache_bytes": l2_bytes}
    reason = (
        f"working_set_bytes={working_set} exceeds L2 cache ({l2_bytes} bytes) ({bound}); "
        f"tiling to {tile_size} elements per dimension"
    )
    return True, params, reason


def _fuse_decision(loop_nest, features):
    adjacent_nests = features.get("adjacent_loop_nests")
    if adjacent_nests:
        this_arrays = set(features.get("arrays", []))
        this_bounds = loop_nest.trip_count
        for other in adjacent_nests:
            shared = this_arrays & set(other.get("arrays", []))
            if shared and other.get("trip_count") == this_bounds:
                params = {"shared_arrays": sorted(shared), "fused_with_header": other.get("header")}
                reason = (
                    f"shares array(s) {sorted(shared)} with adjacent loop nest "
                    f"(header={other.get('header')}) with matching bounds {this_bounds}"
                )
                return True, params, reason

    return False, {}, "fusion candidate detection not yet implemented"


def decide(loop_nest, features):
    """Returns {"tile"|"vectorize"|"unroll"|"fuse": (profitable, params, reason)} for loop_nest."""
    return {
        "vectorize": _simd_unroll_decision("vectorize", features),
        "unroll": _simd_unroll_decision("unroll", features),
        "tile": _tile_decision(loop_nest, features),
        "fuse": _fuse_decision(loop_nest, features),
    }
