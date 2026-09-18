"""Estimates cache behavior (hit/miss rates, working-set fit) for a given loop's access patterns."""

import time
from dataclasses import dataclass
from typing import List, Optional, Union

import islpy as isl

from loopcost.ir_features.loop_info import _int_trip_count

DEFAULT_CACHE_LINE_BYTES = 64
DEFAULT_TIME_BUDGET_S = 0.1


@dataclass
class BytesMovedEstimate:
    """Result of estimate_bytes_moved: the byte count plus which computation path produced it."""

    bytes_moved: int
    path: str  # "exact" or "fallback"
    reason: str


def _chain_trip_count(loop_nest):
    """Multiplies trip counts across loop_nest.bounds (the whole enclosing chain); None if any is symbolic."""
    total = 1
    for _, start, stop, step in loop_nest.bounds:
        if not (isinstance(start, int) and isinstance(stop, int) and isinstance(step, int)):
            return None
        total *= _int_trip_count(start, stop, step)
    return total


def build_access_map(loop_nest, access_record):
    """Builds an islpy BasicMap from this access's loop indices to its linear byte offset.

    Only supported for affine, unit-stride ("contiguous") accesses whose address does not
    depend on any loop variable through a non-unit-stride dimension (which would require
    array-shape/stride information this record doesn't carry). Returns None whenever the
    map can't be built: non-affine access, non-unit-stride access, address depending on an
    orthogonal dimension, symbolic (non-constant) loop bounds, or a malformed expression.
    """
    if not access_record.is_affine or access_record.classification != "contiguous":
        return None
    if access_record.dim_coeffs is None or access_record.unit_dim is None or access_record.itemsize is None:
        return None

    dim_coeffs = access_record.dim_coeffs
    unit_dim = access_record.unit_dim
    for i, coeffs in enumerate(dim_coeffs):
        if i == unit_dim:
            continue
        if any(k != "const" for k in coeffs):
            return None  # address also depends on an orthogonal dimension; unsupported here

    formula = dim_coeffs[unit_dim]
    loop_vars = [v for v in formula if v != "const"]

    bounds_by_var = {var: (var, start, stop, step) for var, start, stop, step in loop_nest.bounds}
    if not all(v in bounds_by_var for v in loop_vars):
        return None
    relevant_bounds = [bounds_by_var[v] for v in loop_vars]
    if not all(isinstance(b[1], int) and isinstance(b[2], int) and isinstance(b[3], int) for b in relevant_bounds):
        return None

    const = formula.get("const", 0)
    if loop_vars:
        terms = " + ".join(f"{formula[v]}*{v}" for v in loop_vars)
        element_expr = f"{terms} + {const}" if const else terms
        dims = ",".join(loop_vars)
        constraints = " and ".join(
            f"{start} <= {var} < {stop}" if step == 1 else f"exists q0: {start} <= {var} < {stop} and {var} - {start} = {step}*q0"
            for var, start, stop, step in relevant_bounds
        )
        expr = f"{{ [{dims}] -> [o] : o = ({element_expr}) * {access_record.itemsize} and {constraints} }}"
    else:
        expr = f"{{ [] -> [o] : o = {const * access_record.itemsize} }}"

    try:
        return isl.BasicMap(expr)
    except Exception:
        return None


def _count_distinct_cache_lines(byte_offset_map, cache_line_bytes):
    """Given a BasicMap onto byte offsets, returns the number of distinct cache lines those offsets span."""
    offsets = isl.Set.from_basic_set(byte_offset_map.range())
    line_map = isl.BasicMap(f"{{ [o] -> [line] : line = floor(o / {cache_line_bytes}) }}")
    lines = isl.Map.from_basic_map(line_map).intersect_domain(offsets).range()
    return int(str(lines.count_val()))


def working_set_bytes(loop_nest, access_records):
    """Conservative footprint: full array size if known, else (loop trip count * itemsize) per distinct array."""
    total_iterations = _chain_trip_count(loop_nest)
    per_array_bytes = {}
    for access in access_records:
        if access.array in per_array_bytes:
            continue
        itemsize = access.itemsize or 0
        if access.array_shape is not None:
            num_elements = 1
            for extent in access.array_shape:
                num_elements *= extent
        elif total_iterations is not None:
            num_elements = total_iterations  # worst case: every iteration touches a new element
        else:
            num_elements = 0
        per_array_bytes[access.array] = num_elements * itemsize
    return sum(per_array_bytes.values())


def estimate_bytes_moved(
    loop_nest,
    access_records,
    cache_line_bytes=DEFAULT_CACHE_LINE_BYTES,
    time_budget_s=DEFAULT_TIME_BUDGET_S,
):
    """Exact islpy-based byte count for affine unit-stride accesses; conservative fallback otherwise.

    Falls back to working_set_bytes() when any access is non-affine or not unit-stride, when
    building or measuring its islpy access map fails, or when the wall-clock time budget is
    exceeded -- whichever happens first.
    """
    start_time = time.perf_counter()
    exact_bytes = 0

    for access in access_records:
        if access.classification != "contiguous" or not access.is_affine:
            return BytesMovedEstimate(
                working_set_bytes(loop_nest, access_records),
                "fallback",
                f"non-affine or non-unit-stride access to '{access.array}'",
            )

        access_map = build_access_map(loop_nest, access)
        if access_map is None:
            return BytesMovedEstimate(
                working_set_bytes(loop_nest, access_records),
                "fallback",
                f"could not build an affine access map for '{access.array}'",
            )

        if time.perf_counter() - start_time > time_budget_s:
            return BytesMovedEstimate(
                working_set_bytes(loop_nest, access_records),
                "fallback",
                "exceeded time budget while building access maps",
            )

        try:
            lines = _count_distinct_cache_lines(access_map, cache_line_bytes)
        except Exception:
            return BytesMovedEstimate(
                working_set_bytes(loop_nest, access_records),
                "fallback",
                f"islpy cardinality computation failed for '{access.array}'",
            )

        if time.perf_counter() - start_time > time_budget_s:
            return BytesMovedEstimate(
                working_set_bytes(loop_nest, access_records),
                "fallback",
                "exceeded time budget while counting cache lines",
            )

        exact_bytes += lines * cache_line_bytes

    return BytesMovedEstimate(exact_bytes, "exact", "all accesses were affine and unit-stride")


def operational_intensity(flop_count, bytes_moved):
    """Returns flop_count / bytes_moved (FLOP/byte).

    Guards division by zero without conflating two different situations: no measured data
    movement with real compute present (flop_count > 0) is the most compute-bound case there
    is -- unboundedly so -- and returns +inf, not 0.0 (which would misclassify it as
    memory-bound, the opposite conclusion). Only the genuinely uninformative case, where
    there's neither measured bytes nor measured flops, returns 0.0.
    """
    if bytes_moved is None or bytes_moved <= 0:
        return float("inf") if flop_count and flop_count > 0 else 0.0
    return flop_count / bytes_moved
