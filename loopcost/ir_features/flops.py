"""Counts and categorizes floating-point and integer arithmetic operations performed within a loop body."""

import operator
from dataclasses import dataclass
from typing import Dict, Union

from numba.core import ir, types

from loopcost.ir_features.loop_info import _int_trip_count

_ADD_OPS = {operator.add, operator.sub, operator.iadd, operator.isub}
_MUL_OPS = {operator.mul, operator.imul}
_DIV_OPS = {operator.truediv, operator.itruediv}
_INT_DIV_OPS = {operator.floordiv, operator.ifloordiv, operator.mod, operator.imod}
_TRANSCENDENTAL_NAMES = {
    "exp", "expm1", "log", "log1p", "log2", "log10", "sqrt",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "pow",
}


@dataclass
class FlopCount:
    """Per-iteration op counts and their trip-count-weighted totals (int if constant, else symbolic)."""

    per_iteration: Dict[str, int]
    total: Dict[str, Union[int, str]]


def _is_float_type(t):
    return isinstance(t, (types.Float, types.Complex))


def _is_int_type(t):
    return isinstance(t, types.Integer)


def _call_target_name(func_var, func_ir):
    try:
        defn = func_ir.get_definition(func_var)
    except Exception:
        return None
    if isinstance(defn, ir.Expr) and defn.op == "getattr":
        return defn.attr
    if isinstance(defn, (ir.Global, ir.FreeVar)):
        return defn.name
    return None


def _weight_by_trip_count(per_iteration, trip_count):
    total = {}
    for op_name, count in per_iteration.items():
        if isinstance(trip_count, int):
            total[op_name] = count * trip_count
        elif count == 0:
            total[op_name] = 0
        else:
            total[op_name] = f"{count} * {trip_count}"
    return total


def count_flops(loop_nest, func_ir, typemap):
    """Counts add/mul/div/transcendental float ops in loop_nest's own body, weighted by its trip count."""
    per_iteration = {"add": 0, "mul": 0, "div": 0, "transcendental": 0}

    for label in loop_nest.own_blocks:
        block = func_ir.blocks[label]
        for stmt in block.body:
            if not (isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr)):
                continue
            expr = stmt.value
            result_type = typemap.get(stmt.target.name)
            if not _is_float_type(result_type):
                continue
            if expr.op in ("binop", "inplace_binop"):
                if expr.fn in _ADD_OPS:
                    per_iteration["add"] += 1
                elif expr.fn in _MUL_OPS:
                    per_iteration["mul"] += 1
                elif expr.fn in _DIV_OPS:
                    per_iteration["div"] += 1
            elif expr.op == "call":
                name = _call_target_name(expr.func.name, func_ir)
                if name in _TRANSCENDENTAL_NAMES:
                    per_iteration["transcendental"] += 1

    total = _weight_by_trip_count(per_iteration, loop_nest.trip_count)
    return FlopCount(per_iteration=per_iteration, total=total)


def _full_chain_trip_count(loop_nest):
    """Product of every bound's trip count across the whole enclosing chain, or None if any
    bound isn't a concrete integer. This is the number of times loop_nest's own body actually
    executes end-to-end -- unlike count_flops/count_intops's own per-record "total" field,
    which weights by this level's own trip count only (see their docstrings), and so
    undercounts by the enclosing levels' trip counts for a genuinely nested loop.
    """
    total = 1
    for _, start, stop, step in loop_nest.bounds:
        if not (isinstance(start, int) and isinstance(stop, int) and isinstance(step, int)):
            return None
        total *= _int_trip_count(start, stop, step)
    return total


def total_ops(loop_nest, func_ir, typemap):
    """Returns the total arithmetic-op count (FLOPs + integer ops) across loop_nest's full
    enclosing chain -- the numerator this project's operational intensity uses: AI = (FLOPs +
    IntOps) / bytes accessed. Returns 0 when the chain's trip count isn't fully concrete.
    """
    flop_count = count_flops(loop_nest, func_ir, typemap)
    intop_count = count_intops(loop_nest, func_ir, typemap)
    per_iteration_total = sum(flop_count.per_iteration.values()) + sum(intop_count.per_iteration.values())

    chain_trip_count = _full_chain_trip_count(loop_nest)
    return per_iteration_total * chain_trip_count if chain_trip_count else 0


def count_intops(loop_nest, func_ir, typemap):
    """Counts add/mul/(floordiv-or-mod) integer ops in loop_nest's own body, weighted by trip count.

    These are typically index/address arithmetic (loop bounds, strided or offset indices)
    rather than "real" numerical work, but this project's operational-intensity definition
    counts them alongside floating-point ops in the numerator -- AI = (FLOPs + IntOps) /
    bytes accessed -- since they're genuine executed arithmetic contributing to a loop's cost,
    and excluding them entirely understated the intensity of index-heavy kernels.
    """
    per_iteration = {"add": 0, "mul": 0, "div": 0}

    for label in loop_nest.own_blocks:
        block = func_ir.blocks[label]
        for stmt in block.body:
            if not (isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr)):
                continue
            expr = stmt.value
            result_type = typemap.get(stmt.target.name)
            if not _is_int_type(result_type):
                continue
            if expr.op in ("binop", "inplace_binop"):
                if expr.fn in _ADD_OPS:
                    per_iteration["add"] += 1
                elif expr.fn in _MUL_OPS:
                    per_iteration["mul"] += 1
                elif expr.fn in _INT_DIV_OPS:
                    per_iteration["div"] += 1

    total = _weight_by_trip_count(per_iteration, loop_nest.trip_count)
    return FlopCount(per_iteration=per_iteration, total=total)
