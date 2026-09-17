"""Counts and categorizes floating-point operations performed within a loop body."""

import operator
from dataclasses import dataclass
from typing import Dict, Union

from numba.core import ir, types

_ADD_OPS = {operator.add, operator.sub, operator.iadd, operator.isub}
_MUL_OPS = {operator.mul, operator.imul}
_DIV_OPS = {operator.truediv, operator.itruediv}
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

    trip_count = loop_nest.trip_count
    total = {}
    for op_name, count in per_iteration.items():
        if isinstance(trip_count, int):
            total[op_name] = count * trip_count
        elif count == 0:
            total[op_name] = 0
        else:
            total[op_name] = f"{count} * {trip_count}"
    return FlopCount(per_iteration=per_iteration, total=total)
