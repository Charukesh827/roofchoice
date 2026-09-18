"""Counts and categorizes floating-point and integer arithmetic operations performed within a loop body."""

import operator
import re
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
    "sinh", "cosh", "tanh", "pow", "erf", "erfc",
}

# A scalar transcendental call (exp/log/sqrt/erf/...) costs far more than a single add/mul on
# real hardware -- typically a multi-term polynomial/rational approximation plus range
# reduction in libm, not one instruction. Counting it as "1 op" (as add/mul/div are) badly
# undercounts real compute cost for transcendental-heavy kernels, which showed up as a
# systematic static-vs-measured compute-bound/memory-bound misclassification (see
# evaluate/bound_validation.py): black_scholes and correlation were both called memory-bound
# statically while the real PAPI-measured AI put them solidly compute-bound. This weight is a
# literature-typical approximation for a non-vectorized scalar transcendental (rough order of
# magnitude, not a per-function-exact cost) -- the actual cost varies by function and libm
# implementation, but "count it as 1" is far more wrong than "count it as ~20".
TRANSCENDENTAL_FLOP_WEIGHT = 20


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


_CHAIN_ENUMERATION_BUDGET = 200_000  # cap on values enumerated for a dependent (triangular) bound


def _eval_bound_expr(value, env):
    """Resolves a bound's start/stop/step to a concrete int, given already-known outer loop
    variable values (env). `value` may already be a concrete int, or a symbolic expression
    string (e.g. "(a.1 + 1)") referencing an enclosing loop's induction variable.

    Numba's SSA-renamed variable names (e.g. "a.1") contain a dot, which is not a valid Python
    identifier -- eval(expr, {}, env) can't resolve "a.1" as a NAME token at all (it parses as
    attribute access on "a", a SyntaxError). So known variables are textually substituted into
    the expression *before* evaluating, rather than passed as eval locals.
    """
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    text = value
    for var, val in env.items():
        text = re.sub(rf"\b{re.escape(var)}\b", f"({val})", text)
    try:
        result = eval(text, {"__builtins__": {}}, {})
        return int(result) if isinstance(result, (int, float)) else None
    except Exception:
        return None


def _bound_references_var(var, stop_expr):
    if not isinstance(stop_expr, str):
        return False
    return re.search(rf"\b{re.escape(var)}\b", stop_expr) is not None


def _full_chain_trip_count(loop_nest):
    """Total number of times loop_nest's own body executes end-to-end across its whole
    enclosing chain -- unlike count_flops/count_intops's own per-record "total" field, which
    weights by this level's own trip count only (see their docstrings), and so undercounts by
    the enclosing levels' trip counts for a genuinely nested loop.

    Independent levels (the common case) multiply directly. A level whose bound depends on an
    enclosing loop's induction variable (a triangular/staircase nest, e.g.
    `for a in range(n): for b in range(a + 1): ...`) is resolved by enumerating the enclosing
    variable's small concrete range and summing that level's trip count per value -- exact, and
    only pays the enumeration cost where a real dependency exists. Returns None if any bound
    still can't be resolved (e.g. depends on a runtime array's shape via a local variable with
    no traceable provenance to the call arguments).
    """
    return _chain_trip_count_rec(loop_nest.bounds, {})


def _chain_trip_count_rec(bounds, env):
    if not bounds:
        return 1

    var, start, stop, step = bounds[0]
    start_v = _eval_bound_expr(start, env)
    step_v = _eval_bound_expr(step, env)
    if start_v is None or step_v is None or step_v == 0:
        return None

    stop_v = _eval_bound_expr(stop, env)
    if stop_v is None:
        return None  # can't resolve even this level's own extent

    remaining = bounds[1:]
    depended_on = any(_bound_references_var(var, b[2]) for b in remaining)

    if not depended_on:
        trip = _int_trip_count(start_v, stop_v, step_v)
        if trip <= 0:
            return 0
        inner = _chain_trip_count_rec(remaining, env)
        return None if inner is None else trip * inner

    # a later bound's extent depends on this loop's induction variable (triangular/staircase
    # nest) -- enumerate this level's (small) concrete range and sum the inner trip count per
    # value, since the levels no longer multiply independently
    values = range(start_v, stop_v, step_v)
    if len(values) > _CHAIN_ENUMERATION_BUDGET:
        return None
    total = 0
    for v in values:
        inner = _chain_trip_count_rec(remaining, {**env, var: v})
        if inner is None:
            return None
        total += inner
    return total


def total_ops(loop_nest, func_ir, typemap):
    """Returns the total arithmetic-op count (FLOPs + integer ops) across loop_nest's full
    enclosing chain -- the numerator this project's operational intensity uses: AI = (FLOPs +
    IntOps) / bytes accessed. Transcendental float ops (exp/log/sqrt/erf/...) are weighted by
    TRANSCENDENTAL_FLOP_WEIGHT, not counted as 1 op like add/mul/div (see that constant's
    docstring). Returns 0 when the chain's trip count isn't fully resolvable.
    """
    flop_count = count_flops(loop_nest, func_ir, typemap)
    intop_count = count_intops(loop_nest, func_ir, typemap)
    weighted_flops = (
        flop_count.per_iteration["add"]
        + flop_count.per_iteration["mul"]
        + flop_count.per_iteration["div"]
        + TRANSCENDENTAL_FLOP_WEIGHT * flop_count.per_iteration["transcendental"]
    )
    per_iteration_total = weighted_flops + sum(intop_count.per_iteration.values())

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
