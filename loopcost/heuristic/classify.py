"""Classifies loops into performance categories based on extracted IR features."""

from numba.core import ir


def classify_bound(oi, ridge_point):
    """Classifies a loop as "compute-bound" or "memory-bound" by comparing OI to the ridge point."""
    return "compute-bound" if oi >= ridge_point else "memory-bound"


def _operand_names(defn):
    return [v.name for v in defn.list_vars()]


def _depends_on(var_name, target_name, func_ir, visited, max_depth):
    if var_name == target_name:
        return True
    if var_name in visited or max_depth <= 0:
        return False
    visited.add(var_name)
    try:
        defn = func_ir.get_definition(var_name)
    except Exception:
        return False
    if not isinstance(defn, ir.Expr):
        return False
    return any(
        _depends_on(name, target_name, func_ir, visited, max_depth - 1)
        for name in _operand_names(defn)
    )


def has_loop_carried_dependency(loop_nest, func_ir, max_depth=20):
    """Detects a phi node at the loop header whose loop-internal incoming value feeds back on itself.

    This is the classic running-accumulator pattern (e.g. `total += x`, `prod *= x`, a running
    max/min): the header's phi target flows into an expression, computed inside the loop body,
    that is then fed back into that same phi -- creating a dependency between consecutive
    iterations that blocks vectorization/unrolling.
    """
    header_block = func_ir.blocks[loop_nest.header]
    body_set = set(loop_nest.body_blocks)

    for stmt in header_block.body:
        if not (isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and stmt.value.op == "phi"):
            continue
        phi = stmt.value
        target_name = stmt.target.name
        for value, block_label in zip(phi.incoming_values, phi.incoming_blocks):
            if block_label not in body_set or not isinstance(value, ir.Var):
                continue
            if _depends_on(value.name, target_name, func_ir, set(), max_depth):
                return True
    return False
