"""Extracts structural loop metadata such as nesting depth, bounds, and trip counts."""

import operator
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

from numba.core import ir
from numba.core.analysis import compute_cfg_from_blocks

_BINOP_SYMBOLS = {
    operator.add: "+",
    operator.sub: "-",
    operator.mul: "*",
    operator.floordiv: "//",
}


@dataclass
class LoopNest:
    """One level of a loop nest: its induction-variable chain, own trip count, and CFG position."""

    header: int
    depth: int
    induction_vars: List[str]
    trip_count: Union[int, str]
    own_blocks: List[int]
    body_blocks: List[int]
    bounds: List[Tuple[str, Any, Any, Any]]  # (var, start, stop, step) for this level and every enclosing one


def _resolve_scalar(name, func_ir):
    """Best-effort resolution of a scalar SSA value to a constant int or a symbolic expression string."""
    try:
        defn = func_ir.get_definition(name)
    except Exception:
        return name
    if isinstance(defn, ir.Const):
        return defn.value
    if isinstance(defn, (ir.Global, ir.FreeVar)):
        return defn.value if isinstance(defn.value, int) else str(defn.value)
    if isinstance(defn, ir.Arg):
        return defn.name
    if isinstance(defn, ir.Expr):
        op = defn.op
        if op == "getattr" and defn.attr == "shape":
            return f"{defn.value.name}.shape"
        if op == "static_getitem":
            base = _resolve_scalar(defn.value.name, func_ir)
            return f"{base}[{defn.index}]"
        if op in ("exhaust_iter", "cast"):
            return _resolve_scalar(defn.value.name, func_ir)
        if op == "unary" and defn.fn is operator.neg:
            inner = _resolve_scalar(defn.value.name, func_ir)
            return -inner if isinstance(inner, int) else f"-({inner})"
        if op == "binop":
            lhs = _resolve_scalar(defn.lhs.name, func_ir)
            rhs = _resolve_scalar(defn.rhs.name, func_ir)
            if isinstance(lhs, int) and isinstance(rhs, int):
                try:
                    return defn.fn(lhs, rhs)
                except Exception:
                    pass
            sym = _BINOP_SYMBOLS.get(defn.fn)
            if sym:
                return f"({lhs} {sym} {rhs})"
    return name


def _int_trip_count(start, stop, step):
    if step > 0:
        return max(0, -(-(stop - start) // step))
    return max(0, -(-(start - stop) // (-step)))


def _range_trip_count(start, stop, step):
    if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
        return _int_trip_count(start, stop, step)
    if start == 0 and step == 1:
        return stop
    return f"ceil(({stop} - {start}) / {step})"


def _find_for_loop_header_info(func_ir, header):
    """Locates the iternext/pair_first/branch pattern that identifies a for-loop header block."""
    block = func_ir.blocks[header]
    iternext_target = None
    iterator_name = None
    pair_first_target = None
    body_label = None
    for stmt in block.body:
        if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
            if stmt.value.op == "iternext":
                iternext_target = stmt.target.name
                iterator_name = stmt.value.value.name
            elif stmt.value.op == "pair_first" and iternext_target is not None:
                if stmt.value.value.name == iternext_target:
                    pair_first_target = stmt.target.name
        if isinstance(stmt, ir.Branch):
            body_label = stmt.truebr
    if pair_first_target is None or iterator_name is None or body_label is None:
        return None
    return iterator_name, pair_first_target, body_label


def _find_induction_var(func_ir, header_label, pair_first_target, body_label):
    """Finds the plain-named variable copied from the loop's phi target, possibly via the header block."""
    candidates = {pair_first_target}
    for label in (header_label, body_label):
        block = func_ir.blocks[label]
        changed = True
        while changed:
            changed = False
            for stmt in block.body:
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var):
                    if stmt.value.name in candidates and stmt.target.name not in candidates:
                        candidates.add(stmt.target.name)
                        changed = True
    named = [c for c in candidates if not c.startswith("$")]
    return named[0] if named else None


def _resolve_range_args(func_ir, iterator_name):
    """Traces a for-loop's iterator back to its range() call and resolves start/stop/step."""
    defn = func_ir.get_definition(iterator_name)
    if not (isinstance(defn, ir.Expr) and defn.op == "getiter"):
        return None
    call_defn = func_ir.get_definition(defn.value.name)
    if not (isinstance(call_defn, ir.Expr) and call_defn.op == "call"):
        return None
    func_defn = func_ir.get_definition(call_defn.func.name)
    if not (isinstance(func_defn, ir.Global) and func_defn.value is range):
        return None
    args = [_resolve_scalar(a.name, func_ir) for a in call_defn.args]
    if len(args) == 1:
        return 0, args[0], 1
    if len(args) == 2:
        return args[0], args[1], 1
    if len(args) == 3:
        return args[0], args[1], args[2]
    return None


def find_loop_nests(func_ir):
    """Returns one LoopNest record per CFG loop level, with induction-variable chains and trip counts."""
    cfg = compute_cfg_from_blocks(func_ir.blocks)
    loops = cfg.loops()

    depths = {header: len(cfg.in_loops(header)) for header in loops}
    order = sorted(loops, key=lambda h: depths[h])

    var_chains = {}
    bound_chains = {}
    own_blocks_map = {}
    trip_counts = {}

    for header in order:
        loop = loops[header]

        parent = None
        for other_header, other_loop in loops.items():
            if other_header != header and header in other_loop.body:
                if parent is None or depths[other_header] > depths[parent]:
                    parent = other_header

        info = _find_for_loop_header_info(func_ir, header)
        own_var = None
        own_bound = None
        trip_counts[header] = "unknown"
        if info is not None:
            iterator_name, pair_first_target, body_label = info
            own_var = _find_induction_var(func_ir, header, pair_first_target, body_label)
            range_args = _resolve_range_args(func_ir, iterator_name)
            if range_args is not None:
                trip_counts[header] = _range_trip_count(*range_args)
                if own_var is not None:
                    own_bound = (own_var, *range_args)

        var_chains[header] = var_chains.get(parent, []) + ([own_var] if own_var else [])
        bound_chains[header] = bound_chains.get(parent, []) + ([own_bound] if own_bound else [])

        child_bodies = set()
        for other_header, other_loop in loops.items():
            if other_header != header and other_header in loop.body:
                child_bodies |= set(other_loop.body)
        own_blocks_map[header] = sorted(set(loop.body) - child_bodies)

    records = []
    for header in order:
        loop = loops[header]
        records.append(
            LoopNest(
                header=header,
                depth=depths[header],
                induction_vars=var_chains[header],
                trip_count=trip_counts[header],
                own_blocks=own_blocks_map[header],
                body_blocks=sorted(loop.body),
                bounds=bound_chains[header],
            )
        )
    return records
