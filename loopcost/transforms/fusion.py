"""Applies loop fusion transformations to combine adjacent compatible loops.

Mirrors loopcost.heuristic.decide's own stance on fusion: deciding to fuse requires knowing
about an *adjacent* loop nest sharing an array with compatible bounds, and a single-function
compilation pipeline has no visibility into "the next kernel" to fuse with -- there is no
Numba compiler-extension hook that hands a pass information about other, unrelated functions.
Consequently `rewrite_source` only attempts a rewrite when the caller explicitly supplies a
matching adjacent loop's source (mirroring decide.py's `adjacent_loop_nests` feature key); with
no such candidate, it returns None rather than guessing -- the same conservative stance as
decide.py's "fusion candidate detection not yet implemented" stub.
"""

import ast
import inspect
import textwrap

from loopcost.transforms.tiling import _find_target_for_loop


def rewrite_source(func, var_name, adjacent_loop_source=None):
    """Fuses `func`'s `var_name`-indexed loop with an explicitly supplied adjacent loop's body.

    `adjacent_loop_source` must be the source text of a function with a `for {var_name} in
    range(...):` loop with identical bounds -- the only case fusion is provably safe without
    deeper alias/dependence analysis this project doesn't implement. Returns None whenever
    that candidate isn't supplied or its bounds don't match, exactly as decide.py's own fuse
    rule does.
    """
    if adjacent_loop_source is None:
        return None

    try:
        this_tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        other_tree = ast.parse(textwrap.dedent(adjacent_loop_source))
    except SyntaxError:
        return None

    this_def = this_tree.body[0]
    other_def = other_tree.body[0]
    this_loop = _find_target_for_loop(this_def, var_name)
    other_loop = _find_target_for_loop(other_def, var_name)
    if this_loop is None or other_loop is None:
        return None
    if ast.dump(this_loop.iter) != ast.dump(other_loop.iter):
        return None  # bounds don't match -- not safely fusible

    fused_loop = ast.For(
        target=this_loop.target,
        iter=this_loop.iter,
        body=this_loop.body + other_loop.body,
        orelse=[],
    )
    this_def.body = [fused_loop if stmt is this_loop else stmt for stmt in this_def.body]
    this_def.decorator_list = []
    this_def.name = f"{this_def.name}_fused"

    ast.fix_missing_locations(this_tree)
    namespace = dict(func.__globals__)
    exec(compile(this_tree, filename=f"<loopcost-fused-{func.__name__}>", mode="exec"), namespace)
    return namespace[this_def.name]
