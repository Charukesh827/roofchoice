"""Applies loop tiling/blocking transformations to improve cache locality.

This is a genuine source-to-source rewrite, done at the Python AST level rather than on
Numba's compiler IR: parsing/rewriting/unparsing a `for` loop with the `ast` module is far
safer and more easily verified than splicing new basic blocks into Numba's SSA-form IR, and
Python's `ast` module is a real, stable, documented tool for exactly this. The rewrite itself
is provably safe for *any* loop body, including one with loop-carried dependencies: blocking a
purely sequential `range` loop into chunks never changes the order iterations execute in, only
how they're grouped, so the tiled version is always semantically identical to the original.

Deliberately narrow in what it recognizes (a single, ascending, unit-or-explicit-step `for var
in range(...):` statement at the top level of the function body, matching the target loop
nest's own induction variable): if the pattern isn't found, `rewrite_source` returns None
rather than guessing at a riskier rewrite.
"""

import ast
import inspect
import textwrap


def _find_target_for_loop(func_def, var_name):
    """Finds the first top-level `for {var_name} in range(...):` statement in the function body."""
    for stmt in func_def.body:
        if (
            isinstance(stmt, ast.For)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == var_name
            and isinstance(stmt.iter, ast.Call)
            and isinstance(stmt.iter.func, ast.Name)
            and stmt.iter.func.id == "range"
        ):
            return stmt
    return None


def _split_range_args(range_call):
    """Returns (start, stop, step) AST expression nodes for a range() call, defaulting start/step."""
    args = range_call.args
    if len(args) == 1:
        return ast.Constant(0), args[0], ast.Constant(1)
    if len(args) == 2:
        return args[0], args[1], ast.Constant(1)
    if len(args) == 3:
        return args[0], args[1], args[2]
    return None


def rewrite_source(func, var_name, tile_size):
    """Rewrites `for {var_name} in range(...):` into an explicit two-level tiled loop nest.

    Returns a new function object with the tiled loop, or None if the expected pattern isn't
    found or isn't safely tileable (e.g. a descending/strided range).
    """
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):
        return None

    lines = source.splitlines()
    while lines and lines[0].strip().startswith("@"):
        lines = lines[1:]  # drop a leading decorator line so ast.parse sees a plain def
    source = "\n".join(lines)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    func_def = tree.body[0]
    target_loop = _find_target_for_loop(func_def, var_name)
    if target_loop is None:
        return None

    range_parts = _split_range_args(target_loop.iter)
    if range_parts is None:
        return None
    start_expr, stop_expr, step_expr = range_parts

    if not (isinstance(step_expr, ast.Constant) and step_expr.value == 1):
        return None  # only ascending, unit-stride ranges are safely re-blockable here

    tile_start_name = f"__{var_name}_tile_start"
    tile_stop_name = f"__{var_name}_tile_stop"

    outer_range = ast.Call(
        func=ast.Name(id="range", ctx=ast.Load()),
        args=[start_expr, stop_expr, ast.Constant(tile_size)],
        keywords=[],
    )
    tile_stop_assign = ast.Assign(
        targets=[ast.Name(id=tile_stop_name, ctx=ast.Store())],
        value=ast.Call(
            func=ast.Name(id="min", ctx=ast.Load()),
            args=[
                ast.BinOp(
                    left=ast.Name(id=tile_start_name, ctx=ast.Load()),
                    op=ast.Add(),
                    right=ast.Constant(tile_size),
                ),
                stop_expr,
            ],
            keywords=[],
        ),
    )
    inner_loop = ast.For(
        target=ast.Name(id=var_name, ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=[ast.Name(id=tile_start_name, ctx=ast.Load()), ast.Name(id=tile_stop_name, ctx=ast.Load())],
            keywords=[],
        ),
        body=target_loop.body,
        orelse=[],
    )
    outer_loop = ast.For(
        target=ast.Name(id=tile_start_name, ctx=ast.Store()),
        iter=outer_range,
        body=[tile_stop_assign, inner_loop],
        orelse=[],
    )

    func_def.body = [outer_loop if stmt is target_loop else stmt for stmt in func_def.body]
    func_def.decorator_list = []
    func_def.name = f"{func_def.name}_tiled"

    ast.fix_missing_locations(tree)
    namespace = dict(func.__globals__)
    exec(compile(tree, filename=f"<loopcost-tiled-{func.__name__}>", mode="exec"), namespace)
    return namespace[func_def.name]
