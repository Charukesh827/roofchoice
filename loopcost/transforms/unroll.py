"""Applies loop unrolling transformations at a specified factor.

Two independent mechanisms, for two different purposes:

1. `make_pipeline_class()` -- Numba has no public flag for a loop unroll factor; unrolling is
   entirely LLVM's own optimizer's decision, not something Numba's Flags expose. The closest
   genuine, verified-live-read lever is `forceinline`: NativeLowering reads
   `state.flags.forceinline` directly (not from the pre-baked targetctx, unlike fastmath --
   see vectorize.py) when building the function descriptor, so a pass can still influence it
   after type inference. Forcing inlining at call sites is a classic *enabler* of unrolling --
   an indirect connection, documented as such rather than overclaiming a fictional flag.

2. `rewrite_source()` -- an actual, explicit unroll-by-N: a genuine Python-AST source-to-source
   rewrite (same technique as tiling.py), duplicating the loop body `factor` times with each
   copy's uses of the loop variable offset by 0..factor-1. This is only safe for independent
   iterations -- callers must confirm there's no loop-carried dependency before calling it,
   exactly as loopcost.heuristic.decide already requires for vectorize/unroll to be profitable.
"""

import ast
import copy
import inspect
import textwrap

from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
import numba.core.typed_passes as typed_passes

from loopcost.transforms.tiling import _find_target_for_loop, _split_range_args

FLAG_NAME = "forceinline"


@register_pass(mutates_CFG=False, analysis_only=False)
class _SetForceInline(FunctionPass):
    """Sets state.flags.forceinline, read live by NativeLowering, as an unrolling enabler."""

    _name = "loopcost_unroll_set_forceinline"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        state.flags.forceinline = True
        return True


def make_pipeline_class():
    """Returns a CompilerBase subclass that forces inlining as an unrolling enabler."""

    class _UnrollPipeline(CompilerBase):
        def define_pipelines(self):
            pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
            pm.add_pass_after(_SetForceInline, typed_passes.NopythonTypeInference)
            pm.finalize()
            return [pm]

    return _UnrollPipeline


class _OffsetLoopVariable(ast.NodeTransformer):
    """Replaces every load of `var_name` with `(var_name + offset)`, leaving offset=0 untouched."""

    def __init__(self, var_name, offset):
        self.var_name = var_name
        self.offset = offset

    def visit_Name(self, node):
        if node.id == self.var_name and isinstance(node.ctx, ast.Load) and self.offset != 0:
            return ast.BinOp(left=node, op=ast.Add(), right=ast.Constant(self.offset))
        return node


def rewrite_source(func, var_name, factor):
    """Unrolls `for {var_name} in range(...):` by `factor`: duplicates the body straight-line,
    offsetting each copy's uses of the loop variable, then handles the remainder with a plain
    trailing loop. Returns None if the expected pattern isn't found or isn't safely unrollable
    (e.g. a descending/strided range, or factor <= 1) -- the same conservative stance as
    tiling.rewrite_source. Safe only when the loop body has no cross-iteration dependency.
    """
    if factor <= 1:
        return None
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):
        return None

    lines = source.splitlines()
    while lines and lines[0].strip().startswith("@"):
        lines = lines[1:]
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
        return None

    unrolled_end_name = f"__{var_name}_unrolled_end"
    unrolled_end_assign = ast.Assign(
        targets=[ast.Name(id=unrolled_end_name, ctx=ast.Store())],
        value=ast.BinOp(
            left=copy.deepcopy(start_expr),
            op=ast.Add(),
            right=ast.BinOp(
                left=ast.BinOp(
                    left=ast.BinOp(left=copy.deepcopy(stop_expr), op=ast.Sub(), right=copy.deepcopy(start_expr)),
                    op=ast.FloorDiv(),
                    right=ast.Constant(factor),
                ),
                op=ast.Mult(),
                right=ast.Constant(factor),
            ),
        ),
    )
    var_init = ast.Assign(targets=[ast.Name(id=var_name, ctx=ast.Store())], value=copy.deepcopy(start_expr))

    unrolled_body = []
    for offset in range(factor):
        for stmt in target_loop.body:
            unrolled_body.append(_OffsetLoopVariable(var_name, offset).visit(copy.deepcopy(stmt)))
    unrolled_body.append(
        ast.AugAssign(target=ast.Name(id=var_name, ctx=ast.Store()), op=ast.Add(), value=ast.Constant(factor))
    )
    while_loop = ast.While(
        test=ast.Compare(
            left=ast.Name(id=var_name, ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[ast.Name(id=unrolled_end_name, ctx=ast.Load())],
        ),
        body=unrolled_body,
        orelse=[],
    )
    remainder_loop = ast.For(
        target=ast.Name(id=var_name, ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=[ast.Name(id=unrolled_end_name, ctx=ast.Load()), copy.deepcopy(stop_expr)],
            keywords=[],
        ),
        body=target_loop.body,
        orelse=[],
    )

    replacement = [unrolled_end_assign, var_init, while_loop, remainder_loop]
    new_body = []
    for stmt in func_def.body:
        if stmt is target_loop:
            new_body.extend(replacement)
        else:
            new_body.append(stmt)
    func_def.body = new_body
    func_def.decorator_list = []
    func_def.name = f"{func_def.name}_unrolled{factor}"

    ast.fix_missing_locations(tree)
    namespace = dict(func.__globals__)
    exec(compile(tree, filename=f"<loopcost-unrolled-{func.__name__}>", mode="exec"), namespace)
    return namespace[func_def.name]
