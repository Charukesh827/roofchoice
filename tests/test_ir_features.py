"""Tests for loop_info/access_pattern/flops against hand-computed values on toy njit functions."""
import math

import numpy as np
from numba import njit
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
import numba.core.typed_passes as typed_passes

from loopcost.ir_features.access_pattern import classify_accesses
from loopcost.ir_features.flops import count_flops
from loopcost.ir_features.loop_info import find_loop_nests

_captured = {}


@register_pass(mutates_CFG=False, analysis_only=False)
class _CaptureTypedState(FunctionPass):
    """Stashes a snapshot of func_ir/typemap right after nopython type inference, for testing."""

    _name = "loopcost_test_capture_typed_state"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        _captured["func_ir"] = state.func_ir.copy()
        _captured["typemap"] = dict(state.typemap)
        return False


class _CaptureAfterTypeInference(CompilerBase):
    """Custom pipeline that inserts _CaptureTypedState right after NopythonTypeInference."""

    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        pm.add_pass_after(_CaptureTypedState, typed_passes.NopythonTypeInference)
        pm.finalize()
        return [pm]


def _compile_and_capture(func, args):
    _captured.clear()
    njit(pipeline_class=_CaptureAfterTypeInference)(func)(*args)
    return _captured["func_ir"], _captured["typemap"]


def contiguous_1d(a, c):
    for i in range(a.shape[0]):
        c[i] = a[i] * 2.0


def strided_2d(a, b):
    n, m = a.shape
    for i in range(n):
        for j in range(0, m, 2):
            b[j, i] = a[i, j]


def exp_loop(a, c):
    for i in range(10):
        c[i] = a[i] + math.exp(a[i])


# --- contiguous_1d: single loop, unit-stride accesses, one float mul per iteration ---


def test_contiguous_1d_loop_info():
    func_ir, _ = _compile_and_capture(contiguous_1d, (np.arange(5.0), np.zeros(5)))
    nests = find_loop_nests(func_ir)
    assert len(nests) == 1
    (nest,) = nests
    assert nest.depth == 1
    assert nest.induction_vars == ["i"]
    assert nest.trip_count == "a.shape[0]"


def test_contiguous_1d_access_pattern():
    func_ir, typemap = _compile_and_capture(contiguous_1d, (np.arange(5.0), np.zeros(5)))
    (nest,) = find_loop_nests(func_ir)
    accesses = classify_accesses(nest, func_ir, typemap)
    by_array = {(a.array, a.kind): a for a in accesses}
    assert by_array[("a", "read")].classification == "contiguous"
    assert by_array[("a", "read")].coefficients == {"dim0": 1}
    assert by_array[("c", "write")].classification == "contiguous"
    assert by_array[("c", "write")].coefficients == {"dim0": 1}


def test_contiguous_1d_flops():
    func_ir, typemap = _compile_and_capture(contiguous_1d, (np.arange(5.0), np.zeros(5)))
    (nest,) = find_loop_nests(func_ir)
    flops = count_flops(nest, func_ir, typemap)
    assert flops.per_iteration == {"add": 0, "mul": 1, "div": 0, "transcendental": 0}
    assert flops.total["mul"] == "1 * a.shape[0]"
    assert flops.total["add"] == 0


# --- strided_2d: outer loop (i) carries no arithmetic; inner loop (i, j) has the two array refs ---


def test_strided_2d_loop_info():
    func_ir, _ = _compile_and_capture(strided_2d, (np.zeros((4, 6)), np.zeros((6, 4))))
    nests = sorted(find_loop_nests(func_ir), key=lambda n: n.depth)
    assert len(nests) == 2
    outer, inner = nests
    assert outer.depth == 1
    assert outer.induction_vars == ["i"]
    assert outer.trip_count == "a.shape[0]"
    assert inner.depth == 2
    assert inner.induction_vars == ["i", "j"]
    assert inner.trip_count == "ceil((a.shape[1] - 0) / 2)"


def test_strided_2d_access_pattern():
    func_ir, typemap = _compile_and_capture(strided_2d, (np.zeros((4, 6)), np.zeros((6, 4))))
    nests = sorted(find_loop_nests(func_ir), key=lambda n: n.depth)
    outer, inner = nests

    # the outer loop's own body (loop management only) touches no arrays
    assert classify_accesses(outer, func_ir, typemap) == []

    accesses = classify_accesses(inner, func_ir, typemap)
    by_array = {(a.array, a.kind): a for a in accesses}
    # a[i, j]: innermost loop var j has coeff 1 in the last (unit-stride, C-layout) dimension
    assert by_array[("a", "read")].classification == "contiguous"
    assert by_array[("a", "read")].coefficients == {"dim0": 0, "dim1": 1}
    # b[j, i]: innermost loop var j has coeff 1 in the *first* dimension, stride = row length
    assert by_array[("b", "write")].classification == "strided-constant"
    assert by_array[("b", "write")].coefficients == {"dim0": 1, "dim1": 0}


def test_strided_2d_flops_are_zero():
    func_ir, typemap = _compile_and_capture(strided_2d, (np.zeros((4, 6)), np.zeros((6, 4))))
    for nest in find_loop_nests(func_ir):
        flops = count_flops(nest, func_ir, typemap)
        assert flops.per_iteration == {"add": 0, "mul": 0, "div": 0, "transcendental": 0}


# --- exp_loop: constant trip count of 10, one add and one math.exp call per iteration ---


def test_exp_loop_flops():
    func_ir, typemap = _compile_and_capture(exp_loop, (np.arange(20.0), np.zeros(10)))
    (nest,) = find_loop_nests(func_ir)
    assert nest.trip_count == 10

    flops = count_flops(nest, func_ir, typemap)
    assert flops.per_iteration == {"add": 1, "mul": 0, "div": 0, "transcendental": 1}
    assert flops.total == {"add": 10, "mul": 0, "div": 0, "transcendental": 10}


def test_exp_loop_access_pattern():
    func_ir, typemap = _compile_and_capture(exp_loop, (np.arange(20.0), np.zeros(10)))
    (nest,) = find_loop_nests(func_ir)
    accesses = classify_accesses(nest, func_ir, typemap)
    assert len(accesses) == 3  # two reads of a[i], one write to c[i]
    assert all(acc.classification == "contiguous" for acc in accesses)
    assert all(acc.coefficients == {"dim0": 1} for acc in accesses)
