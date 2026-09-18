"""Applies vectorization transformations to eligible loop bodies.

Numba's `fastmath` flag looks like the obvious lever, but it can't actually be toggled from a
custom pass: `CompilerBase.__init__` bakes `flags.fastmath` into `state.targetctx` (via
`_make_subtarget`) before any pass -- including a pass inserted after type inference -- ever
runs, so mutating `state.flags.fastmath` later has no effect (verified empirically: the
generated LLVM IR was byte-for-byte identical with and without the mutation). It would also
change floating-point rounding, which is exactly the "blanket fastmath regressed compute-bound
kernels" failure mode this project is trying to avoid.

`noalias` is different: NativeLowering reads `state.flags.noalias` live, at lowering time, not
from the pre-baked targetctx. Setting it adds LLVM `noalias` attributes to the array-data
pointer arguments (verified: 12 occurrences vs. 2 baseline on a 3-array kernel), which is the
classic prerequisite for LLVM's loop vectorizer to auto-vectorize a loop touching multiple
arrays -- without changing a single floating-point operation, so results stay bit-identical.
"""

from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
import numba.core.typed_passes as typed_passes

FLAG_NAME = "noalias"


@register_pass(mutates_CFG=False, analysis_only=False)
class _SetNoAlias(FunctionPass):
    """Sets state.flags.noalias, read live by NativeLowering to mark array pointers non-aliasing."""

    _name = "loopcost_vectorize_set_noalias"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        state.flags.noalias = True
        return True


def make_pipeline_class():
    """Returns a CompilerBase subclass that enables the noalias vectorization hint."""

    class _VectorizePipeline(CompilerBase):
        def define_pipelines(self):
            pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
            pm.add_pass_after(_SetNoAlias, typed_passes.NopythonTypeInference)
            pm.finalize()
            return [pm]

    return _VectorizePipeline


def choose_vector_length(dtype_bytes=8):
    """Heuristically chooses a SIMD vector length (elements per register) for the host CPU.

    Reuses loopcost.heuristic.ridge_point's own host-CPU-feature detection (llvmlite's
    get_host_cpu_features): 512-bit registers under AVX-512, 256-bit under AVX2, 128-bit
    under SSE2, else scalar -- the same detection used to size ridge_point's own FMA
    micro-benchmark, so the chosen length matches what this machine can actually execute.
    """
    from loopcost.heuristic.ridge_point import _get_simd_width

    return _get_simd_width(dtype_bytes)
