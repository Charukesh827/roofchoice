"""Orchestrates the end-to-end flow from IR feature extraction through transformation decision and application."""

from pathlib import Path

import numba
from numba import njit
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
import numba.core.typed_passes as typed_passes

from loopcost.benchmarks.sweep import _best_effort_trip_count, _select_representative
from loopcost.heuristic.classify import classify_bound, has_loop_carried_dependency
from loopcost.heuristic.decide import decide
from loopcost.heuristic.ridge_point import get_ridge_point
from loopcost.ir_features.access_pattern import classify_accesses
from loopcost.ir_features.cache_model import estimate_bytes_moved, operational_intensity, working_set_bytes
from loopcost.ir_features.flops import total_ops
from loopcost.ir_features.loop_info import find_loop_nests
from loopcost.ml.predict import predict_profitability
from loopcost.transforms import fusion, tiling, unroll, vectorize

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TRANSFORM_TYPES = ("tile", "vectorize", "unroll", "fuse")
APPLY_PRIORITY = ("tile", "fuse", "vectorize", "unroll")  # structural transforms first

# Side channel tests (and callers) can inspect: one entry per compile through loopcost.jit.
DECISION_LOG = []


def clear_decision_log():
    DECISION_LOG.clear()


def _model_path(transform_type):
    return MODELS_DIR / f"{transform_type}_model.pkl"


def _extract_features(func_ir, typemap):
    """Runs Step 3-5's feature extraction on func_ir's most work-heavy loop nest.

    Unlike loopcost.benchmarks.sweep's offline extraction, this runs *inside* an actual
    compile, so there are no concrete argument values available to substitute into symbolic
    bounds (the compiler only ever sees argument types, never values) -- bounds/trip counts
    stay symbolic (defaulting to 0/"memory-bound") unless the loop bound is a literal constant
    in the source. This is an inherent limitation of live compiler-pass integration, not a bug.
    """
    loop_nests = find_loop_nests(func_ir)
    representative = _select_representative(loop_nests, func_ir, typemap)
    _, _, ridge_point = get_ridge_point()

    if representative is None:
        features = {
            "oi": 0.0,
            "ridge_point": ridge_point,
            "bound": classify_bound(0.0, ridge_point),
            "trip_count": 0,
            "working_set_bytes": 0,
            "has_loop_carried_dep": False,
            "itemsize": 8,
        }
        return representative, features

    accesses = classify_accesses(representative, func_ir, typemap)
    # AI = (FLOPs + IntOps) / bytes accessed. Note: unlike sweep.py's offline extraction,
    # `representative` here is never concretized against argument values (none are available
    # inside a live compile), so total_ops()'s full-chain-trip-count weighting will only
    # resolve to a concrete number when the loop's bound is a literal constant in the source.
    total_flops = total_ops(representative, func_ir, typemap)
    bytes_estimate = estimate_bytes_moved(representative, accesses)
    oi = operational_intensity(total_flops, bytes_estimate.bytes_moved)
    itemsize = next((a.itemsize for a in accesses if a.itemsize), 8)

    features = {
        "oi": oi,
        "ridge_point": ridge_point,
        "bound": classify_bound(oi, ridge_point),
        "trip_count": _best_effort_trip_count(representative),
        "working_set_bytes": working_set_bytes(representative, accesses),
        "has_loop_carried_dep": has_loop_carried_dependency(representative, func_ir),
        "itemsize": itemsize,
    }
    return representative, features


def _decide_with_ml_override(loop_nest, features):
    """Runs the Step 7 heuristic, then overrides with the Step 8 ML model wherever one exists.

    Returns (heuristic_decision, ml_decision, final_decision, disagreements). ml_decision[t]
    is None when models/{t}_model.pkl doesn't exist (or fails to load/predict) -- the
    heuristic is the fallback in that case, exactly as the task requires.
    """
    heuristic_decision = decide(loop_nest, features)
    final_decision = {}
    ml_decision = {}
    disagreements = []

    for transform_type in TRANSFORM_TYPES:
        h_profitable, h_params, h_reason = heuristic_decision[transform_type]
        final_decision[transform_type] = (h_profitable, h_params, h_reason)
        ml_decision[transform_type] = None

        if not _model_path(transform_type).exists():
            continue
        try:
            ml_profitable, confidence, explanation = predict_profitability(features, transform_type)
        except Exception:
            continue  # missing/corrupt model: fall back to the heuristic, as already recorded

        ml_decision[transform_type] = (ml_profitable, confidence, explanation)
        if ml_profitable != h_profitable:
            disagreements.append(transform_type)
        final_decision[transform_type] = (
            ml_profitable,
            h_params,
            f"ML model overrides heuristic (confidence={confidence:.2f}): {explanation}",
        )

    return heuristic_decision, ml_decision, final_decision, disagreements


def _run_analysis_and_log(state, use_ml):
    """Shared body for the two analysis passes below: extracts features, decides, logs the result."""
    representative, features = _extract_features(state.func_ir, state.typemap)

    if use_ml:
        heuristic_decision, ml_decision, final_decision, disagreements = _decide_with_ml_override(
            representative, features
        )
    else:
        heuristic_decision = decide(representative, features)
        ml_decision = {t: None for t in TRANSFORM_TYPES}
        final_decision = heuristic_decision
        disagreements = []

    applied_transform = next((t for t in APPLY_PRIORITY if final_decision[t][0]), None)

    for transform_type in disagreements:
        h_profitable = heuristic_decision[transform_type][0]
        ml_profitable = ml_decision[transform_type][0]
        print(
            f"[loopcost] {state.func_id.func_name}: heuristic and ML disagree on "
            f"'{transform_type}' (heuristic={h_profitable}, ml={ml_profitable}); using ML"
        )

    DECISION_LOG.append(
        {
            "func_name": state.func_id.func_name,
            "tier": "ml" if use_ml else "heuristic",
            "features": features,
            "representative_loop_nest": representative,
            "heuristic_decision": heuristic_decision,
            "ml_decision": ml_decision,
            "final_decision": final_decision,
            "disagreements": disagreements,
            "applied_transform": applied_transform,
            "applied_params": final_decision[applied_transform][1] if applied_transform else {},
            "transform_succeeded": False,  # filled in by _LoopcostJIT once it knows
        }
    )


@register_pass(mutates_CFG=False, analysis_only=False)
class _LoopcostAnalysis(FunctionPass):
    """Tier-2: after type inference/SSA reconstruction, decides with the ML override and logs it."""

    _name = "loopcost_analysis_ml"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        _run_analysis_and_log(state, use_ml=True)
        return False


@register_pass(mutates_CFG=False, analysis_only=False)
class _LoopcostHeuristicOnlyAnalysis(FunctionPass):
    """Tier-1: after type inference/SSA reconstruction, decides with the heuristic alone and logs it."""

    _name = "loopcost_analysis_heuristic"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        _run_analysis_and_log(state, use_ml=False)
        return False


class _AnalysisPipeline(CompilerBase):
    """Tier-2 analysis-only pipeline: the normal nopython pipeline plus _LoopcostAnalysis."""

    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        pm.add_pass_after(_LoopcostAnalysis, typed_passes.NopythonTypeInference)
        pm.finalize()
        return [pm]


class _HeuristicOnlyAnalysisPipeline(CompilerBase):
    """Tier-1 analysis-only pipeline: the normal nopython pipeline plus _LoopcostHeuristicOnlyAnalysis."""

    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        pm.add_pass_after(_LoopcostHeuristicOnlyAnalysis, typed_passes.NopythonTypeInference)
        pm.finalize()
        return [pm]


class _LoopcostJIT:
    """Lazily analyzes, decides, and compiles `func` on first call; caches the dispatcher after."""

    def __init__(self, func, njit_kwargs, use_ml=True):
        self._func = func
        self._njit_kwargs = njit_kwargs
        self._use_ml = use_ml
        self._compiled = None
        self.py_func = func

    def __call__(self, *args, **kwargs):
        if self._compiled is None:
            self._compiled = self._compile(args)
        return self._compiled(*args, **kwargs)

    def _compile(self, sample_args):
        analysis_pipeline = _AnalysisPipeline if self._use_ml else _HeuristicOnlyAnalysisPipeline
        analysis_fn = njit(pipeline_class=analysis_pipeline)(self._func)
        signature = tuple(numba.typeof(a) for a in sample_args)
        analysis_fn.compile(signature)  # compiles (running _LoopcostAnalysis) without executing
        decision = DECISION_LOG[-1]
        transform_type = decision["applied_transform"]

        if transform_type == "vectorize":
            compiled = njit(pipeline_class=vectorize.make_pipeline_class(), **self._njit_kwargs)(self._func)
            decision["transform_succeeded"] = True
            return compiled

        if transform_type == "unroll":
            compiled = njit(pipeline_class=unroll.make_pipeline_class(), **self._njit_kwargs)(self._func)
            decision["transform_succeeded"] = True
            return compiled

        if transform_type == "tile":
            loop_nest = decision["representative_loop_nest"]
            var_name = loop_nest.induction_vars[-1] if loop_nest and loop_nest.induction_vars else None
            tile_size = decision["applied_params"].get("tile_size")
            rewritten = (
                tiling.rewrite_source(self._func, var_name, tile_size)
                if var_name and tile_size
                else None
            )
            if rewritten is not None:
                decision["transform_succeeded"] = True
                return njit(**self._njit_kwargs)(rewritten)

        if transform_type == "fuse":
            # No adjacent-loop-nest source is available to a single-function pipeline; mirrors
            # decide.py's own conservative fuse stance. rewrite_source() safely returns None.
            rewritten = fusion.rewrite_source(self._func, None, adjacent_loop_source=None)
            if rewritten is not None:
                decision["transform_succeeded"] = True
                return njit(**self._njit_kwargs)(rewritten)

        return njit(**self._njit_kwargs)(self._func)


def jit(*args, use_ml=True, **kwargs):
    """Drop-in decorator wrapping numba.njit(pipeline_class=...).

    On first call, runs the Step 3-5 feature extraction and the Step 7 heuristic decision.
    With `use_ml=True` (the default, "tier-2"), that decision is overridden by the Step 8 ML
    model wherever models/{transform_type}_model.pkl exists; with `use_ml=False` ("tier-1"),
    the heuristic alone is authoritative. Either way, the outcome is logged to
    loopcost.pipeline.DECISION_LOG, and the winning transform is applied: Numba Flags for
    vectorize/unroll, or a source-level rewrite (falling back to a plain compile if the
    rewrite can't be safely applied) for tile/fuse. The resulting dispatcher is cached and
    reused for later calls, like a normal numba.njit dispatcher.
    """

    def decorator(func):
        return _LoopcostJIT(func, kwargs, use_ml=use_ml)

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return decorator(args[0])
    return decorator
