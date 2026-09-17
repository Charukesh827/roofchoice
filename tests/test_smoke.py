"""Smoke test asserting every loopcost module can be imported without raising."""
import importlib

MODULES = [
    "loopcost",
    "loopcost.ir_features",
    "loopcost.ir_features.loop_info",
    "loopcost.ir_features.access_pattern",
    "loopcost.ir_features.flops",
    "loopcost.ir_features.cache_model",
    "loopcost.heuristic",
    "loopcost.heuristic.ridge_point",
    "loopcost.heuristic.classify",
    "loopcost.heuristic.decide",
    "loopcost.ml",
    "loopcost.ml.dataset",
    "loopcost.ml.train",
    "loopcost.ml.predict",
    "loopcost.transforms",
    "loopcost.transforms.vectorize",
    "loopcost.transforms.unroll",
    "loopcost.transforms.tiling",
    "loopcost.transforms.fusion",
    "loopcost.pipeline",
    "loopcost.benchmarks",
    "loopcost.benchmarks.financial_kernels",
    "loopcost.benchmarks.harness",
    "loopcost.benchmarks.sweep",
    "loopcost.evaluate",
    "loopcost.evaluate.compare",
]


def test_all_modules_import():
    for module_name in MODULES:
        importlib.import_module(module_name)
