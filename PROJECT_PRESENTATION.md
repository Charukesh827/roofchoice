# RoofPick: Static Roofline-Guided Optimization Selection for Numba Kernels

## Abstract

RoofPick is a static optimization advisor for Numba `@njit` kernels. Instead of relying on timing runs or profiling-guided trial-and-error, it analyzes a kernel’s typed IR before execution and estimates its memory traffic, arithmetic intensity, loop structure, and bandwidth/compute limits using a roofline model. Based on these static features, it recommends whether a loop should be tiled, vectorized, unrolled, fused, or whether a different LLVM optimization variant is more appropriate.

The system is designed for practical use: it integrates directly into the Numba compilation flow through a custom compiler pipeline, can choose transformation heuristics automatically, and supports an optional ML override for decision-making. This makes RoofPick a bridge between compiler analysis and performance tuning, helping users choose efficient loop optimizations without needing expensive runtime profiling.

## Motivation

Optimizing numerical kernels is difficult because performance depends on multiple interacting factors:

- memory-bound versus compute-bound behavior,
- loop structure and dependencies,
- array access patterns,
- cache residency,
- vectorization and unrolling effectiveness,
- and the specific LLVM optimization settings used during JIT compilation.

In many real workloads, the best optimization is not obvious from the source code alone. Developers often rely on benchmarking and trial-and-error, which is slow and expensive, especially when kernels must be tested across multiple hardware platforms and compiler settings.

This project addresses that pain point by introducing a static, analysis-driven method. The key idea is simple: inspect the kernel before it runs, estimate the important performance characteristics, and apply the most promising optimization based on the roofline model rather than on empirical guesswork.

This is especially useful for scientific and numerical code, where loops dominate runtime and small optimization decisions can have large performance implications.

## Approach

RoofPick follows a structured workflow inspired by static compiler analysis and roofline performance modeling:

1. IR Inspection
   - The project captures the typed IR of a Numba kernel before execution.
   - It identifies loop nests, induction variables, trip counts, and access patterns.

2. Static Feature Extraction
   - It computes arithmetic intensity using FLOP and integer-operation counts.
   - It estimates bytes moved and working-set size using cache-aware accounting.
   - It classifies whether each loop is memory-bound or compute-bound relative to the machine roofline.

3. Decision Heuristics
   - The heuristic layer combines roofline classification with loop dependency checks.
   - It decides whether transformations such as tiling, vectorization, unrolling, or fusion are likely to be profitable.
   - It also classifies kernel style into categories such as vectorizable, transcendental, or non-FP.

4. LLVM Variant Recommendation
   - For kernels dominated by vectorizable floating-point computation, the tool recommends specific LLVM optimization configurations.
   - For transcendental or non-FP code, it chooses settings with minimal needless optimization overhead, matching the empirical behavior observed in the project’s sweep.

5. Optional ML Override
   - In addition to the static heuristic, the repo supports trained ML models as an override layer.
   - This allows the system to blend expert rules with learned predictions when model artifacts are available.

The central benefit is that RoofPick decides before execution, with no need for a profiling run to make the first recommendation.

## Architecture

The project is organized into several modules that mirror the optimization pipeline.

### Core components

- `loopcost/ir_features/`
  - Contains analyzers for loops, access patterns, FLOP counting, and cache-aware byte accounting.
  - These modules extract the raw static performance features from the Numba typed IR.

- `loopcost/heuristic/`
  - Implements roofline-based classification and transformation decisions.
  - Includes logic for ridge-point calibration, loop-bound classification, dependency checks, and LLVM variant selection.

- `loopcost/transforms/`
  - Provides source-to-source transformations such as tiling, vectorization, unrolling, and fusion.
  - These transformations are applied when the decisions suggest they are worthwhile.

- `loopcost/pipeline.py`
  - Integrates the analysis and decision process into a custom Numba compiler pipeline.
  - Acts as the runtime glue between static analysis and compilation.

- `loopcost/ml/`
  - Contains training, prediction, and dataset generation code for model-based optimization.
  - Stores model artifacts in the `models/` directory.

- `loopcost/evaluate/`
  - Produces benchmark comparisons, roofline plots, and validation reports.

- `demo/`
  - Demonstrates the approach on representative kernels across compute-bound and memory-bound regimes.

- `tests/`
  - Verifies correctness, logging, transformation application, and pipeline behavior.

### Execution flow

The project follows this overall flow:

1. A Numba kernel is compiled.
2. The compiler captures the typed IR and identifies its important loop nest.
3. Feature extraction computes arithmetic intensity and data movement estimates.
4. The heuristic determines whether the loop is compute-bound or memory-bound.
5. A transformation or LLVM setting is chosen.
6. The transformation is applied or the LLVM variant is configured.
7. The result is compiled and validated against baseline correctness.

This architecture allows the system to operate as both a research prototype and a practical optimizer that can be dropped into a JIT workflow.

## Validation & Results

The repository includes both validation tests and hardware-aware benchmarking.

### Test validation

The `tests/` directory contains checks for:

- correctness of the custom JIT pipeline,
- decision logging,
- repeated-call caching,
- ML override behavior,
- loop tiling rewrites,
- and real vectorization pipeline integration.

These tests confirm that the project does not merely produce recommendations; it also maintains numerical correctness in the compiled output.

### Benchmark-driven evaluation

The project is validated against a measured sweep based on PAPI hardware counters. The README states that the system is calibrated against a sibling `../experiments/sweep/` project containing a 504-row measured dataset covering:

- 21 kernels,
- 4 Numba JIT flag combinations,
- 6 LLVM variants.

This gives the heuristic a real-world basis rather than relying only on synthetic assumptions.

### Empirical observations

The code and documentation indicate that the heuristic performs well across kernel categories:

- For vectorizable floating-point code, the best LLVM setting is often a loop-vectorized path, with bounds-checking effects significantly affecting performance.
- For transcendental-heavy kernels, the recommendation is typically a no-vectorization path because LLVM’s vectorizer does not help much.
- For non-FP kernels, the choice often has little practical impact, and the system defaults to a stable, high-performing setting.

The repository also includes generated comparison reports and plots in the `data/` folder, which summarize the measured improvements and the gap between the heuristic and the default compiler settings.

### Practical significance

The main result is not simply that RoofPick picks a transformation, but that it does so early, statically, and with hardware-aware reasoning. This reduces the need for exhaustive benchmarking while still aligning optimization decisions with real performance characteristics.

In summary, RoofPick demonstrates that static IR analysis combined with roofline modeling can enable fast, automated optimization choices for Numba kernels without profiling the code at runtime.

## Conclusion

RoofPick presents a compelling approach to compiler-guided kernel optimization. It combines static analysis, roofline reasoning, transform selection, LLVM variant tuning, and optional ML support into one coherent system. The result is a practical performance tool for Python/Numba workloads that want to get better results without the cost of repeated empirical tuning.

This makes the project relevant both as a research contribution and as a useful optimization framework for computational kernels in scientific and numerical computing.

---

Prepared for project presentation based on the repository structure and implementation in `roofpick`.

Key project references:
- `README.md`
- `loopcost/pipeline.py`
- `loopcost/heuristic/decide.py`
- `loopcost/heuristic/llvm_choice.py`
- `tests/test_pipeline.py`
- `pyproject.toml`




































































































































































































































































































































































































































































































































