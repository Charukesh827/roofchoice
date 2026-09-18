# roofchoice (loopcost)

Static, roofline-guided cost/benefit analysis for Numba `@njit` kernels: analyze a kernel's
typed IR *before* it runs, and use that to pick a loop transformation (tiling / vectorization /
unrolling / fusion) or the best LLVM optimization pipeline (vectorization width, SLP, etc.) for
it — no profiling, no execution required to make the recommendation.

See the [hackathon writeup](https://claude.ai/artifact/BgsUJY69DbVwkUbP39JYBf) for the full
story: the roofline model, the LLVM-choice heuristic and its validation against real hardware
measurements, and the ML experiments we tried on top of it.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires Python >= 3.10. Key dependencies: `numba`, `islpy` (exact polyhedral byte-accounting),
`scikit-learn` + `pandas` (ML experiments), `matplotlib` (plots), `pytest`.

Run the test suite from the repo root:

```bash
pytest tests/ -q
```

Some tests are skipped unless a sibling `../experiments/sweep/` project (the 504-row PAPI sweep
this project is calibrated against) is present — see [Data provenance](#data-provenance) below.

## Folder-by-folder guide

### `loopcost/ir_features/` — static IR analyzers (Steps 3-5)

Pure analysis functions that operate on a Numba `CompilerBase` state's typed IR. No execution.

- `loop_info.py` — `find_loop_nests(func_ir)`: locates every loop, its induction variables,
  nesting depth, and bounds from the CFG.
- `access_pattern.py` — `classify_accesses(loop_nest, func_ir, typemap)`: classifies each array
  access in a loop as `contiguous`, `strided-constant`, or `irregular`.
- `flops.py` — `count_flops` / `count_intops` / `total_ops`: counts float and integer arithmetic
  ops per iteration and across a loop's full enclosing chain.
- `cache_model.py` — `estimate_bytes_moved`, `working_set_bytes`, `operational_intensity`: byte
  accounting (exact islpy cache-line counting when possible, else a conservative fallback) and
  arithmetic intensity (AI = (FLOPs+IntOps)/bytes).

Use these directly if you want raw static features for your own analysis; everything else in
the project is built on top of them.

### `loopcost/heuristic/` — decision rules built on the static features

- `ridge_point.py` — `get_ridge_point()`: calibrates (and caches, per-CPU-model, in
  `data/calibration.json`) this machine's peak GFLOP/s, peak memory bandwidth, and roofline
  ridge point.
- `classify.py` — `classify_bound(oi, ridge_point)` (compute-bound vs memory-bound),
  `has_loop_carried_dependency(...)`.
- `decide.py` — `decide(loop_nest, features)`: picks a loop transformation (tile / vectorize /
  unroll / fuse / none) from a kernel's static features.
- `llvm_choice.py` — `classify_kernel_style(fn, args)` + a lookup table:
  `choose_llvm_variant(fn, args, boundscheck, fastmath)` recommends one of 6 LLVM optimization
  variants (O0 through forced vector widths), calibrated against the real sweep. This is the
  heuristic used by `demo/`.

### `loopcost/benchmarks/` — kernels and measurement harnesses

- `financial_kernels.py` — 11 plain (non-jitted) Python kernels used as this project's own
  benchmark suite (Black-Scholes, GARCH, Monte Carlo option pricing, portfolio risk, etc.).
- `harness.py` — `time_kernel(fn, args, repeats, reference)` for wall-clock timing, plus
  `PapiHarness` for core-pinned hardware performance-counter measurement.
- `papi_ctypes.py` — a ctypes wrapper around `libpapi.so` (hardware counters: FLOPs, DRAM
  bytes via uncore counters).
- `sweep.py` — `run_sweep()` and the internal `_capture_ir` / `extract_features` machinery
  (compiles a kernel with a custom pipeline that captures typed IR right after type inference,
  purely for static analysis — nothing here executes the kernel for real unless you call
  `run_sweep()` or the harness functions).

### `loopcost/transforms/` — AST-based loop transformations

`tiling.py`, `unroll.py`, `vectorize.py`, `fusion.py` — safe source-to-source rewrites (Python
`ast` module) that implement the transformation `heuristic/decide.py` recommends. `tiling.py`
and `fusion.py` expose `rewrite_source(func, ...)`; `unroll.py` and `vectorize.py` additionally
expose `make_pipeline_class()` (a Flags-based custom pipeline) used by `pipeline.py`.

### `loopcost/pipeline.py` — the custom Numba compiler pipeline

`jit(*args, use_ml=True, **kwargs)`: a drop-in replacement for `numba.njit` that runs the static
analysis + heuristic (or an optional ML override) as a compiler pass and logs its decision
(`DECISION_LOG`) before compiling. Use this if you want the analysis to run automatically at
`@njit`-compile time instead of calling the analyzers by hand.

### `loopcost/ml/` — trained models

Two independent ML efforts, both following the same pattern (`dataset.py` builds features/
labels, `train.py` compares models and saves the winner to `models/`, `predict.py` loads a
saved model and predicts):

- `dataset.py` / `train.py` / `predict.py` — Step 8's original loop-transformation
  profitability models (`models/{tile,vectorize,unroll,fuse}_model.pkl`).
- `llvm_dataset.py` / `llvm_train.py` / `llvm_predict.py` — the LLVM-variant-choice ML
  experiment described in the writeup: builds a dataset from static features + the real sweep's
  measured labels, compares 14 classifiers via leave-one-kernel-out cross-validation, and saves
  the best *trained* model to `models/llvm_choice_model.pkl` (spoiler: the hand-derived
  heuristic in `heuristic/llvm_choice.py` still wins — see `models/llvm_choice_ml_report.txt`).

Run either training script directly to reproduce:

```bash
python3 -m loopcost.ml.train          # loop-transformation profitability models
python3 -m loopcost.ml.llvm_train     # LLVM-choice ML comparison (needs ../experiments/sweep)
```

### `loopcost/evaluate/` — reporting and plots

- `compare.py` — `run_comparison()`, `run_bound_directed_comparison()`, `run_papi_roofline()`:
  runs the benchmark kernels naive-vs-optimized and produces the roofline/comparison plots in
  `data/`.
- `llvm_choice_report.py` — `run_llvm_choice_experiment()`: validates the LLVM-choice heuristic
  against the real sweep and produces `data/llvm_choice_improvements.{csv,png}` and
  `data/llvm_choice_summary.png`.

```bash
python3 -m loopcost.evaluate.compare
python3 -m loopcost.evaluate.llvm_choice_report   # needs ../experiments/sweep
```

### `demo/` — hackathon live demo

Self-contained: 17 hand-written kernels (none from the calibration sweep) spanning all three
static styles (vectorizable / transcendental / non_fp) and both compute- and memory-bound
regimes.

- `demo/kernels/*.py` — one kernel per file, each exporting `KERNEL` (the plain Python function)
  and `ARGS` (sample arguments).
- `demo/run_demo.py` — discovers every kernel module, prints its static IR numbers (AI, roofline
  performance ceiling, bound classification, access-pattern breakdown) and the heuristic's
  suggested LLVM variant for all 4 njit flag combinations, then tabulates everything.
- `demo/run_demo.sh` — activates the venv and runs it end-to-end (~2.5s).

```bash
./demo/run_demo.sh
```

### `data/` — calibration cache and generated reports

`calibration.json` (per-CPU-model roofline calibration, regenerated automatically if missing or
for a new machine) plus every CSV/PNG produced by the `evaluate/` scripts. Safe to delete and
regenerate; nothing here is hand-authored.

### `models/` — saved model artifacts

`*.pkl` (joblib-serialized models), `*_rules.txt` (human-readable decision-tree rules), and
`llvm_choice_ml_report.txt` (the full ML-vs-heuristic comparison table). Regenerated by the
`loopcost/ml/` training scripts.

### `tests/`

One test file per module under `loopcost/`, run with `pytest tests/ -q` from the repo root.
Tests that depend on the sibling sweep project or real hardware calibration are marked
`skipif` and skip gracefully when that data isn't present.

## Data provenance

The 504-row PAPI-measured sweep this project's heuristic and ML experiments are calibrated
against (21 kernels x 4 njit flag combinations x 6 LLVM variants) lives in a sibling project,
`../experiments/sweep/` (`results/sweep_combined.csv` + `kernel_registry.py`), not inside this
repo. Anything that reads it (`heuristic/llvm_choice.py`'s `validate_against_sweep()`,
`evaluate/llvm_choice_report.py`, `ml/llvm_dataset.py`, and their tests) skips or raises a clear
`FileNotFoundError` if that sibling project isn't present alongside this one.
