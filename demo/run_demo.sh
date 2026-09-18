#!/usr/bin/env bash
# Hackathon demo: suggests the best LLVM optimization pipeline for 15 kernels using
# loopcost's static LLVM-choice heuristic (no ML, no new hardware measurement).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python3 demo/run_demo.py
