#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${TMPDIR:-/tmp}/kernel-agent-bench-cpp-opencl-test"
mkdir -p "$BUILD_DIR"

g++ -std=c++17 -O2 "$ROOT/test/reference_test.cpp" -o "$BUILD_DIR/reference_test"
"$BUILD_DIR/reference_test"

if [[ "${KAB_SKIP_GPU_TESTS:-0}" == "1" ]]; then
  echo "Skipping OpenCL device tests (KAB_SKIP_GPU_TESTS=1)"
  exit 0
fi

OPENCL_FLAGS=(-I"$ROOT")
if [[ "$(uname -s)" == "Darwin" ]]; then
  OPENCL_LIBS=(-framework OpenCL)
else
  OPENCL_LIBS=(-lOpenCL)
fi

g++ -std=c++17 -O2 "${OPENCL_FLAGS[@]}" "$ROOT/test/opencl_correctness_test.cpp" -o "$BUILD_DIR/opencl_correctness_test" "${OPENCL_LIBS[@]}"
"$BUILD_DIR/opencl_correctness_test"
