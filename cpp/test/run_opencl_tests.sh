#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${TMPDIR:-/tmp}/kernel-agent-bench-cpp-opencl-test"
mkdir -p "$BUILD_DIR"

# Harness initial condition (always required).
"${CXX:-c++}" -std=c++17 -O2 "$ROOT/test/notebook_test.cpp" -o "$BUILD_DIR/notebook_test"
"$BUILD_DIR/notebook_test"

if [[ "${KAB_SKIP_GPU_TESTS:-0}" == "1" ]]; then
  echo "[ Info ] OpenCL device tests skipped (KAB_SKIP_GPU_TESTS=1)"
  exit 0
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  OPENCL_FLAGS=()
  OPENCL_LIBS=(-framework OpenCL)
else
  OPENCL_FLAGS=(-I"${CUDA_HOME:-/usr/local/cuda}/include")
  OPENCL_LIBS=(-L"${CUDA_HOME:-/usr/local/cuda}/lib64" -lOpenCL)
fi

# Compile errors in the candidate fail the gate even without a GPU.
"${CXX:-c++}" -std=c++17 -O2 ${OPENCL_FLAGS[@]+"${OPENCL_FLAGS[@]}"} "$ROOT/test/opencl_device_probe.cpp" \
  -o "$BUILD_DIR/opencl_device_probe" "${OPENCL_LIBS[@]}"
"${CXX:-c++}" -std=c++17 -O2 ${OPENCL_FLAGS[@]+"${OPENCL_FLAGS[@]}"} "$ROOT/test/opencl_correctness_test.cpp" \
  -o "$BUILD_DIR/opencl_correctness_test" "${OPENCL_LIBS[@]}"

probe_rc=0
"$BUILD_DIR/opencl_device_probe" || probe_rc=$?
if [[ "$probe_rc" -eq 2 ]]; then
  echo "[ Info ] No double-precision OpenCL GPU on this host; device tests skipped (H200 evaluation is the correctness gate)"
  exit 0
fi
if [[ "$probe_rc" -ne 0 ]]; then
  echo "OpenCL device probe failed with exit $probe_rc" >&2
  exit "$probe_rc"
fi

"$BUILD_DIR/opencl_correctness_test"
echo "OpenCL public tests passed"
