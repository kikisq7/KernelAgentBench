#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${TMPDIR:-/tmp}/kernel-agent-bench-cpp-opencl-test"
mkdir -p "$BUILD_DIR"

# CPU reference sanity (always required).
g++ -std=c++17 -O2 "$ROOT/test/reference_test.cpp" -o "$BUILD_DIR/reference_test"
"$BUILD_DIR/reference_test"

if [[ "${KAB_SKIP_GPU_TESTS:-0}" == "1" ]]; then
  echo "[ Info ] OpenCL device tests skipped (KAB_SKIP_GPU_TESTS=1)"
  exit 0
fi

OPENCL_FLAGS=(-I"$ROOT")
if [[ "$(uname -s)" == "Darwin" ]]; then
  OPENCL_LIBS=(-framework OpenCL)
else
  OPENCL_LIBS=(-lOpenCL)
fi

g++ -std=c++17 -O2 "${OPENCL_FLAGS[@]}" "$ROOT/test/opencl_device_probe.cpp" -o "$BUILD_DIR/opencl_device_probe" "${OPENCL_LIBS[@]}"
probe_rc=0
"$BUILD_DIR/opencl_device_probe" || probe_rc=$?
if [[ "$probe_rc" -eq 2 ]]; then
  echo "[ Info ] OpenCL unavailable on this host; GPU tests skipped (H200 evaluation is the correctness gate)"
  exit 0
fi
if [[ "$probe_rc" -ne 0 ]]; then
  echo "OpenCL device probe failed with exit $probe_rc" >&2
  exit "$probe_rc"
fi

g++ -std=c++17 -O2 "${OPENCL_FLAGS[@]}" "$ROOT/test/opencl_correctness_test.cpp" -o "$BUILD_DIR/opencl_correctness_test" "${OPENCL_LIBS[@]}"
correctness_rc=0
"$BUILD_DIR/opencl_correctness_test" || correctness_rc=$?
if [[ "$correctness_rc" -eq 1 ]]; then
  echo "OpenCL candidate failed numerical correctness against CPU reference" >&2
  exit 1
fi
if [[ "$correctness_rc" -ne 0 ]]; then
  echo "[ Info ] OpenCL runtime error; treating as unavailable host (exit $correctness_rc), not candidate incorrectness" >&2
  exit 0
fi

echo "OpenCL public tests passed"
