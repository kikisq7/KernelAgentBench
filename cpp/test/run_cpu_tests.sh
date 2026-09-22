#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${TMPDIR:-/tmp}/kernel-agent-bench-cpp-test"
mkdir -p "$BUILD_DIR"

"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror \
  "$ROOT/test/reference_test.cpp" \
  -o "$BUILD_DIR/reference_test"
"$BUILD_DIR/reference_test"

# The CUDA candidate cannot be compiled without nvcc, but it must remain a
# non-empty source artifact for the H200 correctness gate.
test -s "$ROOT/gray_scott/candidate.cuh"
