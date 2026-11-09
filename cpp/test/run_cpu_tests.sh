#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${TMPDIR:-/tmp}/kernel-agent-bench-cpp-cpu-test"
mkdir -p "$BUILD_DIR"

"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror \
  "$ROOT/test/reference_test.cpp" \
  -o "$BUILD_DIR/reference_test"
"$BUILD_DIR/reference_test"

"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror -I"$ROOT/gray_scott/cpu" \
  "$ROOT/test/cpu_correctness_test.cpp" \
  -o "$BUILD_DIR/cpu_correctness_test"
"$BUILD_DIR/cpu_correctness_test"
