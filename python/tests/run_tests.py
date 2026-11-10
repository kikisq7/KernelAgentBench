#!/usr/bin/env python3
"""Public tests for the Python tracks: ``run_tests.py cpu`` or ``run_tests.py triton``.

The candidate must reproduce the baseline it is timed against, as the evaluator
checks. Without a CUDA GPU the Triton device checks are skipped (the H200
evaluation is the gate) but the candidate must still import cleanly.
"""

from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.evaluate_candidate import CORRECTNESS_STEPS, compare  # noqa: E402
from python.gray_scott.notebook import GrayScottParams, initialize_fields  # noqa: E402

SHAPES = ((31, 29), (64, 64), (127, 65))


def check_initial_condition() -> None:
    # Shared with julia/test and cpp/test: the three harnesses start bit-identical.
    U, V = initialize_fields(31, 29)
    assert U.shape == (29, 31)
    assert [round(value * 1000) for value in U.ravel()[:8]] == [70, 10, 16, 94, 47, 3, 53, 17]
    assert U.ravel()[-1] == 0.036
    assert V.sum() == 30.25
    assert U[13, 14] == 0.5 and V[13, 14] == 0.25


def check_candidate(package: str) -> None:
    baseline = importlib.import_module(f"{package}.baseline")
    candidate = importlib.import_module(f"{package}.candidate")
    params = GrayScottParams()
    failures = []
    for shape in SHAPES:
        initial = initialize_fields(*shape)
        for steps in CORRECTNESS_STEPS:
            base_state = baseline.prepare_baseline(*map(np.copy, initial))
            cand_state = candidate.prepare_candidate(*map(np.copy, initial))
            expected = baseline.baseline_fields(baseline.run_baseline(base_state, params, steps))
            actual = candidate.candidate_fields(candidate.run_candidate(cand_state, params, steps))
            result = compare(actual, expected, initial, 1e-10, 1e-8)
            if not result["correct"]:
                failures.append(f"{shape[0]}x{shape[1]} after {steps} steps: {result}")
    if failures:
        raise AssertionError("candidate differs from baseline:\n" + "\n".join(failures))


def main() -> int:
    backend = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    check_initial_condition()
    if backend == "cpu":
        check_candidate("python.gray_scott.cpu")
        print("Python CPU candidate matches the baseline")
        return 0
    py_compile.compile(str(ROOT / "python" / "gray_scott" / "gpu" / "candidate.py"), doraise=True)
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        print("[ Info ] torch/triton not installed here; Triton device tests skipped")
        return 0
    if not torch.cuda.is_available():
        print("[ Info ] No CUDA GPU here; Triton device tests skipped")
        return 0
    check_candidate("python.gray_scott.gpu")
    print("Triton candidate matches the baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
