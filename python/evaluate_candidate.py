#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.gray_scott.reference import (
    GrayScottParams,
    cpu_simulate,
    initialize_fields,
    relative_l2_error,
)


def parse_shapes(value: str) -> list[tuple[int, int]]:
    return [tuple(map(int, shape.split("x"))) for shape in value.split(",")]  # type: ignore[misc]


def field_errors(
    actual_u: np.ndarray,
    actual_v: np.ndarray,
    expected_u: np.ndarray,
    expected_v: np.ndarray,
) -> tuple[float, float]:
    max_abs = max(
        float(np.max(np.abs(actual_u - expected_u))),
        float(np.max(np.abs(actual_v - expected_v))),
    )
    relative = max(
        relative_l2_error(actual_u, expected_u),
        relative_l2_error(actual_v, expected_v),
    )
    return max_abs, relative


def timed_run(cuda: Any, runner: Callable[..., Any], state: Any, params: Any, steps: int) -> float:
    start, stop = cuda.event(timing=True), cuda.event(timing=True)
    start.record()
    runner(state, params, steps)
    stop.record()
    stop.synchronize()
    return float(cuda.event_elapsed_time(start, stop))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from numba import cuda

    from python.gray_scott.baseline import (
        baseline_fields,
        prepare_baseline,
        run_baseline,
    )
    from python.gray_scott.candidate import (
        candidate_fields,
        prepare_candidate,
        run_candidate,
    )

    if not cuda.is_available():
        raise RuntimeError("numba-cuda reports no functional CUDA device")
    dtype = np.float32 if args.dtype == "Float32" else np.float64
    params = GrayScottParams()

    compile_started = time.perf_counter()
    compile_u, compile_v = initialize_fields((32, 32))
    run_candidate(prepare_candidate(compile_u.astype(dtype), compile_v.astype(dtype)), params, 1)
    cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_started

    correctness_cases: list[dict[str, Any]] = []
    for shape in parse_shapes(args.correctness_shapes):
        initial_u, initial_v = initialize_fields(shape)
        initial_u, initial_v = initial_u.astype(dtype), initial_v.astype(dtype)
        for steps in (1, 5):
            expected_u, expected_v = cpu_simulate(initial_u, initial_v, params, steps)
            state = prepare_candidate(initial_u, initial_v)
            run_candidate(state, params, steps)
            cuda.synchronize()
            u_device, v_device = candidate_fields(state)
            actual_u, actual_v = u_device.copy_to_host(), v_device.copy_to_host()
            max_abs, relative = field_errors(actual_u, actual_v, expected_u, expected_v)
            correct = bool(
                np.allclose(actual_u, expected_u, atol=args.atol, rtol=args.rtol)
                and np.allclose(actual_v, expected_v, atol=args.atol, rtol=args.rtol)
                and np.isfinite(actual_u).all()
                and np.isfinite(actual_v).all()
            )
            correctness_cases.append(
                {
                    "shape": list(shape),
                    "steps": steps,
                    "correct": correct,
                    "max_abs_error": max_abs,
                    "relative_l2_error": relative,
                }
            )

    workloads: list[dict[str, Any]] = []
    if all(case["correct"] for case in correctness_cases):
        for shape in parse_shapes(args.shapes):
            initial_u, initial_v = initialize_fields(shape)
            initial_u, initial_v = initial_u.astype(dtype), initial_v.astype(dtype)
            for _ in range(args.warmups):
                timed_run(
                    cuda,
                    run_baseline,
                    prepare_baseline(initial_u, initial_v),
                    params,
                    min(args.steps, 10),
                )
                timed_run(
                    cuda,
                    run_candidate,
                    prepare_candidate(initial_u, initial_v),
                    params,
                    min(args.steps, 10),
                )

            baseline_samples: list[float] = []
            candidate_samples: list[float] = []
            last_candidate = None
            for repetition in range(args.repetitions):
                baseline = prepare_baseline(initial_u, initial_v)
                candidate = prepare_candidate(initial_u, initial_v)
                operations = (
                    (
                        (run_baseline, baseline, baseline_samples),
                        (run_candidate, candidate, candidate_samples),
                    )
                    if repetition % 2 == 0
                    else (
                        (run_candidate, candidate, candidate_samples),
                        (run_baseline, baseline, baseline_samples),
                    )
                )
                for runner, state, samples in operations:
                    samples.append(timed_run(cuda, runner, state, params, args.steps))
                last_candidate = candidate

            trusted = prepare_baseline(initial_u, initial_v)
            run_baseline(trusted, params, args.steps)
            cuda.synchronize()
            actual_u_device, actual_v_device = candidate_fields(last_candidate)
            expected_u_device, expected_v_device = baseline_fields(trusted)
            actual_u, actual_v = actual_u_device.copy_to_host(), actual_v_device.copy_to_host()
            expected_u, expected_v = (
                expected_u_device.copy_to_host(),
                expected_v_device.copy_to_host(),
            )
            max_abs, relative = field_errors(actual_u, actual_v, expected_u, expected_v)
            correct = bool(
                np.allclose(actual_u, expected_u, atol=args.atol, rtol=args.rtol)
                and np.allclose(actual_v, expected_v, atol=args.atol, rtol=args.rtol)
            )
            baseline_median = statistics.median(baseline_samples)
            candidate_median = statistics.median(candidate_samples)
            workloads.append(
                {
                    "shape": list(shape),
                    "baseline_samples_ms": baseline_samples,
                    "candidate_samples_ms": candidate_samples,
                    "baseline_median_ms": baseline_median,
                    "candidate_median_ms": candidate_median,
                    "baseline_std_ms": statistics.stdev(baseline_samples)
                    if len(baseline_samples) > 1
                    else 0.0,
                    "candidate_std_ms": statistics.stdev(candidate_samples)
                    if len(candidate_samples) > 1
                    else 0.0,
                    "speedup": baseline_median / candidate_median,
                    "correct": correct,
                    "max_abs_error": max_abs,
                    "relative_l2_error": relative,
                }
            )

    correct = all(case["correct"] for case in correctness_cases) and all(
        workload["correct"] for workload in workloads
    )
    speedup = (
        math.exp(statistics.mean(math.log(workload["speedup"]) for workload in workloads))
        if correct and workloads
        else None
    )
    device = cuda.get_current_device()
    return {
        "schema_version": "2.0",
        "candidate_id": args.candidate_id,
        "language": "python",
        "evaluator_version": "gray-scott-numba-cuda-v1",
        "status": "passed" if correct else "failed",
        "correct": correct,
        "correctness_cases": correctness_cases,
        "workloads": workloads,
        "geometric_mean_speedup": speedup,
        "compile_seconds": compile_seconds,
        "environment": {
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "numba_cuda": getattr(cuda, "__version__", "unknown"),
            "gpu_name": device.name.decode()
            if isinstance(device.name, bytes)
            else str(device.name),
            "compute_capability": list(device.compute_capability),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "error": None,
        "created_at": datetime.now(UTC).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", default="manual")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("Float32", "Float64"), default="Float32")
    parser.add_argument("--shapes", default="1024x1024")
    parser.add_argument("--correctness-shapes", default="31x29,64x64,127x65")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-4)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        result = evaluate(args)
    except Exception:
        result = {
            "schema_version": "2.0",
            "candidate_id": args.candidate_id,
            "language": "python",
            "evaluator_version": "gray-scott-numba-cuda-v1",
            "status": "error",
            "correct": False,
            "correctness_cases": [],
            "workloads": [],
            "geometric_mean_speedup": None,
            "compile_seconds": None,
            "environment": {},
            "error": traceback.format_exc(),
            "created_at": datetime.now(UTC).isoformat(),
        }
    result["end_to_end_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
