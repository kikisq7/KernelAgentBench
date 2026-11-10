#!/usr/bin/env python3
"""Score the Python Gray-Scott candidate against the notebook baseline.

Mirrors julia/evaluate_candidate.jl: the baseline run from the same initial
fields is the correctness oracle, and speedup is baseline median / candidate median.
"""

from __future__ import annotations

import argparse
import importlib
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
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.gray_scott.notebook import GrayScottParams, initialize_fields  # noqa: E402

# Small grids are checked after each of these step counts; timed grids after `steps`.
CORRECTNESS_STEPS = (1, 5, 50)


def parse_shapes(value: str) -> list[tuple[int, int]]:
    return [
        (int(nx), int(ny)) for nx, ny in (item.split("x") for item in value.split(",") if item)
    ]


def relative_l2_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float64).eps)
    return float(np.linalg.norm(actual - expected)) / denominator


def compare(
    actual: tuple[np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray],
    initial: tuple[np.ndarray, np.ndarray],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    actual = tuple(np.asarray(field) for field in actual)
    pairs = list(zip(actual, expected, strict=True))
    if any(a.shape != e.shape or a.dtype != np.float64 for a, e in pairs):
        return {"correct": False, "max_abs_error": math.inf, "relative_l2_error": math.inf}
    correct = all(
        np.isfinite(a).all()
        and np.allclose(a, e, atol=atol, rtol=rtol)
        and np.array_equal(a[[0, -1], :], i[[0, -1], :])
        and np.array_equal(a[:, [0, -1]], i[:, [0, -1]])
        for a, e, i in zip(actual, expected, initial, strict=True)
    )
    return {
        "correct": bool(correct),
        "max_abs_error": max(float(np.max(np.abs(a - e))) for a, e in pairs),
        "relative_l2_error": max(relative_l2_error(a, e) for a, e in pairs),
    }


class Track:
    """Loads baseline and candidate for one backend and times them the same way."""

    def __init__(self, backend: str, threads: int) -> None:
        self.backend = backend
        package = f"python.gray_scott.{'cpu' if backend == 'cpu' else 'gpu'}"
        self.baseline = importlib.import_module(f"{package}.baseline")
        self.candidate = importlib.import_module(f"{package}.candidate")
        if backend == "cpu":
            import numba

            if threads:
                numba.set_num_threads(threads)
            self.threads = numba.get_num_threads()
            self.timer = "cpu_ns"
        else:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("Triton evaluation requires a CUDA GPU")
            self.torch = torch
            self.threads = 0
            self.timer = "cuda_event"

    def api(self, which: str) -> tuple[Callable, Callable, Callable]:
        module: ModuleType = self.baseline if which == "baseline" else self.candidate
        return (
            getattr(module, f"prepare_{which}"),
            getattr(module, f"run_{which}"),
            getattr(module, f"{which}_fields"),
        )

    def synchronize(self) -> None:
        if self.backend != "cpu":
            self.torch.cuda.synchronize()

    def fields(self, which: str, initial, params, steps: int):
        prepare, run, fields = self.api(which)
        state = run(prepare(initial[0].copy(), initial[1].copy()), params, steps)
        self.synchronize()
        return tuple(np.asarray(field) for field in fields(state))

    def timed(self, which: str, initial, params, steps: int):
        prepare, run, fields = self.api(which)
        state = prepare(initial[0].copy(), initial[1].copy())
        self.synchronize()
        if self.backend == "cpu":
            started = time.perf_counter_ns()
            state = run(state, params, steps)
            elapsed = (time.perf_counter_ns() - started) / 1e6
        else:
            start = self.torch.cuda.Event(enable_timing=True)
            stop = self.torch.cuda.Event(enable_timing=True)
            start.record()
            state = run(state, params, steps)
            stop.record()
            stop.synchronize()
            elapsed = float(start.elapsed_time(stop))
        self.synchronize()
        return elapsed, state, fields

    def environment(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "backend": self.backend,
            "timer": self.timer,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        }
        if self.backend == "cpu":
            import numba

            meta.update(numba_version=numba.__version__, threads=self.threads,
                        threading_layer=_threading_layer())
        else:
            import triton

            meta.update(
                torch_version=self.torch.__version__,
                triton_version=triton.__version__,
                gpu_name=self.torch.cuda.get_device_name(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            )
        return meta


def _threading_layer() -> str:
    import numba

    try:
        return numba.threading_layer()
    except ValueError:
        return "unset"


def benchmark_workload(track: Track, shape, params, args) -> dict[str, Any]:
    initial = initialize_fields(*shape)
    for _ in range(args.warmups):
        track.timed("baseline", initial, params, min(args.steps, 10))
        track.timed("candidate", initial, params, min(args.steps, 10))

    samples: dict[str, list[float]] = {"baseline": [], "candidate": []}
    last: dict[str, Any] = {}
    for repetition in range(args.repetitions):
        order = ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")
        for which in order:
            elapsed, state, fields = track.timed(which, initial, params, args.steps)
            samples[which].append(elapsed)
            last[which] = (state, fields)
    expected = tuple(np.asarray(field) for field in last["baseline"][1](last["baseline"][0]))
    actual = tuple(np.asarray(field) for field in last["candidate"][1](last["candidate"][0]))
    comparison = compare(actual, expected, initial, args.atol, args.rtol)

    baseline_median = statistics.median(samples["baseline"])
    candidate_median = statistics.median(samples["candidate"])
    return {
        "shape": list(shape),
        "baseline_samples_ms": samples["baseline"],
        "candidate_samples_ms": samples["candidate"],
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "baseline_std_ms": statistics.stdev(samples["baseline"]) if args.repetitions > 1 else 0.0,
        "candidate_std_ms": statistics.stdev(samples["candidate"]) if args.repetitions > 1 else 0.0,
        "speedup": baseline_median / candidate_median,
        **comparison,
        "timer": track.timer,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.dtype != "Float64":
        raise ValueError("the notebook baseline is Float64 only")
    params = GrayScottParams()
    track = Track(args.backend, args.threads)

    compile_started = time.perf_counter()
    small = initialize_fields(32, 32)
    track.fields("baseline", small, params, 1)
    track.fields("candidate", small, params, 1)
    compile_seconds = time.perf_counter() - compile_started

    cases: list[dict[str, Any]] = []
    for shape in parse_shapes(args.correctness_shapes):
        initial = initialize_fields(*shape)
        for steps in CORRECTNESS_STEPS:
            expected = track.fields("baseline", initial, params, steps)
            actual = track.fields("candidate", initial, params, steps)
            cases.append({
                "dtype": "Float64",
                "shape": list(shape),
                "steps": steps,
                "atol": args.atol,
                "rtol": args.rtol,
                **compare(actual, expected, initial, args.atol, args.rtol),
            })
    correct = all(case["correct"] for case in cases)
    workloads = (
        [benchmark_workload(track, shape, params, args) for shape in parse_shapes(args.shapes)]
        if correct
        else []
    )
    correct = correct and all(workload["correct"] for workload in workloads)
    speedup = (
        math.exp(statistics.mean(math.log(workload["speedup"]) for workload in workloads))
        if correct and workloads
        else None
    )
    return {
        "status": "passed" if correct else "failed",
        "correct": correct,
        "correctness_cases": cases,
        "workloads": workloads,
        "geometric_mean_speedup": speedup,
        "compile_seconds": compile_seconds,
        "environment": track.environment(),
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "triton"), required=True)
    parser.add_argument("--language", default="")
    parser.add_argument("--candidate-id", default="manual")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dtype", default="Float64")
    parser.add_argument("--shapes", default="1024x1024")
    parser.add_argument("--correctness-shapes", default="31x29,64x64,127x65")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--threads", type=int, default=0)
    # Accepted for a uniform evaluator command line; Float64 is the only dtype.
    parser.add_argument("--correctness-dtypes", default="Float64")
    parser.add_argument("--atol64", type=float, default=None)
    parser.add_argument("--rtol64", type=float, default=None)
    args = parser.parse_args()
    language = args.language or f"python-{args.backend}"
    output = args.output or ROOT / "results" / f"{args.candidate_id}.json"
    started = time.perf_counter()
    header = {
        "schema_version": "3.0",
        "candidate_id": args.candidate_id,
        "language": language,
        "backend": args.backend,
        "evaluator_version": f"gray-scott-notebook-python-{args.backend}-v3",
    }
    try:
        result = {**header, **evaluate(args)}
    except Exception:
        result = {
            **header,
            "status": "error",
            "correct": False,
            "correctness_cases": [],
            "workloads": [],
            "geometric_mean_speedup": None,
            "compile_seconds": None,
            "environment": {"backend": args.backend},
            "error": traceback.format_exc(),
        }
    result["end_to_end_seconds"] = time.perf_counter() - started
    result["created_at"] = datetime.now(UTC).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
