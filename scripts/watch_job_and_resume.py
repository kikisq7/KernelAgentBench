#!/usr/bin/env python3
"""Wait for an already-submitted Slurm job, then download its JSON and resume."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from kernel_agent_bench.config import load_orcd, project_root
from kernel_agent_bench.graph import TrialGraph
from kernel_agent_bench.orcd import (
    _run,
    _ssh_stdout,
    remote_transfer,
    rsync_download_command,
    ssh_master_running,
)
from kernel_agent_bench.tracing import configure_tracing


_TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "SPECIAL_EXIT",
}


def _log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def _job_state(orcd, job_id: str) -> str:
    queued = _ssh_stdout(orcd, f"squeue -j {job_id} -h -o %T")
    if queued:
        return queued.split()[0]
    try:
        accounted = _ssh_stdout(orcd, f"sacct -j {job_id} -n -X -o State")
    except RuntimeError:
        return "COMPLETED"
    return (accounted.split()[0] if accounted else "COMPLETED").rstrip("+")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--orcd", required=True, type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log.open("a", encoding="utf-8", buffering=1)
        sys.stdout = log_handle
        sys.stderr = log_handle

    configure_tracing()
    project = project_root()
    orcd = load_orcd(args.orcd)
    bundle = args.bundle if args.bundle.is_absolute() else project / args.bundle
    transfer = remote_transfer(orcd, bundle, project)
    transfer.local_result.parent.mkdir(parents=True, exist_ok=True)

    _log(f"watching {args.job} for {transfer.candidate_id}")
    while True:
        if not ssh_master_running(orcd):
            _log("SSH mux is down; waiting for it to come back")
            time.sleep(args.poll_seconds)
            continue
        try:
            state = _job_state(orcd, args.job)
        except RuntimeError as exc:
            _log(f"queue check failed: {exc}")
            time.sleep(args.poll_seconds)
            continue
        _log(f"{args.job} {state}")
        if state in _TERMINAL:
            break
        time.sleep(args.poll_seconds)

    if not ssh_master_running(orcd):
        _log("SSH mux is down after the job ended; cannot download yet")
        return 3

    last_error: Exception | None = None
    for _ in range(5):
        try:
            _run(rsync_download_command(orcd, transfer))
        except RuntimeError as exc:
            last_error = exc
            _log(f"download failed: {exc}")
            time.sleep(15)
            continue
        if transfer.local_result.is_file():
            break
        time.sleep(15)
    if not transfer.local_result.is_file():
        _log(f"no result JSON after {args.job} ended in {state}")
        if last_error is not None:
            raise last_error
        return 2

    _log(f"downloaded {transfer.local_result}")
    runner = TrialGraph(project)
    try:
        result = runner.resume(args.trial, transfer.local_result)
    finally:
        runner.close()
    _log(json.dumps({"status": result.get("status"), "candidate_id": result.get("candidate_id")}))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
