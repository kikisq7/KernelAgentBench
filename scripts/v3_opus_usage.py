#!/usr/bin/env python3
"""Print cumulative Opus spend (USD) and billed tokens for gray-scott-notebook-v3."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_agent_bench.models import disjoint_token_usage

STUDY = "gray-scott-notebook-v3"
trial_usd: dict[str, float] = {}
tokens = 0
path = ROOT / "results" / "agent_runs.jsonl"
if path.is_file():
    for line in path.read_text(encoding="utf-8").splitlines():
        if STUDY not in line:
            continue
        row = json.loads(line)
        if row.get("event") != "agent_run" or row.get("model_alias") != "opus-5.5":
            continue
        tid = row["trial_id"]
        cost = row.get("billed_cost") or {}
        usd = cost.get("total_cost_usd")
        if usd is not None:
            trial_usd[tid] = max(trial_usd.get(tid, 0.0), float(usd))
        usage = disjoint_token_usage(row["token_usage"], row["resolved_model_id"])
        tokens += int(usage["total_tokens"])

print(f"{sum(trial_usd.values()):.4f} {tokens}")
