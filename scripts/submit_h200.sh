#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CONFIG_ORCD_YAML CANDIDATE_BUNDLE" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
BUNDLE="$2"
CANDIDATE_ID="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1] + "/manifest.json"))["candidate_id"])' \
  "$BUNDLE")"
SCRIPT="$ROOT/.kernel-agent-bench/slurm/${CANDIDATE_ID}.sbatch"
mkdir -p "$(dirname "$SCRIPT")"

python3 -m kernel_agent_bench.cli render-slurm \
  --orcd "$CONFIG" \
  --bundle "$BUNDLE" \
  --output "$SCRIPT"

# Canonical ORCD H200 request. Override with SALLOC_JOB_ID to reuse an open allocation.
if [[ -n "${SALLOC_JOB_ID:-}" ]]; then
  srun --jobid="$SALLOC_JOB_ID" --ntasks=1 --gres=gpu:h200:1 bash "$SCRIPT"
else
  salloc -N 1 -n 32 --cpus-per-task=1 --mem=128GB --time=06:00:00 -G h200:1 \
    -- srun --ntasks=1 --gres=gpu:h200:1 bash "$SCRIPT"
fi
