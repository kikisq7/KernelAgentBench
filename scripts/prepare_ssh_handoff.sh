#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CONFIG_ORCD_YAML CANDIDATE_BUNDLE" >&2
  exit 2
fi

python3 -m kernel_agent_bench.cli ssh-handoff \
  --orcd "$1" \
  --bundle "$2"
