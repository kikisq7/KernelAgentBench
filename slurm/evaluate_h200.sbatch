#!/usr/bin/env bash
#
# Rendered by `kernel-agent-bench render-slurm`. Do not submit this template
# with sbatch. Allocate the H200 with:
#
#   salloc -N 1 -n 32 --cpus-per-task=1 --mem=128GB --time=06:00:00 -G h200:1
#
# then `srun` the rendered evaluator, or use scripts/submit_h200.sh.

set -euo pipefail
echo "Render this template first; see scripts/submit_h200.sh" >&2
exit 2
