#!/usr/bin/env bash
# gray-scott-notebook-v3 trials on one pinned baseline, replicate-major so every
# model x track cell gets replicate 1 before any cell gets replicate 2.
#   scripts/run_v3_trials.sh cpu   # julia-cpu, cpp-cpu, python-cpu (scored locally)
#   scripts/run_v3_trials.sh gpu   # julia-cuda, julia-ka, cpp, python-triton (H200)
# Each trial logs to logs/<trial>.log. A failed trial does not stop the queue.
# Rerunning skips trials already in results/trials.jsonl and continues trials
# whose last step was interrupted (e.g. an H200 eval) instead of restarting them.
# The gpu queue holds one H200 allocation for all evals and releases it at the end.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

QUEUE="${1:?usage: run_v3_trials.sh cpu|gpu}"
BASELINE="${KAB_BASELINE:?set KAB_BASELINE to the pinned v3 commit}"
REPLICATES="${KAB_REPLICATES:-1 2 3}"
MODELS="${KAB_MODELS:-opus-5.5 gpt-6-astra composer-2.5 kimi-k3}"
KAB=.venv/bin/kernel-agent-bench
STUDY=config/study.yaml
ORCD=config/orcd.yaml
STUDY_ID=gray-scott-notebook-v3

set -a
# shellcheck disable=SC1091
source .env
set +a
mkdir -p logs

case "$QUEUE" in
  cpu)
    LANGS="julia-cpu cpp-cpu python-cpu"
    ORCD_ARGS=()
    ;;
  gpu)
    LANGS="julia-cuda julia-ka cpp python-triton"
    ORCD_ARGS=(--orcd "$ORCD")
    HOLD=$("$KAB" hold-h200 --orcd "$ORCD") || exit 1
    echo "$(date '+%F %T') holding H200 allocation $HOLD"
    ;;
  *)
    echo "queue must be cpu or gpu" >&2
    exit 2
    ;;
esac

for REP in $REPLICATES; do
  for TRACK in $LANGS; do
    for MODEL in $MODELS; do
      TRIAL="$STUDY_ID-$TRACK-$MODEL-r$REP"
      LOG="logs/$TRIAL.log"
      if grep -q "\"trial_id\": *\"$TRIAL\"" results/trials.jsonl 2>/dev/null; then
        echo "$(date '+%F %T') skip $TRIAL (already recorded)"
        builtin continue
      fi
      if [[ "$QUEUE" == gpu ]] && ! ssh -O check -o "ControlPath=$HOME/.ssh/kab-%r@%h:%p" \
        sqcheng@orcd-login.mit.edu 2>/dev/null; then
        echo "$(date '+%F %T') ORCD ControlMaster is gone; rerun ssh-check, then this queue" >&2
        exit 1
      fi
      if "$KAB" interrupted --trial "$TRIAL" >/dev/null 2>&1; then
        echo "$(date '+%F %T') continue $TRIAL"
        "$KAB" continue --trial "$TRIAL" >>"$LOG" 2>&1
        echo "$(date '+%F %T') exit=$? $TRIAL"
        builtin continue
      fi
      START_ARGS=(--study "$STUDY")
      if ((${#ORCD_ARGS[@]} > 0)); then
        START_ARGS+=("${ORCD_ARGS[@]}")
      fi
      echo "$(date '+%F %T') start $TRIAL baseline=$BASELINE"
      "$KAB" start \
        "${START_ARGS[@]}" \
        --model "$MODEL" \
        --language "$TRACK" \
        --replicate "$REP" \
        --baseline "$BASELINE" >"$LOG" 2>&1
      rc=$?
      echo "$(date '+%F %T') exit=$rc $TRIAL"
    done
  done
done
if [[ "$QUEUE" == gpu ]]; then
  "$KAB" release-h200 --orcd "$ORCD"
fi
echo "$(date '+%F %T') $QUEUE queue done"
