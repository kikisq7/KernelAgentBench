# Kernel Agent Bench

Kernel Agent Bench measures how effectively coding agents optimize GPU kernels
in Julia, Python, and CUDA C++. The first study uses a corrected
two-dimensional, five-point Gray-Scott solver on one NVIDIA H200. Every one of
the three models runs against every language, giving nine conditions per
replicate. KernelAbstractions and AMD are later, separate tracks.

The benchmark has two planes:

- The local control plane runs a checkpointed LangGraph, invokes Cursor coding
  agents, records patches, and sends traces to LangSmith.
- The ORCD evaluation plane has no model credentials. It reconstructs a
  candidate from a baseline commit and patch, checks correctness, and measures
  baseline and candidate in the same H200 allocation.

## What is measured

Correctness is a hard gate. The primary metric is the geometric mean of
steady-state timestep speedup over all required grid sizes. CUDA event timing
excludes package loading, JIT compilation, allocations, and transfers.
Compilation and end-to-end time are recorded separately.

The supplied baseline uses separate input and output fields on each timestep.
This intentionally fixes the source notebook's in-place cross-block data race.
Initialization is deterministic, boundaries are fixed, and scored computation
uses `Float32` by default.

Each trial records the resolved model ID, source and prompt hashes, complete
candidate patch lineage, raw timing samples, H200 environment, H200 evaluation
count, wall time, and Cursor-reported input/output/cache/reasoning tokens.
Cursor billed cost is also captured when it has settled.

## Setup

Requirements:

- Python 3.11+
- Julia 1.11
- Git
- On ORCD: Julia/CUDA.jl, Python/numba-cuda, and `nvcc`
- A Cursor API key with the requested models enabled
- Optional LangSmith API key

```bash
cd KernelAgentBench
cp config/study.example.yaml config/study.yaml
cp config/orcd.example.yaml config/orcd.yaml
cp .env.example .env

python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
# On the ORCD GPU environment:
pip install -e '.[gpu]'
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

Load `.env` with your preferred secret manager or shell. Do not submit model or
LangSmith keys to Slurm.

Validate configuration:

```bash
kernel-agent-bench validate-config --study config/study.yaml
kernel-agent-bench validate-config \
  --study config/study.yaml \
  --orcd config/orcd.yaml
kernel-agent-bench models --study config/study.yaml
kernel-agent-bench matrix --study config/study.yaml
```

Model selectors are matched against the live `Cursor.models.list()` catalog.
The exact newest Composer, exact GPT-5.6 match, and newest Anthropic/Claude
match are frozen in trial records. A missing model is an error; it is never
silently replaced.

## Run the checkpointed loop

The project must be committed before a real trial starts because every trial
uses a detached worktree at the specified baseline:

```bash
kernel-agent-bench start \
  --study config/study.yaml \
  --model composer \
  --language python \
  --replicate 1 \
  --baseline HEAD
```

The graph runs one coding-agent turn, executes public tests, validates that
only that language's candidate file changed, packages the binary Git patch,
and stops with status `awaiting_h200_result`. The editable files are
`kernels/gray_scott/candidate.jl`, `python/gray_scott/candidate.py`, and
`cpp/gray_scott/candidate.cuh`.

List bundles that do not yet have ingested results:

```bash
kernel-agent-bench pending
```

After H200 evaluation, resume the same checkpoint:

```bash
kernel-agent-bench resume \
  --trial gray-scott-cuda-h200-v1-python-composer-r1 \
  --result results/CANDIDATE_ID.json
```

Each trial is capped at **five** agent turns and **five** H200 evaluations.
The graph keeps the best correct incumbent and continues only when geometric-mean
speedup improves by at least 5% over that incumbent (or over 1.0× on the first
try). A correct candidate that is slower, equal, or only marginally faster ends
the trial early with `no_significant_improvement`, so a plateau uses fewer than
five GPU jobs. Incorrect candidates may still be repaired until the cap.

## ORCD H200

The H200 is requested with keyed SSH and:

```bash
salloc -N 1 -n 32 --cpus-per-task=1 --mem=128GB --time=06:00:00 -G h200:1
```

Fill `config/orcd.yaml`:

- `ssh_user` and `remote_project`: your Kerberos user and the persistent
  ORCD clone of `KernelAgentBench`.
- `scratch_root`: node-visible scratch for temporary worktrees.
- `identity_file`: only if the key is not the default SSH identity.
- `salloc_job_id`: set this if you already hold an `salloc`, so later evals
  use `srun` on that job instead of allocating again.
- `auto_evaluate`: `true` (default) folds rsync + `salloc`/`srun` + result
  download into the LangGraph loop.
- `remote_project` must be the absolute `KernelAgentBench` directory. Transfers
  never touch the rest of the ORCD clone or your home directory.

Rsync is scoped on purpose. Upload writes only
`$remote_project/candidates/<candidate_id>/` (the patch and manifest).
Download copies only `$remote_project/results/<candidate_id>.json` onto the
matching local file. There is no `--delete`, and the persistent kernels,
baselines, git checkout, and other candidate directories are not synced.

One-time remote setup:

```bash
ssh YOUR_USER@orcd-login.mit.edu
git clone YOUR_REPOSITORY_URL /path/to/UROPJulia
cd /path/to/UROPJulia/KernelAgentBench
# create .venv, pip install -e '.[gpu]', instantiate Julia
```

With keyed SSH and `auto_evaluate: true`, start the full loop from your laptop:

```bash
kernel-agent-bench start \
  --study config/study.yaml \
  --orcd config/orcd.yaml \
  --model composer \
  --language julia \
  --replicate 1 \
  --baseline HEAD
```

That command is the closed laptop→H200 loop for one Julia/Composer replicate:

1. Loads `config/study.yaml` and `config/orcd.yaml`.
2. Resolves `composer` against the live Cursor model catalog.
3. Creates a detached local git worktree at `--baseline HEAD`.
4. Runs the Cursor agent in that worktree with the Julia prompt. The agent
   may edit only `kernels/gray_scott/candidate.jl`.
5. Runs the public Julia tests. Illegal path changes or a failed test retry
   the agent, up to five turns.
6. Packages a patch + manifest under `candidates/<id>/`.
7. Rsyncs **only that folder** to
   `$remote_project/candidates/<id>/` (no `--delete`, repo untouched).
8. Over SSH, writes one sbatch file and runs
   `salloc -N 1 -n 32 --cpus-per-task=1 --mem=128GB --time=06:00:00 -G h200:1 -- srun ...`.
   The job makes a scratch worktree of the existing ORCD clone, applies the
   patch, and writes `$remote_project/results/<id>.json`.
9. Rsyncs that one JSON file back, ingests it, and either asks the agent to
   try again or stops under the 5-try / 5% rule.

`config/study.yaml` and `config/orcd.yaml` must exist (copy the example
files). The project must be committed so `--baseline HEAD` is a real commit.
The ORCD clone must already contain that commit.
To reuse one six-hour allocation across those evals, start `salloc` yourself,
put the job id in `salloc_job_id`, then start the trial.

Manual fallback (`auto_evaluate: false`) still prints the same commands:

```bash
scripts/prepare_ssh_handoff.sh config/orcd.yaml candidates/CANDIDATE_ID
```

On the login node you can also run:

```bash
scripts/submit_h200.sh config/orcd.yaml candidates/CANDIDATE_ID
```

Run a language evaluator directly for diagnostics:

```bash
julia --project=julia julia/evaluate_candidate.jl \
  --candidate-id manual \
  --output results/manual.json \
  --shapes 1024x1024,2048x2048,4096x4096 \
  --steps 1000 \
  --warmups 3 \
  --repetitions 10

python3 python/evaluate_candidate.py \
  --candidate-id manual-python \
  --output results/manual-python.json \
  --shapes 1024x1024,2048x2048,4096x4096

nvcc -O3 -std=c++17 cpp/evaluate_candidate.cu -o /tmp/evaluate-gray-scott
/tmp/evaluate-gray-scott \
  --candidate-id manual-cpp \
  --output results/manual-cpp.json \
  --shapes 1024x1024,2048x2048,4096x4096
```

## Tracing and privacy

When `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` is present, each agent
turn and validation stage creates LangSmith spans. Prompt/source content is
included by default. Set `trace_content: false` in the study file to redact
prompt, response, source, and patch payloads while retaining IDs, timing,
status, and token metadata. Secrets are never intentionally added to traces.

Local JSONL in `results/` is the authoritative record; LangSmith is not the
only token ledger. Missing SDK usage or delayed billed cost is recorded as
unavailable rather than estimated.

## Tests and result export

```bash
ruff check .
mypy src
pytest
pytest python/tests
bash cpp/test/run_cpu_tests.sh
julia --project=julia julia/test/runtests.jl
```

CUDA tests skip when CUDA is unavailable. Generate machine-readable schemas or
summaries with:

```bash
kernel-agent-bench write-schemas --output schemas.json
kernel-agent-bench summary --output results/summary.json
kernel-agent-bench summary --output results/summary.csv
```

Generated worktrees, candidates, traces, and results are ignored by Git. Copy
or archive them deliberately when a study must be retained.

## Study rollout

1. Run the unchanged baseline repeatedly on H200 and verify timing variance.
2. Complete one trial with one model and one language.
3. Let the graph stop at five H200 evaluations or earlier when speedup does not
   improve by at least 5%.
4. Run the full Julia/Python/C++ × Composer/GPT-5.6/Anthropic matrix with at
   least three independent replicates and identical budgets.
5. Add a KernelAbstractions baseline only after the CUDA study is stable.
6. Add AMD after hardware and the portability-versus-peak-performance objective
   are specified.
