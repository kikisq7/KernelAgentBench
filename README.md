# Kernel Agent Bench

Kernel Agent Bench measures how effectively coding agents optimize a corrected
two-dimensional, five-point Gray-Scott solver in **Julia**. The active study
tests three backends:

1. **CPU** — Polyester.jl threaded loops
2. **CUDA** — CUDA.jl `@cuda` on one NVIDIA GPU
3. **KernelAbstractions** — portable `@kernel` on CPU and NVIDIA
   (`CUDABackend` now; AMD `ROCBackend` later)

KernelAbstractions is the focus track. Python/numba-cuda and CUDA C++ remain
in the repository but are not part of this study.

Agents still run the existing loop: edit the backend candidate, pass public
tests, package a patch, evaluate against the same-backend baseline, and stop
after five tries or when speedup does not improve by at least 5%.

The benchmark has two planes:

- The local control plane runs a checkpointed LangGraph, invokes Cursor coding
  agents, records patches, and optionally sends traces to LangSmith.
- GPU evaluation (CUDA.jl and KernelAbstractions on NVIDIA) uses keyed SSH to
  ORCD. The CPU backend evaluates locally and does not request a GPU.

## What is measured

Correctness is a hard gate. The primary metric is the geometric mean of
steady-state timestep speedup over all required grid sizes. GPU timings use
CUDA events (`timer: cuda_event`) and exclude package loading, JIT compilation,
allocations, and host/device transfers. CPU and KernelAbstractions-on-CPU use
synchronized wall time (`timer: cpu_ns`). Compilation and end-to-end time are
recorded separately. Each trial uses 3 unscored warmups of 10 steps, then 10
interleaved 1000-step repetitions; the score is the **median**, not the mean.
The CPU backend uses smaller grids and fewer reps so local eval stays practical.

The supplied baseline uses separate input and output fields on each timestep.
This intentionally fixes the source notebook's in-place cross-block data race.
Initialization is deterministic, boundaries are fixed, and scored computation
uses `Float32` by default.

Each trial records the resolved model ID, source and prompt hashes, complete
candidate patch lineage, raw timing samples, machine environment, evaluation
count, wall time, and Cursor-reported input/output/cache/reasoning tokens.

## Setup

Requirements:

- Python 3.11+
- Julia 1.11
- Git
- On ORCD, for GPU tracks: Julia with CUDA.jl and KernelAbstractions
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
The study pins `claude-opus-5-5`, `gpt-6-astra`, and `composer-2.5` exactly; the
resolved IDs are frozen in trial records. A missing model is an error; it is never
silently replaced.

## Test the three Julia backends

Public tests (the CPU backend always runs; CUDA.jl skips if no GPU is present.
KernelAbstractions **requires** a CUDA GPU and fails if none is found):

```bash
julia --project=julia julia/test/runtests.jl cpu
julia --project=julia julia/test/runtests.jl cuda
julia --project=julia julia/test/runtests.jl ka
julia --project=julia julia/test/runtests.jl   # all three
```

Direct evaluator (no agent). CPU does not need a GPU. KernelAbstractions
`--device auto` requires a CUDA GPU and uses `CUDABackend()`:

```bash
julia --project=julia julia/evaluate_candidate.jl \
  --backend cpu \
  --language julia-cpu \
  --candidate-id manual-cpu \
  --output results/manual-cpu.json \
  --shapes 256x256,512x512 \
  --steps 50 \
  --warmups 1 \
  --repetitions 3

julia --project=julia julia/evaluate_candidate.jl \
  --backend ka \
  --language julia-ka \
  --candidate-id manual-ka \
  --output results/manual-ka.json \
  --shapes 64x64 \
  --steps 5 \
  --warmups 1 \
  --repetitions 1
```

## Run the checkpointed agent loop

The project must be committed before a real trial starts because every trial
uses a detached worktree at the specified baseline. KernelAbstractions is the
default focus:

```bash
kernel-agent-bench start \
  --study config/study.yaml \
  --model composer-2.5 \
  --language julia-ka \
  --replicate 1 \
  --baseline HEAD
```

`--language` is one of `julia-ka`, `julia-cuda`, `julia-cpu`, `cpp`, or `cpp-cpu`. The graph runs
one coding-agent turn, executes that backend's public tests, validates that
only that backend's candidate file changed, packages the binary Git patch, and
either evaluates or stops with status `awaiting_h200_result`.

Editable files:

- `kernels/gray_scott/ka/candidate.jl`
- `kernels/gray_scott/cuda/candidate.jl`
- `kernels/gray_scott/cpu/candidate.jl`
- `cpp/gray_scott/candidate_kernel.cl` (OpenCL on the H200)
- `cpp/gray_scott/cpu/candidate.hpp` (host CPU, scored locally)

The agent cwd is a temp copy of that backend's `workspace_paths` only.

List bundles that do not yet have ingested results:

```bash
kernel-agent-bench pending
```

After GPU evaluation, resume the same checkpoint:

```bash
kernel-agent-bench resume \
  --trial gray-scott-julia-backends-v2-julia-ka-composer-2.5-r1 \
  --result results/CANDIDATE_ID.json
```

Each trial is capped at **five** agent turns and **five** scored evaluations.
The graph keeps the best correct incumbent and continues only when geometric-mean
speedup improves by at least 5% over that incumbent (or over 1.0× on the first
try). After **two consecutive** correct evaluations that fail that bar, the trial
ends with `no_significant_improvement` (configurable via
`budgets.no_improvement_streak_to_stop`, default 2). Incorrect candidates may
still be repaired until the cap. `julia-cpu` and `cpp-cpu` evaluations run locally
on the host CPU. `cpp` stays OpenCL on the H200.

## ORCD H200 (CUDA.jl and KernelAbstractions GPU)

The H200 is requested with keyed SSH. The laptop submits with `sbatch` and
only polls `squeue`; it does not hold `salloc` over SSH. The job asks for
1 node, 1 CPU, 8GB, 20 minutes, and `h200:1`. To reuse an existing
allocation, start `salloc` yourself and set `salloc_job_id`.

Fill `config/orcd.yaml`:

- `ssh_user` and `remote_project`: your Kerberos user and the persistent
  ORCD clone of `KernelAgentBench`.
- `scratch_root`: node-visible scratch for temporary worktrees.
- `identity_file`: only if the key is not the default SSH identity.
  ORCD login is usually Kerberos + Duo. Run
  `kernel-agent-bench ssh-check --orcd config/orcd.yaml` once from a
  terminal; later rsync reuses that `ControlPersist` socket.
- `salloc_job_id`: set this if you already hold an `salloc`, so later evals
  use `srun` on that job instead of allocating again.
- `auto_evaluate`: `true` (default) folds rsync + `sbatch` + short `squeue`
  polls + result download into the LangGraph loop.
- `remote_project` must be the absolute `KernelAgentBench` directory. Transfers
  never touch the rest of the ORCD clone or your home directory.

Rsync is scoped on purpose. Upload writes only
`$remote_project/candidates/<candidate_id>/` (the patch and manifest).
Download copies only `$remote_project/results/<candidate_id>.json` onto the
matching local file. There is no `--delete`, and the persistent kernels,
baselines, git checkout, and other candidate directories are not synced.

With keyed SSH and `auto_evaluate: true`, start the KernelAbstractions GPU loop:

```bash
kernel-agent-bench start \
  --study config/study.yaml \
  --orcd config/orcd.yaml \
  --model composer-2.5 \
  --language julia-ka \
  --replicate 1 \
  --baseline HEAD
```

`config/study.yaml` and `config/orcd.yaml` must exist (copy the example
files). The project must be committed so `--baseline HEAD` is a real commit.
The ORCD clone must already contain that commit.

Manual fallback (`auto_evaluate: false`) still prints the same commands:

```bash
scripts/prepare_ssh_handoff.sh config/orcd.yaml candidates/CANDIDATE_ID
```

On the login node you can also run:

```bash
scripts/submit_h200.sh config/orcd.yaml candidates/CANDIDATE_ID
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

1. Test CPU, CUDA.jl, and KernelAbstractions locally (`runtests.jl`).
2. Time the unchanged KernelAbstractions baseline on NVIDIA (H200) and on CPU.
3. Run agent trials with `--language julia-ka` (focus). Same 5-try / 5% loop.
4. Compare CUDA.jl and CPU cells against the KernelAbstractions result.
5. Add AMD/`ROCBackend` when that hardware track is specified.
