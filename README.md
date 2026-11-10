# Kernel Agent Bench

Kernel Agent Bench measures how well coding agents optimize a two-dimensional,
five-point Gray-Scott solver. Every track starts from the same fixed baseline:
the most basic parallel version from the SC25 Julia HPC tutorial
([`parts/gpu/gray-scott.ipynb`](https://github.com/JuliaParallel/julia-hpc-tutorial-sc25/blob/main/parts/gpu/gray-scott.ipynb)),
copied in Julia and translated directly into C++ and Python. The active study
(`gray-scott-notebook-v3`) has seven tracks:

| Track | Baseline | Runs on |
| --- | --- | --- |
| `julia-cpu` | notebook `calculate_gs_cpu_mt!` (`Threads.@threads`) | local CPU, 4 threads |
| `julia-cuda` | notebook `@cuda` kernel, two-pass | H200 |
| `julia-ka` | notebook KernelAbstractions `@kernel`, two-pass | H200 |
| `cpp-cpu` | `std::thread` translation | local CPU, 4 threads |
| `cpp` | OpenCL translation (double precision) | H200 |
| `python-cpu` | Numba `prange` translation | local CPU, 4 threads |
| `python-triton` | Triton translation on CUDA torch tensors | H200 |

`kernels/gray_scott/UPSTREAM.md` lists what is copied verbatim and the few
deliberate changes: the GPU kernels run in two passes because the notebook's
in-place GPU update races across blocks, and the initial noise comes from a
portable seeded generator.

Agents edit only their track's candidate file, pass public tests, package a
patch, and are evaluated against that track's baseline. Each trial stops after
five tries or when speedup stops improving by at least 5%.

The benchmark has two planes:

- The local control plane runs a checkpointed LangGraph, invokes coding agents,
  records patches, and optionally sends traces to LangSmith.
- GPU tracks are evaluated on ORCD over keyed SSH. CPU tracks are evaluated
  locally and do not request a GPU.

## What is measured

The baseline is fixed and is never regenerated. It is both the timing reference
and the correctness oracle. For every evaluation, the baseline and the candidate
start from identical Float64 initial fields. The candidate must match the
baseline's output after 1, 5, and 50 steps on the correctness grids, and after
every timed workload. The tolerances are `atol=1e-10` and `rtol=1e-8`, and
boundary cells must stay unchanged. Reassociation drift over 1000 steps is about
`4e-15`, so legitimate reorderings pass and wrong stencils do not.

Correctness is a hard gate. The primary metric is the geometric mean of
steady-state timestep speedup over the performance grids. GPU timings use device
events (`timer: cuda_event`), or wall time around a full-queue `clFinish` for
OpenCL (`timer: wall_clock`). They exclude
package loading, JIT compilation, allocation, and host/device transfers. CPU
tracks use synchronized wall time (`timer: cpu_ns`). The score is the **median**
of interleaved baseline/candidate repetitions, not the mean. CPU tracks use
256–1024 grids and 200 steps; GPU tracks use 1024–4096 grids and 1000 steps.

All three CPU baselines pin the same thread count (`cpu_threads: 4`, the M1
performance cores). With `-ffp-contract=off` for C++, the Julia, C++, and Numba
CPU baselines produce bit-identical output.

Each trial records the resolved model ID, source and prompt hashes, complete
candidate patch lineage, raw timing samples, machine environment, evaluation
count, wall time, and Cursor-reported input/output/cache/reasoning tokens.

## Setup

Requirements:

- Python 3.11+ (Numba is installed with the package for `python-cpu`)
- Julia 1.11
- A C++17 compiler for `cpp-cpu`
- Git
- On ORCD, for GPU tracks: Julia with CUDA.jl and KernelAbstractions, an
  OpenCL ICD for `cpp`, and a Python virtualenv with `python/requirements-gpu.txt`
  for `python-triton` (created automatically at `triton_venv` on first use)
- A Cursor API key for Composer and Kimi
- An Anthropic API key for Claude Opus (`opus-5.5`). That trial calls the
  Anthropic API directly and does not use the Cursor key
- An OpenAI API key and the Codex CLI for GPT (`gpt-6-astra`). That trial calls
  the OpenAI API directly and does not use the Cursor key:
  `npm install --prefix .kernel-agent-bench/tools @openai/codex@0.167.0`
- Optional LangSmith API key

```bash
cd KernelAgentBench
cp config/study.example.yaml config/study.yaml
cp config/orcd.example.yaml config/orcd.yaml
test -f .env || cp .env.example .env   # do not clobber an existing .env

python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
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

Cursor model selectors are matched against the live `Cursor.models.list()` catalog.
`opus-5.5` is the exception: `provider: anthropic` pins `claude-opus-5-5` and
`effort: high`, and each turn runs the Claude Agent SDK with `ANTHROPIC_API_KEY`.
`gpt-6-astra` works the same way with `provider: openai`: each turn runs
`codex exec` with `OPENAI_API_KEY` and an empty `CODEX_HOME`, so a local ChatGPT
login is never used. Codex reports tokens but not price, so `charged_cents` is null.
The study also pins `composer-2.5` and `kimi-k3`
(Moonshot open-weight Kimi) exactly; the
resolved IDs are frozen in trial records. A missing model is an error; it is never
silently replaced.

## Public tests and direct evaluation

Each track's public test compares its candidate against its baseline. The
agent runs the same command:

```bash
julia --threads=4 --project=julia julia/test/runtests.jl cpu
julia --project=julia julia/test/runtests.jl cuda      # skips without a GPU
julia --threads=4 --project=julia julia/test/runtests.jl ka  # KA on CPU locally
bash cpp/test/run_cpu_tests.sh
bash cpp/test/run_opencl_tests.sh                      # needs an fp64 OpenCL device
python3 python/tests/run_tests.py cpu
python3 python/tests/run_tests.py triton               # needs CUDA + Triton
```

Every evaluator takes the same arguments and writes a schema 3.0 result:

```bash
julia --threads=4 --project=julia julia/evaluate_candidate.jl \
  --backend cpu --language julia-cpu --candidate-id manual-cpu \
  --output results/manual-cpu.json \
  --shapes 256x256,512x512 --steps 50 --warmups 1 --repetitions 3 --threads 4

python3 python/evaluate_candidate.py \
  --backend cpu --language python-cpu --candidate-id manual-py \
  --output results/manual-py.json \
  --shapes 256x256 --steps 50 --warmups 1 --repetitions 3 --threads 4
```

## Run the checkpointed agent loop

The project must be committed before a real trial starts because every trial
uses a detached worktree at the specified baseline:

```bash
kernel-agent-bench start \
  --study config/study.yaml \
  --model composer-2.5 \
  --language julia-ka \
  --replicate 1 \
  --baseline HEAD
```

`--language` is one of `julia-cpu`, `julia-cuda`, `julia-ka`, `cpp-cpu`, `cpp`,
`python-cpu`, or `python-triton`. The graph runs
one coding-agent turn, executes that backend's public tests, validates that
only that backend's candidate file changed, packages the binary Git patch, and
either evaluates or stops with status `awaiting_h200_result`.

Editable files:

- `kernels/gray_scott/ka/candidate.jl`
- `kernels/gray_scott/cuda/candidate.jl`
- `kernels/gray_scott/cpu/candidate.jl`
- `cpp/gray_scott/candidate.hpp` and `cpp/gray_scott/candidate_kernel.cl` (OpenCL on the H200)
- `cpp/gray_scott/cpu/candidate.hpp` (host CPU, scored locally)
- `python/gray_scott/cpu/candidate.py` (Numba, scored locally)
- `python/gray_scott/gpu/candidate.py` (Triton on the H200)

The agent cwd is a temp copy of that backend's `workspace_paths` only.

List bundles that do not yet have ingested results:

```bash
kernel-agent-bench pending
```

After GPU evaluation, resume the same checkpoint:

```bash
kernel-agent-bench resume \
  --trial gray-scott-notebook-v3-julia-ka-composer-2.5-r1 \
  --result results/CANDIDATE_ID.json
```

Each trial is capped at **five** agent turns and **five** scored evaluations.
The graph keeps the best correct incumbent and continues only when geometric-mean
speedup improves by at least 5% over that incumbent (or over 1.0× on the first
try). After **two consecutive** correct evaluations that fail that bar, the trial
ends with `no_significant_improvement` (configurable via
`budgets.no_improvement_streak_to_stop`, default 2). Incorrect candidates may
still be repaired until the cap. The `*-cpu` tracks are evaluated locally on the
host CPU; the other tracks are evaluated on the H200.

## ORCD H200 (GPU tracks)

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

CUDA tests skip when CUDA is unavailable. Results from the earlier
`gray-scott-julia-backends-v2` study used a different baseline and Float32, so
they are kept as-is and not compared with v3. Generate machine-readable schemas or
summaries with:

```bash
kernel-agent-bench write-schemas --output schemas.json
kernel-agent-bench summary --output results/summary.json
kernel-agent-bench summary --output results/summary.csv
```

Generated worktrees, candidates, traces, and results are ignored by Git. Copy
or archive them deliberately when a study must be retained.

## Study rollout

1. Run every track's public test locally (GPU tests on ORCD).
2. Time the unchanged baselines: CPU tracks locally, GPU tracks on the H200.
3. Run agent trials per track and model with the same 5-try / 5% loop.
4. Compare tracks within a language (CPU vs GPU) and across languages per device.
