# Upstream provenance

The task is taken from:

- Repository: https://github.com/JuliaParallel/julia-hpc-tutorial-sc25
- Commit: `2dc277d04843de8e469b848ebfa3131257108467` (re-verified 2026-11-10;
  last notebook change `2c112f73`)
- Notebook: `parts/gpu/gray-scott.ipynb`

Each baseline is the notebook's most basic parallel version, transcribed as
closely as possible. The baseline is fixed and is never regenerated. Its own
output is the correctness oracle: candidates are compared against it on the
same initial fields.

## What is copied verbatim

- The equations, the `laplacian(A, x, y, dx)` five-point stencil, the parameter
  defaults (`D_u=0.16, D_v=0.08, F=0.035, k=0.06, dt=1.0, dx=1.0`), and Float64
  storage.
- The initial condition: `U = rand(0.0:0.001:0.1, n, n)`, `V = zeros`, and a
  centered 11×11 square set to `U=0.5, V=0.25`.
- `cpu/baseline.jl` is `calculate_gs_cpu_mt!`: two passes, `Threads.@threads`
  over rows, with `DU`/`DV` scratch arrays and boundary cells untouched.
- `cuda/baseline.jl` uses `@cuda threads=(16, 16) blocks=(cld(nx, 16), cld(ny, 16))`
  with the notebook's interior `if` guard.
- `ka/baseline.jl` uses the notebook's `@kernel` with `@index(Global, NTuple)`.

## Deviations, and why

1. **No cross-block race on the GPU.** The notebook's CUDA and KA kernels update
   `U`/`V` in place and only call `sync_threads()`, which synchronizes within a
   block. Neighboring blocks can read cells that were already updated, so the
   output depends on scheduling. Both GPU baselines use the notebook's own CPU
   structure instead: one launch computes `DU`/`DV`, and a second launch applies
   the `dt` update.
2. **KA workgroup size is pinned to `(16, 16)`.** The notebook's `cld(n, 16)`
   workgroup exceeds the 1024-thread limit for grids larger than 512.
3. **`@index(Global, NTuple)` without the `[1:2]` slice.** The slice stops KA
   from rewriting the index on the CPU backend.
4. **Deterministic initial noise.** The notebook calls `rand(...)` directly. The
   harness uses a portable `splitmix64` generator (`SEED = 20260918`) that gives
   the same distribution and is bit-identical in Julia, C++, and Python.
5. **`noise` is dropped.** It is 0.0 in the notebook's default runs. A random
   forcing term would make outputs unrepeatable.

## Translations

The C++ and Python tracks translate the same two-pass algorithm, using the same
parameters, initial fields, and row-major `x + nx*y` layout:

| Track | Baseline |
| --- | --- |
| `cpp-cpu` | `cpp/gray_scott/cpu/baseline.hpp`: persistent `std::thread` team, rows split across threads |
| `cpp` | `cpp/gray_scott/baseline_kernel.cl`: OpenCL `calculate_gs` + `update_gs` in double precision |
| `python-cpu` | `python/gray_scott/cpu/baseline.py`: Numba `@njit(parallel=True)` with `prange` over rows |
| `python-triton` | `python/gray_scott/gpu/baseline.py`: Triton two-pass kernels on CUDA torch tensors |

Triton passes Python float scalars as fp32, so the Triton baseline reads its six
parameters from a Float64 device tensor. C++ is compiled with
`-ffp-contract=off`. With that flag, the Julia, C++, and Numba CPU baselines
agree bit for bit (maximum difference 0.0 after 200 steps on a 64×48 grid).
