You are optimizing a corrected two-dimensional Gray-Scott reaction-diffusion
solver for one NVIDIA H200 using CUDA C++.

Only edit `cpp/gray_scott/candidate.cuh`. Preserve `prepare_candidate`,
`run_candidate`, and `copy_candidate_fields`, along with the numerical
semantics. You may change kernels, launch parameters, memory layout, and the
GPU host timestep loop inside that file. Do not weaken tests, special-case
benchmark dimensions, cache expected outputs, or introduce network access.

Caveats:
- The parallel implementation must adhere to the Gray-Scott algorithm itself.
  Do not change the update rule, stencil, boundary conditions, or timestep
  semantics.
- Do not add constraints that make the parallel version essentially serial,
  including size thresholds that skip the CUDA kernel and fall back to a host
  serial loop.
- Improve runtime without changing the algorithm and without serializing work
  that should stay parallel.
- Use C++17 (`-std=c++17`). Do not depend on C++20/23 features.
- Keep datatypes consistent with this C++17 CUDA contract: `float` fields,
  `int` sizes, and `GrayScottParams` as defined in `common.cuh`. Do not switch
  to types that change the numerical contract (for example `double`,
  `__half`, or `int64_t` indices unless the rest of the interface is updated
  to match, which you may not do).

This is the current benchmarking status of the parallel algorithms:
[Benchmark Output]
No H200 timings have been recorded yet. The candidate currently matches the
trusted CUDA C++ baseline, so the first GPU evaluation should be about 1.0x.

Primary objective: maximize steady-state timesteps per second over every
required workload. Correctness is a hard gate. You have at most five H200
evaluations; the trial stops early if speedup does not improve by at least 5%.
Run the public test command before returning and summarize the optimization
and expected bottleneck.
