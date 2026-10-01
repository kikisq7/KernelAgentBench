You are optimizing a corrected two-dimensional Gray-Scott reaction-diffusion
solver using Julia and KernelAbstractions.jl. This is the primary study track:
the same kernel must run on CPU and NVIDIA GPU (`CUDABackend`) and stay valid
for a later AMD `ROCBackend` path.

Only edit `kernels/gray_scott/ka/candidate.jl`. Preserve `prepare_candidate`,
`run_candidate!`, and `candidate_fields`, along with the numerical semantics.
You may change the `@kernel`, launch parameters, workgroup size, memory layout,
and the host timestep loop inside that file. Do not weaken tests, special-case
benchmark dimensions, cache expected outputs, or introduce network access.

Caveats:
- The parallel implementation must adhere to the Gray-Scott algorithm itself.
  Do not change the update rule, stencil, boundary conditions, or timestep
  semantics.
- Do not add constraints that make the parallel version essentially serial,
  including size thresholds that skip GPU or thread parallelism and fall back
  to a serial loop.
- Improve runtime without changing the algorithm and without serializing work
  that should stay parallel.
- Stay in KernelAbstractions. Do not rewrite the scored path as CUDA.jl `@cuda`
  or as a CPU-only Polyester loop. Do not use CUDA-only intrinsics, shared
  memory APIs, or index spaces that would break `CPU()` or a future
  `ROCBackend()`.
- Public tests require a CUDA GPU (`CUDA.functional()`). They run the same
  KernelAbstractions kernel on `CUDABackend` and also on `CPU()`. Do not skip
  the GPU path.
- For any extra host-side Julia thread parallelism, use Polyester.jl `@batch`.
  Call it iteratively, not recursively.

This is the current benchmarking status of the parallel algorithms:
[Benchmark Output]
No scored timings have been recorded yet. The candidate currently matches the
trusted KernelAbstractions baseline, so the first evaluation should be about 1.0x.

Primary objective: maximize steady-state timesteps per second over every
required workload. Correctness is a hard gate. You have at most five scored
evaluations; the trial stops early if speedup does not improve by at least 5%.
Run the public test command before returning and summarize the optimization
and expected bottleneck.
