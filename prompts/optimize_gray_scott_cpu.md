You are optimizing a corrected two-dimensional Gray-Scott reaction-diffusion
solver using Julia on the CPU with Polyester.jl.

Only edit `kernels/gray_scott/cpu/candidate.jl`. Preserve `prepare_candidate`,
`run_candidate!`, and `candidate_fields`, along with the numerical semantics.
You may change loops, memory layout, and threading inside that file. Do not
weaken tests, special-case benchmark dimensions, cache expected outputs, or
introduce network access.

Caveats:
- The parallel implementation must adhere to the Gray-Scott algorithm itself.
  Do not change the update rule, stencil, boundary conditions, or timestep
  semantics.
- Do not add constraints that make the parallel version essentially serial,
  including size thresholds that skip thread parallelism and fall back to a
  serial loop.
- Improve runtime without changing the algorithm and without serializing work
  that should stay parallel.
- The scored path is CPU. Do not move the candidate onto CUDA.jl or require a
  GPU.
- For Julia CPU thread parallelism, use Polyester.jl `@batch`. Call it
  iteratively, not recursively. Do not switch to a different threading API.

This is the current benchmarking status of the parallel algorithms:
[Benchmark Output]
No scored timings have been recorded yet. The candidate currently matches the
trusted CPU baseline, so the first evaluation should be about 1.0x.

Primary objective: maximize steady-state timesteps per second over every
required workload. Correctness is a hard gate. You have at most five scored
evaluations; the trial stops early if speedup does not improve by at least 5%.
Correctness is checked in both Float32 and Float64, so keep the code generic
over the element type (no hard-coded Float32 literals, arrays, or conversions);
timing uses Float32.
Run the public test command before returning and summarize the optimization
and expected bottleneck.
