You are optimizing a corrected two-dimensional Gray-Scott reaction-diffusion
solver for one NVIDIA GPU using Julia and CUDA.jl.

Only edit `kernels/gray_scott/cuda/candidate.jl`. Preserve `prepare_candidate`,
`run_candidate!`, and `candidate_fields`, along with the numerical semantics.
You may change kernels, launch parameters, memory layout, and the GPU host
timestep loop inside that file. Do not weaken tests, special-case benchmark
dimensions, cache expected outputs, or introduce network access.

Caveats:
- The parallel implementation must adhere to the Gray-Scott algorithm itself.
  Do not change the update rule, stencil, boundary conditions, or timestep
  semantics.
- Do not add constraints that make the parallel version essentially serial,
  including size thresholds that skip GPU or thread parallelism and fall back
  to a serial loop.
- Improve runtime without changing the algorithm and without serializing work
  that should stay parallel.
- The scored path is CUDA.jl. Do not replace the GPU kernel with a serial CPU
  implementation or with KernelAbstractions for this backend.
- For any host-side Julia thread parallelism, use Polyester.jl `@batch`. Call
  it iteratively, not recursively.

This is the current benchmarking status of the parallel algorithms:
[Benchmark Output]
No GPU timings have been recorded yet. The candidate currently matches the
trusted CUDA.jl baseline, so the first GPU evaluation should be about 1.0x.

Primary objective: maximize steady-state timesteps per second over every
required workload. Correctness is a hard gate. You have at most five GPU
evaluations; the trial stops early if speedup does not improve by at least 5%.
Correctness is checked in both Float32 and Float64, so keep the code generic
over the element type (no hard-coded Float32 literals, arrays, or conversions);
timing uses Float32.
Run the public test command before returning and summarize the optimization
and expected bottleneck.
