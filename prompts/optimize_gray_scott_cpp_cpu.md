You are optimizing a corrected two-dimensional Gray-Scott reaction-diffusion
solver using C++17 on the CPU.

Only edit `cpp/gray_scott/cpu/candidate.hpp`. Preserve the function
`gray_scott_cpu_candidate::run` and the `CpuGrayScottState` / `CpuGrayScottParams`
types, along with the numerical semantics. You may change loops, memory layout,
and threading inside that file. Do not weaken tests, special-case benchmark
dimensions, cache expected outputs, or introduce network access.

Caveats:
- The parallel implementation must adhere to the Gray-Scott algorithm itself.
  Do not change the update rule, stencil, boundary conditions, or timestep
  semantics.
- Do not add constraints that make the parallel version essentially serial,
  including size thresholds that skip thread parallelism and fall back to a
  serial loop.
- Improve runtime without changing the algorithm and without serializing work
  that should stay parallel.
- The scored path is the host CPU. Do not move the candidate onto OpenCL, CUDA,
  or any GPU API.
- For thread parallelism, use `std::thread` from the C++17 standard library.
  Do not require OpenMP, TBB, or a nonstandard runtime. The initial candidate
  partitions rows with `cpu_parallel_rows`.

This is the current benchmarking status of the parallel algorithms:
[Benchmark Output]
No scored timings have been recorded yet. The candidate currently matches the
trusted C++ CPU baseline, so the first evaluation should be about 1.0x.

Primary objective: maximize steady-state timesteps per second over every
required workload. Correctness is a hard gate. You have at most five scored
evaluations; the trial stops early if speedup does not improve by at least 5%.
Correctness is checked in both Float32 and Float64 against a serial reference,
so keep the arithmetic generic over the element type. Timing uses Float32 and
runs only after that correctness gate passes.
Run the public test command before returning and summarize the optimization
and expected bottleneck.
