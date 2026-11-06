You are optimizing a corrected two-dimensional Gray-Scott reaction-diffusion
solver for one NVIDIA H200 using **OpenCL C++**.

Only edit `cpp/gray_scott/candidate_kernel.cl`. Preserve the kernel name
`step_kernel` and the argument list. You may change the OpenCL kernel body and
launch sizing in `candidate.hpp` only if the public tests still pass (normally
edit the `.cl` file only). Do not weaken tests, special-case benchmark
dimensions, cache expected outputs, or introduce network access.

Caveats:
- The parallel implementation must adhere to the Gray-Scott algorithm itself.
  Do not change the update rule, stencil, boundary conditions, or timestep
  semantics.
- Do not add constraints that make the parallel version essentially serial,
  including size thresholds that skip the OpenCL kernel and fall back to a host
  serial loop.
- Improve runtime without changing the algorithm and without serializing work
  that should stay parallel.
- Use C++17 on the host (`-std=c++17`). Kernel language is OpenCL C.
- Keep datatypes consistent with the contract: `float` fields and the same
  `GrayScottParams` as the baseline.

This is the current benchmarking status of the parallel algorithms:
[Benchmark Output]
No H200 timings have been recorded yet. The candidate currently matches the
trusted OpenCL baseline, so the first GPU evaluation should be about 1.0x.

Primary objective: maximize steady-state timesteps per second over every
required workload. Correctness is a hard gate. You have at most five H200
evaluations; the trial stops after two consecutive scored runs where speedup does
not improve by at least 5% over the incumbent (or 1.0× on the first scored run).
Run the public test command before returning and summarize the optimization
and expected bottleneck.
