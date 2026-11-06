You are optimizing a corrected two-dimensional Gray-Scott reaction-diffusion
solver for one NVIDIA H200 using **OpenCL C++**.

Only edit `cpp/gray_scott/candidate_kernel.cl`. Preserve the kernel name
`step_kernel` and the argument list. The host launcher reads each kernel's
`reqd_work_group_size` (default 16×16 if unset). You may set
`reqd_work_group_size` for tiled kernels; do not change host C++ files. Do not weaken tests, special-case benchmark
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

On a laptop without a working OpenCL GPU, public tests skip device checks
(similar to Julia CUDA.jl tests when CUDA is unavailable). That skip is not
a numerical failure; scored Float32/Float64 correctness runs on the H200 evaluator.
