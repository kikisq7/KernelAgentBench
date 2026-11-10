You are optimizing a two-dimensional Gray-Scott reaction-diffusion solver for
one NVIDIA H200 using OpenCL with a C++17 host.

The baseline is the most rudimentary parallel version of the SC25 HPC tutorial
notebook (`parts/gpu/gray-scott.ipynb`, JuliaParallel/julia-hpc-tutorial-sc25):
a two-pass update that computes DU and DV from the current U and V, then adds
DU*dt and DV*dt, with D_u=0.16, D_v=0.08, F=0.035, k=0.06, dt=1, dx=1. Your
candidate starts as an exact copy of that baseline.

Only edit `cpp/gray_scott/candidate.hpp` (host code) and
`cpp/gray_scott/candidate_kernel.cl` (kernels). Preserve the
`gray_scott_opencl_candidate` namespace with `State`, `prepare`, `run`,
`fields`, and `release`. You may change kernels, work-group sizes, launch
structure, buffers, and the host timestep loop. Do not weaken tests, special-case benchmark dimensions, cache expected
outputs, or introduce network access.

Caveats:
- The parallel implementation must adhere to the Gray-Scott algorithm itself.
  Do not change the update rule, the five-point Laplacian, the fixed ghost cells,
  or the timestep semantics: every cell's step-n+1 value depends only on
  step-n values.
- Do not add constraints that make the parallel version essentially serial,
  including size thresholds that skip the OpenCL kernels and fall back to a serial loop.
- Improve runtime without changing the algorithm and without serializing work
  that should stay parallel.
- Enqueue work on `OpenCLRuntime::instance().queue`. The evaluator times
  `run` with a wall clock from an idle queue until that queue is finished.
- Kernels are built with no extra compiler options. Fields are `double`.

Correctness is a hard gate, checked in Float64 against the baseline itself:
from the same initial fields, the candidate's U and V must match the baseline's
within atol=1e-10 and rtol=1e-8 after 1, 5, and 50 steps on 31x29, 64x64, and
127x65 grids, and again after every timed run. The outer ring of cells must not
change. Reordering floating-point operations is fine at that tolerance;
changing the update is not. Grids are nx-by-ny with x the contiguous dimension
and need not be square or a multiple of any tile size.

This is the current benchmarking status of the parallel algorithms:
[Benchmark Output]
No scored timings have been recorded yet. The candidate currently matches the
baseline, so the first evaluation should be about 1.0x.

Primary objective: maximize steady-state timesteps per second over every
required workload. Scored runs are on one NVIDIA H200, on 1024x1024, 2048x2048, and 4096x4096 grids for 1000 steps. Speedup is the baseline's median time divided by
the candidate's, as a geometric mean over the grids. You have at most five
scored evaluations; the trial stops after two consecutive scored runs that do
not improve speedup by at least 5% over the incumbent (or 1.0x on the first).
Run the public test command before returning and summarize the optimization
and expected bottleneck.

On a laptop without a suitable GPU, public tests skip the device checks. That skip is not a numerical failure; the H200 evaluator is the gate.
