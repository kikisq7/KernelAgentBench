You are optimizing a two-dimensional Gray-Scott reaction-diffusion solver in
Python with Numba on the CPU.

The baseline is the most rudimentary parallel version of the SC25 HPC tutorial
notebook (`parts/gpu/gray-scott.ipynb`, JuliaParallel/julia-hpc-tutorial-sc25):
a two-pass update that computes DU and DV from the current U and V, then adds
DU*dt and DV*dt, with D_u=0.16, D_v=0.08, F=0.035, k=0.06, dt=1, dx=1. Your
candidate starts as an exact copy of that baseline.

Only edit `python/gray_scott/cpu/candidate.py`. Preserve `prepare_candidate`,
`run_candidate`, and `candidate_fields`. You may change the jitted functions,
loops, memory layout, allocation, and the host timestep loop inside that file. Do not weaken tests, special-case benchmark dimensions, cache expected
outputs, or introduce network access.

Caveats:
- The parallel implementation must adhere to the Gray-Scott algorithm itself.
  Do not change the update rule, the five-point Laplacian, the fixed ghost cells,
  or the timestep semantics: every cell's step-n+1 value depends only on
  step-n values.
- Do not add constraints that make the parallel version essentially serial,
  including size thresholds that skip `prange` parallelism and fall back to a serial loop.
- Improve runtime without changing the algorithm and without serializing work
  that should stay parallel.
- The scored path is the CPU. Use Numba (`njit`, `prange`); do not move the work
  onto a GPU or into another compiled language. The evaluator sets Numba to 4
  threads. Fields are float64 NumPy arrays of shape (ny, nx).

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
required workload. Scored runs are on the host CPU with 4 threads, on 256x256, 512x512, and 1024x1024 grids for 200 steps. Speedup is the baseline's median time divided by
the candidate's, as a geometric mean over the grids. You have at most five
scored evaluations; the trial stops after two consecutive scored runs that do
not improve speedup by at least 5% over the incumbent (or 1.0x on the first).
Run the public test command before returning and summarize the optimization
and expected bottleneck.
