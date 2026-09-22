# Upstream provenance

The initial task is derived from:

- Repository: https://github.com/JuliaParallel/julia-hpc-tutorial-sc25
- Commit: `2dc277d04843de8e469b848ebfa3131257108467`
- Notebook: `parts/gpu/gray-scott.ipynb`
- Resolved: 2026-09-18

The scored baseline is a transcription of the model equations, not a byte-for-
byte copy of the notebook. It deliberately changes the numerical execution
contract:

1. Every timestep reads `u` and `v` and writes `u_next` and `v_next`, followed
   by a host-side swap. Blocks never observe partially updated neighbor data.
2. No conditional block barrier is used.
3. The five-point Laplacian uses unit grid spacing.
4. Initialization noise is deterministic and generated on the CPU.
5. Boundary cells remain fixed.

These corrections are part of the supplied baseline. Agents are evaluated on
optimization, not on discovering the source notebook's race.
