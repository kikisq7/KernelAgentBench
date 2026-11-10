module GrayScottBaseline

# "GPU (KernelAbstractions) Implementation of the Gray-Scott Equation" from
# parts/gpu/gray-scott.ipynb. Two changes make it runnable and deterministic:
#
# 1. The kernel body is split at its @synchronize into two launches. In the
#    notebook, @synchronize only orders one workgroup (and sits inside a
#    divergent branch), so neighbouring workgroups race on U and V.
# 2. The workgroup size is (16, 16). The notebook passes (cld(n, 16), cld(n, 16)),
#    which exceeds 1024 threads per workgroup for n > 512 and cannot launch.
#
# The noise term is removed, and so is the no-op `[1:2]` after
# `@index(Global, NTuple)`, which KernelAbstractions' CPU backend cannot compile.

using KernelAbstractions
using ..GrayScottKABackend

export baseline_fields, prepare_baseline, run_baseline!

function laplacian(A, x, y, dx)
    @inbounds DAx = (A[x-1, y] - (2 * A[x, y]) + A[x+1, y]) / dx
    @inbounds DAy = (A[x, y-1] - (2 * A[x, y]) + A[x, y+1]) / dx
    return DAx + DAy
end

@kernel function calculate_gs_ka!(DU, DV, U, V, D_u, D_v, F, k, dx, nx, ny)
    x, y = @index(Global, NTuple)

    # Skip ghost cells on the boundaries
    if (1 < x < nx) && (1 < y < ny)
        # Compute diffusion
        ΔU = laplacian(U, x, y, dx)
        ΔV = laplacian(V, x, y, dx)

        # Update concentration fields
        UVV = @inbounds U[x, y] * (V[x, y] ^ 2)
        @inbounds DU[x, y] = D_u * ΔU - UVV + F * (1 - U[x, y])
        @inbounds DV[x, y] = D_v * ΔV + UVV - (F + k) * V[x, y]
    end
end

@kernel function update_gs_ka!(U, V, DU, DV, dt, nx, ny)
    x, y = @index(Global, NTuple)

    # Skip ghost cells on the boundaries
    if (1 < x < nx) && (1 < y < ny)
        # Update concentration fields
        @inbounds U[x, y] += DU[x, y] * dt
        @inbounds V[x, y] += DV[x, y] * dt
    end
end

struct BaselineState{B,A}
    backend::B
    U::A
    V::A
    DU::A
    DV::A
end

function prepare_baseline(
    U::AbstractMatrix{Float64},
    V::AbstractMatrix{Float64};
    backend=select_backend(),
)
    # Transfer to the GPU
    U_gpu = to_device(backend, U)
    V_gpu = to_device(backend, V)
    DU = KernelAbstractions.allocate(backend, Float64, size(U)...)
    DV = KernelAbstractions.allocate(backend, Float64, size(U)...)
    return BaselineState(backend, U_gpu, V_gpu, DU, DV)
end

function run_baseline!(state::BaselineState, p, steps::Int)
    nx, ny = size(state.U)
    # Compile KA kernels
    calculate! = calculate_gs_ka!(state.backend, (16, 16))
    update! = update_gs_ka!(state.backend, (16, 16))
    for _ in 1:steps
        calculate!(
            state.DU, state.DV, state.U, state.V, p.D_u, p.D_v, p.F, p.k, p.dx, nx, ny;
            ndrange=(nx, ny),
        )
        update!(state.U, state.V, state.DU, state.DV, p.dt, nx, ny; ndrange=(nx, ny))
    end
    KernelAbstractions.synchronize(state.backend)
    return state
end

baseline_fields(state::BaselineState) = (state.U, state.V)

end
