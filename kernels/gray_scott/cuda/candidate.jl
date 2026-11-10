module GrayScottCandidate

# Editable candidate. It starts identical to baseline.jl, which is the scored
# reference for both speed and correctness. Keep prepare_candidate,
# run_candidate!, and candidate_fields.

using CUDA

export candidate_fields, prepare_candidate, run_candidate!

function laplacian(A, x, y, dx)
    @inbounds DAx = (A[x-1, y] - (2 * A[x, y]) + A[x+1, y]) / dx
    @inbounds DAy = (A[x, y-1] - (2 * A[x, y]) + A[x, y+1]) / dx
    return DAx + DAy
end

function calculate_gs_cuda!(DU, DV, U, V, D_u, D_v, F, k, dx, nx, ny)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

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

    return
end

function update_gs_cuda!(U, V, DU, DV, dt, nx, ny)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    # Skip ghost cells on the boundaries
    if (1 < x < nx) && (1 < y < ny)
        # Update concentration fields
        @inbounds U[x, y] += DU[x, y] * dt
        @inbounds V[x, y] += DV[x, y] * dt
    end

    return
end

struct CandidateState{A}
    U::A
    V::A
    DU::A
    DV::A
end

function prepare_candidate(U::AbstractMatrix{Float64}, V::AbstractMatrix{Float64})
    # Transfer to the GPU
    U_gpu = CuArray(U)
    V_gpu = CuArray(V)
    return CandidateState(U_gpu, V_gpu, similar(U_gpu), similar(V_gpu))
end

function run_candidate!(state::CandidateState, p, steps::Int)
    nx, ny = size(state.U)
    for _ in 1:steps
        @cuda threads=(16, 16) blocks=(cld(nx, 16), cld(ny, 16)) calculate_gs_cuda!(
            state.DU, state.DV, state.U, state.V, p.D_u, p.D_v, p.F, p.k, p.dx, nx, ny,
        )
        @cuda threads=(16, 16) blocks=(cld(nx, 16), cld(ny, 16)) update_gs_cuda!(
            state.U, state.V, state.DU, state.DV, p.dt, nx, ny,
        )
    end
    return state
end

candidate_fields(state::CandidateState) = (state.U, state.V)

end
