module GrayScottCandidate

# Editable candidate. It starts identical to baseline.jl, which is the scored
# reference for both speed and correctness. Keep prepare_candidate,
# run_candidate!, and candidate_fields.

export candidate_fields, prepare_candidate, run_candidate!

function laplacian(A, x, y, dx)
    @inbounds DAx = (A[x-1, y] - (2 * A[x, y]) + A[x+1, y]) / dx
    @inbounds DAy = (A[x, y-1] - (2 * A[x, y]) + A[x, y+1]) / dx
    return DAx + DAy
end

function calculate_gs_cpu_mt!(U, V, D_u, D_v, F, k, dt, dx, nx, ny)
    # Allocate U and V temporaries for updates
    DU = Matrix{Float64}(undef, nx, ny)
    DV = Matrix{Float64}(undef, nx, ny)

    # Compute diffusion and concentration fields changes
    Threads.@threads for y in 2:(ny-1)
        for x in 2:(nx-1)
            # Compute diffusion
            ΔU = laplacian(U, x, y, dx)
            ΔV = laplacian(V, x, y, dx)

            # Compute concentration fields changes
            UVV = @inbounds U[x, y] * (V[x, y] ^ 2)
            @inbounds DU[x, y] = D_u * ΔU - UVV + F * (1 - U[x, y])
            @inbounds DV[x, y] = D_v * ΔV + UVV - (F + k) * V[x, y]
        end
    end

    # Update concentration fields
    Threads.@threads for y in 2:(ny-1)
        for x in 2:(nx-1)
            @inbounds U[x, y] += DU[x, y] * dt
            @inbounds V[x, y] += DV[x, y] * dt
        end
    end

    return
end

struct CandidateState{A}
    U::A
    V::A
end

prepare_candidate(U::AbstractMatrix{Float64}, V::AbstractMatrix{Float64}) =
    CandidateState(copy(U), copy(V))

function run_candidate!(state::CandidateState, p, steps::Int)
    nx, ny = size(state.U)
    for _ in 1:steps
        calculate_gs_cpu_mt!(state.U, state.V, p.D_u, p.D_v, p.F, p.k, p.dt, p.dx, nx, ny)
    end
    return state
end

candidate_fields(state::CandidateState) = (state.U, state.V)

end
