module GrayScottCandidate

using Polyester

export candidate_fields, prepare_candidate, run_candidate!

# This file is the only source file coding agents may edit for the CPU backend.
# The initial candidate intentionally matches the trusted corrected baseline.
mutable struct CandidateState{A}
    u::A
    v::A
    u_next::A
    v_next::A
end

function prepare_candidate(u::AbstractMatrix{T}, v::AbstractMatrix{T}) where {T<:AbstractFloat}
    return CandidateState(copy(u), copy(v), copy(u), copy(v))
end

function step!(
    u_next::AbstractMatrix{T},
    v_next::AbstractMatrix{T},
    u::AbstractMatrix{T},
    v::AbstractMatrix{T},
    p,
) where {T<:AbstractFloat}
    copyto!(u_next, u)
    copyto!(v_next, v)
    height, width = size(u)
    @batch for x in 2:width-1
        for y in 2:height-1
            u_center = u[y, x]
            v_center = v[y, x]
            lap_u = u[y - 1, x] + u[y + 1, x] + u[y, x - 1] + u[y, x + 1] -
                    T(4) * u_center
            lap_v = v[y - 1, x] + v[y + 1, x] + v[y, x - 1] + v[y, x + 1] -
                    T(4) * v_center
            reaction = u_center * v_center * v_center
            u_next[y, x] = u_center +
                           p.dt * (p.diffusion_u * lap_u - reaction + p.feed * (one(T) - u_center))
            v_next[y, x] = v_center +
                           p.dt * (p.diffusion_v * lap_v + reaction - (p.feed + p.kill) * v_center)
        end
    end
    return nothing
end

function run_candidate!(state::CandidateState, p, steps::Int)
    for _ in 1:steps
        step!(state.u_next, state.v_next, state.u, state.v, p)
        state.u, state.u_next = state.u_next, state.u
        state.v, state.v_next = state.v_next, state.v
    end
    return state
end

candidate_fields(state::CandidateState) = (state.u, state.v)

end
