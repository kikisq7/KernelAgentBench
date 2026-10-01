module GrayScottCandidate

using KernelAbstractions
using ..GrayScottKABackend

export candidate_fields, prepare_candidate, run_candidate!

# This file is the only source file coding agents may edit for KernelAbstractions.
# The initial candidate intentionally matches the trusted corrected baseline.
# The same kernel must remain valid on CPU and NVIDIA GPU (CUDABackend), and
# must not use CUDA-only intrinsics that would block a later ROCBackend/AMD path.
mutable struct CandidateState{B,A}
    backend::B
    u::A
    v::A
    u_next::A
    v_next::A
end

function prepare_candidate(
    u::AbstractMatrix{T},
    v::AbstractMatrix{T};
    backend=select_backend(),
) where {T<:AbstractFloat}
    u_dev = to_device(backend, u)
    v_dev = to_device(backend, v)
    return CandidateState(backend, u_dev, v_dev, copy(u_dev), copy(v_dev))
end

@kernel function candidate_kernel!(u_next, v_next, u, v, diffusion_u, diffusion_v, feed, kill, dt)
    y, x = @index(Global, NTuple)
    height, width = size(u)
    if 1 < y < height && 1 < x < width
        u_center = u[y, x]
        v_center = v[y, x]
        lap_u = u[y - 1, x] + u[y + 1, x] + u[y, x - 1] + u[y, x + 1] -
                4 * u_center
        lap_v = v[y - 1, x] + v[y + 1, x] + v[y, x - 1] + v[y, x + 1] -
                4 * v_center
        reaction = u_center * v_center * v_center
        u_next[y, x] = u_center +
                       dt * (diffusion_u * lap_u - reaction + feed * (1 - u_center))
        v_next[y, x] = v_center +
                       dt * (diffusion_v * lap_v + reaction - (feed + kill) * v_center)
    end
end

function run_candidate!(state::CandidateState, p, steps::Int)
    kernel! = candidate_kernel!(state.backend)
    ndrange = size(state.u)
    for _ in 1:steps
        kernel!(
            state.u_next,
            state.v_next,
            state.u,
            state.v,
            p.diffusion_u,
            p.diffusion_v,
            p.feed,
            p.kill,
            p.dt;
            ndrange,
        )
        KernelAbstractions.synchronize(state.backend)
        state.u, state.u_next = state.u_next, state.u
        state.v, state.v_next = state.v_next, state.v
    end
    return state
end

candidate_fields(state::CandidateState) = (state.u, state.v)

end
