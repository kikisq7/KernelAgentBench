module GrayScottBaseline

using KernelAbstractions
using ..GrayScottKABackend

export baseline_fields, prepare_baseline, run_baseline!

# Match kernels/gray_scott/cuda/baseline.jl: 16×16 tiles over the full grid.
const BASELINE_TILE = (16, 16)

mutable struct BaselineState{B,A}
    backend::B
    u::A
    v::A
    u_next::A
    v_next::A
end

function prepare_baseline(
    u::AbstractMatrix{T},
    v::AbstractMatrix{T};
    backend=select_backend(),
) where {T<:AbstractFloat}
    u_dev = to_device(backend, u)
    v_dev = to_device(backend, v)
    return BaselineState(backend, u_dev, v_dev, copy(u_dev), copy(v_dev))
end

@kernel function baseline_kernel!(u_next, v_next, u, v, diffusion_u, diffusion_v, feed, kill, dt)
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

function baseline_launch_kwargs(backend, height::Int, width::Int)
    ndrange = (height, width)
    if backend isa CPU
        return ndrange, NamedTuple()
    end
    return ndrange, (workgroupsize=BASELINE_TILE,)
end

function run_baseline!(state::BaselineState, p, steps::Int)
    kernel! = baseline_kernel!(state.backend)
    height, width = size(state.u)
    ndrange, launch_kwargs = baseline_launch_kwargs(state.backend, height, width)
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
            launch_kwargs...,
        )
        state.u, state.u_next = state.u_next, state.u
        state.v, state.v_next = state.v_next, state.v
    end
    KernelAbstractions.synchronize(state.backend)
    return state
end

baseline_fields(state::BaselineState) = (state.u, state.v)

end
