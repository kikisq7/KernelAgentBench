module GrayScottBaseline

using CUDA

export baseline_fields, prepare_baseline, run_baseline!

mutable struct BaselineState{A}
    u::A
    v::A
    u_next::A
    v_next::A
end

function prepare_baseline(u::AbstractMatrix{T}, v::AbstractMatrix{T}) where {T<:AbstractFloat}
    u_gpu, v_gpu = CuArray(u), CuArray(v)
    return BaselineState(u_gpu, v_gpu, copy(u_gpu), copy(v_gpu))
end

function baseline_kernel!(u_next, v_next, u, v, diffusion_u, diffusion_v, feed, kill, dt)
    y = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    x = (blockIdx().y - 1) * blockDim().y + threadIdx().y
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
    return
end

function run_baseline!(state::BaselineState, p, steps::Int)
    threads = (16, 16)
    blocks = cld.(size(state.u), threads)
    for _ in 1:steps
        @cuda threads=threads blocks=blocks baseline_kernel!(
            state.u_next,
            state.v_next,
            state.u,
            state.v,
            p.diffusion_u,
            p.diffusion_v,
            p.feed,
            p.kill,
            p.dt,
        )
        state.u, state.u_next = state.u_next, state.u
        state.v, state.v_next = state.v_next, state.v
    end
    return state
end

baseline_fields(state::BaselineState) = (state.u, state.v)

end
