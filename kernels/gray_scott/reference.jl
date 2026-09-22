module GrayScottReference

export GrayScottParams, cpu_simulate, initialize_fields, relative_l2_error

struct GrayScottParams{T<:AbstractFloat}
    diffusion_u::T
    diffusion_v::T
    feed::T
    kill::T
    dt::T
end

GrayScottParams(::Type{T}=Float32) where {T<:AbstractFloat} =
    GrayScottParams(T(0.16), T(0.08), T(0.035), T(0.065), T(1.0))

function initialize_fields(
    ::Type{T},
    shape::Tuple{Int,Int};
    seed::Int=20260918,
) where {T<:AbstractFloat}
    height, width = shape
    height >= 5 && width >= 5 || throw(ArgumentError("shape must be at least 5×5"))
    u = ones(T, height, width)
    v = zeros(T, height, width)
    radius_y = max(1, height ÷ 10)
    radius_x = max(1, width ÷ 10)
    center_y, center_x = (height + 1) ÷ 2, (width + 1) ÷ 2

    @inbounds for x in max(2, center_x - radius_x):min(width - 1, center_x + radius_x)
        for y in max(2, center_y - radius_y):min(height - 1, center_y + radius_y)
            # Deterministic perturbation without device-side RNG.
            noise = T(0.01) * T(sin((y * 73856093 + x * 19349663 + seed) * 0.001))
            u[y, x] = T(0.5) + noise
            v[y, x] = T(0.25) - noise
        end
    end
    return u, v
end

function step!(
    u_next::Matrix{T},
    v_next::Matrix{T},
    u::Matrix{T},
    v::Matrix{T},
    p::GrayScottParams{T},
) where {T<:AbstractFloat}
    copyto!(u_next, u)
    copyto!(v_next, v)
    height, width = size(u)
    @inbounds for x in 2:width-1
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

function cpu_simulate(
    u_initial::Matrix{T},
    v_initial::Matrix{T},
    p::GrayScottParams{T},
    steps::Int,
) where {T<:AbstractFloat}
    steps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    u, v = copy(u_initial), copy(v_initial)
    u_next, v_next = similar(u), similar(v)
    for _ in 1:steps
        step!(u_next, v_next, u, v, p)
        u, u_next = u_next, u
        v, v_next = v_next, v
    end
    return u, v
end

function relative_l2_error(actual::AbstractArray, expected::AbstractArray)
    denominator = max(sqrt(sum(abs2, expected)), eps(Float64))
    return sqrt(sum(abs2, actual .- expected)) / denominator
end

end
