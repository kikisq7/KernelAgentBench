module GrayScottNotebook

# Harness-side pieces of parts/gpu/gray-scott.ipynb: the model parameters and the
# initial condition. The timed code lives in <backend>/baseline.jl.

export GrayScottParams, initialize_fields, relative_l2_error, SEED

# Defaults of simulate_gs_* in the notebook. `noise` is fixed at 0.0 there by
# default and is dropped: a random forcing term would make outputs unrepeatable.
Base.@kwdef struct GrayScottParams
    D_u::Float64 = 0.16
    D_v::Float64 = 0.08
    F::Float64 = 0.035
    k::Float64 = 0.06
    dt::Float64 = 1.0
    dx::Float64 = 1.0
end

const SEED = 20260918

function splitmix64(z::UInt64)
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

# The notebook's `U = rand(0.0:0.001:0.1, n, n)` with a portable generator, so the
# C++ and Python harnesses build bit-identical fields. Element (x, y) uses the
# 0-based linear index x + nx*y; x is the contiguous dimension in every language.
function initialize_fields(nx::Integer, ny::Integer; seed::Integer=SEED, init_multi=0.5)
    nx >= 16 && ny >= 16 || throw(ArgumentError("grid must be at least 16×16"))
    U = Matrix{Float64}(undef, nx, ny)
    for y in 1:ny, x in 1:nx
        index = UInt64((x - 1) + nx * (y - 1))
        draw = splitmix64(UInt64(seed) + index * 0x9e3779b97f4a7c15) % UInt64(101)
        U[x, y] = Float64(draw) / 1000
    end
    V = zeros(nx, ny)

    # Set up an initial small disturbance
    nx_mid, ny_mid = div(nx, 2), div(ny, 2)
    U[nx_mid-5:nx_mid+5, ny_mid-5:ny_mid+5] .= 1.0 .* init_multi
    V[nx_mid-5:nx_mid+5, ny_mid-5:ny_mid+5] .= 0.5 .* init_multi
    return U, V
end

function relative_l2_error(actual::AbstractArray, expected::AbstractArray)
    denominator = max(sqrt(sum(abs2, expected)), eps(Float64))
    return sqrt(sum(abs2, actual .- expected)) / denominator
end

end
