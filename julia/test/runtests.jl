using Test
using CUDA

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "kernels", "gray_scott", "reference.jl"))
using .GrayScottReference

const REQUESTED = isempty(ARGS) ? "all" : ARGS[1]
const SHAPES = ((31, 29), (64, 64), (127, 65))
# Matches the evaluator defaults: (atol, rtol) per precision.
const TOLERANCES = (Float32 => (2e-5, 2e-4), Float64 => (1e-12, 1e-10))

function check_backend(prepare, run!, fields_fn; prepare_kwargs=NamedTuple())
    for (T, (atol, rtol)) in TOLERANCES
        params = GrayScottParams(T)
        for shape in SHAPES
            u, v = initialize_fields(T, shape)
            expected_u, expected_v = cpu_simulate(u, v, params, 5)
            state = prepare(copy(u), copy(v); prepare_kwargs...)
            run!(state, params, 5)
            actual_u, actual_v = Array.(fields_fn(state))
            @test eltype(actual_u) == T && eltype(actual_v) == T
            @test actual_u ≈ expected_u atol=atol rtol=rtol
            @test actual_v ≈ expected_v atol=atol rtol=rtol
            for (actual, initial) in ((actual_u, u), (actual_v, v))
                @test actual[[1, end], :] == initial[[1, end], :]
                @test actual[:, [1, end]] == initial[:, [1, end]]
            end
        end
    end
end

@testset "Gray-Scott CPU reference" begin
    params = GrayScottParams(Float32)
    u, v = initialize_fields(Float32, (31, 29))
    @test size(u) == (31, 29)
    @test all(u[[1, end], :] .== 1)
    @test all(v[:, [1, end]] .== 0)

    result_u, result_v = cpu_simulate(u, v, params, 5)
    repeated_u, repeated_v = cpu_simulate(u, v, params, 5)
    @test result_u == repeated_u
    @test result_v == repeated_v
    @test all(isfinite, result_u)
    @test all(isfinite, result_v)
    @test result_u[[1, end], :] == u[[1, end], :]
    @test result_u[:, [1, end]] == u[:, [1, end]]
    @test result_v[[1, end], :] == v[[1, end], :]
    @test result_v[:, [1, end]] == v[:, [1, end]]
end

REQUESTED in ("all", "cpu") && include("cpu_tests.jl")
REQUESTED in ("all", "cuda") && include("cuda_tests.jl")
REQUESTED in ("all", "ka") && include("ka_tests.jl")
