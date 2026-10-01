using Test
using CUDA

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "kernels", "gray_scott", "reference.jl"))
using .GrayScottReference

const REQUESTED = isempty(ARGS) ? "all" : ARGS[1]
const SHAPES = ((31, 29), (64, 64), (127, 65))
const ATOL = 2f-5
const RTOL = 2f-4

function check_backend(prepare, run!, fields_fn; prepare_kwargs=NamedTuple())
    params = GrayScottParams(Float32)
    for shape in SHAPES
        u, v = initialize_fields(Float32, shape)
        expected_u, expected_v = cpu_simulate(u, v, params, 5)
        state = prepare(u, v; prepare_kwargs...)
        run!(state, params, 5)
        actual_u, actual_v = Array.(fields_fn(state))
        @test actual_u ≈ expected_u atol=ATOL rtol=RTOL
        @test actual_v ≈ expected_v atol=ATOL rtol=RTOL
        @test actual_u[[1, end], :] == u[[1, end], :]
        @test actual_v[:, [1, end]] == v[:, [1, end]]
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
