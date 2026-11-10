using Test
using CUDA

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "kernels", "gray_scott", "notebook.jl"))
using .GrayScottNotebook

const REQUESTED = isempty(ARGS) ? "all" : ARGS[1]
const SHAPES = ((31, 29), (64, 64), (127, 65))
const STEPS = (1, 5, 50)
# Matches the evaluator defaults.
const ATOL, RTOL = 1e-10, 1e-8
const PARAMS = GrayScottParams()

function check_fields(actual_u, actual_v, expected_u, expected_v, initial_u, initial_v)
    @test eltype(actual_u) == Float64 && eltype(actual_v) == Float64
    @test all(isfinite, actual_u) && all(isfinite, actual_v)
    @test actual_u ≈ expected_u atol=ATOL rtol=RTOL
    @test actual_v ≈ expected_v atol=ATOL rtol=RTOL
    for (actual, initial) in ((actual_u, initial_u), (actual_v, initial_v))
        @test actual[[1, end], :] == initial[[1, end], :]
        @test actual[:, [1, end]] == initial[:, [1, end]]
    end
end

run_fields(prepare, run!, fields, u, v, steps; kwargs...) =
    Array.(fields(run!(prepare(copy(u), copy(v); kwargs...), PARAMS, steps)))

# The candidate must reproduce the baseline it is timed against.
function check_backend(baseline, candidate; prepare_kwargs=NamedTuple())
    for shape in SHAPES, steps in STEPS
        u, v = initialize_fields(shape...)
        expected = run_fields(baseline..., u, v, steps; prepare_kwargs...)
        actual = run_fields(candidate..., u, v, steps; prepare_kwargs...)
        check_fields(actual..., expected..., u, v)
    end
end

@testset "Gray-Scott notebook initial condition" begin
    u, v = initialize_fields(31, 29)
    @test size(u) == (31, 29)
    # Shared with cpp/test and python/test: the three harnesses start bit-identical.
    @test round.(Int, vec(u)[1:8] .* 1000) == [70, 10, 16, 94, 47, 3, 53, 17]
    @test vec(u)[end] == 0.036
    @test sum(v) == 30.25
    @test u[15, 14] == 0.5 && v[15, 14] == 0.25
    @test initialize_fields(31, 29) == (u, v)
    @test_throws ArgumentError initialize_fields(15, 32)
end

REQUESTED in ("all", "cpu") && include("cpu_tests.jl")
REQUESTED in ("all", "cuda") && include("cuda_tests.jl")
REQUESTED in ("all", "ka") && include("ka_tests.jl")
