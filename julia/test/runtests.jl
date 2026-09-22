using Test
using CUDA

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "kernels", "gray_scott", "reference.jl"))
using .GrayScottReference

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

@testset "Gray-Scott candidate" begin
    if CUDA.functional()
        include(joinpath(ROOT, "kernels", "gray_scott", "baseline.jl"))
        include(joinpath(ROOT, "kernels", "gray_scott", "candidate.jl"))
        params = GrayScottParams(Float32)
        for shape in ((31, 29), (64, 64), (127, 65))
            u, v = initialize_fields(Float32, shape)
            expected_u, expected_v = cpu_simulate(u, v, params, 5)
            baseline = GrayScottBaseline.prepare_baseline(u, v)
            GrayScottBaseline.run_baseline!(baseline, params, 5)
            state = GrayScottCandidate.prepare_candidate(u, v)
            GrayScottCandidate.run_candidate!(state, params, 5)
            CUDA.synchronize()
            baseline_u_gpu, baseline_v_gpu = GrayScottBaseline.baseline_fields(baseline)
            actual_u_gpu, actual_v_gpu = GrayScottCandidate.candidate_fields(state)
            @test Array(baseline_u_gpu) ≈ expected_u atol=2f-5 rtol=2f-4
            @test Array(baseline_v_gpu) ≈ expected_v atol=2f-5 rtol=2f-4
            @test Array(actual_u_gpu) ≈ expected_u atol=2f-5 rtol=2f-4
            @test Array(actual_v_gpu) ≈ expected_v atol=2f-5 rtol=2f-4
        end
    else
        @info "CUDA unavailable; candidate GPU tests skipped"
    end
end
