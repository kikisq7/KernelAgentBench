module CUDAKernels
include(joinpath(Main.ROOT, "kernels", "gray_scott", "cuda", "baseline.jl"))
include(joinpath(Main.ROOT, "kernels", "gray_scott", "cuda", "candidate.jl"))
end

@testset "Gray-Scott CUDA.jl" begin
    if CUDA.functional()
        Main.check_backend(
            (
                CUDAKernels.GrayScottBaseline.prepare_baseline,
                CUDAKernels.GrayScottBaseline.run_baseline!,
                CUDAKernels.GrayScottBaseline.baseline_fields,
            ),
            (
                CUDAKernels.GrayScottCandidate.prepare_candidate,
                CUDAKernels.GrayScottCandidate.run_candidate!,
                CUDAKernels.GrayScottCandidate.candidate_fields,
            ),
        )
    else
        @info "CUDA unavailable; CUDA.jl GPU tests skipped"
    end
end
