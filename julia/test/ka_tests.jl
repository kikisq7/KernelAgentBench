module KAKernels
include(joinpath(Main.ROOT, "kernels", "gray_scott", "ka", "backend.jl"))
include(joinpath(Main.ROOT, "kernels", "gray_scott", "ka", "baseline.jl"))
include(joinpath(Main.ROOT, "kernels", "gray_scott", "ka", "candidate.jl"))
end

# Scored KernelAbstractions eval always requires a GPU. Local public tests on a
# laptop without CUDA still run the portable CPU kernel so the agent loop can
# package a patch; the H200 evaluator then enforces CUDA.functional().
if CUDA.functional()
    @testset "Gray-Scott KernelAbstractions CUDA" begin
        @test CUDA.functional()
        gpu = KAKernels.GrayScottKABackend.select_backend("cuda")
        Main.check_backend(
            KAKernels.GrayScottBaseline.prepare_baseline,
            KAKernels.GrayScottBaseline.run_baseline!,
            KAKernels.GrayScottBaseline.baseline_fields;
            prepare_kwargs=(backend=gpu,),
        )
        Main.check_backend(
            KAKernels.GrayScottCandidate.prepare_candidate,
            KAKernels.GrayScottCandidate.run_candidate!,
            KAKernels.GrayScottCandidate.candidate_fields;
            prepare_kwargs=(backend=gpu,),
        )
    end
elseif !isempty(get(ENV, "SLURM_JOB_ID", ""))
    KAKernels.GrayScottKABackend.require_cuda_gpu()
else
    @info "CUDA unavailable locally; KernelAbstractions NVIDIA public tests skipped (evaluator still requires a GPU)"
end

@testset "Gray-Scott KernelAbstractions CPU" begin
    cpu = KAKernels.GrayScottKABackend.select_backend("cpu")
    Main.check_backend(
        KAKernels.GrayScottBaseline.prepare_baseline,
        KAKernels.GrayScottBaseline.run_baseline!,
        KAKernels.GrayScottBaseline.baseline_fields;
        prepare_kwargs=(backend=cpu,),
    )
    Main.check_backend(
        KAKernels.GrayScottCandidate.prepare_candidate,
        KAKernels.GrayScottCandidate.run_candidate!,
        KAKernels.GrayScottCandidate.candidate_fields;
        prepare_kwargs=(backend=cpu,),
    )
end
