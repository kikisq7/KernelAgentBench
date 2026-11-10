module KAKernels
include(joinpath(Main.ROOT, "kernels", "gray_scott", "ka", "backend.jl"))
include(joinpath(Main.ROOT, "kernels", "gray_scott", "ka", "baseline.jl"))
include(joinpath(Main.ROOT, "kernels", "gray_scott", "ka", "candidate.jl"))
end

const KA_BASELINE = (
    KAKernels.GrayScottBaseline.prepare_baseline,
    KAKernels.GrayScottBaseline.run_baseline!,
    KAKernels.GrayScottBaseline.baseline_fields,
)
const KA_CANDIDATE = (
    KAKernels.GrayScottCandidate.prepare_candidate,
    KAKernels.GrayScottCandidate.run_candidate!,
    KAKernels.GrayScottCandidate.candidate_fields,
)

# Scored KernelAbstractions eval always requires a GPU. Local public tests on a
# laptop without CUDA still run the portable CPU kernel so the agent loop can
# package a patch; the H200 evaluator then enforces CUDA.functional().
if CUDA.functional()
    @testset "Gray-Scott KernelAbstractions CUDA" begin
        gpu = KAKernels.GrayScottKABackend.select_backend("cuda")
        Main.check_backend(KA_BASELINE, KA_CANDIDATE; prepare_kwargs=(backend=gpu,))
    end
elseif !isempty(get(ENV, "SLURM_JOB_ID", ""))
    KAKernels.GrayScottKABackend.require_cuda_gpu()
else
    @info "CUDA unavailable locally; KernelAbstractions NVIDIA public tests skipped (evaluator still requires a GPU)"
end

module KAThreadsReference
include(joinpath(Main.ROOT, "kernels", "gray_scott", "cpu", "baseline.jl"))
end

@testset "Gray-Scott KernelAbstractions CPU" begin
    cpu = KAKernels.GrayScottKABackend.select_backend("cpu")
    Main.check_backend(KA_BASELINE, KA_CANDIDATE; prepare_kwargs=(backend=cpu,))
    # The two-launch KA baseline computes what the notebook's threaded CPU code does.
    reference = KAThreadsReference.GrayScottBaseline
    for shape in Main.SHAPES
        u, v = Main.initialize_fields(shape...)
        expected = Main.run_fields(
            reference.prepare_baseline, reference.run_baseline!, reference.baseline_fields,
            u, v, 50,
        )
        actual = Main.run_fields(KA_BASELINE..., u, v, 50; backend=cpu)
        Main.check_fields(actual..., expected..., u, v)
    end
end
