module CPUKernels
include(joinpath(Main.ROOT, "kernels", "gray_scott", "cpu", "baseline.jl"))
include(joinpath(Main.ROOT, "kernels", "gray_scott", "cpu", "candidate.jl"))
end

@testset "Gray-Scott CPU Polyester" begin
    Main.check_backend(
        CPUKernels.GrayScottBaseline.prepare_baseline,
        CPUKernels.GrayScottBaseline.run_baseline!,
        CPUKernels.GrayScottBaseline.baseline_fields,
    )
    Main.check_backend(
        CPUKernels.GrayScottCandidate.prepare_candidate,
        CPUKernels.GrayScottCandidate.run_candidate!,
        CPUKernels.GrayScottCandidate.candidate_fields,
    )
end
