module CPUKernels
include(joinpath(Main.ROOT, "kernels", "gray_scott", "cpu", "baseline.jl"))
include(joinpath(Main.ROOT, "kernels", "gray_scott", "cpu", "candidate.jl"))
end

const CPU_BASELINE = (
    CPUKernels.GrayScottBaseline.prepare_baseline,
    CPUKernels.GrayScottBaseline.run_baseline!,
    CPUKernels.GrayScottBaseline.baseline_fields,
)

@testset "Gray-Scott CPU (Threads.@threads)" begin
    Main.check_backend(
        CPU_BASELINE,
        (
            CPUKernels.GrayScottCandidate.prepare_candidate,
            CPUKernels.GrayScottCandidate.run_candidate!,
            CPUKernels.GrayScottCandidate.candidate_fields,
        ),
    )
end
