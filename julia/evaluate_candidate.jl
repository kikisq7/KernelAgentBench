#!/usr/bin/env julia

using CUDA
using JSON3
using LinearAlgebra
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "kernels", "gray_scott", "reference.jl"))
include(joinpath(ROOT, "kernels", "gray_scott", "baseline.jl"))
include(joinpath(ROOT, "kernels", "gray_scott", "candidate.jl"))

using .GrayScottReference
using .GrayScottBaseline
using .GrayScottCandidate

function parse_args(args)
    parsed = Dict{String,String}()
    index = 1
    while index <= length(args)
        startswith(args[index], "--") || error("expected option, got $(args[index])")
        index == length(args) && error("missing value for $(args[index])")
        parsed[args[index][3:end]] = args[index + 1]
        index += 2
    end
    return parsed
end

parse_shapes(value) = [
    Tuple(parse.(Int, split(shape, 'x'))) for shape in split(value, ',') if !isempty(shape)
]

function errors(actual_u, actual_v, expected_u, expected_v)
    max_abs = max(
        maximum(abs.(actual_u .- expected_u)),
        maximum(abs.(actual_v .- expected_v)),
    )
    relative = max(
        relative_l2_error(actual_u, expected_u),
        relative_l2_error(actual_v, expected_v),
    )
    return Float64(max_abs), Float64(relative)
end

function candidate_correctness(dtype, shapes, p, atol, rtol)
    cases = []
    for shape in shapes
        initial_u, initial_v = initialize_fields(dtype, shape)
        for steps in (1, 5)
            expected_u, expected_v = cpu_simulate(initial_u, initial_v, p, steps)
            state = prepare_candidate(initial_u, initial_v)
            run_candidate!(state, p, steps)
            CUDA.synchronize()
            actual_u_gpu, actual_v_gpu = candidate_fields(state)
            actual_u, actual_v = Array(actual_u_gpu), Array(actual_v_gpu)
            max_abs, relative = errors(actual_u, actual_v, expected_u, expected_v)
            correct = isapprox(actual_u, expected_u; atol=atol, rtol=rtol) &&
                      isapprox(actual_v, expected_v; atol=atol, rtol=rtol) &&
                      all(isfinite, actual_u) && all(isfinite, actual_v) &&
                      actual_u[[1, end], :] == initial_u[[1, end], :] &&
                      actual_v[:, [1, end]] == initial_v[:, [1, end]]
            push!(cases, Dict(
                "shape" => collect(shape),
                "steps" => steps,
                "correct" => correct,
                "max_abs_error" => max_abs,
                "relative_l2_error" => relative,
            ))
        end
    end
    return all(case["correct"] for case in cases), cases
end

function timed_run!(runner, state, p, steps)
    seconds = CUDA.@elapsed begin
        runner(state, p, steps)
    end
    return Float64(seconds * 1000)
end

function benchmark_workload(dtype, shape, p, steps, warmups, repetitions, atol, rtol)
    initial_u, initial_v = initialize_fields(dtype, shape)

    for _ in 1:warmups
        timed_run!(run_baseline!, prepare_baseline(initial_u, initial_v), p, min(steps, 10))
        timed_run!(run_candidate!, prepare_candidate(initial_u, initial_v), p, min(steps, 10))
    end

    baseline_samples = Float64[]
    candidate_samples = Float64[]
    last_candidate = nothing
    for repetition in 1:repetitions
        baseline_state = prepare_baseline(initial_u, initial_v)
        candidate_state = prepare_candidate(initial_u, initial_v)
        if isodd(repetition)
            push!(baseline_samples, timed_run!(run_baseline!, baseline_state, p, steps))
            push!(candidate_samples, timed_run!(run_candidate!, candidate_state, p, steps))
        else
            push!(candidate_samples, timed_run!(run_candidate!, candidate_state, p, steps))
            push!(baseline_samples, timed_run!(run_baseline!, baseline_state, p, steps))
        end
        last_candidate = candidate_state
    end

    trusted_state = prepare_baseline(initial_u, initial_v)
    run_baseline!(trusted_state, p, steps)
    CUDA.synchronize()
    actual_u_gpu, actual_v_gpu = candidate_fields(last_candidate)
    expected_u_gpu, expected_v_gpu = baseline_fields(trusted_state)
    actual_u, actual_v = Array(actual_u_gpu), Array(actual_v_gpu)
    expected_u, expected_v = Array(expected_u_gpu), Array(expected_v_gpu)
    max_abs, relative = errors(actual_u, actual_v, expected_u, expected_v)
    correct = isapprox(actual_u, expected_u; atol=atol, rtol=rtol) &&
              isapprox(actual_v, expected_v; atol=atol, rtol=rtol)

    baseline_median = median(baseline_samples)
    candidate_median = median(candidate_samples)
    return Dict(
        "shape" => collect(shape),
        "baseline_samples_ms" => baseline_samples,
        "candidate_samples_ms" => candidate_samples,
        "baseline_median_ms" => baseline_median,
        "candidate_median_ms" => candidate_median,
        "baseline_std_ms" => length(baseline_samples) > 1 ? std(baseline_samples) : 0.0,
        "candidate_std_ms" => length(candidate_samples) > 1 ? std(candidate_samples) : 0.0,
        "speedup" => baseline_median / candidate_median,
        "correct" => correct,
        "max_abs_error" => max_abs,
        "relative_l2_error" => relative,
    )
end

function environment_metadata()
    return Dict(
        "hostname" => gethostname(),
        "julia_version" => string(VERSION),
        "cuda_version" => string(CUDA.runtime_version()),
        "gpu_name" => CUDA.name(CUDA.device()),
        "gpu_capability" => string(CUDA.capability(CUDA.device())),
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
        "cuda_visible_devices" => get(ENV, "CUDA_VISIBLE_DEVICES", ""),
    )
end

function main()
    options = parse_args(ARGS)
    candidate_id = get(options, "candidate-id", "manual")
    output = get(options, "output", joinpath(ROOT, "results", "$candidate_id.json"))
    performance_shapes = parse_shapes(get(options, "shapes", "1024x1024"))
    correctness_shapes = parse_shapes(get(options, "correctness-shapes", "31x29,64x64,127x65"))
    steps = parse(Int, get(options, "steps", "100"))
    warmups = parse(Int, get(options, "warmups", "2"))
    repetitions = parse(Int, get(options, "repetitions", "5"))
    atol = parse(Float64, get(options, "atol", "2e-5"))
    rtol = parse(Float64, get(options, "rtol", "2e-4"))
    dtype_name = get(options, "dtype", "Float32")
    dtype = dtype_name == "Float32" ? Float32 :
            dtype_name == "Float64" ? Float64 :
            error("dtype must be Float32 or Float64")
    p = GrayScottParams(dtype)
    started = time()

    result = try
        CUDA.functional() || error("CUDA is not functional on this host")
        compile_started = time()
        compile_state = prepare_candidate(initialize_fields(dtype, (32, 32))...)
        run_candidate!(compile_state, p, 1)
        CUDA.synchronize()
        compile_seconds = time() - compile_started

        correct, correctness_cases =
            candidate_correctness(dtype, correctness_shapes, p, atol, rtol)
        workloads = correct ? [
            benchmark_workload(dtype, shape, p, steps, warmups, repetitions, atol, rtol)
            for shape in performance_shapes
        ] : []
        all_correct = correct && all(workload["correct"] for workload in workloads)
        speedup = all_correct && !isempty(workloads) ?
                  exp(mean(log(workload["speedup"]) for workload in workloads)) : nothing
        Dict(
            "schema_version" => "2.0",
            "candidate_id" => candidate_id,
            "language" => "julia",
            "evaluator_version" => "gray-scott-cuda-v1",
            "status" => all_correct ? "passed" : "failed",
            "correct" => all_correct,
            "correctness_cases" => correctness_cases,
            "workloads" => workloads,
            "geometric_mean_speedup" => speedup,
            "compile_seconds" => compile_seconds,
            "end_to_end_seconds" => time() - started,
            "environment" => environment_metadata(),
            "error" => nothing,
            "created_at" => string(Dates.now(Dates.UTC)),
        )
    catch exception
        Dict(
            "schema_version" => "2.0",
            "candidate_id" => candidate_id,
            "language" => "julia",
            "evaluator_version" => "gray-scott-cuda-v1",
            "status" => "error",
            "correct" => false,
            "workloads" => [],
            "geometric_mean_speedup" => nothing,
            "compile_seconds" => nothing,
            "end_to_end_seconds" => time() - started,
            "environment" => Dict{String,String}(),
            "error" => sprint(showerror, exception, catch_backtrace()),
            "created_at" => string(Dates.now(Dates.UTC)),
        )
    end

    mkpath(dirname(output))
    open(output, "w") do handle
        JSON3.pretty(handle, result)
        write(handle, '\n')
    end
    println(output)
    result["status"] == "passed" || exit(2)
end

using Dates
main()
