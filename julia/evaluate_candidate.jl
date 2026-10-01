#!/usr/bin/env julia

using Dates
using JSON3
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "kernels", "gray_scott", "reference.jl"))
using .GrayScottReference

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

const OPTIONS = parse_args(ARGS)
const BACKEND = get(OPTIONS, "backend", "ka")
BACKEND in ("cpu", "cuda", "ka") || error("backend must be cpu, cuda, or ka")
const KERNEL_DIR = joinpath(ROOT, "kernels", "gray_scott", BACKEND)

if BACKEND != "cpu"
    using CUDA
end
if BACKEND == "ka"
    include(joinpath(KERNEL_DIR, "backend.jl"))
    using .GrayScottKABackend
end
include(joinpath(KERNEL_DIR, "baseline.jl"))
include(joinpath(KERNEL_DIR, "candidate.jl"))
using .GrayScottBaseline
using .GrayScottCandidate

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

function host_fields(fields)
    return Array(fields[1]), Array(fields[2])
end

function make_states(initial_u, initial_v, backend, ka_device)
    if backend == "ka"
        return (
            prepare_baseline(initial_u, initial_v; backend=ka_device),
            prepare_candidate(initial_u, initial_v; backend=ka_device),
        )
    end
    return prepare_baseline(initial_u, initial_v), prepare_candidate(initial_u, initial_v)
end

uses_cuda_events(backend, state) = backend == "cuda" ||
    (backend == "ka" && nameof(typeof(state.backend)) == :CUDABackend)

timer_name(backend, state) = uses_cuda_events(backend, state) ? "cuda_event" : "cpu_ns"

if BACKEND == "cpu"
    synchronize_run(backend, state) = nothing
    function timed_run!(backend, runner, state, p, steps)
        started = time_ns()
        runner(state, p, steps)
        return Float64((time_ns() - started) / 1e6)
    end
else
    include(joinpath(@__DIR__, "timing_gpu.jl"))
end

function candidate_correctness(backend, ka_device, dtype, shapes, p, atol, rtol)
    cases = []
    for shape in shapes
        initial_u, initial_v = initialize_fields(dtype, shape)
        for steps in (1, 5)
            expected_u, expected_v = cpu_simulate(initial_u, initial_v, p, steps)
            _, candidate_state = make_states(initial_u, initial_v, backend, ka_device)
            run_candidate!(candidate_state, p, steps)
            synchronize_run(backend, candidate_state)
            actual_u, actual_v = host_fields(candidate_fields(candidate_state))
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

function benchmark_workload(backend, ka_device, dtype, shape, p, steps, warmups, repetitions, atol, rtol)
    initial_u, initial_v = initialize_fields(dtype, shape)
    warmup_steps = min(steps, 10)

    for _ in 1:warmups
        baseline_state, candidate_state = make_states(initial_u, initial_v, backend, ka_device)
        timed_run!(backend, run_baseline!, baseline_state, p, warmup_steps)
        timed_run!(backend, run_candidate!, candidate_state, p, warmup_steps)
    end

    baseline_samples = Float64[]
    candidate_samples = Float64[]
    last_candidate = nothing
    for repetition in 1:repetitions
        baseline_state, candidate_state = make_states(initial_u, initial_v, backend, ka_device)
        if isodd(repetition)
            push!(baseline_samples, timed_run!(backend, run_baseline!, baseline_state, p, steps))
            push!(candidate_samples, timed_run!(backend, run_candidate!, candidate_state, p, steps))
        else
            push!(candidate_samples, timed_run!(backend, run_candidate!, candidate_state, p, steps))
            push!(baseline_samples, timed_run!(backend, run_baseline!, baseline_state, p, steps))
        end
        last_candidate = candidate_state
    end

    trusted_state, _ = make_states(initial_u, initial_v, backend, ka_device)
    run_baseline!(trusted_state, p, steps)
    synchronize_run(backend, trusted_state)
    actual_u, actual_v = host_fields(candidate_fields(last_candidate))
    expected_u, expected_v = host_fields(baseline_fields(trusted_state))
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
        "timer" => timer_name(backend, last_candidate),
    )
end

function environment_metadata(backend, ka_device)
    meta = Dict{String,Any}(
        "hostname" => gethostname(),
        "julia_version" => string(VERSION),
        "backend" => backend,
        "julia_threads" => Threads.nthreads(),
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
        "cuda_visible_devices" => get(ENV, "CUDA_VISIBLE_DEVICES", ""),
    )
    if backend == "cpu"
        meta["timer"] = "cpu_ns"
    elseif backend == "cuda"
        meta["timer"] = "cuda_event"
    else
        meta["timer"] = ka_device !== nothing && nameof(typeof(ka_device)) == :CUDABackend ?
                        "cuda_event" : "cpu_ns"
        meta["ka_device"] = string(typeof(ka_device))
    end
    if meta["timer"] == "cuda_event"
        cuda = getfield(Main, :CUDA)
        meta["cuda_version"] = string(cuda.runtime_version())
        meta["gpu_name"] = cuda.name(cuda.device())
        meta["gpu_capability"] = string(cuda.capability(cuda.device()))
    end
    return meta
end

function main()
    options = OPTIONS
    backend = BACKEND
    language = get(options, "language", "julia-$backend")
    device = get(options, "device", "auto")
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
    evaluator_version = "gray-scott-julia-$(backend)-v1"

    result = try
        if backend == "ka"
            getfield(Main, :CUDA).functional() ||
                error("KernelAbstractions requires a CUDA GPU")
            ka_device = select_backend(device)
        else
            ka_device = nothing
        end
        if backend == "cuda"
            getfield(Main, :CUDA).functional() || error("CUDA is not functional on this host")
        end

        compile_started = time()
        _, compile_state = make_states(initialize_fields(dtype, (32, 32))..., backend, ka_device)
        run_candidate!(compile_state, p, 1)
        synchronize_run(backend, compile_state)
        compile_seconds = time() - compile_started

        correct, correctness_cases =
            candidate_correctness(backend, ka_device, dtype, correctness_shapes, p, atol, rtol)
        workloads = correct ? [
            benchmark_workload(
                backend, ka_device, dtype, shape, p, steps, warmups, repetitions, atol, rtol,
            )
            for shape in performance_shapes
        ] : []
        all_correct = correct && all(workload["correct"] for workload in workloads)
        speedup = all_correct && !isempty(workloads) ?
                  exp(mean(log(workload["speedup"]) for workload in workloads)) : nothing
        Dict(
            "schema_version" => "2.1",
            "candidate_id" => candidate_id,
            "language" => language,
            "backend" => backend,
            "evaluator_version" => evaluator_version,
            "status" => all_correct ? "passed" : "failed",
            "correct" => all_correct,
            "correctness_cases" => correctness_cases,
            "workloads" => workloads,
            "geometric_mean_speedup" => speedup,
            "compile_seconds" => compile_seconds,
            "end_to_end_seconds" => time() - started,
            "environment" => environment_metadata(backend, ka_device),
            "error" => nothing,
            "created_at" => string(Dates.now(Dates.UTC)),
        )
    catch exception
        Dict(
            "schema_version" => "2.1",
            "candidate_id" => candidate_id,
            "language" => language,
            "backend" => backend,
            "evaluator_version" => evaluator_version,
            "status" => "error",
            "correct" => false,
            "workloads" => [],
            "geometric_mean_speedup" => nothing,
            "compile_seconds" => nothing,
            "end_to_end_seconds" => time() - started,
            "environment" => Dict{String,Any}(
                "backend" => backend,
                "julia_threads" => Threads.nthreads(),
            ),
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

main()
