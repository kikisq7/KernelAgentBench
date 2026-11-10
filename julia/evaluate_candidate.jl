#!/usr/bin/env julia

using Dates
using JSON3
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "kernels", "gray_scott", "notebook.jl"))
using .GrayScottNotebook

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
# Small grids are checked after each of these step counts; timed grids after `steps`.
const CORRECTNESS_STEPS = (1, 5, 50)

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

host_fields(fields) = (Array(fields[1]), Array(fields[2]))

boundary_fixed(actual, initial) =
    actual[[1, end], :] == initial[[1, end], :] && actual[:, [1, end]] == initial[:, [1, end]]

function matches(actual_u, actual_v, expected_u, expected_v, initial_u, initial_v, atol, rtol)
    return eltype(actual_u) == Float64 && eltype(actual_v) == Float64 &&
           size(actual_u) == size(expected_u) && size(actual_v) == size(expected_v) &&
           all(isfinite, actual_u) && all(isfinite, actual_v) &&
           isapprox(actual_u, expected_u; atol=atol, rtol=rtol) &&
           isapprox(actual_v, expected_v; atol=atol, rtol=rtol) &&
           boundary_fixed(actual_u, initial_u) && boundary_fixed(actual_v, initial_v)
end

# Fresh copies so a candidate that aliases its inputs cannot mutate the shared initial fields.
function make_states(initial_u, initial_v, backend, ka_device)
    initial_u, initial_v = copy(initial_u), copy(initial_v)
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

function run_both(initial_u, initial_v, backend, ka_device, p, steps)
    baseline_state, candidate_state = make_states(initial_u, initial_v, backend, ka_device)
    run_baseline!(baseline_state, p, steps)
    synchronize_run(backend, baseline_state)
    run_candidate!(candidate_state, p, steps)
    synchronize_run(backend, candidate_state)
    return host_fields(baseline_fields(baseline_state)),
           host_fields(candidate_fields(candidate_state))
end

function candidate_correctness(backend, ka_device, shapes, p, atol, rtol)
    cases = []
    for shape in shapes
        initial_u, initial_v = initialize_fields(shape...)
        for steps in CORRECTNESS_STEPS
            (expected_u, expected_v), (actual_u, actual_v) =
                run_both(initial_u, initial_v, backend, ka_device, p, steps)
            max_abs, relative = errors(actual_u, actual_v, expected_u, expected_v)
            correct = matches(
                actual_u, actual_v, expected_u, expected_v, initial_u, initial_v, atol, rtol,
            )
            push!(cases, Dict(
                "dtype" => "Float64",
                "shape" => collect(shape),
                "steps" => steps,
                "correct" => correct,
                "atol" => atol,
                "rtol" => rtol,
                "max_abs_error" => max_abs,
                "relative_l2_error" => relative,
            ))
        end
    end
    return all(case["correct"] for case in cases), cases
end

function benchmark_workload(backend, ka_device, shape, p, steps, warmups, repetitions, atol, rtol)
    initial_u, initial_v = initialize_fields(shape...)
    warmup_steps = min(steps, 10)

    for _ in 1:warmups
        baseline_state, candidate_state = make_states(initial_u, initial_v, backend, ka_device)
        timed_run!(backend, run_baseline!, baseline_state, p, warmup_steps)
        timed_run!(backend, run_candidate!, candidate_state, p, warmup_steps)
    end

    baseline_samples = Float64[]
    candidate_samples = Float64[]
    last_baseline = nothing
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
        last_baseline = baseline_state
        last_candidate = candidate_state
    end

    synchronize_run(backend, last_baseline)
    synchronize_run(backend, last_candidate)
    actual_u, actual_v = host_fields(candidate_fields(last_candidate))
    expected_u, expected_v = host_fields(baseline_fields(last_baseline))
    max_abs, relative = errors(actual_u, actual_v, expected_u, expected_v)
    correct = matches(
        actual_u, actual_v, expected_u, expected_v, initial_u, initial_v, atol, rtol,
    )

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
    warmups = parse(Int, get(options, "warmups", "1"))
    repetitions = parse(Int, get(options, "repetitions", "3"))
    atol = parse(Float64, get(options, "atol", "1e-10"))
    rtol = parse(Float64, get(options, "rtol", "1e-8"))
    threads = parse(Int, get(options, "threads", "0"))
    p = GrayScottParams()
    started = time()
    evaluator_version = "gray-scott-notebook-julia-$(backend)-v3"

    result = try
        get(options, "dtype", "Float64") == "Float64" ||
            error("the notebook baseline is Float64 only")
        threads == 0 || Threads.nthreads() == threads ||
            error("expected $(threads) Julia threads, got $(Threads.nthreads()); start julia with --threads=$(threads)")
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
        run_both(initialize_fields(32, 32)..., backend, ka_device, p, 1)
        compile_seconds = time() - compile_started

        correct, correctness_cases =
            candidate_correctness(backend, ka_device, correctness_shapes, p, atol, rtol)
        workloads = correct ? [
            benchmark_workload(backend, ka_device, shape, p, steps, warmups, repetitions, atol, rtol)
            for shape in performance_shapes
        ] : []
        all_correct = correct && all(workload["correct"] for workload in workloads)
        speedup = all_correct && !isempty(workloads) ?
                  exp(mean(log(workload["speedup"]) for workload in workloads)) : nothing
        Dict(
            "schema_version" => "3.0",
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
            "schema_version" => "3.0",
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
