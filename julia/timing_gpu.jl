function synchronize_run(backend, state)
    if backend == "cuda"
        CUDA.synchronize()
    elseif backend == "ka"
        synchronize_backend(state.backend)
    end
    return nothing
end

function timed_run!(backend, runner, state, p, steps)
    synchronize_run(backend, state)
    if uses_cuda_events(backend, state)
        seconds = CUDA.@elapsed runner(state, p, steps)
        return Float64(seconds * 1000)
    end
    started = time_ns()
    runner(state, p, steps)
    synchronize_run(backend, state)
    return Float64((time_ns() - started) / 1e6)
end
