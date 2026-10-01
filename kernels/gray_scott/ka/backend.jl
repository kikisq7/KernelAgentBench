module GrayScottKABackend

using KernelAbstractions
using CUDA

export select_backend, require_cuda_gpu, to_device, to_host, synchronize_backend

function require_cuda_gpu()
    CUDA.functional() || error("KernelAbstractions requires a CUDA GPU")
    return nothing
end

function select_backend(name::AbstractString="auto")
    requested = lowercase(name)
    if requested == "cpu"
        return CPU()
    elseif requested in ("cuda", "auto")
        require_cuda_gpu()
        return CUDABackend()
    elseif requested in ("rocm", "amd")
        error(
            "AMD/ROCm is not enabled in this Julia project yet. Keep the KernelAbstractions " *
            "kernel portable for ROCBackend; do not add CUDA-only intrinsics.",
        )
    end
    error("unknown KernelAbstractions device $(name)")
end

function to_device(backend, host::AbstractArray{T}) where {T}
    device = KernelAbstractions.allocate(backend, T, size(host)...)
    KernelAbstractions.copyto!(backend, device, host)
    KernelAbstractions.synchronize(backend)
    return device
end

to_host(array) = Array(array)

synchronize_backend(backend) = KernelAbstractions.synchronize(backend)

end
