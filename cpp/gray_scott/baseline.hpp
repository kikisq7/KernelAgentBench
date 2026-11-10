#pragma once

// Host side of the OpenCL baseline: the notebook's simulate_gs_cuda loop with
// `@cuda threads=(16, 16) blocks=(cld(n, 16), cld(n, 16))`, one launch per kernel.

#include "opencl_common.hpp"

namespace gray_scott_opencl_baseline {

struct State {
    int nx = 0;
    int ny = 0;
    cl_mem U = nullptr;
    cl_mem V = nullptr;
    cl_mem DU = nullptr;
    cl_mem DV = nullptr;
};

struct Kernels {
    cl_program program;
    cl_kernel calculate;
    cl_kernel update;
};

inline Kernels& kernels() {
    static Kernels compiled = [] {
        cl_program program = build_program_from_file(opencl_kernel_path("baseline_kernel.cl"));
        return Kernels{program, create_kernel(program, "calculate_gs"), create_kernel(program, "update_gs")};
    }();
    return compiled;
}

inline State prepare(const GrayScottFields& fields) {
    require_fp64_device();
    // Transfer to the GPU
    const std::size_t count = fields.U.size();
    return State{fields.nx, fields.ny, device_copy(fields.U), device_copy(fields.V),
                 device_alloc(count), device_alloc(count)};
}

// Returns once every launch has finished.
inline void run(State& state, const GrayScottParams& p, int steps) {
    Kernels& k = kernels();
    cl_command_queue queue = OpenCLRuntime::instance().queue;
    const std::size_t local[2] = {16, 16};
    const std::size_t global[2] = {
        16 * ((static_cast<std::size_t>(state.nx) + 15) / 16),
        16 * ((static_cast<std::size_t>(state.ny) + 15) / 16),
    };
    set_arg(k.calculate, 0, state.DU);
    set_arg(k.calculate, 1, state.DV);
    set_arg(k.calculate, 2, state.U);
    set_arg(k.calculate, 3, state.V);
    set_arg(k.calculate, 4, p.D_u);
    set_arg(k.calculate, 5, p.D_v);
    set_arg(k.calculate, 6, p.F);
    set_arg(k.calculate, 7, p.k);
    set_arg(k.calculate, 8, p.dx);
    set_arg(k.calculate, 9, state.nx);
    set_arg(k.calculate, 10, state.ny);
    set_arg(k.update, 0, state.U);
    set_arg(k.update, 1, state.V);
    set_arg(k.update, 2, state.DU);
    set_arg(k.update, 3, state.DV);
    set_arg(k.update, 4, p.dt);
    set_arg(k.update, 5, state.nx);
    set_arg(k.update, 6, state.ny);
    for (int step = 0; step < steps; ++step) {
        cl_check(clEnqueueNDRangeKernel(queue, k.calculate, 2, nullptr, global, local, 0, nullptr, nullptr),
                 "enqueue calculate_gs");
        cl_check(clEnqueueNDRangeKernel(queue, k.update, 2, nullptr, global, local, 0, nullptr, nullptr),
                 "enqueue update_gs");
    }
    cl_check(clFinish(queue), "clFinish");
}

inline GrayScottFields fields(const State& state) {
    const std::size_t count = static_cast<std::size_t>(state.nx) * static_cast<std::size_t>(state.ny);
    return GrayScottFields{state.nx, state.ny, host_copy(state.U, count), host_copy(state.V, count)};
}

inline void release(State& state) {
    release_buffer(state.U);
    release_buffer(state.V);
    release_buffer(state.DU);
    release_buffer(state.DV);
}

}  // namespace gray_scott_opencl_baseline
