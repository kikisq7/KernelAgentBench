#pragma once

#include "opencl_common.hpp"

#include <filesystem>
#include <mutex>
#include <utility>

inline std::filesystem::path kernel_directory() {
    static const std::filesystem::path directory =
        std::filesystem::path(__FILE__).parent_path();
    return directory;
}

struct OpenCLKernelRunner {
    cl_program program = nullptr;
    cl_kernel kernel_f32 = nullptr;
    cl_kernel kernel_f64 = nullptr;

    explicit OpenCLKernelRunner(const std::filesystem::path& kernel_path) {
        const std::string source = read_text_file(kernel_path.string());
        program = build_program_from_source(source);
        cl_int error = 0;
        kernel_f32 = clCreateKernel(program, "step_kernel", &error);
        cl_check(error, "clCreateKernel step_kernel");
        kernel_f64 = clCreateKernel(program, "step_kernel_f64", &error);
        if (error != CL_SUCCESS) {
            kernel_f64 = nullptr;
        }
    }

    bool has_f64() const { return kernel_f64 != nullptr; }

    ~OpenCLKernelRunner() {
        if (kernel_f32) clReleaseKernel(kernel_f32);
        if (kernel_f64) clReleaseKernel(kernel_f64);
        if (program) clReleaseProgram(program);
    }

    OpenCLKernelRunner(const OpenCLKernelRunner&) = delete;
    OpenCLKernelRunner& operator=(const OpenCLKernelRunner&) = delete;

    void enqueue_f32(DeviceFields& state, GrayScottParams params, cl_event* event) {
        OpenCLRuntime& runtime = OpenCLRuntime::instance();
        const std::size_t local_x = 16;
        const std::size_t local_y = 16;
        const std::size_t global_x =
            ((static_cast<std::size_t>(state.width) + local_x - 1) / local_x) * local_x;
        const std::size_t global_y =
            ((static_cast<std::size_t>(state.height) + local_y - 1) / local_y) * local_y;
        const std::size_t global[2] = {global_x, global_y};
        const std::size_t local[2] = {local_x, local_y};

        cl_int error = 0;
        error |= clSetKernelArg(kernel_f32, 0, sizeof(cl_mem), &state.u_next);
        error |= clSetKernelArg(kernel_f32, 1, sizeof(cl_mem), &state.v_next);
        error |= clSetKernelArg(kernel_f32, 2, sizeof(cl_mem), &state.u);
        error |= clSetKernelArg(kernel_f32, 3, sizeof(cl_mem), &state.v);
        error |= clSetKernelArg(kernel_f32, 4, sizeof(int), &state.height);
        error |= clSetKernelArg(kernel_f32, 5, sizeof(int), &state.width);
        error |= clSetKernelArg(kernel_f32, 6, sizeof(float), &params.diffusion_u);
        error |= clSetKernelArg(kernel_f32, 7, sizeof(float), &params.diffusion_v);
        error |= clSetKernelArg(kernel_f32, 8, sizeof(float), &params.feed);
        error |= clSetKernelArg(kernel_f32, 9, sizeof(float), &params.kill);
        error |= clSetKernelArg(kernel_f32, 10, sizeof(float), &params.dt);
        cl_check(error, "clSetKernelArg f32");
        cl_check(
            clEnqueueNDRangeKernel(
                runtime.queue, kernel_f32, 2, nullptr, global, local, 0, nullptr, event
            ),
            "clEnqueueNDRangeKernel f32"
        );
        std::swap(state.u, state.u_next);
        std::swap(state.v, state.v_next);
    }

    void enqueue_f64(DeviceFieldsF64& state, GrayScottParams params, cl_event* event) {
        OpenCLRuntime& runtime = OpenCLRuntime::instance();
        const std::size_t local_x = 16;
        const std::size_t local_y = 16;
        const std::size_t global_x =
            ((static_cast<std::size_t>(state.width) + local_x - 1) / local_x) * local_x;
        const std::size_t global_y =
            ((static_cast<std::size_t>(state.height) + local_y - 1) / local_y) * local_y;
        const std::size_t global[2] = {global_x, global_y};
        const std::size_t local[2] = {local_x, local_y};
        const double diffusion_u = params.diffusion_u;
        const double diffusion_v = params.diffusion_v;
        const double feed = params.feed;
        const double kill = params.kill;
        const double dt = params.dt;

        cl_int error = 0;
        error |= clSetKernelArg(kernel_f64, 0, sizeof(cl_mem), &state.u_next);
        error |= clSetKernelArg(kernel_f64, 1, sizeof(cl_mem), &state.v_next);
        error |= clSetKernelArg(kernel_f64, 2, sizeof(cl_mem), &state.u);
        error |= clSetKernelArg(kernel_f64, 3, sizeof(cl_mem), &state.v);
        error |= clSetKernelArg(kernel_f64, 4, sizeof(int), &state.height);
        error |= clSetKernelArg(kernel_f64, 5, sizeof(int), &state.width);
        error |= clSetKernelArg(kernel_f64, 6, sizeof(double), &diffusion_u);
        error |= clSetKernelArg(kernel_f64, 7, sizeof(double), &diffusion_v);
        error |= clSetKernelArg(kernel_f64, 8, sizeof(double), &feed);
        error |= clSetKernelArg(kernel_f64, 9, sizeof(double), &kill);
        error |= clSetKernelArg(kernel_f64, 10, sizeof(double), &dt);
        cl_check(error, "clSetKernelArg f64");
        cl_check(
            clEnqueueNDRangeKernel(
                runtime.queue, kernel_f64, 2, nullptr, global, local, 0, nullptr, event
            ),
            "clEnqueueNDRangeKernel f64"
        );
        std::swap(state.u, state.u_next);
        std::swap(state.v, state.v_next);
    }

    void run_f32(DeviceFields& state, GrayScottParams params, int steps) {
        for (int step = 0; step < steps; ++step) {
            cl_event event = nullptr;
            enqueue_f32(state, params, &event);
            clWaitForEvents(1, &event);
            clReleaseEvent(event);
        }
        clFinish(OpenCLRuntime::instance().queue);
    }

    void run_f64(DeviceFieldsF64& state, GrayScottParams params, int steps) {
        if (kernel_f64 == nullptr) {
            require_fp64_device();
            throw std::runtime_error("step_kernel_f64 is unavailable on this OpenCL device");
        }
        for (int step = 0; step < steps; ++step) {
            cl_event event = nullptr;
            enqueue_f64(state, params, &event);
            clWaitForEvents(1, &event);
            clReleaseEvent(event);
        }
        clFinish(OpenCLRuntime::instance().queue);
    }

    float timed_f32(DeviceFields& state, GrayScottParams params, int steps) {
        cl_event start_event = nullptr;
        cl_event end_event = nullptr;
        for (int step = 0; step < steps; ++step) {
            cl_event event = nullptr;
            enqueue_f32(state, params, &event);
            if (step == 0) {
                start_event = event;
            } else if (step == steps - 1) {
                end_event = event;
            } else {
                clReleaseEvent(event);
            }
        }
        if (steps == 1) {
            end_event = start_event;
        }
        clWaitForEvents(1, &end_event);
        clFinish(OpenCLRuntime::instance().queue);
        const float milliseconds = elapsed_ms(end_event);
        if (start_event != end_event && start_event) {
            clReleaseEvent(start_event);
        }
        if (end_event) {
            clReleaseEvent(end_event);
        }
        return milliseconds;
    }
};

inline OpenCLKernelRunner& baseline_runner() {
    static std::once_flag once;
    static OpenCLKernelRunner* runner = nullptr;
    std::call_once(once, [] {
        runner = new OpenCLKernelRunner(kernel_directory() / "baseline_kernel.cl");
    });
    return *runner;
}

inline OpenCLKernelRunner& candidate_runner() {
    static std::once_flag once;
    static OpenCLKernelRunner* runner = nullptr;
    std::call_once(once, [] {
        runner = new OpenCLKernelRunner(kernel_directory() / "candidate_kernel.cl");
    });
    return *runner;
}
