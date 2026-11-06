#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct GrayScottParams {
    float diffusion_u = 0.16f;
    float diffusion_v = 0.08f;
    float feed = 0.035f;
    float kill = 0.065f;
    float dt = 1.0f;
};

inline void cl_check(cl_int status, const char* operation) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with OpenCL error " +
                                 std::to_string(status));
    }
}

inline std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to read " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

struct OpenCLRuntime {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;

    static OpenCLRuntime& instance() {
        static OpenCLRuntime runtime;
        runtime.ensure_initialized();
        return runtime;
    }

    void ensure_initialized() {
        if (context != nullptr) {
            return;
        }
        cl_uint platform_count = 0;
        cl_check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs count");
        if (platform_count == 0) {
            throw std::runtime_error("no OpenCL platforms found");
        }
        std::vector<cl_platform_id> platforms(platform_count);
        cl_check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");
        platform = platforms.front();

        cl_uint device_count = 0;
        cl_int device_status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        if (device_status != CL_SUCCESS || device_count == 0) {
            throw std::runtime_error("no OpenCL GPU devices found");
        }
        std::vector<cl_device_id> devices(device_count);
        cl_check(
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr),
            "clGetDeviceIDs"
        );
        device = devices.front();

        cl_int context_error = 0;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &context_error);
        cl_check(context_error, "clCreateContext");
        cl_int queue_error = 0;
        queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &queue_error);
        cl_check(queue_error, "clCreateCommandQueue");
    }

    std::string device_name() const {
        char name[512] = {};
        cl_check(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr), "device name");
        return name;
    }

    std::string platform_version() const {
        char version[512] = {};
        cl_check(
            clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(version), version, nullptr),
            "platform version"
        );
        return version;
    }
};

struct DeviceFields {
    cl_mem u = nullptr;
    cl_mem v = nullptr;
    cl_mem u_next = nullptr;
    cl_mem v_next = nullptr;
    int height = 0;
    int width = 0;
};

struct DeviceFieldsF64 {
    cl_mem u = nullptr;
    cl_mem v = nullptr;
    cl_mem u_next = nullptr;
    cl_mem v_next = nullptr;
    int height = 0;
    int width = 0;
};

inline bool device_supports_fp64() {
    OpenCLRuntime& runtime = OpenCLRuntime::instance();
    cl_ulong config = 0;
    if (clGetDeviceInfo(
            runtime.device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(config), &config, nullptr
        ) != CL_SUCCESS) {
        return false;
    }
    return config != 0;
}

inline void require_fp64_device() {
    if (!device_supports_fp64()) {
        throw std::runtime_error("OpenCL device does not support double-precision kernels");
    }
}

inline DeviceFields prepare_fields(
    const std::vector<float>& u,
    const std::vector<float>& v,
    int height,
    int width
) {
    OpenCLRuntime& runtime = OpenCLRuntime::instance();
    DeviceFields state;
    state.height = height;
    state.width = width;
    const std::size_t bytes = u.size() * sizeof(float);
    cl_int error = 0;
    state.u = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE, bytes, nullptr, &error);
    cl_check(error, "clCreateBuffer u");
    state.v = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE, bytes, nullptr, &error);
    cl_check(error, "clCreateBuffer v");
    state.u_next = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE, bytes, nullptr, &error);
    cl_check(error, "clCreateBuffer u_next");
    state.v_next = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE, bytes, nullptr, &error);
    cl_check(error, "clCreateBuffer v_next");
    cl_check(
        clEnqueueWriteBuffer(runtime.queue, state.u, CL_TRUE, 0, bytes, u.data(), 0, nullptr, nullptr),
        "write u"
    );
    cl_check(
        clEnqueueWriteBuffer(runtime.queue, state.v, CL_TRUE, 0, bytes, v.data(), 0, nullptr, nullptr),
        "write v"
    );
    cl_check(
        clEnqueueWriteBuffer(
            runtime.queue, state.u_next, CL_TRUE, 0, bytes, u.data(), 0, nullptr, nullptr
        ),
        "write u_next"
    );
    cl_check(
        clEnqueueWriteBuffer(
            runtime.queue, state.v_next, CL_TRUE, 0, bytes, v.data(), 0, nullptr, nullptr
        ),
        "write v_next"
    );
    return state;
}

inline void release_fields(DeviceFields& state) {
    if (state.u) clReleaseMemObject(state.u);
    if (state.v) clReleaseMemObject(state.v);
    if (state.u_next) clReleaseMemObject(state.u_next);
    if (state.v_next) clReleaseMemObject(state.v_next);
    state = {};
}

inline DeviceFieldsF64 prepare_fields_f64(
    const std::vector<double>& u,
    const std::vector<double>& v,
    int height,
    int width
) {
    require_fp64_device();
    OpenCLRuntime& runtime = OpenCLRuntime::instance();
    DeviceFieldsF64 state;
    state.height = height;
    state.width = width;
    const std::size_t bytes = u.size() * sizeof(double);
    cl_int error = 0;
    state.u = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE, bytes, nullptr, &error);
    cl_check(error, "clCreateBuffer u f64");
    state.v = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE, bytes, nullptr, &error);
    cl_check(error, "clCreateBuffer v f64");
    state.u_next = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE, bytes, nullptr, &error);
    cl_check(error, "clCreateBuffer u_next f64");
    state.v_next = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE, bytes, nullptr, &error);
    cl_check(error, "clCreateBuffer v_next f64");
    cl_check(
        clEnqueueWriteBuffer(runtime.queue, state.u, CL_TRUE, 0, bytes, u.data(), 0, nullptr, nullptr),
        "write u f64"
    );
    cl_check(
        clEnqueueWriteBuffer(runtime.queue, state.v, CL_TRUE, 0, bytes, v.data(), 0, nullptr, nullptr),
        "write v f64"
    );
    cl_check(
        clEnqueueWriteBuffer(
            runtime.queue, state.u_next, CL_TRUE, 0, bytes, u.data(), 0, nullptr, nullptr
        ),
        "write u_next f64"
    );
    cl_check(
        clEnqueueWriteBuffer(
            runtime.queue, state.v_next, CL_TRUE, 0, bytes, v.data(), 0, nullptr, nullptr
        ),
        "write v_next f64"
    );
    return state;
}

inline void release_fields_f64(DeviceFieldsF64& state) {
    if (state.u) clReleaseMemObject(state.u);
    if (state.v) clReleaseMemObject(state.v);
    if (state.u_next) clReleaseMemObject(state.u_next);
    if (state.v_next) clReleaseMemObject(state.v_next);
    state = {};
}

inline std::pair<std::vector<double>, std::vector<double>> copy_fields_f64(const DeviceFieldsF64& state) {
    OpenCLRuntime& runtime = OpenCLRuntime::instance();
    const std::size_t count = static_cast<std::size_t>(state.height) * state.width;
    const std::size_t bytes = count * sizeof(double);
    std::vector<double> u(count), v(count);
    cl_check(
        clEnqueueReadBuffer(runtime.queue, state.u, CL_TRUE, 0, bytes, u.data(), 0, nullptr, nullptr),
        "read u f64"
    );
    cl_check(
        clEnqueueReadBuffer(runtime.queue, state.v, CL_TRUE, 0, bytes, v.data(), 0, nullptr, nullptr),
        "read v f64"
    );
    return {std::move(u), std::move(v)};
}

inline std::pair<std::vector<float>, std::vector<float>> copy_fields(const DeviceFields& state) {
    OpenCLRuntime& runtime = OpenCLRuntime::instance();
    const std::size_t count = static_cast<std::size_t>(state.height) * state.width;
    const std::size_t bytes = count * sizeof(float);
    std::vector<float> u(count), v(count);
    cl_check(
        clEnqueueReadBuffer(runtime.queue, state.u, CL_TRUE, 0, bytes, u.data(), 0, nullptr, nullptr),
        "read u"
    );
    cl_check(
        clEnqueueReadBuffer(runtime.queue, state.v, CL_TRUE, 0, bytes, v.data(), 0, nullptr, nullptr),
        "read v"
    );
    return {std::move(u), std::move(v)};
}

inline cl_program build_program_from_source(const std::string& source) {
    OpenCLRuntime& runtime = OpenCLRuntime::instance();
    const char* sources[] = {source.c_str()};
    const std::size_t lengths[] = {source.size()};
    cl_int error = 0;
    cl_program program = clCreateProgramWithSource(runtime.context, 1, sources, lengths, &error);
    cl_check(error, "clCreateProgramWithSource");
    error = clBuildProgram(program, 1, &runtime.device, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (error != CL_SUCCESS) {
        std::size_t log_size = 0;
        clGetProgramBuildInfo(program, runtime.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(
            program, runtime.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr
        );
        std::string message(log.data());
        clReleaseProgram(program);
        throw std::runtime_error("OpenCL build failed:\n" + message);
    }
    return program;
}

inline float elapsed_ms(cl_event event) {
    cl_ulong start = 0;
    cl_ulong end = 0;
    cl_check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr),
             "profiling start");
    cl_check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr),
             "profiling end");
    return static_cast<float>((end - start) * 1e-6);
}
