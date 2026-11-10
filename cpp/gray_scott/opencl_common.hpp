#pragma once

// OpenCL plumbing shared by the baseline, the candidate, and the evaluator.

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "notebook.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

inline void cl_check(cl_int status, const char* operation) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with OpenCL error " +
                                 std::to_string(status));
    }
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

        cl_int error = 0;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
        cl_check(error, "clCreateContext");
        queue = clCreateCommandQueue(context, device, 0, &error);
        cl_check(error, "clCreateCommandQueue");
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

inline void require_fp64_device() {
    cl_device_fp_config config = 0;
    const cl_int status = clGetDeviceInfo(
        OpenCLRuntime::instance().device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(config), &config, nullptr
    );
    if (status != CL_SUCCESS || config == 0) {
        throw std::runtime_error("no OpenCL GPU with double precision on this host");
    }
}

// Kernel sources sit next to this header.
inline std::filesystem::path opencl_kernel_path(const std::string& name) {
    return std::filesystem::path(__FILE__).parent_path() / name;
}

inline cl_program build_program_from_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to read " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string source = buffer.str();
    OpenCLRuntime& runtime = OpenCLRuntime::instance();
    const char* sources[] = {source.c_str()};
    const std::size_t lengths[] = {source.size()};
    cl_int error = 0;
    cl_program program = clCreateProgramWithSource(runtime.context, 1, sources, lengths, &error);
    cl_check(error, "clCreateProgramWithSource");
    error = clBuildProgram(program, 1, &runtime.device, "", nullptr, nullptr);
    if (error != CL_SUCCESS) {
        std::size_t log_size = 0;
        clGetProgramBuildInfo(program, runtime.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, '\0');
        clGetProgramBuildInfo(program, runtime.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        clReleaseProgram(program);
        throw std::runtime_error("OpenCL build failed for " + path.string() + ":\n" + log.data());
    }
    return program;
}

inline cl_kernel create_kernel(cl_program program, const char* name) {
    cl_int error = 0;
    cl_kernel kernel = clCreateKernel(program, name, &error);
    cl_check(error, name);
    return kernel;
}

inline cl_mem device_copy(const std::vector<double>& host) {
    OpenCLRuntime& runtime = OpenCLRuntime::instance();
    cl_int error = 0;
    cl_mem buffer = clCreateBuffer(
        runtime.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        host.size() * sizeof(double), const_cast<double*>(host.data()), &error
    );
    cl_check(error, "clCreateBuffer");
    return buffer;
}

inline cl_mem device_alloc(std::size_t count) {
    cl_int error = 0;
    cl_mem buffer = clCreateBuffer(
        OpenCLRuntime::instance().context, CL_MEM_READ_WRITE, count * sizeof(double), nullptr, &error
    );
    cl_check(error, "clCreateBuffer");
    return buffer;
}

inline std::vector<double> host_copy(cl_mem buffer, std::size_t count) {
    std::vector<double> host(count);
    cl_check(
        clEnqueueReadBuffer(
            OpenCLRuntime::instance().queue, buffer, CL_TRUE, 0, count * sizeof(double),
            host.data(), 0, nullptr, nullptr
        ),
        "clEnqueueReadBuffer"
    );
    return host;
}

inline void release_buffer(cl_mem& buffer) {
    if (buffer != nullptr) {
        clReleaseMemObject(buffer);
        buffer = nullptr;
    }
}

template <typename T>
inline void set_arg(cl_kernel kernel, cl_uint index, const T& value) {
    cl_check(clSetKernelArg(kernel, index, sizeof(T), &value), "clSetKernelArg");
}
