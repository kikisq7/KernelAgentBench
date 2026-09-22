#pragma once

#include <cuda_runtime.h>

#include <cmath>
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

inline void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceFields {
    float* u = nullptr;
    float* v = nullptr;
    float* u_next = nullptr;
    float* v_next = nullptr;
    int height = 0;
    int width = 0;
};

inline DeviceFields prepare_fields(
    const std::vector<float>& u,
    const std::vector<float>& v,
    int height,
    int width
) {
    DeviceFields state;
    state.height = height;
    state.width = width;
    const std::size_t bytes = u.size() * sizeof(float);
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&state.u), bytes), "cudaMalloc u");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&state.v), bytes), "cudaMalloc v");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&state.u_next), bytes), "cudaMalloc u_next");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&state.v_next), bytes), "cudaMalloc v_next");
    cuda_check(cudaMemcpy(state.u, u.data(), bytes, cudaMemcpyHostToDevice), "copy u");
    cuda_check(cudaMemcpy(state.v, v.data(), bytes, cudaMemcpyHostToDevice), "copy v");
    cuda_check(cudaMemcpy(state.u_next, u.data(), bytes, cudaMemcpyHostToDevice), "copy u_next");
    cuda_check(cudaMemcpy(state.v_next, v.data(), bytes, cudaMemcpyHostToDevice), "copy v_next");
    return state;
}

inline void release_fields(DeviceFields& state) {
    cudaFree(state.u);
    cudaFree(state.v);
    cudaFree(state.u_next);
    cudaFree(state.v_next);
    state = {};
}

inline std::pair<std::vector<float>, std::vector<float>> copy_fields(const DeviceFields& state) {
    const std::size_t count = static_cast<std::size_t>(state.height) * state.width;
    const std::size_t bytes = count * sizeof(float);
    std::vector<float> u(count), v(count);
    cuda_check(cudaMemcpy(u.data(), state.u, bytes, cudaMemcpyDeviceToHost), "copy result u");
    cuda_check(cudaMemcpy(v.data(), state.v, bytes, cudaMemcpyDeviceToHost), "copy result v");
    return {std::move(u), std::move(v)};
}
