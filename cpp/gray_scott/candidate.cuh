#pragma once

#include "common.cuh"

namespace candidate {

// This is the only CUDA C++ file coding agents may edit.
__global__ void step_kernel(
    float* u_next,
    float* v_next,
    const float* u,
    const float* v,
    int height,
    int width,
    GrayScottParams params
) {
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y <= 0 || y >= height - 1 || x <= 0 || x >= width - 1) {
        return;
    }
    const int index = y * width + x;
    const float u_center = u[index];
    const float v_center = v[index];
    const float lap_u =
        u[index - width] + u[index + width] + u[index - 1] + u[index + 1] - 4.0f * u_center;
    const float lap_v =
        v[index - width] + v[index + width] + v[index - 1] + v[index + 1] - 4.0f * v_center;
    const float reaction = u_center * v_center * v_center;
    u_next[index] = u_center + params.dt *
        (params.diffusion_u * lap_u - reaction + params.feed * (1.0f - u_center));
    v_next[index] = v_center + params.dt *
        (params.diffusion_v * lap_v + reaction - (params.feed + params.kill) * v_center);
}

inline DeviceFields prepare_candidate(
    const std::vector<float>& u,
    const std::vector<float>& v,
    int height,
    int width
) {
    return prepare_fields(u, v, height, width);
}

inline void run_candidate(DeviceFields& state, GrayScottParams params, int steps) {
    const dim3 threads(16, 16);
    const dim3 blocks(
        (state.width + threads.x - 1) / threads.x,
        (state.height + threads.y - 1) / threads.y
    );
    for (int step = 0; step < steps; ++step) {
        step_kernel<<<blocks, threads>>>(
            state.u_next,
            state.v_next,
            state.u,
            state.v,
            state.height,
            state.width,
            params
        );
        cuda_check(cudaGetLastError(), "candidate kernel launch");
        std::swap(state.u, state.u_next);
        std::swap(state.v, state.v_next);
    }
}

inline std::pair<std::vector<float>, std::vector<float>> copy_candidate_fields(
    const DeviceFields& state
) {
    return copy_fields(state);
}

}  // namespace candidate
