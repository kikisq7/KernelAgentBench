#pragma once

#include "types.hpp"

// Trusted C++ CPU baseline. The candidate starts as a copy of this file.
// Parallelize across rows; the inner x loop stays contiguous in row-major layout.

namespace gray_scott_cpu_baseline {

template <typename T>
inline void run(CpuGrayScottState<T>& state, CpuGrayScottParams<T> params, int steps) {
    const int height = state.height;
    const int width = state.width;
    for (int step = 0; step < steps; ++step) {
        state.u_next = state.u;
        state.v_next = state.v;
        cpu_parallel_rows(1, height - 1, [&](int y_begin, int y_end) {
            for (int y = y_begin; y < y_end; ++y) {
                for (int x = 1; x < width - 1; ++x) {
                    const std::size_t index = static_cast<std::size_t>(y * width + x);
                    const T u_center = state.u[index];
                    const T v_center = state.v[index];
                    const T lap_u = state.u[index - static_cast<std::size_t>(width)] +
                        state.u[index + static_cast<std::size_t>(width)] +
                        state.u[index - 1] + state.u[index + 1] - T(4) * u_center;
                    const T lap_v = state.v[index - static_cast<std::size_t>(width)] +
                        state.v[index + static_cast<std::size_t>(width)] +
                        state.v[index - 1] + state.v[index + 1] - T(4) * v_center;
                    const T reaction = u_center * v_center * v_center;
                    state.u_next[index] = u_center + params.dt * (
                        params.diffusion_u * lap_u - reaction + params.feed * (T(1) - u_center)
                    );
                    state.v_next[index] = v_center + params.dt * (
                        params.diffusion_v * lap_v + reaction -
                        (params.feed + params.kill) * v_center
                    );
                }
            }
        });
        state.u.swap(state.u_next);
        state.v.swap(state.v_next);
    }
}

}  // namespace gray_scott_cpu_baseline
