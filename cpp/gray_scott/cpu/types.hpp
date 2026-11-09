#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

// Host Gray-Scott types shared by the C++ CPU baseline, candidate, and evaluator.
// The timed kernels live in baseline.hpp and candidate.hpp. This header is the
// serial reference and the row partition used by the initial parallel code.

template <typename T>
struct CpuGrayScottParams {
    T diffusion_u = T(0.16);
    T diffusion_v = T(0.08);
    T feed = T(0.035);
    T kill = T(0.065);
    T dt = T(1);
};

template <typename T>
struct CpuGrayScottState {
    int height = 0;
    int width = 0;
    std::vector<T> u;
    std::vector<T> v;
    std::vector<T> u_next;
    std::vector<T> v_next;
};

template <typename T>
inline CpuGrayScottState<T> cpu_initialize_fields(int height, int width, int seed = 20260918) {
    if (height < 5 || width < 5) {
        throw std::invalid_argument("shape must be at least 5x5");
    }
    const std::size_t count = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    CpuGrayScottState<T> state;
    state.height = height;
    state.width = width;
    state.u.assign(count, T(1));
    state.v.assign(count, T(0));
    state.u_next.resize(count);
    state.v_next.resize(count);
    const int radius_y = std::max(1, height / 10);
    const int radius_x = std::max(1, width / 10);
    const int center_y = (height - 1) / 2;
    const int center_x = (width - 1) / 2;
    for (int y = std::max(1, center_y - radius_y);
         y <= std::min(height - 2, center_y + radius_y);
         ++y) {
        for (int x = std::max(1, center_x - radius_x);
             x <= std::min(width - 2, center_x + radius_x);
             ++x) {
            const T noise = T(0.01) * std::sin(
                (T(y + 1) * T(73856093) + T(x + 1) * T(19349663) + T(seed)) * T(0.001)
            );
            const int index = y * width + x;
            state.u[static_cast<std::size_t>(index)] = T(0.5) + noise;
            state.v[static_cast<std::size_t>(index)] = T(0.25) - noise;
        }
    }
    return state;
}

template <typename Fn>
inline void cpu_parallel_rows(int y_begin, int y_end, Fn&& body) {
    const int rows = y_end - y_begin;
    if (rows <= 0) {
        return;
    }
    const unsigned hardware = std::thread::hardware_concurrency();
    int workers = hardware == 0 ? 1 : static_cast<int>(hardware);
    workers = std::max(1, std::min(workers, rows));
    if (workers == 1) {
        body(y_begin, y_end);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers));
    const int chunk = (rows + workers - 1) / workers;
    for (int worker = 0; worker < workers; ++worker) {
        const int start = y_begin + worker * chunk;
        const int stop = std::min(y_end, start + chunk);
        if (start >= stop) {
            break;
        }
        threads.emplace_back(body, start, stop);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
}

// Serial oracle. Correctness compares the parallel candidate with this, not
// with the timed baseline, so a shared bug in both parallel codes still fails.
template <typename T>
inline CpuGrayScottState<T> cpu_serial_simulate(
    CpuGrayScottState<T> state,
    CpuGrayScottParams<T> params,
    int steps
) {
    if (steps < 0) {
        throw std::invalid_argument("steps must be nonnegative");
    }
    const int height = state.height;
    const int width = state.width;
    for (int step = 0; step < steps; ++step) {
        state.u_next = state.u;
        state.v_next = state.v;
        for (int y = 1; y < height - 1; ++y) {
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
                    params.diffusion_v * lap_v + reaction - (params.feed + params.kill) * v_center
                );
            }
        }
        state.u.swap(state.u_next);
        state.v.swap(state.v_next);
    }
    return state;
}
