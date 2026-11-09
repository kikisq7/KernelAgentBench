#pragma once

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
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

// Persistent worker team, the C++ counterpart of Polyester's thread pool in the
// Julia CPU baseline. Threads live for the whole process, so a timestep pays a
// wake-up and a join barrier, not thread creation.
class CpuThreadTeam {
public:
    explicit CpuThreadTeam(int size) : size_(std::max(1, size)) {
        for (int index = 1; index < size_; ++index) {
            threads_.emplace_back([this, index] { work(index); });
        }
    }

    ~CpuThreadTeam() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        wake_.notify_all();
        for (std::thread& thread : threads_) {
            thread.join();
        }
    }

    CpuThreadTeam(const CpuThreadTeam&) = delete;
    CpuThreadTeam& operator=(const CpuThreadTeam&) = delete;

    void run(int begin, int end, const std::function<void(int, int)>& body) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            body_ = &body;
            begin_ = begin;
            end_ = end;
            pending_ = size_ - 1;
            ++generation_;
        }
        wake_.notify_all();
        chunk(0);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return pending_ == 0; });
        body_ = nullptr;
    }

private:
    void chunk(int index) {
        const int rows = end_ - begin_;
        const int per = (rows + size_ - 1) / size_;
        const int start = begin_ + index * per;
        const int stop = std::min(end_, start + per);
        if (start < stop) {
            (*body_)(start, stop);
        }
    }

    void work(int index) {
        unsigned long seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return generation_ != seen; });
                seen = generation_;
                if (stop_) {
                    return;
                }
            }
            chunk(index);
            std::lock_guard<std::mutex> lock(mutex_);
            if (--pending_ == 0) {
                done_.notify_one();
            }
        }
    }

    int size_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    const std::function<void(int, int)>* body_ = nullptr;
    int begin_ = 0;
    int end_ = 0;
    int pending_ = 0;
    unsigned long generation_ = 0;
    bool stop_ = false;
};

inline CpuThreadTeam& cpu_thread_team() {
    static CpuThreadTeam team([] {
        const unsigned hardware = std::thread::hardware_concurrency();
        return hardware == 0 ? 1 : static_cast<int>(hardware);
    }());
    return team;
}

template <typename Fn>
inline void cpu_parallel_rows(int y_begin, int y_end, Fn&& body) {
    if (y_end <= y_begin) {
        return;
    }
    const std::function<void(int, int)> rows = [&body](int start, int stop) { body(start, stop); };
    cpu_thread_team().run(y_begin, y_end, rows);
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
