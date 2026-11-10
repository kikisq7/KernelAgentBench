#pragma once

#include "../notebook.hpp"

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

// Host state and the thread pool shared by the C++ CPU baseline, candidate, and
// evaluator. The timed code lives in baseline.hpp and candidate.hpp.

using CpuGrayScottState = GrayScottFields;

// Worker count for every parallel loop. The evaluator sets it from --threads
// before the first loop runs; 0 means std::thread::hardware_concurrency().
inline int& cpu_thread_count() {
    static int count = 0;
    return count;
}

// Persistent worker team: the C++ counterpart of Julia's thread pool behind
// Threads.@threads. Threads live for the whole process, so a parallel loop pays
// a wake-up and a join barrier, not thread creation.
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

    int size() const { return size_; }

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
        const int count = end_ - begin_;
        const int per = (count + size_ - 1) / size_;
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
        if (cpu_thread_count() > 0) {
            return cpu_thread_count();
        }
        const unsigned hardware = std::thread::hardware_concurrency();
        return hardware == 0 ? 1 : static_cast<int>(hardware);
    }());
    return team;
}

// `Threads.@threads for i in begin:(end-1)`: the range is split into one
// contiguous chunk per worker and body(i) runs for every i.
template <typename Body>
inline void threads_for(int begin, int end, Body&& body) {
    if (end <= begin) {
        return;
    }
    const std::function<void(int, int)> chunk = [&body](int start, int stop) {
        for (int i = start; i < stop; ++i) {
            body(i);
        }
    };
    cpu_thread_team().run(begin, end, chunk);
}
