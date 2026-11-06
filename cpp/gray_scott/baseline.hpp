#pragma once

#include "opencl_runner.hpp"

namespace baseline {

inline DeviceFields prepare_baseline(
    const std::vector<float>& u,
    const std::vector<float>& v,
    int height,
    int width
) {
    return prepare_fields(u, v, height, width);
}

inline DeviceFieldsF64 prepare_baseline_f64(
    const std::vector<double>& u,
    const std::vector<double>& v,
    int height,
    int width
) {
    return prepare_fields_f64(u, v, height, width);
}

inline void run_baseline(DeviceFields& state, GrayScottParams params, int steps) {
    baseline_runner().run_f32(state, params, steps);
}

inline void run_baseline_f64(DeviceFieldsF64& state, GrayScottParams params, int steps) {
    baseline_runner().run_f64(state, params, steps);
}

inline float timed_baseline(DeviceFields& state, GrayScottParams params, int steps) {
    return baseline_runner().timed_f32(state, params, steps);
}

inline std::pair<std::vector<float>, std::vector<float>> copy_baseline_fields(
    const DeviceFields& state
) {
    return copy_fields(state);
}

inline std::pair<std::vector<double>, std::vector<double>> copy_baseline_fields_f64(
    const DeviceFieldsF64& state
) {
    return copy_fields_f64(state);
}

}  // namespace baseline
