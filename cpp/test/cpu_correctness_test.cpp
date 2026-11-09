#include "baseline.hpp"
#include "candidate.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

template <typename T>
bool close_to_serial(const char* label, int height, int width, int steps) {
    const CpuGrayScottParams<T> params;
    const CpuGrayScottState<T> initial = cpu_initialize_fields<T>(height, width);
    const CpuGrayScottState<T> expected = cpu_serial_simulate(initial, params, steps);
    CpuGrayScottState<T> baseline = initial;
    CpuGrayScottState<T> candidate = initial;
    gray_scott_cpu_baseline::run(baseline, params, steps);
    gray_scott_cpu_candidate::run(candidate, params, steps);
    const T atol = std::is_same_v<T, double> ? T(1e-12) : T(2e-5);
    const T rtol = std::is_same_v<T, double> ? T(1e-10) : T(2e-4);
    for (const auto& actual : {&baseline, &candidate}) {
        for (std::size_t index = 0; index < expected.u.size(); ++index) {
            for (const auto pair : {
                     std::pair<T, T>{actual->u[index], expected.u[index]},
                     std::pair<T, T>{actual->v[index], expected.v[index]},
                 }) {
                if (!std::isfinite(static_cast<double>(pair.first)) ||
                    std::abs(pair.first - pair.second) > atol + rtol * std::abs(pair.second)) {
                    std::cerr << label << " mismatch at " << height << "x" << width
                              << " steps " << steps << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

int main() {
    const int shapes[][2] = {{31, 29}, {64, 64}, {127, 65}};
    for (const auto& shape : shapes) {
        for (int steps : {1, 5}) {
            if (!close_to_serial<float>("float", shape[0], shape[1], steps) ||
                !close_to_serial<double>("double", shape[0], shape[1], steps)) {
                return 1;
            }
        }
    }
    std::cout << "C++ CPU baseline and candidate match the serial reference\n";
    return 0;
}
