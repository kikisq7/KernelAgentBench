#include "../gray_scott/candidate.hpp"
#include "../gray_scott/opencl_common.hpp"
#include "../gray_scott/opencl_runner.hpp"
#include "../gray_scott/reference.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using FieldsF64 = std::pair<std::vector<double>, std::vector<double>>;

namespace {

constexpr int SHAPES[][2] = {{31, 29}, {64, 64}, {127, 65}};

bool close_f32(float a, float e, float atol, float rtol) {
    return std::isfinite(a) && std::abs(a - e) <= atol + rtol * std::abs(e);
}

bool close_f64(double a, double e, double atol, double rtol) {
    return std::isfinite(a) && std::abs(a - e) <= atol + rtol * std::abs(e);
}

bool boundary_fixed_f32(
    const std::vector<float>& actual,
    const std::vector<float>& initial,
    int height,
    int width
) {
    for (int x = 0; x < width; ++x) {
        if (actual[x] != initial[x] || actual[(height - 1) * width + x] != initial[(height - 1) * width + x]) {
            return false;
        }
    }
    for (int y = 0; y < height; ++y) {
        if (actual[y * width] != initial[y * width] ||
            actual[y * width + (width - 1)] != initial[y * width + (width - 1)]) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    using namespace gray_scott_reference;
    GrayScottParams params;
    constexpr float atol32 = 2e-5f;
    constexpr float rtol32 = 2e-4f;
    constexpr double atol64 = 1e-12;
    constexpr double rtol64 = 1e-10;

    for (const auto& shape : SHAPES) {
        const int height = shape[0];
        const int width = shape[1];
        for (int steps : {1, 5}) {
            const Fields initial = initialize_fields(height, width);
            const Fields expected = simulate(initial, height, width, Params{}, steps);
            DeviceFields state = candidate::prepare_candidate(
                initial.first, initial.second, height, width
            );
            candidate::run_candidate(state, params, steps);
            const auto actual_pair = candidate::copy_candidate_fields(state);
            release_fields(state);
            for (std::size_t index = 0; index < actual_pair.first.size(); ++index) {
                if (!close_f32(actual_pair.first[index], expected.first[index], atol32, rtol32) ||
                    !close_f32(actual_pair.second[index], expected.second[index], atol32, rtol32)) {
                    std::cerr << "Float32 mismatch at shape " << height << 'x' << width << " steps "
                              << steps << '\n';
                    return 1;
                }
            }
            if (!boundary_fixed_f32(actual_pair.first, initial.first, height, width) ||
                !boundary_fixed_f32(actual_pair.second, initial.second, height, width)) {
                std::cerr << "Float32 boundary violation\n";
                return 1;
            }
        }
    }

    OpenCLRuntime::instance().ensure_initialized();
    if (!device_supports_fp64() || !candidate_runner().has_f64()) {
        std::cout << "Float64 OpenCL kernels unavailable; Float32 correctness passed\n";
        return 0;
    }
    try {
        for (const auto& shape : SHAPES) {
            const int height = shape[0];
            const int width = shape[1];
            for (int steps : {1, 5}) {
                std::vector<double> u(height * width, 1.0);
                std::vector<double> v(height * width, 0.0);
                const Fields initial_f32 = initialize_fields(height, width);
                for (std::size_t index = 0; index < u.size(); ++index) {
                    u[index] = initial_f32.first[index];
                    v[index] = initial_f32.second[index];
                }
                FieldsF64 initial{u, v};
                FieldsF64 expected = initial;
                Params p64;
                for (int step = 0; step < steps; ++step) {
                    FieldsF64 next = expected;
                    for (int y = 1; y < height - 1; ++y) {
                        for (int x = 1; x < width - 1; ++x) {
                            const int index = y * width + x;
                            const double uc = expected.first[index];
                            const double vc = expected.second[index];
                            const double lap_u =
                                expected.first[index - width] + expected.first[index + width] +
                                expected.first[index - 1] + expected.first[index + 1] - 4.0 * uc;
                            const double lap_v =
                                expected.second[index - width] + expected.second[index + width] +
                                expected.second[index - 1] + expected.second[index + 1] - 4.0 * vc;
                            const double reaction = uc * vc * vc;
                            next.first[index] = uc + p64.dt *
                                (p64.diffusion_u * lap_u - reaction + p64.feed * (1.0 - uc));
                            next.second[index] = vc + p64.dt *
                                (p64.diffusion_v * lap_v + reaction - (p64.feed + p64.kill) * vc);
                        }
                    }
                    expected = next;
                }
                DeviceFieldsF64 state = candidate::prepare_candidate_f64(
                    initial.first, initial.second, height, width
                );
                candidate::run_candidate_f64(state, params, steps);
                const auto actual_pair = candidate::copy_candidate_fields_f64(state);
                release_fields_f64(state);
                for (std::size_t index = 0; index < actual_pair.first.size(); ++index) {
                    if (!close_f64(
                            actual_pair.first[index], expected.first[index], atol64, rtol64
                        ) ||
                        !close_f64(
                            actual_pair.second[index], expected.second[index], atol64, rtol64
                        )) {
                        std::cerr << "Float64 mismatch at shape " << height << 'x' << width << " steps "
                                  << steps << '\n';
                        return 1;
                    }
                }
            }
        }
    } catch (const std::exception& exception) {
        std::cerr << "Float64 OpenCL correctness skipped: " << exception.what() << '\n';
        return 1;
    }

    std::cout << "OpenCL correctness tests passed (Float32 and Float64)\n";
    return 0;
}
