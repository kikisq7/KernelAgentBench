#pragma once

// Harness-side pieces of parts/gpu/gray-scott.ipynb, shared by the C++ CPU and
// OpenCL tracks: model parameters and the initial condition. Mirrors
// kernels/gray_scott/notebook.jl bit for bit.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Defaults of simulate_gs_* in the notebook. The noise term (0.0 there) is dropped.
struct GrayScottParams {
    double D_u = 0.16;
    double D_v = 0.08;
    double F = 0.035;
    double k = 0.06;
    double dt = 1.0;
    double dx = 1.0;
};

// Element (x, y) lives at x + nx * y: x is contiguous, as in Julia's U[x, y].
struct GrayScottFields {
    int nx = 0;
    int ny = 0;
    std::vector<double> U;
    std::vector<double> V;
};

constexpr std::uint64_t GRAY_SCOTT_SEED = 20260918;

inline std::uint64_t splitmix64(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// The notebook's `U = rand(0.0:0.001:0.1, n, n)` with a portable generator.
inline GrayScottFields initialize_fields(
    int nx, int ny, std::uint64_t seed = GRAY_SCOTT_SEED, double init_multi = 0.5
) {
    if (nx < 16 || ny < 16) {
        throw std::invalid_argument("grid must be at least 16x16");
    }
    GrayScottFields fields;
    fields.nx = nx;
    fields.ny = ny;
    const std::size_t count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    fields.U.resize(count);
    fields.V.assign(count, 0.0);
    for (std::size_t index = 0; index < count; ++index) {
        const std::uint64_t draw = splitmix64(seed + index * 0x9e3779b97f4a7c15ULL) % 101;
        fields.U[index] = static_cast<double>(draw) / 1000.0;
    }

    // Set up an initial small disturbance (Julia's n_mid-5:n_mid+5, 0-based here).
    const int nx_mid = nx / 2 - 1;
    const int ny_mid = ny / 2 - 1;
    for (int y = ny_mid - 5; y <= ny_mid + 5; ++y) {
        for (int x = nx_mid - 5; x <= nx_mid + 5; ++x) {
            const std::size_t index = static_cast<std::size_t>(x) + static_cast<std::size_t>(nx) * y;
            fields.U[index] = 1.0 * init_multi;
            fields.V[index] = 0.5 * init_multi;
        }
    }
    return fields;
}
