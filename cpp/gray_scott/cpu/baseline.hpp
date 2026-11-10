#pragma once

#include "types.hpp"

#include <cstddef>
#include <memory>

// C++ translation of "CPU (Multi-Threaded) Implementation of the Gray-Scott
// Equation" in parts/gpu/gray-scott.ipynb, without the noise term. Julia's
// U[x, y] is U[x + nx*y] here, and indices are 0-based.

namespace gray_scott_cpu_baseline {

inline double laplacian(const double* A, int x, int y, double dx, int nx) {
    const auto at = [A, nx](int i, int j) { return A[i + static_cast<std::size_t>(nx) * j]; };
    const double DAx = (at(x - 1, y) - (2 * at(x, y)) + at(x + 1, y)) / dx;
    const double DAy = (at(x, y - 1) - (2 * at(x, y)) + at(x, y + 1)) / dx;
    return DAx + DAy;
}

inline void calculate_gs_cpu_mt(
    double* U, double* V, double D_u, double D_v, double F, double k, double dt, double dx,
    int nx, int ny
) {
    const std::size_t count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    // Allocate U and V temporaries for updates
    std::unique_ptr<double[]> DU(new double[count]);
    std::unique_ptr<double[]> DV(new double[count]);

    // Compute diffusion and concentration fields changes
    threads_for(1, ny - 1, [&](int y) {
        for (int x = 1; x < nx - 1; ++x) {
            const std::size_t i = x + static_cast<std::size_t>(nx) * y;
            // Compute diffusion
            const double DeltaU = laplacian(U, x, y, dx, nx);
            const double DeltaV = laplacian(V, x, y, dx, nx);

            // Compute concentration fields changes
            const double UVV = U[i] * (V[i] * V[i]);
            DU[i] = D_u * DeltaU - UVV + F * (1 - U[i]);
            DV[i] = D_v * DeltaV + UVV - (F + k) * V[i];
        }
    });

    // Update concentration fields
    threads_for(1, ny - 1, [&](int y) {
        for (int x = 1; x < nx - 1; ++x) {
            const std::size_t i = x + static_cast<std::size_t>(nx) * y;
            U[i] += DU[i] * dt;
            V[i] += DV[i] * dt;
        }
    });
}

inline void run(CpuGrayScottState& state, const GrayScottParams& p, int steps) {
    for (int step = 0; step < steps; ++step) {
        calculate_gs_cpu_mt(
            state.U.data(), state.V.data(), p.D_u, p.D_v, p.F, p.k, p.dt, p.dx,
            state.nx, state.ny
        );
    }
}

}  // namespace gray_scott_cpu_baseline
