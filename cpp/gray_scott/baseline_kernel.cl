// OpenCL translation of "GPU (CUDA) Implementation of the Gray-Scott Equation" in
// parts/gpu/gray-scott.ipynb. As in kernels/gray_scott/cuda/baseline.jl, the
// notebook kernel is split at its CUDA.sync_threads() into two launches (the
// in-place version races across blocks), and the noise term is removed.
// Julia's U[x, y] is U[x + nx*y] here, and indices are 0-based.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

double laplacian(__global const double* A, int x, int y, double dx, int nx) {
    const double DAx = (A[(x - 1) + nx * y] - (2 * A[x + nx * y]) + A[(x + 1) + nx * y]) / dx;
    const double DAy = (A[x + nx * (y - 1)] - (2 * A[x + nx * y]) + A[x + nx * (y + 1)]) / dx;
    return DAx + DAy;
}

__kernel void calculate_gs(
    __global double* DU, __global double* DV,
    __global const double* U, __global const double* V,
    const double D_u, const double D_v, const double F, const double k, const double dx,
    const int nx, const int ny
) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);

    // Skip ghost cells on the boundaries
    if ((0 < x && x < nx - 1) && (0 < y && y < ny - 1)) {
        const int i = x + nx * y;
        // Compute diffusion
        const double DeltaU = laplacian(U, x, y, dx, nx);
        const double DeltaV = laplacian(V, x, y, dx, nx);

        // Update concentration fields
        const double UVV = U[i] * (V[i] * V[i]);
        DU[i] = D_u * DeltaU - UVV + F * (1 - U[i]);
        DV[i] = D_v * DeltaV + UVV - (F + k) * V[i];
    }
}

__kernel void update_gs(
    __global double* U, __global double* V,
    __global const double* DU, __global const double* DV,
    const double dt, const int nx, const int ny
) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);

    // Skip ghost cells on the boundaries
    if ((0 < x && x < nx - 1) && (0 < y && y < ny - 1)) {
        const int i = x + nx * y;
        // Update concentration fields
        U[i] += DU[i] * dt;
        V[i] += DV[i] * dt;
    }
}
