// Agents may edit this OpenCL kernel. Host entry points live in candidate.hpp.
__kernel void step_kernel(
    __global float* u_next,
    __global float* v_next,
    __global const float* u,
    __global const float* v,
    const int height,
    const int width,
    const float diffusion_u,
    const float diffusion_v,
    const float feed,
    const float kill,
    const float dt
) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (y <= 0 || y >= height - 1 || x <= 0 || x >= width - 1) {
        return;
    }
    const int index = y * width + x;
    const float u_center = u[index];
    const float v_center = v[index];
    const float lap_u =
        u[index - width] + u[index + width] + u[index - 1] + u[index + 1] - 4.0f * u_center;
    const float lap_v =
        v[index - width] + v[index + width] + v[index - 1] + v[index + 1] - 4.0f * v_center;
    const float reaction = u_center * v_center * v_center;
    u_next[index] = u_center + dt *
        (diffusion_u * lap_u - reaction + feed * (1.0f - u_center));
    v_next[index] = v_center + dt *
        (diffusion_v * lap_v + reaction - (feed + kill) * v_center);
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void step_kernel_f64(
    __global double* u_next,
    __global double* v_next,
    __global const double* u,
    __global const double* v,
    const int height,
    const int width,
    const double diffusion_u,
    const double diffusion_v,
    const double feed,
    const double kill,
    const double dt
) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (y <= 0 || y >= height - 1 || x <= 0 || x >= width - 1) {
        return;
    }
    const int index = y * width + x;
    const double u_center = u[index];
    const double v_center = v[index];
    const double lap_u =
        u[index - width] + u[index + width] + u[index - 1] + u[index + 1] - 4.0 * u_center;
    const double lap_v =
        v[index - width] + v[index + width] + v[index - 1] + v[index + 1] - 4.0 * v_center;
    const double reaction = u_center * v_center * v_center;
    u_next[index] = u_center + dt *
        (diffusion_u * lap_u - reaction + feed * (1.0 - u_center));
    v_next[index] = v_center + dt *
        (diffusion_v * lap_v + reaction - (feed + kill) * v_center);
}
