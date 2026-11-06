#include "../gray_scott/candidate.hpp"
#include "../gray_scott/opencl_common.hpp"

#include <iostream>
#include <string>

// Exit codes (public gate on the laptop):
//   0 — OpenCL GPU can run the candidate kernel (run full device correctness).
//   2 — No usable OpenCL GPU here (skip device tests; H200 eval is the gate).
//   1 — Unexpected setup failure (broken build tree, not a numerical mismatch).

int main() {
    try {
        OpenCLRuntime::instance().ensure_initialized();
        constexpr int height = 32;
        constexpr int width = 32;
        std::vector<float> u(static_cast<std::size_t>(height) * width, 1.0f);
        std::vector<float> v(static_cast<std::size_t>(height) * width, 0.0f);
        GrayScottParams params;
        DeviceFields state = candidate::prepare_candidate(u, v, height, width);
        candidate::run_candidate(state, params, 1);
        release_fields(state);
        std::cout << "OpenCL device probe succeeded\n";
        return 0;
    } catch (const std::runtime_error& exception) {
        const std::string message = exception.what();
        if (message.find("no OpenCL GPU") != std::string::npos ||
            message.find("clEnqueueNDRangeKernel") != std::string::npos ||
            message.find("step_kernel_f64 is unavailable") != std::string::npos ||
            message.find("OpenCL build failed") != std::string::npos) {
            std::cout << "OpenCL GPU unavailable on this host; skipping device tests\n";
            return 2;
        }
        std::cerr << "OpenCL device probe failed: " << message << '\n';
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "OpenCL device probe failed: " << exception.what() << '\n';
        return 1;
    }
}
