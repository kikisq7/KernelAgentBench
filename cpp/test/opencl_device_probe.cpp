#include "../gray_scott/candidate.hpp"

#include <iostream>
#include <string>

// Exit codes (public gate on the laptop):
//   0 — an OpenCL GPU with double precision can run the candidate (run device tests).
//   2 — no usable OpenCL GPU here (skip device tests; the H200 eval is the gate).
//   1 — unexpected setup failure (broken build tree, not a numerical mismatch).

int main() {
    try {
        GrayScottFields fields = initialize_fields(32, 32);
        auto state = gray_scott_opencl_candidate::prepare(fields);
        gray_scott_opencl_candidate::run(state, GrayScottParams{}, 1);
        gray_scott_opencl_candidate::release(state);
        std::cout << "OpenCL device probe succeeded\n";
        return 0;
    } catch (const std::runtime_error& exception) {
        const std::string message = exception.what();
        if (message.find("no OpenCL") != std::string::npos) {
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
