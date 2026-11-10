#include "../eval_common.hpp"
#include "../gray_scott/baseline.hpp"
#include "../gray_scott/candidate.hpp"

#include <iostream>

// Public gate for the OpenCL track: the candidate must reproduce the baseline it
// is timed against, as the evaluator checks. Exit 1 means a numerical mismatch.
int main() {
    const GrayScottParams params;
    bool ok = true;
    for (Shape shape : {Shape{31, 29}, Shape{64, 64}, Shape{127, 65}}) {
        const GrayScottFields initial = initialize_fields(shape.nx, shape.ny);
        for (int steps : CORRECTNESS_STEPS) {
            auto baseline = gray_scott_opencl_baseline::prepare(initial);
            gray_scott_opencl_baseline::run(baseline, params, steps);
            const GrayScottFields expected = gray_scott_opencl_baseline::fields(baseline);
            gray_scott_opencl_baseline::release(baseline);

            auto candidate = gray_scott_opencl_candidate::prepare(initial);
            gray_scott_opencl_candidate::run(candidate, params, steps);
            cl_check(clFinish(OpenCLRuntime::instance().queue), "clFinish");
            const GrayScottFields actual = gray_scott_opencl_candidate::fields(candidate);
            gray_scott_opencl_candidate::release(candidate);

            const Comparison comparison = compare_fields(actual, expected, initial, 1e-10, 1e-8);
            if (!comparison.correct) {
                std::cerr << "candidate differs from baseline on " << shape.nx << "x" << shape.ny
                          << " after " << steps << " steps (max abs error "
                          << comparison.max_abs_error << ")\n";
                ok = false;
            }
        }
    }
    if (ok) {
        std::cout << "OpenCL candidate matches the baseline\n";
    }
    return ok ? 0 : 1;
}
