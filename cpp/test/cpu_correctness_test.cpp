#include "baseline.hpp"
#include "candidate.hpp"
#include "../eval_common.hpp"

#include <iostream>

// Public gate for the C++ CPU track: the candidate must reproduce the baseline it
// is timed against, as the evaluator checks.
int main() {
    cpu_thread_count() = 4;
    const GrayScottParams params;
    bool ok = true;
    for (Shape shape : {Shape{31, 29}, Shape{64, 64}, Shape{127, 65}}) {
        const GrayScottFields initial = initialize_fields(shape.nx, shape.ny);
        for (int steps : CORRECTNESS_STEPS) {
            CpuGrayScottState baseline = initial;
            CpuGrayScottState candidate = initial;
            gray_scott_cpu_baseline::run(baseline, params, steps);
            gray_scott_cpu_candidate::run(candidate, params, steps);
            const Comparison comparison = compare_fields(candidate, baseline, initial, 1e-10, 1e-8);
            if (!comparison.correct) {
                std::cerr << "candidate differs from baseline on " << shape.nx << "x" << shape.ny
                          << " after " << steps << " steps (max abs error "
                          << comparison.max_abs_error << ")\n";
                ok = false;
            }
        }
    }
    if (ok) {
        std::cout << "C++ CPU candidate matches the baseline\n";
    }
    return ok ? 0 : 1;
}
