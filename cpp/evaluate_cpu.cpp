#include "baseline.hpp"
#include "candidate.hpp"
#include "eval_common.hpp"

#include <iostream>

struct CpuTrack {
    template <typename Run>
    static GrayScottFields run_fields(Run&& run, const GrayScottFields& initial, const GrayScottParams& params, int steps) {
        CpuGrayScottState state = initial;
        run(state, params, steps);
        return state;
    }

    template <typename Run>
    static double timed(Run&& run, const GrayScottFields& initial, const GrayScottParams& params, int steps, GrayScottFields* out) {
        CpuGrayScottState state = initial;
        const auto started = std::chrono::steady_clock::now();
        run(state, params, steps);
        const double elapsed = milliseconds_since(started);
        if (out != nullptr) {
            *out = std::move(state);
        }
        return elapsed;
    }

    GrayScottFields run_baseline_fields(const GrayScottFields& initial, const GrayScottParams& params, int steps) {
        return run_fields(gray_scott_cpu_baseline::run, initial, params, steps);
    }
    GrayScottFields run_candidate_fields(const GrayScottFields& initial, const GrayScottParams& params, int steps) {
        return run_fields(gray_scott_cpu_candidate::run, initial, params, steps);
    }
    double time_baseline(const GrayScottFields& initial, const GrayScottParams& params, int steps, GrayScottFields* out) {
        return timed(gray_scott_cpu_baseline::run, initial, params, steps, out);
    }
    double time_candidate(const GrayScottFields& initial, const GrayScottParams& params, int steps, GrayScottFields* out) {
        return timed(gray_scott_cpu_candidate::run, initial, params, steps, out);
    }
};

int main(int argc, char** argv) {
    const auto started = std::chrono::steady_clock::now();
    EvalOptions options;
    ResultHeader header{"cpp-cpu", "cpu", "gray-scott-notebook-cpp-cpu-v3", "cpu_ns", ""};
    try {
        options = parse_options(argc, argv, "256x256,512x512,1024x1024");
        cpu_thread_count() = options.threads;
        header.environment_json = "{\"timer\":\"cpu_ns\",\"threads\":" +
            std::to_string(cpu_thread_team().size()) + ",\"slurm_job_id\":\"" +
            json_escape(slurm_job_id()) + "\"}";
        CpuTrack track;
        std::vector<CorrectnessCase> cases;
        std::vector<Workload> workloads;
        const bool correct = evaluate(track, options, cases, workloads);
        write_result(
            options, header, correct ? "passed" : "failed", correct, cases, workloads,
            milliseconds_since(started) / 1000.0, ""
        );
        std::cout << options.output.string() << '\n';
        return correct ? 0 : 2;
    } catch (const std::exception& exception) {
        write_result(options, header, "error", false, {}, {}, milliseconds_since(started) / 1000.0, exception.what());
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
