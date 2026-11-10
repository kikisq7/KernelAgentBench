#include "eval_common.hpp"
#include "gray_scott/baseline.hpp"
#include "gray_scott/candidate.hpp"

#include <iostream>

// Wall-clock time from an idle queue to a finished queue around run() covers every
// launch and host step, measured the same way for baseline and candidate.
struct OpenCLTrack {
    template <typename Api>
    static GrayScottFields run_fields(const GrayScottFields& initial, const GrayScottParams& params, int steps) {
        auto state = Api::prepare(initial);
        Api::run(state, params, steps);
        GrayScottFields result = Api::fields(state);
        Api::release(state);
        return result;
    }

    template <typename Api>
    static double timed(const GrayScottFields& initial, const GrayScottParams& params, int steps, GrayScottFields* out) {
        auto state = Api::prepare(initial);
        cl_check(clFinish(OpenCLRuntime::instance().queue), "clFinish");
        const auto started = std::chrono::steady_clock::now();
        Api::run(state, params, steps);
        cl_check(clFinish(OpenCLRuntime::instance().queue), "clFinish");
        const double elapsed = milliseconds_since(started);
        if (out != nullptr) {
            *out = Api::fields(state);
        }
        Api::release(state);
        return elapsed;
    }

    struct Baseline {
        using State = gray_scott_opencl_baseline::State;
        static State prepare(const GrayScottFields& fields) { return gray_scott_opencl_baseline::prepare(fields); }
        static void run(State& state, const GrayScottParams& p, int steps) { gray_scott_opencl_baseline::run(state, p, steps); }
        static GrayScottFields fields(const State& state) { return gray_scott_opencl_baseline::fields(state); }
        static void release(State& state) { gray_scott_opencl_baseline::release(state); }
    };

    struct Candidate {
        using State = gray_scott_opencl_candidate::State;
        static State prepare(const GrayScottFields& fields) { return gray_scott_opencl_candidate::prepare(fields); }
        static void run(State& state, const GrayScottParams& p, int steps) { gray_scott_opencl_candidate::run(state, p, steps); }
        static GrayScottFields fields(const State& state) { return gray_scott_opencl_candidate::fields(state); }
        static void release(State& state) { gray_scott_opencl_candidate::release(state); }
    };

    GrayScottFields run_baseline_fields(const GrayScottFields& initial, const GrayScottParams& params, int steps) {
        return run_fields<Baseline>(initial, params, steps);
    }
    GrayScottFields run_candidate_fields(const GrayScottFields& initial, const GrayScottParams& params, int steps) {
        return run_fields<Candidate>(initial, params, steps);
    }
    double time_baseline(const GrayScottFields& initial, const GrayScottParams& params, int steps, GrayScottFields* out) {
        return timed<Baseline>(initial, params, steps, out);
    }
    double time_candidate(const GrayScottFields& initial, const GrayScottParams& params, int steps, GrayScottFields* out) {
        return timed<Candidate>(initial, params, steps, out);
    }
};

int main(int argc, char** argv) {
    const auto started = std::chrono::steady_clock::now();
    EvalOptions options;
    ResultHeader header{"cpp", "opencl", "gray-scott-notebook-cpp-opencl-v3", "wall_clock", ""};
    try {
        options = parse_options(argc, argv, "1024x1024");
        OpenCLRuntime& runtime = OpenCLRuntime::instance();
        header.environment_json = "{\"timer\":\"wall_clock\",\"gpu_name\":\"" +
            json_escape(runtime.device_name()) + "\",\"opencl_platform\":\"" +
            json_escape(runtime.platform_version()) + "\",\"slurm_job_id\":\"" +
            json_escape(slurm_job_id()) + "\"}";
        OpenCLTrack track;
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
