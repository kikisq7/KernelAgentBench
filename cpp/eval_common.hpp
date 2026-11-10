#pragma once

// Shared by the C++ CPU and OpenCL evaluators: argument parsing, the comparison
// against the baseline, and the result JSON. Mirrors julia/evaluate_candidate.jl.

#include "gray_scott/notebook.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Small grids are checked after each of these step counts; timed grids after `steps`.
constexpr int CORRECTNESS_STEPS[] = {1, 5, 50};

struct Shape {
    int nx;
    int ny;
};

struct CorrectnessCase {
    Shape shape;
    int steps = 0;
    bool correct = false;
    double max_abs_error = 0.0;
    double relative_l2_error = 0.0;
};

struct Workload {
    Shape shape;
    std::vector<double> baseline_samples;
    std::vector<double> candidate_samples;
    bool correct = false;
    double max_abs_error = 0.0;
    double relative_l2_error = 0.0;
};

struct EvalOptions {
    std::string candidate_id = "manual";
    std::filesystem::path output = "results/manual.json";
    std::vector<Shape> correctness_shapes;
    std::vector<Shape> performance_shapes;
    int steps = 100;
    int warmups = 1;
    int repetitions = 3;
    double atol = 1e-10;
    double rtol = 1e-8;
    int threads = 0;
    double compile_seconds = 0.0;
};

inline std::vector<std::string> parse_csv(const std::string& value) {
    std::vector<std::string> items;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            items.push_back(item);
        }
    }
    return items;
}

inline std::vector<Shape> parse_shapes(const std::string& value) {
    std::vector<Shape> shapes;
    for (const std::string& item : parse_csv(value)) {
        const auto separator = item.find('x');
        if (separator == std::string::npos) {
            throw std::invalid_argument("shape must look like 64x64: " + item);
        }
        shapes.push_back({std::stoi(item.substr(0, separator)), std::stoi(item.substr(separator + 1))});
    }
    return shapes;
}

// Accepts the shared evaluator arguments; unknown ones (for example --language)
// are ignored so every track takes the same command line.
inline EvalOptions parse_options(int argc, char** argv, const std::string& default_shapes) {
    std::map<std::string, std::string> args;
    for (int index = 1; index < argc; index += 2) {
        const std::string key = argv[index];
        if (key.rfind("--", 0) != 0 || index + 1 >= argc) {
            throw std::invalid_argument("arguments must be --name value pairs");
        }
        args[key.substr(2)] = argv[index + 1];
    }
    const auto get = [&args](const std::string& key, const std::string& fallback) {
        const auto found = args.find(key);
        return found == args.end() ? fallback : found->second;
    };
    if (get("dtype", "Float64") != "Float64") {
        throw std::invalid_argument("the notebook baseline is Float64 only");
    }
    EvalOptions options;
    options.candidate_id = get("candidate-id", options.candidate_id);
    options.output = get("output", "results/" + options.candidate_id + ".json");
    options.correctness_shapes = parse_shapes(get("correctness-shapes", "31x29,64x64,127x65"));
    options.performance_shapes = parse_shapes(get("shapes", default_shapes));
    options.steps = std::stoi(get("steps", "100"));
    options.warmups = std::stoi(get("warmups", "1"));
    options.repetitions = std::stoi(get("repetitions", "3"));
    options.atol = std::stod(get("atol", "1e-10"));
    options.rtol = std::stod(get("rtol", "1e-8"));
    options.threads = std::stoi(get("threads", "0"));
    options.compile_seconds = std::stod(get("compile-seconds", "0"));
    return options;
}

inline bool boundary_fixed(const std::vector<double>& actual, const std::vector<double>& initial, int nx, int ny) {
    for (int x = 0; x < nx; ++x) {
        const std::size_t last = x + static_cast<std::size_t>(nx) * (ny - 1);
        if (actual[x] != initial[x] || actual[last] != initial[last]) {
            return false;
        }
    }
    for (int y = 0; y < ny; ++y) {
        const std::size_t first = static_cast<std::size_t>(nx) * y;
        const std::size_t last = first + nx - 1;
        if (actual[first] != initial[first] || actual[last] != initial[last]) {
            return false;
        }
    }
    return true;
}

struct Comparison {
    bool correct = false;
    double max_abs_error = 0.0;
    double relative_l2_error = 0.0;
};

// isapprox-style elementwise check of the candidate against the baseline, plus
// finite values and untouched ghost cells.
inline Comparison compare_fields(
    const GrayScottFields& actual,
    const GrayScottFields& expected,
    const GrayScottFields& initial,
    double atol,
    double rtol
) {
    Comparison result;
    if (actual.U.size() != expected.U.size() || actual.V.size() != expected.V.size()) {
        return result;
    }
    bool close = true;
    double squared_error = 0.0;
    double squared_expected = 0.0;
    for (const auto& pair : {std::make_pair(&actual.U, &expected.U), std::make_pair(&actual.V, &expected.V)}) {
        for (std::size_t index = 0; index < pair.first->size(); ++index) {
            const double a = (*pair.first)[index];
            const double e = (*pair.second)[index];
            const double difference = std::abs(a - e);
            close = close && std::isfinite(a) && difference <= atol + rtol * std::abs(e);
            result.max_abs_error = std::max(result.max_abs_error, difference);
            squared_error += difference * difference;
            squared_expected += e * e;
        }
    }
    result.relative_l2_error = std::sqrt(squared_error) / std::max(std::sqrt(squared_expected), 1e-300);
    result.correct = close &&
        boundary_fixed(actual.U, initial.U, initial.nx, initial.ny) &&
        boundary_fixed(actual.V, initial.V, initial.nx, initial.ny);
    return result;
}

inline double median_of(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("median of empty sample");
    }
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

inline double sample_std(const std::vector<double>& values) {
    if (values.size() < 2) {
        return 0.0;
    }
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double sum = 0.0;
    for (double value : values) {
        sum += (value - mean) * (value - mean);
    }
    return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

inline std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (char character : value) {
        if (character == '"' || character == '\\') {
            output << '\\' << character;
        } else if (character == '\n') {
            output << "\\n";
        } else if (static_cast<unsigned char>(character) < 0x20) {
            output << ' ';
        } else {
            output << character;
        }
    }
    return output.str();
}

inline void write_samples(std::ostream& output, const std::vector<double>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        output << (index ? "," : "") << values[index];
    }
    output << ']';
}

struct ResultHeader {
    std::string language;
    std::string backend;
    std::string evaluator_version;
    std::string timer;
    // Pre-rendered JSON object for "environment".
    std::string environment_json;
};

inline void write_result(
    const EvalOptions& options,
    const ResultHeader& header,
    const std::string& status,
    bool correct,
    const std::vector<CorrectnessCase>& cases,
    const std::vector<Workload>& workloads,
    double end_to_end_seconds,
    const std::string& error
) {
    if (!options.output.parent_path().empty()) {
        std::filesystem::create_directories(options.output.parent_path());
    }
    std::ofstream output(options.output);
    output << std::setprecision(17);
    output << "{\n"
           << "\"schema_version\":\"3.0\",\n"
           << "\"candidate_id\":\"" << json_escape(options.candidate_id) << "\",\n"
           << "\"language\":\"" << header.language << "\",\n"
           << "\"backend\":\"" << header.backend << "\",\n"
           << "\"evaluator_version\":\"" << header.evaluator_version << "\",\n"
           << "\"status\":\"" << status << "\",\n"
           << "\"correct\":" << (correct ? "true" : "false") << ",\n"
           << "\"correctness_cases\":[";
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const CorrectnessCase& item = cases[index];
        output << (index ? "," : "")
               << "{\"dtype\":\"Float64\",\"shape\":[" << item.shape.nx << ',' << item.shape.ny << "],"
               << "\"steps\":" << item.steps << ","
               << "\"correct\":" << (item.correct ? "true" : "false") << ","
               << "\"atol\":" << options.atol << ",\"rtol\":" << options.rtol << ","
               << "\"max_abs_error\":" << item.max_abs_error << ","
               << "\"relative_l2_error\":" << item.relative_l2_error << '}';
    }
    output << "],\n\"workloads\":[";
    double log_speedup = 0.0;
    for (std::size_t index = 0; index < workloads.size(); ++index) {
        const Workload& workload = workloads[index];
        const double baseline_median = median_of(workload.baseline_samples);
        const double candidate_median = median_of(workload.candidate_samples);
        log_speedup += std::log(baseline_median / candidate_median);
        output << (index ? "," : "")
               << "{\"shape\":[" << workload.shape.nx << ',' << workload.shape.ny << "],"
               << "\"baseline_samples_ms\":";
        write_samples(output, workload.baseline_samples);
        output << ",\"candidate_samples_ms\":";
        write_samples(output, workload.candidate_samples);
        output << ",\"baseline_median_ms\":" << baseline_median
               << ",\"candidate_median_ms\":" << candidate_median
               << ",\"baseline_std_ms\":" << sample_std(workload.baseline_samples)
               << ",\"candidate_std_ms\":" << sample_std(workload.candidate_samples)
               << ",\"speedup\":" << baseline_median / candidate_median
               << ",\"correct\":" << (workload.correct ? "true" : "false")
               << ",\"max_abs_error\":" << workload.max_abs_error
               << ",\"relative_l2_error\":" << workload.relative_l2_error
               << ",\"timer\":\"" << header.timer << "\"}";
    }
    output << "],\n\"geometric_mean_speedup\":";
    if (correct && !workloads.empty()) {
        output << std::exp(log_speedup / static_cast<double>(workloads.size()));
    } else {
        output << "null";
    }
    output << ",\n\"compile_seconds\":" << options.compile_seconds
           << ",\n\"end_to_end_seconds\":" << end_to_end_seconds
           << ",\n\"environment\":" << (header.environment_json.empty() ? "{}" : header.environment_json)
           << ",\n\"error\":";
    if (error.empty()) {
        output << "null";
    } else {
        output << '"' << json_escape(error) << '"';
    }
    output << "\n}\n";
}

inline std::string slurm_job_id() {
    const char* job = std::getenv("SLURM_JOB_ID");
    return job ? job : "";
}

// Runs the correctness phase and, if it passes, the timed workloads. `Track`
// supplies prepare/run/fields/release for the baseline and the candidate, plus a
// timed run that returns milliseconds.
template <typename Track>
bool evaluate(
    Track& track,
    const EvalOptions& options,
    std::vector<CorrectnessCase>& cases,
    std::vector<Workload>& workloads
) {
    const GrayScottParams params;
    bool correct = true;
    for (Shape shape : options.correctness_shapes) {
        const GrayScottFields initial = initialize_fields(shape.nx, shape.ny);
        for (int steps : CORRECTNESS_STEPS) {
            const GrayScottFields expected = track.run_baseline_fields(initial, params, steps);
            const GrayScottFields actual = track.run_candidate_fields(initial, params, steps);
            const Comparison comparison = compare_fields(actual, expected, initial, options.atol, options.rtol);
            cases.push_back({shape, steps, comparison.correct, comparison.max_abs_error, comparison.relative_l2_error});
            correct = correct && comparison.correct;
        }
    }
    if (!correct) {
        return false;
    }
    for (Shape shape : options.performance_shapes) {
        const GrayScottFields initial = initialize_fields(shape.nx, shape.ny);
        const int warmup_steps = std::min(options.steps, 10);
        for (int warmup = 0; warmup < options.warmups; ++warmup) {
            track.time_baseline(initial, params, warmup_steps, nullptr);
            track.time_candidate(initial, params, warmup_steps, nullptr);
        }
        Workload workload;
        workload.shape = shape;
        GrayScottFields last_baseline;
        GrayScottFields last_candidate;
        for (int repetition = 0; repetition < options.repetitions; ++repetition) {
            if (repetition % 2 == 0) {
                workload.baseline_samples.push_back(track.time_baseline(initial, params, options.steps, &last_baseline));
                workload.candidate_samples.push_back(track.time_candidate(initial, params, options.steps, &last_candidate));
            } else {
                workload.candidate_samples.push_back(track.time_candidate(initial, params, options.steps, &last_candidate));
                workload.baseline_samples.push_back(track.time_baseline(initial, params, options.steps, &last_baseline));
            }
        }
        const Comparison comparison = compare_fields(last_candidate, last_baseline, initial, options.atol, options.rtol);
        workload.correct = comparison.correct;
        workload.max_abs_error = comparison.max_abs_error;
        workload.relative_l2_error = comparison.relative_l2_error;
        correct = correct && workload.correct;
        workloads.push_back(std::move(workload));
    }
    return correct;
}

inline double milliseconds_since(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}
