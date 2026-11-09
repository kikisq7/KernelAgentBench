#include "baseline.hpp"
#include "candidate.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

struct Shape {
    int height;
    int width;
};

struct Errors {
    double max_abs = 0.0;
    double relative_l2 = 0.0;
};

struct CorrectnessCase {
    std::string dtype;
    Shape shape;
    int steps = 0;
    bool correct = false;
    double atol = 0.0;
    double rtol = 0.0;
    Errors errors;
};

struct Workload {
    Shape shape;
    std::vector<double> baseline_samples;
    std::vector<double> candidate_samples;
    double baseline_median = 0.0;
    double candidate_median = 0.0;
    bool correct = false;
    Errors errors;
};

std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key.rfind("--", 0) != 0 || index + 1 >= argc) {
            throw std::invalid_argument("expected --key value arguments");
        }
        args.emplace(key.substr(2), argv[++index]);
    }
    return args;
}

std::vector<std::string> parse_csv(const std::string& value) {
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

std::vector<Shape> parse_shapes(const std::string& value) {
    std::vector<Shape> shapes;
    for (const std::string& item : parse_csv(value)) {
        const auto separator = item.find('x');
        if (separator == std::string::npos) {
            throw std::invalid_argument("shape must look like 64x64");
        }
        shapes.push_back(Shape{
            std::stoi(item.substr(0, separator)),
            std::stoi(item.substr(separator + 1)),
        });
    }
    return shapes;
}

template <typename T>
Errors field_errors(const std::vector<T>& actual_u, const std::vector<T>& actual_v,
                    const std::vector<T>& expected_u, const std::vector<T>& expected_v) {
    double max_abs = 0.0;
    double squared_error = 0.0;
    double squared_expected = 0.0;
    for (std::size_t index = 0; index < actual_u.size(); ++index) {
        for (const auto pair : {
                 std::pair<T, T>{actual_u[index], expected_u[index]},
                 std::pair<T, T>{actual_v[index], expected_v[index]},
             }) {
            const double difference = static_cast<double>(pair.first) - static_cast<double>(pair.second);
            max_abs = std::max(max_abs, std::abs(difference));
            squared_error += difference * difference;
            squared_expected += static_cast<double>(pair.second) * static_cast<double>(pair.second);
        }
    }
    const double denominator = std::max(std::sqrt(squared_expected), 1e-30);
    return Errors{max_abs, std::sqrt(squared_error) / denominator};
}

template <typename T>
bool fields_close(T atol, T rtol, const std::vector<T>& expected_u,
                  const std::vector<T>& expected_v, const std::vector<T>& actual_u,
                  const std::vector<T>& actual_v) {
    for (std::size_t index = 0; index < actual_u.size(); ++index) {
        for (const auto pair : {
                 std::pair<T, T>{actual_u[index], expected_u[index]},
                 std::pair<T, T>{actual_v[index], expected_v[index]},
             }) {
            if (!std::isfinite(static_cast<double>(pair.first))) {
                return false;
            }
            const T tolerance = atol + rtol * std::abs(pair.second);
            if (std::abs(pair.first - pair.second) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

bool boundary_fixed(const std::vector<float>& actual, const std::vector<float>& initial, int height, int width) {
    for (int x = 0; x < width; ++x) {
        if (actual[static_cast<std::size_t>(x)] != initial[static_cast<std::size_t>(x)]) {
            return false;
        }
        const std::size_t bottom = static_cast<std::size_t>((height - 1) * width + x);
        if (actual[bottom] != initial[bottom]) {
            return false;
        }
    }
    for (int y = 0; y < height; ++y) {
        if (actual[static_cast<std::size_t>(y * width)] != initial[static_cast<std::size_t>(y * width)]) {
            return false;
        }
        const std::size_t right = static_cast<std::size_t>(y * width + width - 1);
        if (actual[right] != initial[right]) {
            return false;
        }
    }
    return true;
}

double median_of(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("median of empty sample");
    }
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[mid];
    }
    return 0.5 * (values[mid - 1] + values[mid]);
}

double sample_std(const std::vector<double>& values) {
    if (values.size() < 2) {
        return 0.0;
    }
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
    double sum = 0.0;
    for (double value : values) {
        sum += (value - mean) * (value - mean);
    }
    return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (char character : value) {
        if (character == '"' || character == '\\') {
            output << '\\' << character;
        } else if (character == '\n') {
            output << "\\n";
        } else {
            output << character;
        }
    }
    return output.str();
}

void write_samples(std::ostream& output, const std::vector<double>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
}

void write_result(
    const std::filesystem::path& path,
    const std::string& candidate_id,
    const std::string& status,
    bool correct,
    const std::vector<CorrectnessCase>& correctness_cases,
    const std::vector<Workload>& workloads,
    double compile_seconds,
    double end_to_end_seconds,
    const std::string& error
) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    output << std::setprecision(10);
    double log_speedup = 0.0;
    for (const Workload& workload : workloads) {
        log_speedup += std::log(workload.baseline_median / workload.candidate_median);
    }
    const double speedup = workloads.empty() ? 0.0 : std::exp(log_speedup / workloads.size());
    output << "{\n"
           << "\"schema_version\":\"2.1\",\n"
           << "\"candidate_id\":\"" << json_escape(candidate_id) << "\",\n"
           << "\"language\":\"cpp-cpu\",\n"
           << "\"backend\":\"cpu\",\n"
           << "\"evaluator_version\":\"gray-scott-cpu-cpp-v1\",\n"
           << "\"status\":\"" << status << "\",\n"
           << "\"correct\":" << (correct ? "true" : "false") << ",\n"
           << "\"correctness_cases\":[";
    for (std::size_t index = 0; index < correctness_cases.size(); ++index) {
        if (index) {
            output << ',';
        }
        const CorrectnessCase& item = correctness_cases[index];
        output << "{\"dtype\":\"" << json_escape(item.dtype) << "\","
               << "\"shape\":[" << item.shape.height << ',' << item.shape.width << "],"
               << "\"steps\":" << item.steps << ","
               << "\"correct\":" << (item.correct ? "true" : "false") << ","
               << "\"atol\":" << item.atol << ","
               << "\"rtol\":" << item.rtol << ","
               << "\"max_abs_error\":" << item.errors.max_abs << ","
               << "\"relative_l2_error\":" << item.errors.relative_l2 << '}';
    }
    output << "],\n\"workloads\":[";
    for (std::size_t index = 0; index < workloads.size(); ++index) {
        if (index) {
            output << ',';
        }
        const Workload& workload = workloads[index];
        output << "{\"shape\":[" << workload.shape.height << ',' << workload.shape.width << "],";
        output << "\"baseline_samples_ms\":";
        write_samples(output, workload.baseline_samples);
        output << ",\"candidate_samples_ms\":";
        write_samples(output, workload.candidate_samples);
        output << ",\"baseline_median_ms\":" << workload.baseline_median
               << ",\"candidate_median_ms\":" << workload.candidate_median
               << ",\"baseline_std_ms\":" << sample_std(workload.baseline_samples)
               << ",\"candidate_std_ms\":" << sample_std(workload.candidate_samples)
               << ",\"speedup\":" << workload.baseline_median / workload.candidate_median
               << ",\"correct\":" << (workload.correct ? "true" : "false")
               << ",\"max_abs_error\":" << workload.errors.max_abs
               << ",\"relative_l2_error\":" << workload.errors.relative_l2
               << ",\"timer\":\"cpu_ns\"}";
    }
    output << "],\n\"geometric_mean_speedup\":";
    if (correct && !workloads.empty()) {
        output << speedup;
    } else {
        output << "null";
    }
    const char* job = std::getenv("SLURM_JOB_ID");
    output << ",\n\"compile_seconds\":" << compile_seconds
           << ",\n\"end_to_end_seconds\":" << end_to_end_seconds
           << ",\n\"environment\":{\"timer\":\"cpu_ns\",\"threads\":"
           << (std::thread::hardware_concurrency() == 0 ? 1 : std::thread::hardware_concurrency())
           << ",\"slurm_job_id\":\"" << json_escape(job ? job : "") << "\"},\n\"error\":";
    if (error.empty()) {
        output << "null";
    } else {
        output << '"' << json_escape(error) << '"';
    }
    output << "\n}\n";
}

template <typename Run>
double timed_run(Run&& run, CpuGrayScottState<float>& state, CpuGrayScottParams<float> params, int steps) {
    const auto started = std::chrono::steady_clock::now();
    run(state, params, steps);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

template <typename T>
CorrectnessCase check_case(const std::string& dtype, Shape shape, int steps, double atol, double rtol) {
    const CpuGrayScottParams<T> params;
    const CpuGrayScottState<T> initial = cpu_initialize_fields<T>(shape.height, shape.width);
    CpuGrayScottState<T> candidate = initial;
    gray_scott_cpu_candidate::run(candidate, params, steps);
    const CpuGrayScottState<T> expected = cpu_serial_simulate(initial, params, steps);
    CorrectnessCase item;
    item.dtype = dtype;
    item.shape = shape;
    item.steps = steps;
    item.atol = atol;
    item.rtol = rtol;
    item.errors = field_errors(candidate.u, candidate.v, expected.u, expected.v);
    item.correct = fields_close(
        static_cast<T>(atol), static_cast<T>(rtol),
        expected.u, expected.v, candidate.u, candidate.v
    );
    if constexpr (std::is_same_v<T, float>) {
        item.correct = item.correct &&
            boundary_fixed(candidate.u, initial.u, shape.height, shape.width) &&
            boundary_fixed(candidate.v, initial.v, shape.height, shape.width);
    }
    return item;
}

int main(int argc, char** argv) {
    const auto started = std::chrono::steady_clock::now();
    std::filesystem::path output_path = "results/manual-cpp-cpu.json";
    std::string candidate_id = "manual-cpp-cpu";
    double compile_seconds = 0.0;
    try {
        const auto args = parse_args(argc, argv);
        auto get = [&args](const std::string& key, const std::string& fallback) {
            const auto iterator = args.find(key);
            return iterator == args.end() ? fallback : iterator->second;
        };
        output_path = get("output", output_path.string());
        candidate_id = get("candidate-id", candidate_id);
        compile_seconds = std::stod(get("compile-seconds", "0"));
        const auto correctness_shapes = parse_shapes(get("correctness-shapes", "31x29,64x64,127x65"));
        const auto performance_shapes = parse_shapes(get("shapes", "256x256,512x512,1024x1024"));
        const int steps = std::stoi(get("steps", "200"));
        const int warmups = std::stoi(get("warmups", "1"));
        const int repetitions = std::stoi(get("repetitions", "3"));
        const double atol = std::stod(get("atol", "2e-5"));
        const double rtol = std::stod(get("rtol", "2e-4"));
        const double atol64 = std::stod(get("atol64", "1e-12"));
        const double rtol64 = std::stod(get("rtol64", "1e-10"));
        const auto correctness_dtypes = parse_csv(get("correctness-dtypes", "Float32,Float64"));

        std::vector<CorrectnessCase> correctness_cases;
        bool correct = true;
        for (const std::string& dtype : correctness_dtypes) {
            for (Shape shape : correctness_shapes) {
                for (int test_steps : {1, 5}) {
                    CorrectnessCase item = dtype == "Float64"
                        ? check_case<double>("Float64", shape, test_steps, atol64, rtol64)
                        : check_case<float>("Float32", shape, test_steps, atol, rtol);
                    if (dtype != "Float32" && dtype != "Float64") {
                        throw std::invalid_argument("unsupported correctness dtype: " + dtype);
                    }
                    correct = correct && item.correct;
                    correctness_cases.push_back(std::move(item));
                }
            }
        }

        std::vector<Workload> workloads;
        if (correct) {
            const CpuGrayScottParams<float> params;
            for (Shape shape : performance_shapes) {
                const CpuGrayScottState<float> initial =
                    cpu_initialize_fields<float>(shape.height, shape.width);
                const int warmup_steps = std::min(steps, 10);
                for (int warmup = 0; warmup < warmups; ++warmup) {
                    CpuGrayScottState<float> baseline_state = initial;
                    CpuGrayScottState<float> candidate_state = initial;
                    timed_run([](auto& state, auto params, int n) {
                        gray_scott_cpu_baseline::run(state, params, n);
                    }, baseline_state, params, warmup_steps);
                    timed_run([](auto& state, auto params, int n) {
                        gray_scott_cpu_candidate::run(state, params, n);
                    }, candidate_state, params, warmup_steps);
                }
                Workload workload;
                workload.shape = shape;
                CpuGrayScottState<float> last_candidate = initial;
                for (int repetition = 0; repetition < repetitions; ++repetition) {
                    CpuGrayScottState<float> baseline_state = initial;
                    CpuGrayScottState<float> candidate_state = initial;
                    if (repetition % 2 == 0) {
                        workload.baseline_samples.push_back(timed_run([](auto& state, auto params, int n) {
                            gray_scott_cpu_baseline::run(state, params, n);
                        }, baseline_state, params, steps));
                        workload.candidate_samples.push_back(timed_run([](auto& state, auto params, int n) {
                            gray_scott_cpu_candidate::run(state, params, n);
                        }, candidate_state, params, steps));
                    } else {
                        workload.candidate_samples.push_back(timed_run([](auto& state, auto params, int n) {
                            gray_scott_cpu_candidate::run(state, params, n);
                        }, candidate_state, params, steps));
                        workload.baseline_samples.push_back(timed_run([](auto& state, auto params, int n) {
                            gray_scott_cpu_baseline::run(state, params, n);
                        }, baseline_state, params, steps));
                    }
                    last_candidate = std::move(candidate_state);
                }
                const CpuGrayScottState<float> expected = cpu_serial_simulate(initial, params, steps);
                workload.errors = field_errors(last_candidate.u, last_candidate.v, expected.u, expected.v);
                workload.correct = fields_close(
                    static_cast<float>(atol), static_cast<float>(rtol),
                    expected.u, expected.v, last_candidate.u, last_candidate.v
                );
                workload.baseline_median = median_of(workload.baseline_samples);
                workload.candidate_median = median_of(workload.candidate_samples);
                correct = correct && workload.correct;
                workloads.push_back(std::move(workload));
            }
        }

        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();
        write_result(
            output_path, candidate_id, correct ? "passed" : "failed", correct,
            correctness_cases, workloads, compile_seconds, elapsed, ""
        );
        std::cout << output_path << '\n';
        return correct ? 0 : 2;
    } catch (const std::exception& exception) {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();
        write_result(output_path, candidate_id, "error", false, {}, {}, compile_seconds, elapsed, exception.what());
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
