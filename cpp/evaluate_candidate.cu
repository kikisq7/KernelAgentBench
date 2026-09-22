#include "gray_scott/baseline.cuh"
#include "gray_scott/candidate.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

struct Shape {
    int height;
    int width;
};

struct Errors {
    double max_abs;
    double relative_l2;
};

struct Workload {
    Shape shape;
    std::vector<float> baseline_samples;
    std::vector<float> candidate_samples;
    double baseline_median;
    double candidate_median;
    bool correct;
    Errors errors;
};

using Fields = std::pair<std::vector<float>, std::vector<float>>;

std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> values;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc || std::string(argv[index]).rfind("--", 0) != 0) {
            throw std::runtime_error("arguments must be --name value pairs");
        }
        values[std::string(argv[index]).substr(2)] = argv[index + 1];
    }
    return values;
}

std::vector<Shape> parse_shapes(const std::string& value) {
    std::vector<Shape> shapes;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto separator = item.find('x');
        if (separator == std::string::npos) {
            throw std::runtime_error("invalid shape: " + item);
        }
        shapes.push_back({
            std::stoi(item.substr(0, separator)),
            std::stoi(item.substr(separator + 1)),
        });
    }
    return shapes;
}

Fields initialize_fields(Shape shape, int seed = 20260918) {
    const std::size_t count = static_cast<std::size_t>(shape.height) * shape.width;
    Fields fields{std::vector<float>(count, 1.0f), std::vector<float>(count, 0.0f)};
    const int radius_y = std::max(1, shape.height / 10);
    const int radius_x = std::max(1, shape.width / 10);
    const int center_y = (shape.height - 1) / 2;
    const int center_x = (shape.width - 1) / 2;
    for (int y = std::max(1, center_y - radius_y);
         y <= std::min(shape.height - 2, center_y + radius_y);
         ++y) {
        for (int x = std::max(1, center_x - radius_x);
             x <= std::min(shape.width - 2, center_x + radius_x);
             ++x) {
            const float noise = 0.01f * std::sin(
                ((y + 1) * 73856093.0 + (x + 1) * 19349663.0 + seed) * 0.001
            );
            const int index = y * shape.width + x;
            fields.first[index] = 0.5f + noise;
            fields.second[index] = 0.25f - noise;
        }
    }
    return fields;
}

Fields cpu_simulate(Fields fields, Shape shape, GrayScottParams params, int steps) {
    Fields next = fields;
    for (int step = 0; step < steps; ++step) {
        next = fields;
        for (int y = 1; y < shape.height - 1; ++y) {
            for (int x = 1; x < shape.width - 1; ++x) {
                const int index = y * shape.width + x;
                const float u_center = fields.first[index];
                const float v_center = fields.second[index];
                const float lap_u =
                    fields.first[index - shape.width] + fields.first[index + shape.width] +
                    fields.first[index - 1] + fields.first[index + 1] - 4.0f * u_center;
                const float lap_v =
                    fields.second[index - shape.width] + fields.second[index + shape.width] +
                    fields.second[index - 1] + fields.second[index + 1] - 4.0f * v_center;
                const float reaction = u_center * v_center * v_center;
                next.first[index] = u_center + params.dt *
                    (params.diffusion_u * lap_u - reaction +
                     params.feed * (1.0f - u_center));
                next.second[index] = v_center + params.dt *
                    (params.diffusion_v * lap_v + reaction -
                     (params.feed + params.kill) * v_center);
            }
        }
        fields.swap(next);
    }
    return fields;
}

Errors field_errors(const Fields& actual, const Fields& expected) {
    double max_abs = 0.0;
    double squared_error = 0.0;
    double squared_expected = 0.0;
    for (std::size_t index = 0; index < actual.first.size(); ++index) {
        for (const auto& pair : {
                 std::pair<const std::vector<float>*, const std::vector<float>*>{
                     &actual.first, &expected.first},
                 std::pair<const std::vector<float>*, const std::vector<float>*>{
                     &actual.second, &expected.second},
             }) {
            const double difference = (*pair.first)[index] - (*pair.second)[index];
            max_abs = std::max(max_abs, std::abs(difference));
            squared_error += difference * difference;
            squared_expected += static_cast<double>((*pair.second)[index]) *
                                (*pair.second)[index];
        }
    }
    return {max_abs, std::sqrt(squared_error / std::max(squared_expected, 1e-30))};
}

bool fields_close(const Fields& actual, const Fields& expected, double atol, double rtol) {
    for (std::size_t index = 0; index < actual.first.size(); ++index) {
        for (const auto& pair : {
                 std::pair<const std::vector<float>*, const std::vector<float>*>{
                     &actual.first, &expected.first},
                 std::pair<const std::vector<float>*, const std::vector<float>*>{
                     &actual.second, &expected.second},
             }) {
            const double a = (*pair.first)[index];
            const double e = (*pair.second)[index];
            if (!std::isfinite(a) || std::abs(a - e) > atol + rtol * std::abs(e)) {
                return false;
            }
        }
    }
    return true;
}

template <typename Runner>
float timed_run(Runner runner, DeviceFields& state, GrayScottParams params, int steps) {
    cudaEvent_t start;
    cudaEvent_t stop;
    cuda_check(cudaEventCreate(&start), "create start event");
    cuda_check(cudaEventCreate(&stop), "create stop event");
    cuda_check(cudaEventRecord(start), "record start event");
    runner(state, params, steps);
    cuda_check(cudaEventRecord(stop), "record stop event");
    cuda_check(cudaEventSynchronize(stop), "synchronize stop event");
    float milliseconds = 0.0f;
    cuda_check(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed time");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

double sample_std(const std::vector<float>& values) {
    if (values.size() < 2) {
        return 0.0;
    }
    const double mean =
        std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double sum = 0.0;
    for (float value : values) {
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

void write_samples(std::ostream& output, const std::vector<float>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << values[index];
    }
    output << ']';
}

void write_result(
    const std::filesystem::path& path,
    const std::string& candidate_id,
    const std::string& status,
    bool correct,
    const std::vector<Workload>& workloads,
    double compile_seconds,
    double end_to_end_seconds,
    const std::string& error
) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    output << std::setprecision(10);
    double log_speedup = 0.0;
    for (const auto& workload : workloads) {
        log_speedup += std::log(workload.baseline_median / workload.candidate_median);
    }
    const double speedup = workloads.empty() ? 0.0 : std::exp(log_speedup / workloads.size());
    int runtime_version = 0;
    int driver_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    output << "{\n"
           << "\"schema_version\":\"2.0\",\n"
           << "\"candidate_id\":\"" << json_escape(candidate_id) << "\",\n"
           << "\"language\":\"cpp\",\n"
           << "\"evaluator_version\":\"gray-scott-cuda-cpp-v1\",\n"
           << "\"status\":\"" << status << "\",\n"
           << "\"correct\":" << (correct ? "true" : "false") << ",\n"
           << "\"correctness_cases\":[],\n"
           << "\"workloads\":[";
    for (std::size_t index = 0; index < workloads.size(); ++index) {
        if (index) output << ',';
        const auto& workload = workloads[index];
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
               << ",\"relative_l2_error\":" << workload.errors.relative_l2 << '}';
    }
    output << "],\n\"geometric_mean_speedup\":";
    if (correct && !workloads.empty()) output << speedup; else output << "null";
    output << ",\n\"compile_seconds\":" << compile_seconds
           << ",\n\"end_to_end_seconds\":" << end_to_end_seconds
           << ",\n\"environment\":{\"gpu_name\":\"" << json_escape(properties.name)
           << "\",\"cuda_runtime_version\":" << runtime_version
           << ",\"cuda_driver_version\":" << driver_version
           << ",\"slurm_job_id\":\""
           << json_escape(std::getenv("SLURM_JOB_ID") ? std::getenv("SLURM_JOB_ID") : "")
           << "\"},\n\"error\":";
    if (error.empty()) output << "null"; else output << '"' << json_escape(error) << '"';
    output << "\n}\n";
}

int main(int argc, char** argv) {
    const auto started = std::chrono::steady_clock::now();
    std::filesystem::path output_path = "results/manual.json";
    std::string candidate_id = "manual";
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
        const auto correctness_shapes =
            parse_shapes(get("correctness-shapes", "31x29,64x64,127x65"));
        const auto performance_shapes = parse_shapes(get("shapes", "1024x1024"));
        const int steps = std::stoi(get("steps", "100"));
        const int warmups = std::stoi(get("warmups", "2"));
        const int repetitions = std::stoi(get("repetitions", "5"));
        const double atol = std::stod(get("atol", "2e-5"));
        const double rtol = std::stod(get("rtol", "2e-4"));
        const GrayScottParams params;

        bool correct = true;
        for (Shape shape : correctness_shapes) {
            for (int test_steps : {1, 5}) {
                const Fields initial = initialize_fields(shape);
                const Fields expected = cpu_simulate(initial, shape, params, test_steps);
                DeviceFields state = candidate::prepare_candidate(
                    initial.first, initial.second, shape.height, shape.width
                );
                candidate::run_candidate(state, params, test_steps);
                cuda_check(cudaDeviceSynchronize(), "correctness synchronize");
                const Fields actual = candidate::copy_candidate_fields(state);
                release_fields(state);
                correct = correct && fields_close(actual, expected, atol, rtol);
            }
        }

        std::vector<Workload> workloads;
        if (correct) {
            for (Shape shape : performance_shapes) {
                const Fields initial = initialize_fields(shape);
                for (int warmup = 0; warmup < warmups; ++warmup) {
                    auto baseline_state = baseline::prepare_baseline(
                        initial.first, initial.second, shape.height, shape.width
                    );
                    auto candidate_state = candidate::prepare_candidate(
                        initial.first, initial.second, shape.height, shape.width
                    );
                    timed_run(baseline::run_baseline, baseline_state, params, std::min(steps, 10));
                    timed_run(candidate::run_candidate, candidate_state, params, std::min(steps, 10));
                    release_fields(baseline_state);
                    release_fields(candidate_state);
                }

                Workload workload{shape};
                Fields last_candidate;
                for (int repetition = 0; repetition < repetitions; ++repetition) {
                    auto baseline_state = baseline::prepare_baseline(
                        initial.first, initial.second, shape.height, shape.width
                    );
                    auto candidate_state = candidate::prepare_candidate(
                        initial.first, initial.second, shape.height, shape.width
                    );
                    if (repetition % 2 == 0) {
                        workload.baseline_samples.push_back(
                            timed_run(baseline::run_baseline, baseline_state, params, steps)
                        );
                        workload.candidate_samples.push_back(
                            timed_run(candidate::run_candidate, candidate_state, params, steps)
                        );
                    } else {
                        workload.candidate_samples.push_back(
                            timed_run(candidate::run_candidate, candidate_state, params, steps)
                        );
                        workload.baseline_samples.push_back(
                            timed_run(baseline::run_baseline, baseline_state, params, steps)
                        );
                    }
                    last_candidate = candidate::copy_candidate_fields(candidate_state);
                    release_fields(baseline_state);
                    release_fields(candidate_state);
                }
                auto trusted = baseline::prepare_baseline(
                    initial.first, initial.second, shape.height, shape.width
                );
                baseline::run_baseline(trusted, params, steps);
                cuda_check(cudaDeviceSynchronize(), "validation synchronize");
                const Fields expected = baseline::copy_baseline_fields(trusted);
                release_fields(trusted);
                workload.errors = field_errors(last_candidate, expected);
                workload.correct = fields_close(last_candidate, expected, atol, rtol);
                workload.baseline_median = median(workload.baseline_samples);
                workload.candidate_median = median(workload.candidate_samples);
                correct = correct && workload.correct;
                workloads.push_back(std::move(workload));
            }
        }
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();
        write_result(
            output_path,
            candidate_id,
            correct ? "passed" : "failed",
            correct,
            workloads,
            compile_seconds,
            elapsed,
            ""
        );
        std::cout << output_path << '\n';
        return correct ? 0 : 2;
    } catch (const std::exception& exception) {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();
        write_result(
            output_path,
            candidate_id,
            "error",
            false,
            {},
            compile_seconds,
            elapsed,
            exception.what()
        );
        std::cerr << exception.what() << '\n';
        return 2;
    }
}
