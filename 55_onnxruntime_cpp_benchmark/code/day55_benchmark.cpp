#include "day54/preprocessing.hpp"
#include "day55/statistics.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::filesystem::path model;
    std::filesystem::path class_map;
    std::string mode;
    std::size_t batch_size{};
    std::size_t warmup_runs{};
    std::size_t measured_runs{};
    int intra_threads{};
    std::filesystem::path raw_output;
    std::filesystem::path summary_output;
    std::filesystem::path logits_output;
    std::vector<std::filesystem::path> images;
};

std::size_t parse_size(std::string_view text, const std::string& option, bool allow_zero) {
    std::size_t value = 0U;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        (!allow_zero && value == 0U)) {
        throw std::invalid_argument("invalid value for " + option + ": " + std::string(text));
    }
    return value;
}

Options parse_options(int argc, char* argv[]) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) throw std::invalid_argument("missing value for " + argument);
            return argv[index];
        };
        if (argument == "--model") options.model = value();
        else if (argument == "--class-map") options.class_map = value();
        else if (argument == "--mode") options.mode = value();
        else if (argument == "--batch-size") options.batch_size = parse_size(value(), argument, false);
        else if (argument == "--warmup") options.warmup_runs = parse_size(value(), argument, true);
        else if (argument == "--runs") options.measured_runs = parse_size(value(), argument, false);
        else if (argument == "--intra-threads") {
            const std::size_t parsed = parse_size(value(), argument, true);
            if (parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("intra thread count is too large");
            }
            options.intra_threads = static_cast<int>(parsed);
        } else if (argument == "--raw-output") options.raw_output = value();
        else if (argument == "--summary-output") options.summary_output = value();
        else if (argument == "--logits-output") options.logits_output = value();
        else if (argument == "--images") {
            while (index + 1 < argc) options.images.emplace_back(argv[++index]);
        } else throw std::invalid_argument("unknown argument: " + argument);
    }
    const bool batch_allowed = options.batch_size == 1U || options.batch_size == 3U ||
        options.batch_size == 6U;
    if (options.model.empty() || options.class_map.empty() || options.raw_output.empty() ||
        options.summary_output.empty() || options.logits_output.empty() ||
        (options.mode != "runtime-only" && options.mode != "end-to-end") ||
        !batch_allowed || options.measured_runs == 0U ||
        options.images.size() < options.batch_size) {
        throw std::invalid_argument("invalid or incomplete benchmark options");
    }
    return options;
}

void require_file(const std::filesystem::path& path, const char* label) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(std::string(label) + " missing: " + path.string());
    }
}

void validate_class_map(const std::filesystem::path& path) {
    cv::FileStorage file(path.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!file.isOpened() || !file.root().isMap() || file.root().size() != 50U) {
        throw std::runtime_error("expected a 50-class JSON map");
    }
    std::array<bool, 50> seen{};
    for (auto iterator = file.root().begin(); iterator != file.root().end(); ++iterator) {
        const int index = static_cast<int>(*iterator);
        if (index < 0 || index >= 50 || seen[static_cast<std::size_t>(index)]) {
            throw std::runtime_error("class indices must be unique and continuous");
        }
        seen[static_cast<std::size_t>(index)] = true;
    }
}

std::vector<float> load_batch(const Options& options) {
    constexpr std::size_t kElementsPerImage = 3U * 224U * 224U;
    std::vector<float> batch;
    batch.reserve(options.batch_size * kElementsPerImage);
    for (std::size_t index = 0; index < options.batch_size; ++index) {
        require_file(options.images[index], "image");
        const cv::Mat image = cv::imread(options.images[index].string(), cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("could not decode image");
        day54::PreprocessResult result = day54::preprocess_bgr(image);
        batch.insert(batch.end(), result.values.begin(), result.values.end());
    }
    return batch;
}

Ort::Value create_input_tensor(
    Ort::MemoryInfo& memory,
    std::vector<float>& input,
    const std::array<std::int64_t, 4>& shape
) {
    return Ort::Value::CreateTensor<float>(
        memory, input.data(), input.size(), shape.data(), shape.size()
    );
}

std::vector<Ort::Value> run_session(Ort::Session& session, Ort::Value& tensor) {
    constexpr std::array<const char*, 1> input_names{"images"};
    constexpr std::array<const char*, 1> output_names{"logits"};
    return session.Run(
        Ort::RunOptions{nullptr}, input_names.data(), &tensor, 1U, output_names.data(), 1U
    );
}

std::vector<float> extract_logits(
    const std::vector<Ort::Value>& values,
    std::size_t batch
) {
    const std::vector<std::int64_t> expected{static_cast<std::int64_t>(batch), 50};
    if (values.size() != 1U || !values[0].IsTensor() ||
        values[0].GetTensorTypeAndShapeInfo().GetShape() != expected) {
        throw std::runtime_error("unexpected output tensor");
    }
    const float* data = values[0].GetTensorData<float>();
    std::vector<float> logits(data, data + batch * 50U);
    if (std::any_of(logits.begin(), logits.end(), [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error("non-finite logits");
    }
    return logits;
}

void consume_top3(const std::vector<float>& logits, std::size_t batch) {
    for (std::size_t image = 0; image < batch; ++image) {
        std::array<std::size_t, 50> indices{};
        std::iota(indices.begin(), indices.end(), 0U);
        const float* row = logits.data() + image * 50U;
        std::partial_sort(indices.begin(), indices.begin() + 3, indices.end(),
            [&](std::size_t left, std::size_t right) { return row[left] > row[right]; });
    }
}

void ensure_parent(const std::filesystem::path& path) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
}

void write_outputs(const Options& options, const std::vector<float>& logits,
                   const std::vector<double>& timings, const day55::TimingSummary& summary,
                   double session_creation_ms) {
    ensure_parent(options.logits_output);
    ensure_parent(options.raw_output);
    ensure_parent(options.summary_output);
    std::ofstream binary(options.logits_output, std::ios::binary);
    binary.write(reinterpret_cast<const char*>(logits.data()),
        static_cast<std::streamsize>(logits.size() * sizeof(float)));
    std::ofstream raw(options.raw_output);
    raw << "mode,batch_size,intra_threads,run_index,elapsed_ms\n";
    for (std::size_t index = 0; index < timings.size(); ++index) {
        raw << options.mode << ',' << options.batch_size << ',' << options.intra_threads << ','
            << index + 1U << ',' << std::fixed << std::setprecision(6) << timings[index] << '\n';
    }
    std::ofstream aggregate(options.summary_output);
    aggregate << "mode,batch_size,intra_threads,session_creation_ms,min_ms,median_ms,p90_ms,max_ms,mean_ms,mean_ms_per_image,images_per_second\n";
    aggregate << options.mode << ',' << options.batch_size << ',' << options.intra_threads
        << std::fixed << std::setprecision(6) << ',' << session_creation_ms << ','
        << summary.minimum_ms << ','
        << summary.median_ms << ',' << summary.p90_ms << ',' << summary.maximum_ms << ','
        << summary.mean_ms << ',' << summary.mean_ms_per_image << ',' << summary.images_per_second << '\n';
    if (!binary || !raw || !aggregate) throw std::runtime_error("could not write outputs");
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const Options options = parse_options(argc, argv);
        require_file(options.model, "model");
        require_file(options.class_map, "class map");
        validate_class_map(options.class_map);
        cv::setNumThreads(1);
        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "day55_benchmark");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        if (options.intra_threads > 0) session_options.SetIntraOpNumThreads(options.intra_threads);
        session_options.SetInterOpNumThreads(1);
        const auto session_start = std::chrono::steady_clock::now();
        Ort::Session session(environment, options.model.c_str(), session_options);
        const double session_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - session_start
        ).count();

        std::vector<float> prepared;
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<std::int64_t, 4> input_shape{
            static_cast<std::int64_t>(options.batch_size), 3, 224, 224,
        };
        std::size_t input_tensor_creations = 0U;
        std::vector<Ort::Value> latest_outputs;
        Ort::Value prepared_tensor{nullptr};
        if (options.mode == "runtime-only") {
            prepared = load_batch(options);
            prepared_tensor = create_input_tensor(memory, prepared, input_shape);
            ++input_tensor_creations;
        }
        std::vector<float> latest_logits;
        auto execute = [&]() {
            if (options.mode == "runtime-only") {
                latest_outputs = run_session(session, prepared_tensor);
            } else {
                std::vector<float> current = load_batch(options);
                Ort::Value current_tensor = create_input_tensor(memory, current, input_shape);
                ++input_tensor_creations;
                latest_outputs = run_session(session, current_tensor);
                latest_logits = extract_logits(latest_outputs, options.batch_size);
                consume_top3(latest_logits, options.batch_size);
            }
        };
        for (std::size_t run = 0; run < options.warmup_runs; ++run) execute();
        std::vector<double> timings;
        timings.reserve(options.measured_runs);
        for (std::size_t run = 0; run < options.measured_runs; ++run) {
            const auto start = std::chrono::steady_clock::now();
            execute();
            timings.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start
            ).count());
        }
        if (options.mode == "runtime-only") {
            latest_logits = extract_logits(latest_outputs, options.batch_size);
            consume_top3(latest_logits, options.batch_size);
        }
        const day55::TimingSummary summary = day55::summarize_timings(timings, options.batch_size);
        write_outputs(options, latest_logits, timings, summary, session_ms);
        std::cout << "Mode: " << options.mode << '\n'
                  << "Batch size: " << options.batch_size << '\n'
                  << "Warm-up runs: " << options.warmup_runs << '\n'
                  << "Measured runs: " << options.measured_runs << '\n'
                  << "Intra-op threads: " << options.intra_threads << '\n'
                  << "Timed boundary: "
                  << (options.mode == "runtime-only"
                      ? "Session::Run only"
                      : "decode + preprocess + tensor + Session::Run + Top-k") << '\n'
                  << "Input tensor creations: " << input_tensor_creations << '\n'
                  << "Session creation ms: " << std::fixed << std::setprecision(6) << session_ms << '\n'
                  << "Median ms: " << summary.median_ms << '\n'
                  << "P90 ms: " << summary.p90_ms << '\n'
                  << "Images/s: " << summary.images_per_second << '\n'
                  << "DAY55_BENCHMARK_OK\n";
        return 0;
    } catch (const std::invalid_argument& error) {
        std::cerr << "Argument error: " << error.what() << '\n';
        return 2;
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
        return 3;
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: " << error.what() << '\n';
        return 4;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 5;
    }
}
