#include "day54/preprocessing.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::filesystem::path model;
    std::size_t batch_size{};
    std::size_t warmup_runs{};
    std::size_t profiled_runs{};
    int intra_threads{};
    std::filesystem::path profile_output;
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
        else if (argument == "--batch-size") options.batch_size = parse_size(value(), argument, false);
        else if (argument == "--warmup") options.warmup_runs = parse_size(value(), argument, true);
        else if (argument == "--runs") options.profiled_runs = parse_size(value(), argument, false);
        else if (argument == "--intra-threads") {
            const std::size_t parsed = parse_size(value(), argument, true);
            if (parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("intra thread count is too large");
            }
            options.intra_threads = static_cast<int>(parsed);
        } else if (argument == "--profile-output") options.profile_output = value();
        else if (argument == "--logits-output") options.logits_output = value();
        else if (argument == "--images") {
            while (index + 1 < argc) options.images.emplace_back(argv[++index]);
        } else throw std::invalid_argument("unknown argument: " + argument);
    }
    const bool batch_allowed = options.batch_size == 1U || options.batch_size == 6U;
    if (options.model.empty() || options.profile_output.empty() || options.logits_output.empty() ||
        !batch_allowed || options.profiled_runs == 0U || options.images.size() < options.batch_size) {
        throw std::invalid_argument("invalid or incomplete profiling options");
    }
    return options;
}

void require_file(const std::filesystem::path& path, const char* label) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(std::string(label) + " missing: " + path.string());
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

Ort::SessionOptions make_session_options(int intra_threads,
                                         const std::filesystem::path* profile_prefix = nullptr) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    if (intra_threads > 0) options.SetIntraOpNumThreads(intra_threads);
    options.SetInterOpNumThreads(1);
    if (profile_prefix != nullptr) options.EnableProfiling(profile_prefix->c_str());
    return options;
}

Ort::Value make_tensor(Ort::MemoryInfo& memory, std::vector<float>& input,
                       const std::array<std::int64_t, 4>& shape) {
    return Ort::Value::CreateTensor<float>(
        memory, input.data(), input.size(), shape.data(), shape.size());
}

std::vector<Ort::Value> run(Ort::Session& session, Ort::Value& tensor) {
    constexpr std::array<const char*, 1> input_names{"images"};
    constexpr std::array<const char*, 1> output_names{"logits"};
    return session.Run(Ort::RunOptions{nullptr}, input_names.data(), &tensor, 1U,
                       output_names.data(), 1U);
}

std::vector<float> extract_logits(const std::vector<Ort::Value>& outputs, std::size_t batch) {
    const std::vector<std::int64_t> expected{static_cast<std::int64_t>(batch), 50};
    if (outputs.size() != 1U || !outputs[0].IsTensor() ||
        outputs[0].GetTensorTypeAndShapeInfo().GetShape() != expected) {
        throw std::runtime_error("unexpected output tensor");
    }
    const float* data = outputs[0].GetTensorData<float>();
    std::vector<float> logits(data, data + batch * 50U);
    if (std::any_of(logits.begin(), logits.end(), [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error("non-finite logits");
    }
    return logits;
}

void ensure_parent(const std::filesystem::path& path) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
}

void move_profile(const char* produced_name, const std::filesystem::path& requested) {
    const std::filesystem::path produced = produced_name;
    if (!std::filesystem::is_regular_file(produced)) {
        throw std::runtime_error("ONNX Runtime did not create profile JSON");
    }
    ensure_parent(requested);
    if (std::filesystem::exists(requested)) std::filesystem::remove(requested);
    std::filesystem::rename(produced, requested);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const Options options = parse_options(argc, argv);
        require_file(options.model, "model");
        ensure_parent(options.profile_output);
        ensure_parent(options.logits_output);
        cv::setNumThreads(1);
        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "day56_profiling");
        std::vector<float> input = load_batch(options);
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<std::int64_t, 4> shape{
            static_cast<std::int64_t>(options.batch_size), 3, 224, 224};

        if (options.warmup_runs > 0U) {
            Ort::SessionOptions warmup_options = make_session_options(options.intra_threads);
            Ort::Session warmup_session(environment, options.model.c_str(), warmup_options);
            Ort::Value tensor = make_tensor(memory, input, shape);
            for (std::size_t index = 0; index < options.warmup_runs; ++index) {
                (void)run(warmup_session, tensor);
            }
        }

        const std::filesystem::path prefix = options.profile_output.parent_path() /
            (options.profile_output.stem().string() + "_ort_");
        Ort::SessionOptions profile_options = make_session_options(options.intra_threads, &prefix);
        Ort::Session profile_session(environment, options.model.c_str(), profile_options);
        Ort::Value tensor = make_tensor(memory, input, shape);
        std::vector<Ort::Value> latest_outputs;
        for (std::size_t index = 0; index < options.profiled_runs; ++index) {
            latest_outputs = run(profile_session, tensor);
        }
        const std::vector<float> logits = extract_logits(latest_outputs, options.batch_size);
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr produced = profile_session.EndProfilingAllocated(allocator);
        move_profile(produced.get(), options.profile_output);

        std::ofstream binary(options.logits_output, std::ios::binary);
        binary.write(reinterpret_cast<const char*>(logits.data()),
                     static_cast<std::streamsize>(logits.size() * sizeof(float)));
        if (!binary) throw std::runtime_error("could not write logits");

        std::cout << "Batch size: " << options.batch_size << '\n'
                  << "Warm-up runs (separate session): " << options.warmup_runs << '\n'
                  << "Profiled runs: " << options.profiled_runs << '\n'
                  << "Intra-op threads: " << options.intra_threads << '\n'
                  << "Profile output: " << options.profile_output.string() << '\n'
                  << "DAY56_PROFILING_OK\n";
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
