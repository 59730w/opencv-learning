#include "day51/image_pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace day51 {
namespace {

bool is_image_extension(const fs::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    return extension == ".jpg" || extension == ".jpeg"
        || extension == ".png" || extension == ".bmp"
        || extension == ".tif" || extension == ".tiff";
}

std::vector<fs::path> list_images(const fs::path& directory) {
    std::vector<fs::path> images;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && is_image_extension(entry.path())) {
            images.push_back(entry.path());
        }
    }
    std::sort(images.begin(), images.end());
    return images;
}

}  // namespace

std::optional<Operation> parse_operation(const std::string& name) {
    if (name == "gray") {
        return Operation::Gray;
    }
    if (name == "blur") {
        return Operation::Blur;
    }
    if (name == "edge") {
        return Operation::Edge;
    }
    return std::nullopt;
}

std::string operation_name(Operation operation) {
    switch (operation) {
        case Operation::Gray:
            return "gray";
        case Operation::Blur:
            return "blur";
        case Operation::Edge:
            return "edge";
    }
    return "unknown";
}

std::optional<ExecutionMode> parse_execution_mode(const std::string& name) {
    if (name == "sequential") {
        return ExecutionMode::Sequential;
    }
    if (name == "threads") {
        return ExecutionMode::Threads;
    }
    return std::nullopt;
}

std::string execution_mode_name(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::Sequential:
            return "sequential";
        case ExecutionMode::Threads:
            return "threads";
    }
    return "unknown";
}

void print_summary(const std::vector<ImageResult>& results) {
    int saved_count = 0;
    int read_failed = 0;
    int write_failed = 0;

    std::cout << "Result summary:\n";
    for (const ImageResult& result : results) {
        if (!result.read_ok) {
            ++read_failed;
            std::cout << "  SKIP (read-failed): "
                      << result.input_path.string() << '\n';
        } else if (!result.saved) {
            ++write_failed;
            std::cout << "  FAIL (write-failed): "
                      << result.output_path.string() << '\n';
        } else {
            ++saved_count;
            std::cout << "  OK: " << result.output_path.string()
                      << "  " << result.width << " x " << result.height
                      << "  type=" << result.type << '\n';
        }
    }

    std::cout << "Saved " << saved_count << " / " << results.size()
              << " image(s); read-failed " << read_failed
              << "; write-failed " << write_failed << ".\n";
}

ImagePipeline::ImagePipeline(
    Operation operation,
    ExecutionMode mode,
    unsigned int worker_count)
    : operation_(operation),
      mode_(mode),
      worker_count_(worker_count) {
    if (worker_count_ == 0) {
        throw std::invalid_argument("worker_count must be positive");
    }
}

PipelineReport ImagePipeline::process_directory(
    const fs::path& input_dir,
    const fs::path& output_dir) const {
    cv::setNumThreads(1);
    const std::vector<fs::path> images = list_images(input_dir);

    PipelineReport report;
    report.mode = mode_;
    report.requested_workers = worker_count_;
    report.effective_workers = effective_worker_count(images.size());
    report.hardware_workers = std::thread::hardware_concurrency();
    report.results.resize(images.size());

    const auto start = std::chrono::steady_clock::now();
    if (mode_ == ExecutionMode::Sequential) {
        process_sequential(images, output_dir, report.results);
    } else {
        process_threads(
            images,
            output_dir,
            report.effective_workers,
            report.results);
    }
    const auto end = std::chrono::steady_clock::now();

    report.elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return report;
}

unsigned int ImagePipeline::effective_worker_count(
    std::size_t image_count) const noexcept {
    if (mode_ == ExecutionMode::Sequential || image_count <= 1) {
        return 1;
    }
    const auto capped_count = std::min<std::size_t>(worker_count_, image_count);
    return static_cast<unsigned int>(capped_count);
}

void ImagePipeline::process_sequential(
    const std::vector<fs::path>& images,
    const fs::path& output_dir,
    std::vector<ImageResult>& results) const {
    for (std::size_t index = 0; index < images.size(); ++index) {
        results[index] = process_image(images[index], output_dir);
    }
}

void ImagePipeline::process_threads(
    const std::vector<fs::path>& images,
    const fs::path& output_dir,
    unsigned int effective_workers,
    std::vector<ImageResult>& results) const {
    if (images.empty()) {
        return;
    }

    std::atomic<std::size_t> next_index{0};
    std::atomic<bool> stop_requested{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;
    std::vector<std::thread> workers;
    workers.reserve(effective_workers);

    for (unsigned int worker = 0; worker < effective_workers; ++worker) {
        workers.emplace_back([&]() {
            try {
                while (!stop_requested.load()) {
                    const std::size_t index = next_index.fetch_add(1);
                    if (index >= images.size()) {
                        break;
                    }
                    results[index] = process_image(images[index], output_dir);
                }
            } catch (...) {
                stop_requested.store(true);
                const std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
        });
    }

    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    if (first_error) {
        std::rethrow_exception(first_error);
    }
}

cv::Mat ImagePipeline::apply_operation(const cv::Mat& color) const {
    cv::Mat result;
    if (operation_ == Operation::Gray) {
        cv::cvtColor(color, result, cv::COLOR_BGR2GRAY);
    } else if (operation_ == Operation::Blur) {
        cv::GaussianBlur(color, result, cv::Size(7, 7), 1.5);
    } else {
        cv::Mat gray;
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, result, 80.0, 160.0);
    }
    return result;
}

fs::path ImagePipeline::output_path_for(
    const fs::path& input,
    const fs::path& output_dir) const {
    return output_dir
        / (input.stem().string() + "_" + operation_name(operation_) + ".jpg");
}

ImageResult ImagePipeline::process_image(
    const fs::path& input,
    const fs::path& output_dir) const {
    ImageResult result;
    result.input_path = input;
    result.output_path = output_path_for(input, output_dir);

    const cv::Mat color = cv::imread(input.string(), cv::IMREAD_COLOR);
    if (color.empty()) {
        return result;
    }
    result.read_ok = true;

    const cv::Mat processed = apply_operation(color);
    result.width = processed.cols;
    result.height = processed.rows;
    result.type = processed.type();
    result.saved = cv::imwrite(result.output_path.string(), processed);
    return result;
}

}  // namespace day51
