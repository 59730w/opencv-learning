#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace day52 {

namespace fs = std::filesystem;

enum class Operation {
    Gray,
    Blur,
    Edge,
};

enum class ExecutionMode {
    Sequential,
    Threads,
};

struct ImageResult {
    fs::path input_path;
    fs::path output_path;
    bool read_ok = false;
    bool saved = false;
    int width = 0;
    int height = 0;
    int type = 0;
};

struct PipelineReport {
    std::vector<ImageResult> results;
    ExecutionMode mode = ExecutionMode::Sequential;
    unsigned int requested_workers = 1;
    unsigned int effective_workers = 1;
    unsigned int hardware_workers = 0;
    double elapsed_ms = 0.0;
};

std::optional<Operation> parse_operation(const std::string& name);
std::string operation_name(Operation operation);
std::optional<ExecutionMode> parse_execution_mode(const std::string& name);
std::string execution_mode_name(ExecutionMode mode);

// A pure function is intentionally public so worker-selection policy can be
// unit-tested without touching the filesystem or starting threads.
[[nodiscard]] unsigned int resolve_worker_count(
    ExecutionMode mode,
    unsigned int requested_workers,
    std::size_t image_count);

void print_summary(const std::vector<ImageResult>& results);

class ImagePipeline {
public:
    ImagePipeline(
        Operation operation,
        ExecutionMode mode,
        unsigned int worker_count);

    [[nodiscard]] PipelineReport process_directory(
        const fs::path& input_dir,
        const fs::path& output_dir) const;

private:
    void process_sequential(
        const std::vector<fs::path>& images,
        const fs::path& output_dir,
        std::vector<ImageResult>& results) const;
    void process_threads(
        const std::vector<fs::path>& images,
        const fs::path& output_dir,
        unsigned int effective_workers,
        std::vector<ImageResult>& results) const;
    [[nodiscard]] cv::Mat apply_operation(const cv::Mat& color) const;
    [[nodiscard]] fs::path output_path_for(
        const fs::path& input,
        const fs::path& output_dir) const;
    [[nodiscard]] ImageResult process_image(
        const fs::path& input,
        const fs::path& output_dir) const;

    Operation operation_;
    ExecutionMode mode_;
    unsigned int worker_count_;
};

}  // namespace day52
