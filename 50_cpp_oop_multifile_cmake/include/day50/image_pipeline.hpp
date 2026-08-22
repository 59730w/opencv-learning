#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace day50 {

namespace fs = std::filesystem;

enum class Operation {
    Gray,
    Blur,
    Edge,
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

std::optional<Operation> parse_operation(const std::string& name);
std::string operation_name(Operation operation);
void print_summary(const std::vector<ImageResult>& results);

class ImagePipeline {
public:
    explicit ImagePipeline(Operation operation) noexcept;

    [[nodiscard]] Operation operation() const noexcept;
    [[nodiscard]] std::vector<ImageResult> process_directory(
        const fs::path& input_dir,
        const fs::path& output_dir) const;

private:
    [[nodiscard]] cv::Mat apply_operation(const cv::Mat& color) const;
    [[nodiscard]] fs::path output_path_for(
        const fs::path& input,
        const fs::path& output_dir) const;
    [[nodiscard]] ImageResult process_image(
        const fs::path& input,
        const fs::path& output_dir) const;

    Operation operation_;
};

}  // namespace day50
