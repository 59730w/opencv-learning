#include "day50/image_pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace day50 {
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

ImagePipeline::ImagePipeline(Operation operation) noexcept
    : operation_(operation) {
}

Operation ImagePipeline::operation() const noexcept {
    return operation_;
}

std::vector<ImageResult> ImagePipeline::process_directory(
    const fs::path& input_dir,
    const fs::path& output_dir) const {
    const std::vector<fs::path> images = list_images(input_dir);
    std::vector<ImageResult> results;
    results.reserve(images.size());

    for (const fs::path& image : images) {
        results.push_back(process_image(image, output_dir));
    }
    return results;
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

}  // namespace day50
