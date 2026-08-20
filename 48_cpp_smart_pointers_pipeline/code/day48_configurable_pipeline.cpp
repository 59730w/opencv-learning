#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

enum class Operation { Gray, Blur, Edge };

struct ImageResult {
    fs::path path;
    bool read_ok = false;
    bool saved = false;
    int width = 0;
    int height = 0;
    int type = 0;
    fs::path output_path;
};

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

bool is_image_extension(const fs::path& path) {
    const std::string ext = path.extension().string();
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png"
        || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
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

cv::Mat apply_operation(const cv::Mat& color, Operation operation) {
    cv::Mat result;
    if (operation == Operation::Gray) {
        cv::cvtColor(color, result, cv::COLOR_BGR2GRAY);
    } else if (operation == Operation::Blur) {
        cv::GaussianBlur(color, result, cv::Size(7, 7), 1.5);
    } else {
        cv::Mat gray;
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, result, 80.0, 160.0);
    }
    return result;
}

fs::path output_path_for(const fs::path& input,
                         const fs::path& output_dir,
                         Operation operation) {
    return output_dir
        / (input.stem().string() + "_" + operation_name(operation) + ".jpg");
}

ImageResult process_image(const fs::path& path,
                          const fs::path& output_dir,
                          Operation operation) {
    ImageResult result;
    result.path = path;

    const cv::Mat color = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (color.empty()) {
        return result;
    }
    result.read_ok = true;

    const cv::Mat processed = apply_operation(color, operation);
    result.width = processed.cols;
    result.height = processed.rows;
    result.type = processed.type();
    result.output_path = output_path_for(path, output_dir, operation);
    result.saved = cv::imwrite(result.output_path.string(), processed);
    return result;
}

void print_summary(const std::vector<ImageResult>& results) {
    int saved = 0;
    int read_failed = 0;
    int write_failed = 0;

    std::cout << "Result summary:\n";
    for (const ImageResult& result : results) {
        if (!result.read_ok) {
            ++read_failed;
            std::cout << "  SKIP (read-failed): " << result.path.string() << '\n';
        } else if (!result.saved) {
            ++write_failed;
            std::cout << "  FAIL (write-failed): "
                      << result.output_path.string() << '\n';
        } else {
            ++saved;
            std::cout << "  OK: " << result.output_path.string()
                      << "  " << result.width << " x " << result.height
                      << "  type=" << result.type << '\n';
        }
    }

    std::cout << "Saved " << saved << " / " << results.size()
              << " image(s); read-failed " << read_failed
              << "; write-failed " << write_failed << ".\n";
}

int main(int argc, char* argv[]) {
    if (argc != 5 || std::string(argv[3]) != "--op") {
        std::cerr
            << "Usage: day48_pipeline <input_dir> <output_dir> "
            << "--op <gray|blur|edge>\n";
        return 2;
    }

    const fs::path input_dir{argv[1]};
    const fs::path output_dir{argv[2]};
    const std::optional<Operation> operation = parse_operation(argv[4]);
    if (!operation.has_value()) {
        std::cerr << "Unknown operation: " << argv[4] << '\n';
        return 4;
    }

    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "Not a directory: " << input_dir.string() << '\n';
        return 1;
    }

    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        std::cerr << "Cannot create output directory: " << output_dir.string()
                  << " (" << ec.message() << ")\n";
        return 3;
    }

    const std::vector<fs::path> images = list_images(input_dir);
    std::cout << "Operation: " << operation_name(*operation) << '\n';
    std::cout << "Found " << images.size() << " image(s)\n";

    std::vector<ImageResult> results;
    results.reserve(images.size());
    for (const fs::path& image : images) {
        results.push_back(process_image(image, output_dir, *operation));
    }

    print_summary(results);
    std::cout << "DAY48_PIPELINE_OK\n";
    return 0;
}
