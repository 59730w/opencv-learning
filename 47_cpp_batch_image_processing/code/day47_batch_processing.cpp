#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

struct ImageResult {
    fs::path path;
    bool read_ok = false;
    bool saved = false;
    int width = 0;
    int height = 0;
    int type = 0;
    fs::path output_path;
};

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
    return images;
}

fs::path grayscale_output_path(const fs::path& image_path,
                               const fs::path& output_dir) {
    return output_dir / (image_path.stem().string() + "_gray.jpg");
}

ImageResult process_image(const fs::path& path, const fs::path& output_dir) {
    ImageResult result;
    result.path = path;

    const cv::Mat color = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (color.empty()) {
        return result;
    }
    result.read_ok = true;

    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    result.width = gray.cols;
    result.height = gray.rows;
    result.type = gray.type();

    result.output_path = grayscale_output_path(path, output_dir);
    result.saved = cv::imwrite(result.output_path.string(), gray);
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
            std::cout << "  FAIL (write-failed): " << result.output_path.string() << '\n';
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
    if (argc != 3) {
        std::cerr << "Usage: day47_batch_processing <input_dir> <output_dir>\n";
        return 2;
    }

    const fs::path input_dir{argv[1]};
    const fs::path output_dir{argv[2]};

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
    std::cout << "Found " << images.size() << " image(s)\n";

    std::vector<ImageResult> results;
    results.reserve(images.size());
    for (const fs::path& image : images) {
        results.push_back(process_image(image, output_dir));
    }

    print_summary(results);

    std::cout << "DAY47_BATCH_OK\n";
    return 0;
}
