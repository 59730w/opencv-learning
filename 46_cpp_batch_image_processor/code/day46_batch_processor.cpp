#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

struct ImageResult {
    fs::path path;
    bool loaded = false;
    int width = 0;
    int height = 0;
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

ImageResult load_image(const fs::path& path) {
    ImageResult result;
    result.path = path;
    const cv::Mat mat = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (!mat.empty()) {
        result.loaded = true;
        result.width = mat.cols;
        result.height = mat.rows;
    }
    return result;
}

void print_summary(const std::vector<ImageResult>& results) {
    int loaded = 0;
    std::cout << "Result summary:\n";
    for (const ImageResult& result : results) {
        if (result.loaded) {
            ++loaded;
            std::cout << "  OK:   " << result.path.string()
                      << "  " << result.width << " x " << result.height << '\n';
        } else {
            std::cout << "  SKIP: " << result.path.string() << " (unreadable)\n";
        }
    }
    std::cout << "Loaded " << loaded << " / " << results.size()
              << " image(s); skipped " << (results.size() - loaded)
              << " unreadable file(s).\n";
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: day46_batch_processor <directory>\n";
        return 2;
    }

    const fs::path directory{argv[1]};
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        std::cerr << "Not a directory: " << directory.string() << '\n';
        return 1;
    }

    const std::vector<fs::path> images = list_images(directory);
    std::cout << "Found " << images.size() << " image(s):\n";

    std::vector<ImageResult> results;
    results.reserve(images.size());
    for (const fs::path& image : images) {
        results.push_back(load_image(image));
    }

    print_summary(results);

    std::cout << "DAY46_BATCH_OK\n";
    return 0;
}
