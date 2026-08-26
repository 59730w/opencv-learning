#include "day54/preprocessing.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void write_float_tensor(const std::filesystem::path& output_path, const std::vector<float>& values) {
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("could not open tensor output: " + output_path.string());
    }

    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float))
    );
    if (!output) {
        throw std::runtime_error("could not write tensor output: " + output_path.string());
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 4 || std::string(argv[2]) != "--output") {
        std::cerr << "Usage: day54_preprocess_images <image> --output <tensor.bin>\n";
        return 2;
    }

    const std::filesystem::path image_path = argv[1];
    const std::filesystem::path output_path = argv[3];

    try {
        const cv::Mat bgr_image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (bgr_image.empty()) {
            std::cerr << "Could not read image: " << image_path.string() << '\n';
            return 3;
        }

        const day54::PreprocessResult result = day54::preprocess_bgr(bgr_image);
        write_float_tensor(output_path, result.values);

        std::cout << "Image: " << std::filesystem::absolute(image_path).string() << '\n';
        std::cout << "Source size: " << bgr_image.cols << 'x' << bgr_image.rows << '\n';
        std::cout << "Resized size: " << result.resized_size.width << 'x'
                  << result.resized_size.height << '\n';
        std::cout << "Crop: x=" << result.crop.x << ", y=" << result.crop.y
                  << ", width=" << result.crop.width
                  << ", height=" << result.crop.height << '\n';
        std::cout << "Tensor shape: [1, 3, 224, 224]\n";
        std::cout << "Tensor dtype: float32\n";
        std::cout << "Tensor elements: " << result.values.size() << '\n';
        std::cout << "DAY54_PREPROCESS_OK\n";
        return 0;
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
        return 4;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 5;
    }
}
