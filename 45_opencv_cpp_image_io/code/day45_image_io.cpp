#include <filesystem>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

bool shares_pixel_buffer(const cv::Mat& first, const cv::Mat& second) {
    return first.data == second.data;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: day45_image_io <image_path>\n";
        return 2;
    }

    const std::filesystem::path image_path{argv[1]};
    const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Could not read image: " << image_path.string() << '\n';
        return 1;
    }

    const cv::Mat shallow_copy = image;
    const cv::Mat deep_copy = image.clone();

    const bool shallow_shares_data =
        shares_pixel_buffer(image, shallow_copy);
    const bool deep_has_own_data =
        !shares_pixel_buffer(image, deep_copy);

    std::cout << "Image data address: "
              << static_cast<const void*>(image.data) << '\n';
    std::cout << "Shallow-copy data address: "
              << static_cast<const void*>(shallow_copy.data) << '\n';
    std::cout << "Deep-copy data address: "
              << static_cast<const void*>(deep_copy.data) << '\n';

    std::cout << "Shallow copy shares data: "
              << std::boolalpha << shallow_shares_data << '\n';
    std::cout << "Deep copy owns different data: "
              << deep_has_own_data << '\n';

    if (!shallow_shares_data || !deep_has_own_data) {
        std::cerr << "Unexpected cv::Mat copy behavior.\n";
        return 3;
    }

    std::cout << "DAY45_MAT_OWNERSHIP_OK\n";
    std::cout << "DAY45_CONST_REFERENCE_OK\n";

    std::cout << "OpenCV version: " << CV_VERSION << '\n';
    std::cout << "Image path: " << image_path.string() << '\n';
    std::cout << "Width (cols): " << image.cols << '\n';
    std::cout << "Height (rows): " << image.rows << '\n';
    std::cout << "Channels: " << image.channels() << '\n';
    std::cout << "Depth: " << image.depth() << " (CV_8U = " << CV_8U << ")\n";
    std::cout << "Type: " << image.type() << " (CV_8UC3 = " << CV_8UC3 << ")\n";
    std::cout << "Continuous memory: " << std::boolalpha << image.isContinuous() << '\n';
    cv::imshow("Day45 Input", image);
    std::cout << "Click the image window, then press any key to close it.\n";

    cv::waitKey(0);
    cv::destroyAllWindows();

    std::cout << "DAY45_IMAGE_DISPLAY_OK\n";

    return 0;
}
