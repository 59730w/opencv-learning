#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

int main() {
    constexpr std::array<std::uint8_t, 6> gray_values{
        10, 80, 127,
        128, 180, 255
    };

    cv::Mat gray(2, 3, CV_8U);
    std::copy(gray_values.begin(), gray_values.end(), gray.ptr<std::uint8_t>());
    const cv::Scalar mean_value = cv::mean(gray);

    cv::Mat binary;
    constexpr double threshold_value = 127.0;
    cv::threshold(gray, binary, threshold_value, 255.0, cv::THRESH_BINARY);

    std::cout << "OpenCV version: " << CV_VERSION << '\n';
    std::cout << "Shape: " << gray.rows << " x " << gray.cols << '\n';
    std::cout << "Channels: " << gray.channels() << '\n';
    std::cout << "Mean: " << mean_value[0] << '\n';
    std::cout << "Depth: " << gray.depth() << " (CV_8U = " << CV_8U << ")\n";
    std::cout << "Continuous memory: " << std::boolalpha << gray.isContinuous() << '\n';
    std::cout << "Input pixels:\n" << gray << '\n';
    std::cout << "Binary pixels (threshold = " << threshold_value << "):\n"
              << binary << '\n';

    cv::Mat expected = cv::Mat::zeros(2, 3, CV_8U);
    expected.at<std::uint8_t>(1, 0) = 255;
    expected.at<std::uint8_t>(1, 1) = 255;
    expected.at<std::uint8_t>(1, 2) = 255;

    if (cv::countNonZero(binary != expected) != 0) {
        std::cerr << "Unexpected threshold result.\n";
        return 1;
    }

    std::cout << "DAY44_OPENCV_OK\n";
    return 0;
}
