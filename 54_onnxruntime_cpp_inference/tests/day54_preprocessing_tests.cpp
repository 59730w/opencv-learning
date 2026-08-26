#include "day54/preprocessing.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect_true(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
void expect_near(float actual, float expected, float tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

}  // namespace

int main() {
    try {
        const cv::Mat constant_bgr(300, 400, CV_8UC3, cv::Scalar(10, 20, 30));
        const day54::PreprocessResult result = day54::preprocess_bgr(constant_bgr);

        const std::array<std::int64_t, 4> expected_shape{1, 3, 224, 224};
        expect_true(result.shape == expected_shape, "NCHW shape mismatch");
        expect_true(result.values.size() == 3U * 224U * 224U, "tensor element count mismatch");
        expect_true(result.resized_size == cv::Size(341, 256), "resize-short-side geometry mismatch");
        expect_true(result.crop == cv::Rect(58, 16, 224, 224), "center crop geometry mismatch");

        const float expected_red = (30.0F / 255.0F - 0.485F) / 0.229F;
        const float expected_green = (20.0F / 255.0F - 0.456F) / 0.224F;
        const float expected_blue = (10.0F / 255.0F - 0.406F) / 0.225F;
        const std::size_t channel_size = 224U * 224U;

        expect_near(result.values[0], expected_red, 1.0e-6F, "red channel mismatch");
        expect_near(result.values[channel_size], expected_green, 1.0e-6F, "green channel mismatch");
        expect_near(result.values[2U * channel_size], expected_blue, 1.0e-6F, "blue channel mismatch");
        expect_near(result.values.back(), expected_blue, 1.0e-6F, "CHW tail mismatch");

        bool empty_rejected = false;
        try {
            static_cast<void>(day54::preprocess_bgr(cv::Mat{}));
        } catch (const std::invalid_argument&) {
            empty_rejected = true;
        }
        expect_true(empty_rejected, "empty image must be rejected");

        std::cout << "DAY54_PREPROCESS_UNIT_OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
}
