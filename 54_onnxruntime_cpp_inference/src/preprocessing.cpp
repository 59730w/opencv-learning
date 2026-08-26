#include "day54/preprocessing.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>

namespace day54 {
namespace {

constexpr int kResizeShortSide = 256;
constexpr int kCropSize = 224;
constexpr std::array<float, 3> kImageNetMean{0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kImageNetStd{0.229F, 0.224F, 0.225F};

cv::Size resized_size_for(const cv::Size& source_size) {
    if (source_size.width <= 0 || source_size.height <= 0) {
        throw std::invalid_argument("source image dimensions must be positive");
    }

    if (source_size.width < source_size.height) {
        const int resized_height =
            kResizeShortSide * source_size.height / source_size.width;
        return {kResizeShortSide, resized_height};
    }

    const int resized_width =
        kResizeShortSide * source_size.width / source_size.height;
    return {resized_width, kResizeShortSide};
}

}  // namespace

PreprocessResult preprocess_bgr(const cv::Mat& bgr_image) {
    if (bgr_image.empty()) {
        throw std::invalid_argument("input image must not be empty");
    }
    if (bgr_image.type() != CV_8UC3) {
        throw std::invalid_argument("input image must be an 8-bit three-channel BGR image");
    }

    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

    const cv::Size resized_size = resized_size_for(rgb_image.size());
    const bool shrinking =
        resized_size.width < rgb_image.cols || resized_size.height < rgb_image.rows;

    cv::Mat resized_image;
    cv::resize(
        rgb_image,
        resized_image,
        resized_size,
        0.0,
        0.0,
        shrinking ? cv::INTER_AREA : cv::INTER_LINEAR
    );

    const cv::Rect crop(
        (resized_image.cols - kCropSize) / 2,
        (resized_image.rows - kCropSize) / 2,
        kCropSize,
        kCropSize
    );
    const cv::Mat cropped_image = resized_image(crop);

    cv::Mat float_rgb;
    cropped_image.convertTo(float_rgb, CV_32FC3, 1.0 / 255.0);

    const std::size_t channel_size =
        static_cast<std::size_t>(kCropSize) * static_cast<std::size_t>(kCropSize);
    std::vector<float> tensor_values(3U * channel_size);

    for (int row = 0; row < kCropSize; ++row) {
        const auto* pixels = float_rgb.ptr<cv::Vec3f>(row);
        for (int column = 0; column < kCropSize; ++column) {
            const std::size_t pixel_index =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(kCropSize) +
                static_cast<std::size_t>(column);
            for (int channel = 0; channel < 3; ++channel) {
                const std::size_t channel_index = static_cast<std::size_t>(channel);
                tensor_values[channel_index * channel_size + pixel_index] =
                    (pixels[column][channel] - kImageNetMean[channel_index]) /
                    kImageNetStd[channel_index];
            }
        }
    }

    return {
        std::move(tensor_values),
        {1, 3, kCropSize, kCropSize},
        resized_size,
        crop,
    };
}

}  // namespace day54
