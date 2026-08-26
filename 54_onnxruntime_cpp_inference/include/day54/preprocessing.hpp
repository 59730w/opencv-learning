#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace day54 {

struct PreprocessResult {
    std::vector<float> values;
    std::array<std::int64_t, 4> shape;
    cv::Size resized_size;
    cv::Rect crop;
};

PreprocessResult preprocess_bgr(const cv::Mat& bgr_image);

}  // namespace day54
