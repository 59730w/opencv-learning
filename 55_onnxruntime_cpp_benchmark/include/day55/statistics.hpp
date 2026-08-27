#pragma once

#include <cstddef>
#include <vector>

namespace day55 {

struct TimingSummary {
    double minimum_ms{};
    double median_ms{};
    double p90_ms{};
    double maximum_ms{};
    double mean_ms{};
    double mean_ms_per_image{};
    double images_per_second{};
};

TimingSummary summarize_timings(const std::vector<double>& elapsed_ms, std::size_t batch_size);

}  // namespace day55
