#include "day55/statistics.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_close(double actual, double expected, const char* label) {
    if (std::abs(actual - expected) > 1e-9) {
        throw std::runtime_error(std::string(label) + " mismatch");
    }
}

}  // namespace

int main() {
    try {
        const day55::TimingSummary summary = day55::summarize_timings(
            std::vector<double>{10.0, 1.0, 7.0, 3.0},
            2U
        );
        require_close(summary.minimum_ms, 1.0, "minimum");
        require_close(summary.median_ms, 5.0, "median");
        require_close(summary.p90_ms, 10.0, "p90");
        require_close(summary.maximum_ms, 10.0, "maximum");
        require_close(summary.mean_ms, 5.25, "mean");
        require_close(summary.mean_ms_per_image, 2.625, "mean per image");
        require_close(summary.images_per_second, 2000.0 / 5.25, "throughput");

        bool empty_rejected = false;
        try {
            static_cast<void>(day55::summarize_timings({}, 1U));
        } catch (const std::invalid_argument&) {
            empty_rejected = true;
        }
        if (!empty_rejected) {
            throw std::runtime_error("empty timings were accepted");
        }

        bool zero_batch_rejected = false;
        try {
            static_cast<void>(day55::summarize_timings({1.0}, 0U));
        } catch (const std::invalid_argument&) {
            zero_batch_rejected = true;
        }
        if (!zero_batch_rejected) {
            throw std::runtime_error("zero batch was accepted");
        }

        std::cout << "DAY55_STATISTICS_TESTS_OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
}
