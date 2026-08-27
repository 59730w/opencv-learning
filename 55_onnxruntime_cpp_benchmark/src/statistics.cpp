#include "day55/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace day55 {

TimingSummary summarize_timings(const std::vector<double>& elapsed_ms, std::size_t batch_size) {
    if (elapsed_ms.empty() || batch_size == 0U) {
        throw std::invalid_argument("timings must be non-empty and batch must be positive");
    }
    if (std::any_of(elapsed_ms.begin(), elapsed_ms.end(), [](double value) {
            return !std::isfinite(value) || value <= 0.0;
        })) {
        throw std::invalid_argument("timings must be finite and positive");
    }
    std::vector<double> sorted = elapsed_ms;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t count = sorted.size();
    const double median = count % 2U == 0U
        ? (sorted[count / 2U - 1U] + sorted[count / 2U]) / 2.0
        : sorted[count / 2U];
    const std::size_t p90_rank = static_cast<std::size_t>(
        std::ceil(0.90 * static_cast<double>(count))
    );
    const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
        static_cast<double>(count);
    return {
        sorted.front(), median, sorted[p90_rank - 1U], sorted.back(), mean,
        mean / static_cast<double>(batch_size),
        static_cast<double>(batch_size) * 1000.0 / mean,
    };
}

}  // namespace day55
