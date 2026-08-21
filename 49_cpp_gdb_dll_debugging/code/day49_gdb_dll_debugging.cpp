#include <charconv>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

std::optional<std::size_t> parse_index(std::string_view text) {
    if (text.empty()) {
        return std::nullopt;
    }

    std::size_t value = 0;
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto [position, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || position != end) {
        return std::nullopt;
    }
    return value;
}

int select_summary_value(const std::vector<int>& summary, std::size_t index) {
    return summary.at(index);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: day49_debug_target <image_path> <summary_index>\n";
        return 2;
    }

    const std::optional<std::size_t> index = parse_index(argv[2]);
    if (!index.has_value()) {
        std::cerr << "Invalid summary index: " << argv[2] << '\n';
        return 3;
    }

    const cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Cannot read image: " << argv[1] << '\n';
        return 1;
    }

    const std::vector<int> summary{
        image.cols,
        image.rows,
        image.channels(),
    };

    int selected_value = 0;
    try {
        selected_value = select_summary_value(summary, *index);
    }
    catch (const std::out_of_range&) {
        std::cerr << "Summary index out of range: " << *index << '\n';
        return 4;
    }

    std::cout << "Image: " << image.cols << " x " << image.rows
              << " channels=" << image.channels() << '\n';
    std::cout << "summary[" << *index << "]=" << selected_value << '\n';
    std::cout << "DAY49_DEBUG_OK\n";
    return 0;
}
