#include <charconv>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

#include "day52/image_pipeline.hpp"

namespace {

std::optional<unsigned int> parse_worker_count(std::string_view text) {
    if (text.empty()) {
        return std::nullopt;
    }

    unsigned int value = 0;
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto [position, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || position != end || value == 0) {
        return std::nullopt;
    }
    return value;
}

void print_usage() {
    std::cerr
        << "Usage: day52_parallel_pipeline <input_dir> <output_dir> "
        << "--op <gray|blur|edge> --mode <sequential|threads> "
        << "--workers <positive_integer>\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;

    if (argc != 9
        || std::string(argv[3]) != "--op"
        || std::string(argv[5]) != "--mode"
        || std::string(argv[7]) != "--workers") {
        print_usage();
        return 2;
    }

    const fs::path input_dir{argv[1]};
    const fs::path output_dir{argv[2]};
    const auto operation = day52::parse_operation(argv[4]);
    if (!operation.has_value()) {
        std::cerr << "Unknown operation: " << argv[4] << '\n';
        return 4;
    }

    const auto mode = day52::parse_execution_mode(argv[6]);
    if (!mode.has_value()) {
        std::cerr << "Unknown execution mode: " << argv[6] << '\n';
        return 5;
    }

    const auto worker_count = parse_worker_count(argv[8]);
    if (!worker_count.has_value()) {
        std::cerr << "Invalid worker count: " << argv[8] << '\n';
        return 6;
    }

    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "Not a directory: " << input_dir.string() << '\n';
        return 1;
    }

    if (fs::exists(output_dir) && !fs::is_directory(output_dir)) {
        std::cerr << "Output path is not a directory: "
                  << output_dir.string() << '\n';
        return 3;
    }

    std::error_code error;
    fs::create_directories(output_dir, error);
    if (error) {
        std::cerr << "Cannot create output directory: "
                  << output_dir.string() << " (" << error.message() << ")\n";
        return 3;
    }

    try {
        const day52::ImagePipeline pipeline{
            *operation,
            *mode,
            *worker_count,
        };
        const day52::PipelineReport report =
            pipeline.process_directory(input_dir, output_dir);

        const double throughput = report.results.empty()
            || report.elapsed_ms <= 0.0
            ? 0.0
            : static_cast<double>(report.results.size())
                / (report.elapsed_ms / 1000.0);

        std::cout << "Operation: " << day52::operation_name(*operation) << '\n';
        std::cout << "Mode: " << day52::execution_mode_name(report.mode) << '\n';
        std::cout << "Requested workers: " << report.requested_workers << '\n';
        std::cout << "Effective workers: " << report.effective_workers << '\n';
        std::cout << "Hardware concurrency: " << report.hardware_workers << '\n';
        std::cout << "Files: " << report.results.size() << '\n';
        std::cout << std::fixed << std::setprecision(3)
                  << "Elapsed: " << report.elapsed_ms << " ms\n";
        std::cout << std::fixed << std::setprecision(2)
                  << "Throughput: " << throughput << " images/s\n";
        day52::print_summary(report.results);
        std::cout << "DAY52_PIPELINE_OK\n";
    } catch (const std::exception& exception) {
        std::cerr << "Processing failed: " << exception.what() << '\n';
        return 7;
    }

    return 0;
}
