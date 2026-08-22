#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "day50/image_pipeline.hpp"

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;
    using day50::ImagePipeline;
    using day50::ImageResult;
    using day50::Operation;

    if (argc != 5 || std::string(argv[3]) != "--op") {
        std::cerr << "Usage: day50_oop_pipeline <input_dir> <output_dir> "
                  << "--op <gray|blur|edge>\n";
        return 2;
    }

    const fs::path input_dir{argv[1]};
    const fs::path output_dir{argv[2]};
    const std::optional<Operation> operation = day50::parse_operation(argv[4]);
    if (!operation.has_value()) {
        std::cerr << "Unknown operation: " << argv[4] << '\n';
        return 4;
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

    ImagePipeline pipeline{*operation};
    const std::vector<ImageResult> results =
        pipeline.process_directory(input_dir, output_dir);

    std::cout << "Operation: " << day50::operation_name(pipeline.operation())
              << '\n';
    std::cout << "Found " << results.size() << " image(s)\n";
    day50::print_summary(results);
    std::cout << "DAY50_OOP_OK\n";
    return 0;
}
