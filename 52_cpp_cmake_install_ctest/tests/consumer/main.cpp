#include <iostream>

#include <day52/image_pipeline.hpp>

int main() {
    const auto operation = day52::parse_operation("edge");
    const auto mode = day52::parse_execution_mode("threads");

    if (!operation.has_value()
        || !mode.has_value()
        || day52::operation_name(*operation) != "edge"
        || day52::resolve_worker_count(*mode, 8, 3) != 3) {
        std::cerr << "Installed day52 package returned an unexpected result.\n";
        return 1;
    }

    std::cout << "Installed target: day52::pipeline\n";
    std::cout << "DAY52_CONSUMER_OK\n";
    return 0;
}
