#include <iostream>
#include <stdexcept>
#include <string>

#include "day52/image_pipeline.hpp"

namespace {

int passed_checks = 0;
int failed_checks = 0;

void expect(bool condition, const std::string& message) {
    if (condition) {
        ++passed_checks;
        std::cout << "[PASS] " << message << '\n';
    } else {
        ++failed_checks;
        std::cerr << "[FAIL] " << message << '\n';
    }
}

template <typename Function>
void expect_invalid_argument(Function&& function, const std::string& message) {
    try {
        function();
        expect(false, message);
    } catch (const std::invalid_argument&) {
        expect(true, message);
    } catch (...) {
        expect(false, message + " (wrong exception type)");
    }
}

}  // namespace

int main() {
    using day52::ExecutionMode;
    using day52::Operation;

    const auto gray = day52::parse_operation("gray");
    const auto blur = day52::parse_operation("blur");
    const auto edge = day52::parse_operation("edge");
    expect(gray == Operation::Gray, "parse gray operation");
    expect(blur == Operation::Blur, "parse blur operation");
    expect(edge == Operation::Edge, "parse edge operation");
    expect(!day52::parse_operation("").has_value(), "reject empty operation");
    expect(!day52::parse_operation("rotate").has_value(),
           "reject unknown operation");

    expect(day52::operation_name(Operation::Gray) == "gray",
           "format gray operation");
    expect(day52::operation_name(Operation::Blur) == "blur",
           "format blur operation");
    expect(day52::operation_name(Operation::Edge) == "edge",
           "format edge operation");

    const auto sequential = day52::parse_execution_mode("sequential");
    const auto threads = day52::parse_execution_mode("threads");
    expect(sequential == ExecutionMode::Sequential,
           "parse sequential execution mode");
    expect(threads == ExecutionMode::Threads,
           "parse threads execution mode");
    expect(!day52::parse_execution_mode("").has_value(),
           "reject empty execution mode");
    expect(!day52::parse_execution_mode("async").has_value(),
           "reject unknown execution mode");
    expect(day52::execution_mode_name(ExecutionMode::Sequential)
               == "sequential",
           "format sequential execution mode");
    expect(day52::execution_mode_name(ExecutionMode::Threads) == "threads",
           "format threads execution mode");

    expect(day52::resolve_worker_count(ExecutionMode::Sequential, 8, 20) == 1,
           "sequential mode always uses one worker");
    expect(day52::resolve_worker_count(ExecutionMode::Threads, 4, 20) == 4,
           "thread mode keeps a valid requested worker count");
    expect(day52::resolve_worker_count(ExecutionMode::Threads, 8, 3) == 3,
           "worker count is capped by image count");
    expect(day52::resolve_worker_count(ExecutionMode::Threads, 4, 1) == 1,
           "single image uses one worker");
    expect(day52::resolve_worker_count(ExecutionMode::Threads, 4, 0) == 1,
           "empty input has a safe effective worker count");
    expect_invalid_argument(
        []() {
            static_cast<void>(day52::resolve_worker_count(
                ExecutionMode::Threads, 0, 10));
        },
        "zero requested workers throw invalid_argument");
    expect_invalid_argument(
        []() {
            const day52::ImagePipeline pipeline{
                Operation::Gray,
                ExecutionMode::Threads,
                0,
            };
            static_cast<void>(pipeline);
        },
        "pipeline rejects zero workers");

    if (failed_checks != 0) {
        std::cerr << "Failed " << failed_checks << " of "
                  << passed_checks + failed_checks << " Day52 unit checks.\n";
        return 1;
    }

    std::cout << "Passed " << passed_checks << " Day52 unit checks.\n";
    std::cout << "DAY52_UNIT_TESTS_OK\n";
    return 0;
}
