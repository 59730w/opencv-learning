#include "day54/preprocessing.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::filesystem::path model;
    std::filesystem::path class_map;
    std::filesystem::path tensor_output;
    std::filesystem::path logits_output;
    std::vector<std::filesystem::path> images;
};

Options parse_options(int argc, char* argv[]) {
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto require_value = [&](const std::string& option_name) -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument("missing value for " + option_name);
            }
            ++index;
            return argv[index];
        };

        if (argument == "--model") {
            options.model = require_value(argument);
        } else if (argument == "--class-map") {
            options.class_map = require_value(argument);
        } else if (argument == "--tensor-output") {
            options.tensor_output = require_value(argument);
        } else if (argument == "--logits-output") {
            options.logits_output = require_value(argument);
        } else if (argument == "--images") {
            while (index + 1 < argc) {
                ++index;
                options.images.emplace_back(argv[index]);
            }
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }

    if (options.model.empty() || options.class_map.empty() ||
        options.tensor_output.empty() || options.logits_output.empty() ||
        options.images.empty()) {
        throw std::invalid_argument(
            "required options: --model, --class-map, --tensor-output, "
            "--logits-output, --images"
        );
    }
    return options;
}

void require_regular_file(const std::filesystem::path& path, const std::string& description) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(description + " does not exist: " + path.string());
    }
}

std::vector<std::string> read_class_names(const std::filesystem::path& class_map_path) {
    cv::FileStorage class_file(
        class_map_path.string(),
        cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON
    );
    if (!class_file.isOpened()) {
        throw std::runtime_error("could not open class map: " + class_map_path.string());
    }

    const cv::FileNode root = class_file.root();
    if (root.empty() || !root.isMap()) {
        throw std::runtime_error("class map must be a non-empty JSON object");
    }

    std::vector<std::pair<std::string, int>> entries;
    entries.reserve(root.size());
    for (auto iterator = root.begin(); iterator != root.end(); ++iterator) {
        entries.emplace_back((*iterator).name(), static_cast<int>(*iterator));
    }

    std::vector<std::string> class_names(entries.size());
    for (const auto& [class_name, class_index] : entries) {
        if (class_index < 0 || static_cast<std::size_t>(class_index) >= class_names.size()) {
            throw std::runtime_error("class indices must be continuous from zero");
        }
        std::string& destination = class_names[static_cast<std::size_t>(class_index)];
        if (!destination.empty()) {
            throw std::runtime_error("duplicate class index in class map");
        }
        destination = class_name;
    }

    if (std::any_of(class_names.begin(), class_names.end(), [](const std::string& name) {
            return name.empty();
        })) {
        throw std::runtime_error("class indices must be continuous from zero");
    }
    return class_names;
}

void write_float_data(const std::filesystem::path& output_path, const std::vector<float>& values) {
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("could not open output: " + output_path.string());
    }
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float))
    );
    if (!output) {
        throw std::runtime_error("could not write output: " + output_path.string());
    }
}

std::vector<std::size_t> top_indices(const float* logits, std::size_t class_count, std::size_t k) {
    std::vector<std::size_t> indices(class_count);
    std::iota(indices.begin(), indices.end(), 0U);
    std::partial_sort(
        indices.begin(),
        indices.begin() + static_cast<std::ptrdiff_t>(k),
        indices.end(),
        [&](std::size_t left, std::size_t right) {
            if (logits[left] == logits[right]) {
                return left < right;
            }
            return logits[left] > logits[right];
        }
    );
    indices.resize(k);
    return indices;
}

std::vector<float> softmax(const float* logits, std::size_t class_count) {
    const float maximum = *std::max_element(logits, logits + class_count);
    std::vector<float> probabilities(class_count);
    float sum = 0.0F;
    for (std::size_t index = 0; index < class_count; ++index) {
        probabilities[index] = std::exp(logits[index] - maximum);
        sum += probabilities[index];
    }
    for (float& probability : probabilities) {
        probability /= sum;
    }
    return probabilities;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const Options options = parse_options(argc, argv);
        require_regular_file(options.model, "model");
        require_regular_file(options.class_map, "class map");

        const std::vector<std::string> class_names = read_class_names(options.class_map);
        if (class_names.size() != 50U) {
            throw std::runtime_error("the Day53 forest model requires exactly 50 classes");
        }

        constexpr std::size_t kElementsPerImage = 3U * 224U * 224U;
        std::vector<float> batch_values;
        batch_values.reserve(options.images.size() * kElementsPerImage);

        for (const std::filesystem::path& image_path : options.images) {
            require_regular_file(image_path, "image");
            const cv::Mat bgr_image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
            if (bgr_image.empty()) {
                throw std::runtime_error("could not read image: " + image_path.string());
            }
            day54::PreprocessResult result = day54::preprocess_bgr(bgr_image);
            batch_values.insert(
                batch_values.end(),
                result.values.begin(),
                result.values.end()
            );
        }

        const std::array<std::int64_t, 4> input_shape{
            static_cast<std::int64_t>(options.images.size()),
            3,
            224,
            224,
        };

        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "day54_cpp_inference");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session(environment, options.model.c_str(), session_options);

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            batch_values.data(),
            batch_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        constexpr std::array<const char*, 1> input_names{"images"};
        constexpr std::array<const char*, 1> output_names{"logits"};
        std::vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            input_names.size(),
            output_names.data(),
            output_names.size()
        );

        if (outputs.size() != 1U || !outputs[0].IsTensor()) {
            throw std::runtime_error("expected exactly one tensor output");
        }
        const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
        const std::vector<std::int64_t> output_shape = output_info.GetShape();
        const std::vector<std::int64_t> expected_output_shape{
            static_cast<std::int64_t>(options.images.size()),
            static_cast<std::int64_t>(class_names.size()),
        };
        if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            output_shape != expected_output_shape) {
            throw std::runtime_error("unexpected logits type or shape");
        }

        const std::size_t logits_count = options.images.size() * class_names.size();
        const float* logits_data = outputs[0].GetTensorData<float>();
        const std::vector<float> logits(logits_data, logits_data + logits_count);

        write_float_data(options.tensor_output, batch_values);
        write_float_data(options.logits_output, logits);

        std::cout << "Batch size: " << options.images.size() << '\n';
        std::cout << "Input shape: [" << options.images.size() << ", 3, 224, 224]\n";
        std::cout << "Output shape: [" << options.images.size() << ", "
                  << class_names.size() << "]\n";

        for (std::size_t image_index = 0; image_index < options.images.size(); ++image_index) {
            const float* image_logits = logits.data() + image_index * class_names.size();
            const std::vector<float> probabilities = softmax(image_logits, class_names.size());
            const std::vector<std::size_t> top3 = top_indices(
                image_logits,
                class_names.size(),
                3U
            );

            std::cout << "Image " << image_index << " Top-1: "
                      << class_names[top3[0]] << '\n';
            std::cout << "Image " << image_index << " Top-3:";
            for (std::size_t rank = 0; rank < top3.size(); ++rank) {
                const std::size_t class_index = top3[rank];
                std::cout << ' ' << (rank + 1U) << '=' << class_names[class_index]
                          << "(" << std::fixed << std::setprecision(6)
                          << probabilities[class_index] << ')';
            }
            std::cout << '\n';
        }

        std::cout << "DAY54_CPP_INFERENCE_OK\n";
        return 0;
    } catch (const std::invalid_argument& error) {
        std::cerr << "Argument error: " << error.what() << '\n';
        return 2;
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
        return 3;
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: " << error.what() << '\n';
        return 4;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 5;
    }
}
