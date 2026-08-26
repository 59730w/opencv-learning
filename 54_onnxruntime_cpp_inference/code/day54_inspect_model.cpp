#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string element_type_name(ONNXTensorElementDataType element_type) {
    switch (element_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return "float64";
        default:
            return "unknown(" + std::to_string(static_cast<int>(element_type)) + ")";
    }
}

std::string shape_string(const Ort::ConstTensorTypeAndShapeInfo& tensor_info) {
    const std::vector<std::int64_t> dimensions = tensor_info.GetShape();
    std::vector<const char*> symbols(dimensions.size(), nullptr);
    tensor_info.GetSymbolicDimensions(symbols.data(), symbols.size());

    std::ostringstream output;
    output << '[';

    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }

        if (dimensions[index] >= 0) {
            output << dimensions[index];
        } else if (symbols[index] != nullptr && symbols[index][0] != '\0') {
            output << symbols[index];
        } else {
            output << '?';
        }
    }

    output << ']';
    return output.str();
}

struct ModelValueInfo {
    std::string name;
    ONNXTensorElementDataType element_type;
    std::string shape;
};

ModelValueInfo input_info(
    const Ort::Session& session,
    std::size_t index,
    Ort::AllocatorWithDefaultOptions& allocator
) {
    const auto name = session.GetInputNameAllocated(index, allocator);
    const Ort::TypeInfo type_info = session.GetInputTypeInfo(index);
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return {name.get(), tensor_info.GetElementType(), shape_string(tensor_info)};
}

ModelValueInfo output_info(
    const Ort::Session& session,
    std::size_t index,
    Ort::AllocatorWithDefaultOptions& allocator
) {
    const auto name = session.GetOutputNameAllocated(index, allocator);
    const Ort::TypeInfo type_info = session.GetOutputTypeInfo(index);
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return {name.get(), tensor_info.GetElementType(), shape_string(tensor_info)};
}

void verify_forest_model_contract(
    std::size_t input_count,
    const ModelValueInfo& input,
    std::size_t output_count,
    const ModelValueInfo& output
) {
    const bool input_matches =
        input_count == 1 &&
        input.name == "images" &&
        input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        input.shape == "[batch, 3, 224, 224]";

    const bool output_matches =
        output_count == 1 &&
        output.name == "logits" &&
        output.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        output.shape == "[batch, 50]";

    if (!input_matches || !output_matches) {
        throw std::runtime_error("The model interface does not match the Day53 forest model contract.");
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: day54_inspect_model <model.onnx>\n";
        return 2;
    }

    const std::filesystem::path model_path = argv[1];
    if (!std::filesystem::is_regular_file(model_path)) {
        std::cerr << "Model file does not exist: " << model_path.string() << '\n';
        return 3;
    }

    try {
        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "day54_model_interface");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session(environment, model_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        const std::size_t input_count = session.GetInputCount();
        const std::size_t output_count = session.GetOutputCount();

        if (input_count == 0 || output_count == 0) {
            throw std::runtime_error("The model must have at least one input and one output.");
        }

        const ModelValueInfo input = input_info(session, 0, allocator);
        const ModelValueInfo output = output_info(session, 0, allocator);

        std::cout << "Model: " << std::filesystem::absolute(model_path).string() << '\n';
        std::cout << "Input count: " << input_count << '\n';
        std::cout << "Input 0 name: " << input.name << '\n';
        std::cout << "Input 0 type: " << element_type_name(input.element_type) << '\n';
        std::cout << "Input 0 shape: " << input.shape << '\n';
        std::cout << "Output count: " << output_count << '\n';
        std::cout << "Output 0 name: " << output.name << '\n';
        std::cout << "Output 0 type: " << element_type_name(output.element_type) << '\n';
        std::cout << "Output 0 shape: " << output.shape << '\n';

        verify_forest_model_contract(input_count, input, output_count, output);
        std::cout << "DAY54_MODEL_INTERFACE_OK\n";
        return 0;
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: " << error.what() << '\n';
        return 4;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 5;
    }
}
