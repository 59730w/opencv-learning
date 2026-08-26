cmake_minimum_required(VERSION 3.22)

project(day54_onnxruntime_cpp_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4.10 REQUIRED COMPONENTS core imgcodecs imgproc)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an extracted ONNX Runtime release")

set(ONNXRUNTIME_HEADER "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")

foreach(required_path
        "${ONNXRUNTIME_HEADER}"
        "${ONNXRUNTIME_LIBRARY}"
        "${ONNXRUNTIME_DLL}")
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR "Required ONNX Runtime file does not exist: ${required_path}")
    endif()
endforeach()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}"
    IMPORTED_LOCATION "${ONNXRUNTIME_DLL}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

add_library(day54_preprocessing STATIC
    src/preprocessing.cpp
)
target_include_directories(day54_preprocessing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(day54_preprocessing
    PUBLIC
        opencv_core
    PRIVATE
        opencv_imgproc
)

add_executable(day54_inspect_model
    code/day54_inspect_model.cpp
)

target_link_libraries(day54_inspect_model PRIVATE onnxruntime)

add_executable(day54_preprocess_images
    code/day54_preprocess_images.cpp
)
target_link_libraries(day54_preprocess_images PRIVATE
    day54_preprocessing
    opencv_imgcodecs
)

add_executable(day54_cpp_inference
    code/day54_cpp_inference.cpp
)
target_link_libraries(day54_cpp_inference PRIVATE
    day54_preprocessing
    onnxruntime
    opencv_imgcodecs
)

if(MSVC)
    target_compile_options(day54_preprocessing PRIVATE /W4 /permissive-)
    target_compile_options(day54_inspect_model PRIVATE /W4 /permissive-)
    target_compile_options(day54_preprocess_images PRIVATE /W4 /permissive-)
    target_compile_options(day54_cpp_inference PRIVATE /W4 /permissive-)
else()
    target_compile_options(day54_preprocessing PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(day54_inspect_model PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(day54_preprocess_images PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(day54_cpp_inference PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_custom_command(
    TARGET day54_inspect_model
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DLL}"
        "$<TARGET_FILE_DIR:day54_inspect_model>"
    COMMENT "Copying onnxruntime.dll next to the executable"
)

function(day54_copy_opencv_runtime target_name)
    add_custom_command(
        TARGET ${target_name}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:opencv_core>"
            "$<TARGET_FILE_DIR:${target_name}>"
        COMMENT "Copying the OpenCV runtime next to ${target_name}"
    )
endfunction()

day54_copy_opencv_runtime(day54_preprocess_images)
day54_copy_opencv_runtime(day54_cpp_inference)

add_custom_command(
    TARGET day54_cpp_inference
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DLL}"
        "$<TARGET_FILE_DIR:day54_cpp_inference>"
    COMMENT "Copying onnxruntime.dll next to day54_cpp_inference"
)

include(CTest)

if(BUILD_TESTING)
    find_program(DAY54_POWERSHELL_EXECUTABLE NAMES pwsh powershell REQUIRED)

    add_executable(day54_preprocessing_tests
        tests/day54_preprocessing_tests.cpp
    )
    target_link_libraries(day54_preprocessing_tests PRIVATE day54_preprocessing)
    if(MSVC)
        target_compile_options(day54_preprocessing_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(day54_preprocessing_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME day54_preprocessing_unit
        COMMAND day54_preprocessing_tests
    )
    day54_copy_opencv_runtime(day54_preprocessing_tests)

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(
        NAME day54_preprocessing_contract
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/day54_preprocessing_contract.py
            --executable $<TARGET_FILE:day54_preprocess_images>
            --day53-code ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/code
            --images
                ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/assets/acacia_IMG_6348.JPG
                ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/assets/adenanthera_microsperma_IMG_5777.JPG
                ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/assets/cananga_odorata_IMG_5483.JPG
    )
    set_tests_properties(day54_preprocessing_contract PROPERTIES
        DEPENDS day54_preprocessing_unit
    )

    add_test(
        NAME day54_inference_contract
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/day54_inference_contract.py
            --executable $<TARGET_FILE:day54_cpp_inference>
            --model ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/artifacts/forest_species_resnet18.onnx
            --class-map ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/assets/class_to_idx.json
            --day53-code ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/code
            --images
                ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/assets/acacia_IMG_6348.JPG
                ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/assets/adenanthera_microsperma_IMG_5777.JPG
                ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/assets/cananga_odorata_IMG_5483.JPG
    )
    set_tests_properties(day54_inference_contract PROPERTIES
        DEPENDS "day54_preprocessing_contract;day54_model_interface_contract"
    )

    add_test(
        NAME day54_model_interface_contract
        COMMAND
            ${DAY54_POWERSHELL_EXECUTABLE}
            -NoProfile
            -ExecutionPolicy Bypass
            -File ${CMAKE_CURRENT_SOURCE_DIR}/tests/day54_contract.ps1
            -Executable $<TARGET_FILE:day54_inspect_model>
            -Model ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/artifacts/forest_species_resnet18.onnx
    )
endif()

message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_ROOT}")
message(STATUS "ONNX Runtime DLL: ${ONNXRUNTIME_DLL}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
