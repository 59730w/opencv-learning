cmake_minimum_required(VERSION 3.22)

project(day55_onnxruntime_cpp_benchmark LANGUAGES CXX)

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

add_library(day55_statistics STATIC src/statistics.cpp)
target_include_directories(day55_statistics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(day55_preprocessing STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../54_onnxruntime_cpp_inference/src/preprocessing.cpp
)
target_include_directories(day55_preprocessing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../54_onnxruntime_cpp_inference/include
)
target_link_libraries(day55_preprocessing PUBLIC opencv_core PRIVATE opencv_imgproc)

add_executable(day55_benchmark code/day55_benchmark.cpp)
target_link_libraries(day55_benchmark PRIVATE
    day55_statistics day55_preprocessing onnxruntime opencv_imgcodecs
)

if(MSVC)
    foreach(target day55_statistics day55_preprocessing day55_benchmark)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    endforeach()
else()
    foreach(target day55_statistics day55_preprocessing day55_benchmark)
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endforeach()
endif()

add_custom_command(TARGET day55_benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DLL}" "$<TARGET_FILE_DIR:day55_benchmark>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:opencv_core>" "$<TARGET_FILE_DIR:day55_benchmark>"
)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(day55_statistics_tests tests/day55_statistics_tests.cpp)
    target_link_libraries(day55_statistics_tests PRIVATE day55_statistics)
    if(MSVC)
        target_compile_options(day55_statistics_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(day55_statistics_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME day55_statistics_unit COMMAND day55_statistics_tests)

    set(DAY54_EXECUTABLE
        "${CMAKE_CURRENT_SOURCE_DIR}/../54_onnxruntime_cpp_inference/build/Release/day54_cpp_inference.exe"
        CACHE FILEPATH "Day54 reference executable"
    )
    if(NOT EXISTS "${DAY54_EXECUTABLE}")
        message(FATAL_ERROR "Day54 reference executable missing: ${DAY54_EXECUTABLE}")
    endif()
    add_test(NAME day55_benchmark_contract COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/day55_benchmark_contract.py
        --executable $<TARGET_FILE:day55_benchmark>
        --day54-executable ${DAY54_EXECUTABLE}
        --model ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/artifacts/forest_species_resnet18.onnx
        --class-map ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/assets/class_to_idx.json
        --images
            ${CMAKE_CURRENT_SOURCE_DIR}/assets/banana_bird_256x256.png
            ${CMAKE_CURRENT_SOURCE_DIR}/assets/hotdog_wrap_398x296.png
            ${CMAKE_CURRENT_SOURCE_DIR}/assets/fruit_strip_140x60.png
            ${CMAKE_CURRENT_SOURCE_DIR}/assets/dog_grooming_400x301.jpg
            ${CMAKE_CURRENT_SOURCE_DIR}/assets/cifar_airplane_32x32.png
            ${CMAKE_CURRENT_SOURCE_DIR}/assets/hotdog_pickle_614x419.png
    )
    set_tests_properties(day55_benchmark_contract PROPERTIES DEPENDS day55_statistics_unit)
endif()

message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_ROOT}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "DAY55_STAGE1_CONFIG_OK")
