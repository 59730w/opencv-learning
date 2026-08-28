cmake_minimum_required(VERSION 3.22)

project(day56_onnxruntime_cpp_profiling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4.10 REQUIRED COMPONENTS core imgcodecs imgproc)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an extracted ONNX Runtime release")
set(ONNXRUNTIME_HEADER "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
foreach(required_path "${ONNXRUNTIME_HEADER}" "${ONNXRUNTIME_LIBRARY}" "${ONNXRUNTIME_DLL}")
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR "Required ONNX Runtime file does not exist: ${required_path}")
    endif()
endforeach()
file(STRINGS "${ONNXRUNTIME_HEADER}" profiling_api_lines
    REGEX "EnableProfiling|EndProfilingAllocated")
list(LENGTH profiling_api_lines profiling_api_count)
if(profiling_api_count LESS 2)
    message(FATAL_ERROR "Required ONNX Runtime profiling C++ APIs are unavailable")
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}"
    IMPORTED_LOCATION "${ONNXRUNTIME_DLL}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")

add_library(day56_preprocessing STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../54_onnxruntime_cpp_inference/src/preprocessing.cpp)
target_include_directories(day56_preprocessing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../54_onnxruntime_cpp_inference/include)
target_link_libraries(day56_preprocessing PUBLIC opencv_core PRIVATE opencv_imgproc)

add_executable(day56_profiling code/day56_profiling.cpp)
target_link_libraries(day56_profiling PRIVATE day56_preprocessing onnxruntime opencv_imgcodecs)

if(MSVC)
    target_compile_options(day56_preprocessing PRIVATE /W4 /permissive-)
    target_compile_options(day56_profiling PRIVATE /W4 /permissive-)
else()
    target_compile_options(day56_preprocessing PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(day56_profiling PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_custom_command(TARGET day56_profiling POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_DLL}" "$<TARGET_FILE_DIR:day56_profiling>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:opencv_core>" "$<TARGET_FILE_DIR:day56_profiling>")

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME day56_profile_analysis_unit COMMAND
        ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -p test_profile_analysis.py -v)
    set(DAY54_EXECUTABLE
        "${CMAKE_CURRENT_SOURCE_DIR}/../54_onnxruntime_cpp_inference/build/Release/day54_cpp_inference.exe"
        CACHE FILEPATH "Day54 reference executable")
    if(NOT EXISTS "${DAY54_EXECUTABLE}")
        message(FATAL_ERROR "Day54 reference executable missing: ${DAY54_EXECUTABLE}")
    endif()
    set(DAY56_ASSETS ${CMAKE_CURRENT_SOURCE_DIR}/../55_onnxruntime_cpp_benchmark/assets)
    add_test(NAME day56_profiling_contract COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/day56_profiling_contract.py
        --executable $<TARGET_FILE:day56_profiling>
        --analyzer ${CMAKE_CURRENT_SOURCE_DIR}/code/day56_analyze_profile.py
        --day54-executable ${DAY54_EXECUTABLE}
        --model ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/artifacts/forest_species_resnet18.onnx
        --class-map ${CMAKE_CURRENT_SOURCE_DIR}/../53_pytorch_onnx_export/assets/class_to_idx.json
        --images
            ${DAY56_ASSETS}/banana_bird_256x256.png
            ${DAY56_ASSETS}/hotdog_wrap_398x296.png
            ${DAY56_ASSETS}/fruit_strip_140x60.png
            ${DAY56_ASSETS}/dog_grooming_400x301.jpg
            ${DAY56_ASSETS}/cifar_airplane_32x32.png
            ${DAY56_ASSETS}/hotdog_pickle_614x419.png)
    set_tests_properties(day56_profiling_contract PROPERTIES DEPENDS day56_profile_analysis_unit)
endif()

message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_ROOT}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "DAY56_PROFILING_API_OK")
