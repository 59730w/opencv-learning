cmake_minimum_required(VERSION 3.22)

project(day48_cpp_smart_pointers_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 5 REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(day48_smart_pointers code/day48_smart_pointers.cpp)
add_executable(day48_pipeline code/day48_configurable_pipeline.cpp)

target_include_directories(day48_pipeline PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(day48_pipeline PRIVATE
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
)

if(MINGW)
    target_compile_options(day48_smart_pointers PRIVATE -Wall -Wextra -pedantic)
    target_compile_options(day48_pipeline PRIVATE -Wall -Wextra -pedantic)
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
