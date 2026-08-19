cmake_minimum_required(VERSION 3.22)

project(day47_cpp_batch_image_processing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 5 REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(day47_batch_processing code/day47_batch_processing.cpp)

target_include_directories(day47_batch_processing PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(day47_batch_processing PRIVATE
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
)

if(MINGW)
    target_compile_options(day47_batch_processing PRIVATE -Wall -Wextra -pedantic)
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
