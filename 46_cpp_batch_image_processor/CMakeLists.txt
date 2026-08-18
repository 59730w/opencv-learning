cmake_minimum_required(VERSION 3.22)

project(day46_cpp_batch_image_processor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 5 REQUIRED COMPONENTS core imgcodecs)

add_executable(day46_batch_processor code/day46_batch_processor.cpp)

target_include_directories(day46_batch_processor PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(day46_batch_processor PRIVATE
    opencv_core
    opencv_imgcodecs
)

if(MINGW)
    target_compile_options(day46_batch_processor PRIVATE -Wall -Wextra -pedantic)
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
