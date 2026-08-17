cmake_minimum_required(VERSION 3.22)

project(day45_opencv_cpp_image_io LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 5 REQUIRED COMPONENTS core imgcodecs highgui)

add_executable(day45_image_io code/day45_image_io.cpp)

target_include_directories(day45_image_io PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(day45_image_io PRIVATE
    opencv_core
    opencv_imgcodecs
    opencv_highgui
)

if(MINGW)
    target_compile_options(day45_image_io PRIVATE -Wall -Wextra -pedantic)
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
