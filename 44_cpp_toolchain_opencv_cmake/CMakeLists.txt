cmake_minimum_required(VERSION 3.22)

project(day44_opencv_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 5 REQUIRED COMPONENTS core imgproc)

add_executable(day44_opencv_cpp code/day44_opencv_cpp.cpp)

target_include_directories(day44_opencv_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(day44_opencv_cpp PRIVATE ${OpenCV_LIBS})

if(MINGW)
    target_compile_options(day44_opencv_cpp PRIVATE -Wall -Wextra -pedantic)
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
