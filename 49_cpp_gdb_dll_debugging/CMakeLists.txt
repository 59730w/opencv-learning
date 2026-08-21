cmake_minimum_required(VERSION 3.22)

project(day49_cpp_gdb_dll_debugging LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 5 REQUIRED COMPONENTS core imgcodecs)

add_executable(day49_debug_target code/day49_gdb_dll_debugging.cpp)

target_include_directories(day49_debug_target PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(day49_debug_target PRIVATE
    opencv_core
    opencv_imgcodecs
)

if(MINGW)
    target_compile_options(day49_debug_target PRIVATE
        -O0
        -g3
        -Wall
        -Wextra
        -pedantic
    )
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
