cmake_minimum_required(VERSION 3.22)

project(day51_cpp_threads_timing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 5 REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(Threads REQUIRED)

add_library(day51_pipeline STATIC
    src/image_pipeline.cpp
)

target_include_directories(day51_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(day51_pipeline
    PUBLIC
        opencv_core
    PRIVATE
        opencv_imgcodecs
        opencv_imgproc
        Threads::Threads
)

add_executable(day51_parallel_pipeline
    code/day51_threads_timing.cpp
)

target_link_libraries(day51_parallel_pipeline
    PRIVATE
        day51_pipeline
)

if(MINGW)
    target_compile_options(day51_pipeline PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
    target_compile_options(day51_parallel_pipeline PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CMake thread libraries: ${CMAKE_THREAD_LIBS_INIT}")
