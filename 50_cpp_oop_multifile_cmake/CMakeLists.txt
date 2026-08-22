cmake_minimum_required(VERSION 3.22)

project(day50_cpp_oop_multifile_cmake LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 5 REQUIRED COMPONENTS core imgcodecs imgproc)

add_library(day50_pipeline STATIC
    src/image_pipeline.cpp
)

target_include_directories(day50_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(day50_pipeline
    PUBLIC
        opencv_core
    PRIVATE
        opencv_imgcodecs
        opencv_imgproc
)

add_executable(day50_oop_pipeline
    code/day50_oop_multifile.cpp
)

target_link_libraries(day50_oop_pipeline
    PRIVATE
        day50_pipeline
)

if(MINGW)
    target_compile_options(day50_pipeline PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
    target_compile_options(day50_oop_pipeline PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
