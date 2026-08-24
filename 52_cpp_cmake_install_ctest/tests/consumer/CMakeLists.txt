cmake_minimum_required(VERSION 3.22)

project(day52_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(day52 1 CONFIG REQUIRED)

add_executable(day52_consumer
    main.cpp
)

target_link_libraries(day52_consumer
    PRIVATE
        day52::pipeline
)

if(MINGW OR CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(day52_consumer PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()
