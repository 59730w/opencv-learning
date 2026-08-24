cmake_minimum_required(VERSION 3.22)

project(day52_cpp_cmake_install_ctest VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

find_package(OpenCV 5 REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(Threads REQUIRED)

add_library(day52_pipeline STATIC
    src/image_pipeline.cpp
)
add_library(day52::pipeline ALIAS day52_pipeline)

target_include_directories(day52_pipeline
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(day52_pipeline
    PUBLIC
        opencv_core
    PRIVATE
        opencv_imgcodecs
        opencv_imgproc
        Threads::Threads
)

set_target_properties(day52_pipeline PROPERTIES
    EXPORT_NAME pipeline
)

add_executable(day52_parallel_pipeline
    code/day52_cmake_install_ctest.cpp
)

target_link_libraries(day52_parallel_pipeline
    PRIVATE
        day52_pipeline
)

function(day52_enable_warnings target_name)
    if(MINGW OR CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    elseif(MSVC)
        target_compile_options(${target_name} PRIVATE /W4)
    endif()
endfunction()

day52_enable_warnings(day52_pipeline)
day52_enable_warnings(day52_parallel_pipeline)

if(BUILD_TESTING)
    add_executable(day52_unit_tests
        tests/day52_unit_tests.cpp
    )
    target_link_libraries(day52_unit_tests PRIVATE day52_pipeline)
    day52_enable_warnings(day52_unit_tests)

    add_test(
        NAME day52_unit_tests
        COMMAND day52_unit_tests
    )

    find_program(DAY52_POWERSHELL_EXECUTABLE NAMES pwsh powershell REQUIRED)
    add_test(
        NAME day52_contract
        COMMAND
            ${DAY52_POWERSHELL_EXECUTABLE}
            -NoProfile
            -ExecutionPolicy Bypass
            -File ${CMAKE_CURRENT_SOURCE_DIR}/tests/day52_contract.ps1
            -Executable $<TARGET_FILE:day52_parallel_pipeline>
    )
    set_tests_properties(day52_contract PROPERTIES
        DEPENDS day52_unit_tests
    )
endif()

install(
    TARGETS
        day52_pipeline
        day52_parallel_pipeline
    EXPORT day52Targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

set(DAY52_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/day52")

install(
    EXPORT day52Targets
    FILE day52Targets.cmake
    NAMESPACE day52::
    DESTINATION ${DAY52_CMAKE_INSTALL_DIR}
)

configure_package_config_file(
    cmake/day52Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/day52Config.cmake
    INSTALL_DESTINATION ${DAY52_CMAKE_INSTALL_DIR}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/day52ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/day52Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/day52ConfigVersion.cmake
    DESTINATION ${DAY52_CMAKE_INSTALL_DIR}
)

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_NAME "day52-cpp-image-pipeline")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_FILE_NAME
    "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-windows-ucrt64"
)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
include(CPack)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
