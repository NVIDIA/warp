# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(apic_visualization_cpu CXX)

enable_testing()

# Find Warp native directory
if(NOT DEFINED WARP_NATIVE_DIR)
    set(WARP_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../native")
endif()

get_filename_component(WARP_INCLUDE "${WARP_NATIVE_DIR}" ABSOLUTE)
if(NOT EXISTS "${WARP_INCLUDE}/builtin.h")
    message(FATAL_ERROR
        "Could not find Warp native directory at: ${WARP_INCLUDE}\n"
        "Set WARP_NATIVE_DIR to override (e.g., cmake -DWARP_NATIVE_DIR=/path/to/warp/native)")
endif()

# Find Warp library
if(NOT DEFINED WARP_BIN_DIR)
    set(WARP_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../bin")
endif()
get_filename_component(WARP_BIN_ABS "${WARP_BIN_DIR}" ABSOLUTE)

find_library(WARP_LIB
    NAMES warp warp.so warp.dll
    HINTS ${WARP_BIN_ABS}
    PATHS ENV WARP_PATH
    PATH_SUFFIXES bin lib
)

if(NOT WARP_LIB)
    message(FATAL_ERROR
        "Could not find Warp library (warp.dll / warp.so / libwarp.dylib)\n"
        "Hints searched: ${WARP_BIN_ABS}\n"
        "Set WARP_BIN_DIR to the directory containing the warp library")
endif()

# Find OpenGL (optional — skip this example on headless CI systems)
find_package(OpenGL QUIET)
if(NOT OPENGL_FOUND)
    message(STATUS "OpenGL not found — skipping 03_apic_visualization_cpu")
    return()
endif()

# Fetch GLFW and GLAD
include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.8
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG        v0.1.36
)
set(GLAD_PROFILE "core" CACHE STRING "" FORCE)
set(GLAD_API "gl=3.3" CACHE STRING "" FORCE)
set(GLAD_GENERATOR "c" CACHE STRING "" FORCE)

FetchContent_MakeAvailable(glfw glad)

# Check that the APIC graph exists
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/wave_sim.wrp")
    message(WARNING
        "No APIC graph found at ${CMAKE_CURRENT_SOURCE_DIR}/generated/wave_sim.wrp\n"
        "Run the following command first to generate the graph:\n"
        "  python capture_wave.py\n"
        "or:\n"
        "  uv run capture_wave.py")
endif()

# Add executable (pure C++ — no CUDA required)
add_executable(03_apic_visualization_cpu main.cpp)

target_compile_features(03_apic_visualization_cpu PRIVATE cxx_std_17)

target_include_directories(03_apic_visualization_cpu SYSTEM PRIVATE
    ${WARP_INCLUDE}
)

target_link_libraries(03_apic_visualization_cpu
    OpenGL::GL
    glfw
    glad
    ${WARP_LIB}
)

# On Windows, copy the warp DLLs to the output directory
if(WIN32)
    get_filename_component(WARP_LIB_DIR "${WARP_LIB}" DIRECTORY)
    add_custom_command(TARGET 03_apic_visualization_cpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${WARP_LIB_DIR}/warp.dll"
            "$<TARGET_FILE_DIR:03_apic_visualization_cpu>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${WARP_LIB_DIR}/warp-clang.dll"
            "$<TARGET_FILE_DIR:03_apic_visualization_cpu>"
        COMMENT "Copying warp.dll and warp-clang.dll to output directory"
    )
endif()

message(STATUS "Warp include: ${WARP_INCLUDE}")
message(STATUS "Warp library: ${WARP_LIB}")
