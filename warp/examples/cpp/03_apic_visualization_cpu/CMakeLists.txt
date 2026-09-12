# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(apic_visualization_cpu CXX C)

enable_testing()

# Find OpenGL (optional — skip this example on headless CI systems)
find_package(OpenGL QUIET)
if(NOT OPENGL_FOUND)
    message(STATUS "OpenGL not found — skipping 03_apic_visualization_cpu")
    return()
endif()

# Find Warp native directory
if(NOT DEFINED WARP_NATIVE_DIR)
    set(WARP_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../native")
endif()

get_filename_component(WARP_INCLUDE "${WARP_NATIVE_DIR}" ABSOLUTE)
if(NOT EXISTS "${WARP_INCLUDE}/builtin.h" OR NOT EXISTS "${WARP_INCLUDE}/warp_clang.h")
    message(FATAL_ERROR
        "Could not find Warp native directory at: ${WARP_INCLUDE}\n"
        "Expected builtin.h and warp_clang.h. Set WARP_NATIVE_DIR to override\n"
        "(e.g., cmake -DWARP_NATIVE_DIR=/path/to/warp/native)")
endif()

# Find Warp library
if(NOT DEFINED WARP_BIN_DIR)
    set(WARP_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../bin")
endif()
get_filename_component(WARP_BIN_ABS "${WARP_BIN_DIR}" ABSOLUTE)

if(WIN32)
    find_file(WARP_LIBRARY
        NAMES warp.dll
        HINTS ${WARP_BIN_ABS}
        PATHS ENV WARP_PATH
        PATH_SUFFIXES bin
    )
    find_library(WARP_IMPLIB
        NAMES warp
        HINTS ${WARP_BIN_ABS}
        PATHS ENV WARP_PATH
        PATH_SUFFIXES bin lib
    )
    find_file(WARP_CLANG_LIBRARY
        NAMES warp-clang.dll
        HINTS ${WARP_BIN_ABS}
        PATHS ENV WARP_PATH
        PATH_SUFFIXES bin
    )
    find_library(WARP_CLANG_IMPLIB
        NAMES warp-clang
        HINTS ${WARP_BIN_ABS}
        PATHS ENV WARP_PATH
        PATH_SUFFIXES bin lib
    )
    if(NOT WARP_LIBRARY OR NOT WARP_IMPLIB OR NOT WARP_CLANG_LIBRARY OR NOT WARP_CLANG_IMPLIB)
        message(FATAL_ERROR
            "Could not find warp.dll, warp.lib, warp-clang.dll, and warp-clang.lib\n"
            "Hints searched: ${WARP_BIN_ABS}\n"
            "Set WARP_BIN_DIR to the directory containing all four files")
    endif()

    add_library(Warp::runtime SHARED IMPORTED)
    set_target_properties(Warp::runtime PROPERTIES
        IMPORTED_LOCATION "${WARP_LIBRARY}"
        IMPORTED_IMPLIB "${WARP_IMPLIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${WARP_INCLUDE}"
    )
    add_library(Warp::clang SHARED IMPORTED)
    set_target_properties(Warp::clang PROPERTIES
        IMPORTED_LOCATION "${WARP_CLANG_LIBRARY}"
        IMPORTED_IMPLIB "${WARP_CLANG_IMPLIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${WARP_INCLUDE}"
    )
else()
    find_library(WARP_LIBRARY
        NAMES warp warp.so
        HINTS ${WARP_BIN_ABS}
        PATHS ENV WARP_PATH
        PATH_SUFFIXES bin lib
    )
    find_library(WARP_CLANG_LIBRARY
        NAMES warp-clang warp-clang.so
        HINTS ${WARP_BIN_ABS}
        PATHS ENV WARP_PATH
        PATH_SUFFIXES bin lib
    )
    if(NOT WARP_LIBRARY OR NOT WARP_CLANG_LIBRARY)
        message(FATAL_ERROR
            "Could not find the Warp and warp-clang shared libraries\n"
            "Hints searched: ${WARP_BIN_ABS}\n"
            "Set WARP_BIN_DIR to the directory containing both libraries")
    endif()

    add_library(Warp::runtime SHARED IMPORTED)
    set_target_properties(Warp::runtime PROPERTIES
        IMPORTED_LOCATION "${WARP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${WARP_INCLUDE}"
    )
    add_library(Warp::clang SHARED IMPORTED)
    set_target_properties(Warp::clang PROPERTIES
        IMPORTED_LOCATION "${WARP_CLANG_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${WARP_INCLUDE}"
    )
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
    GIT_TAG        v2.0.8
    SOURCE_SUBDIR  cmake
)

FetchContent_MakeAvailable(glfw glad)

# glad v2 doesn't create a target on its own — glad_add_library generates
# the loader sources for the requested API. REPRODUCIBLE uses the GL spec
# bundled with glad instead of fetching it from Khronos at build time.
# The target name is shared across examples so the guard avoids a collision
# when multiple examples are configured under the top-level CMakeLists.
if(NOT TARGET glad)
    glad_add_library(glad STATIC REPRODUCIBLE API gl:core=3.3)
endif()

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

target_link_libraries(03_apic_visualization_cpu
    OpenGL::GL
    glfw
    glad
    Warp::runtime
    Warp::clang
)

# On Windows, copy the warp DLLs to the output directory
if(WIN32)
    add_custom_command(TARGET 03_apic_visualization_cpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${WARP_LIBRARY}"
            "$<TARGET_FILE_DIR:03_apic_visualization_cpu>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${WARP_CLANG_LIBRARY}"
            "$<TARGET_FILE_DIR:03_apic_visualization_cpu>"
        COMMENT "Copying warp.dll and warp-clang.dll to output directory"
    )
endif()

message(STATUS "Warp include: ${WARP_INCLUDE}")
message(STATUS "Warp library: ${WARP_LIBRARY}")
message(STATUS "warp-clang library: ${WARP_CLANG_LIBRARY}")
if(WIN32)
    message(STATUS "Warp import library: ${WARP_IMPLIB}")
    message(STATUS "warp-clang import library: ${WARP_CLANG_IMPLIB}")
endif()

# Add headless smoke test (uses --smoke to skip GUI).
# PASS_REGULAR_EXPRESSION asserts the iteration count actually ran — guards against
# a future regression where the smoke branch silently exits 0 without replaying.
add_test(NAME apic_visualization_cpu_smoke
    COMMAND $<TARGET_FILE:03_apic_visualization_cpu> --smoke
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(apic_visualization_cpu_smoke PROPERTIES
    LABELS "cpp_example"
    PASS_REGULAR_EXPRESSION "smoke OK \\(10 replay iterations\\)"
    TIMEOUT 60)
