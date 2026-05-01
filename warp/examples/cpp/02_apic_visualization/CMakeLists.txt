# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)
project(apic_visualization CUDA CXX C)

# Enable testing support
enable_testing()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find OpenGL (optional — skip this example on headless CI systems)
find_package(OpenGL QUIET)
if(NOT OPENGL_FOUND)
    message(STATUS "OpenGL not found — skipping 02_apic_visualization")
    return()
endif()

# Find Warp native directory
# Default: assume repository checkout (examples are in warp/examples/cpp/)
if(NOT DEFINED WARP_NATIVE_DIR)
    set(WARP_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../native")
endif()

# Resolve to absolute path and verify
get_filename_component(WARP_INCLUDE "${WARP_NATIVE_DIR}" ABSOLUTE)
if(NOT EXISTS "${WARP_INCLUDE}/builtin.h")
    message(FATAL_ERROR
        "Could not find Warp native directory at: ${WARP_INCLUDE}\n"
        "Set WARP_NATIVE_DIR to override (e.g., cmake -DWARP_NATIVE_DIR=/path/to/warp/native)")
endif()

# Find Warp library
# Default: look in warp/bin directory (repository structure)
if(NOT DEFINED WARP_BIN_DIR)
    set(WARP_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../bin")
endif()
get_filename_component(WARP_BIN_ABS "${WARP_BIN_DIR}" ABSOLUTE)

# Try to find warp library
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

# Wrap the resolved library path in an IMPORTED target so CMake passes the
# full path to the linker verbatim. Otherwise CMake converts an absolute
# path like /path/to/warp.so to "-L/path/to -lwarp", which fails on Linux
# because the library lacks the conventional lib<name>.so prefix.
add_library(warp::runtime UNKNOWN IMPORTED)
set_target_properties(warp::runtime PROPERTIES
    IMPORTED_LOCATION "${WARP_LIB}"
    IMPORTED_NO_SONAME TRUE
)

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

# Add executable
add_executable(02_apic_visualization main.cu)

# Set CUDA architecture (needed for device code - update_vertices kernel)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # Default to Turing, can override
endif()

# Set C++ standard
target_compile_features(02_apic_visualization PRIVATE cxx_std_17)

# Add includes
# Use SYSTEM to suppress warnings from external headers
target_include_directories(02_apic_visualization SYSTEM PRIVATE
    ${WARP_INCLUDE}
)

# Link libraries
target_link_libraries(02_apic_visualization
    CUDA::cuda_driver
    CUDA::cudart
    OpenGL::GL
    glfw
    glad
    warp::runtime
)

# warp.so contains lazily-resolved Python C API function references that
# are deferred to runtime by -Wl,-z,lazy in the warp.so build, so non-Python
# C++ hosts can dlopen the library. The static linker still tries to resolve
# them at link time; tell it to defer unresolved shared-lib symbols to
# runtime instead. The smoke path never calls these symbols.
# Remove once `nm -D --undefined-only warp/bin/warp.so | grep Py` is empty.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(02_apic_visualization PRIVATE
        "LINKER:--unresolved-symbols=ignore-in-shared-libs")
endif()

# On Windows, copy the warp DLL to the output directory
if(WIN32)
    get_filename_component(WARP_LIB_DIR "${WARP_LIB}" DIRECTORY)
    add_custom_command(TARGET 02_apic_visualization POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${WARP_LIB_DIR}/warp.dll"
            "$<TARGET_FILE_DIR:02_apic_visualization>"
        COMMENT "Copying warp.dll to output directory"
    )
endif()

message(STATUS "Warp include: ${WARP_INCLUDE}")
message(STATUS "Warp library: ${WARP_LIB}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Add headless smoke test (uses --smoke to skip GUI; needs a CUDA-capable runtime).
# PASS_REGULAR_EXPRESSION asserts the iteration count actually ran — guards against
# a future regression where the smoke branch silently exits 0 without launching.
add_test(NAME apic_visualization_smoke
    COMMAND $<TARGET_FILE:02_apic_visualization> --smoke
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(apic_visualization_smoke PROPERTIES
    LABELS "cpp_example"
    PASS_REGULAR_EXPRESSION "smoke OK \\(10 graph launches\\)"
    TIMEOUT 60)
