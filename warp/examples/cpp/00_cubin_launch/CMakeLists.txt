# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(warp_cubin_launch CUDA CXX)

# Enable testing support
enable_testing()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find CUBIN file (must exist before configuration)
file(GLOB CUBIN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/generated/wp___main__.sm*.cubin")
if(NOT CUBIN_FILES)
    message(FATAL_ERROR 
        "No CUBIN file found in ${CMAKE_CURRENT_SOURCE_DIR}/generated/\n"
        "Run the following command first to generate the CUBIN file:\n"
        "  python compile_kernel.py\n"
        "or:\n"
        "  uv run compile_kernel.py")
endif()
list(GET CUBIN_FILES 0 CUBIN_FILE)

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

# Kernel name
set(KERNEL_NAME "saxpy_cuda_kernel_forward")

# Add executable (named after example directory for consistency with Makefile)
add_executable(00_cubin_launch main.cu)

# Set C++ standard (no CUDA architecture needed - host-only code)
target_compile_features(00_cubin_launch PRIVATE cxx_std_17)

# Add includes and definitions
# Use SYSTEM to suppress warnings from Warp headers
target_include_directories(00_cubin_launch SYSTEM PRIVATE ${WARP_INCLUDE})
target_compile_definitions(00_cubin_launch PRIVATE
    CUBIN_FILE="${CUBIN_FILE}"
    KERNEL_NAME="${KERNEL_NAME}"
)

# Link CUDA driver library
target_link_libraries(00_cubin_launch CUDA::cuda_driver)

message(STATUS "Warp include: ${WARP_INCLUDE}")
message(STATUS "CUBIN file: ${CUBIN_FILE}")
message(STATUS "Kernel name: ${KERNEL_NAME}")

# Add test that runs the example executable
add_test(NAME cubin_launch_test COMMAND $<TARGET_FILE:00_cubin_launch>)
set_tests_properties(cubin_launch_test PROPERTIES
    LABELS "cpp_example"
)
