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

# Set CUDA architectures BEFORE project() to prevent CMake auto-detection
# Inherited from parent when running via ctest, or set here for standalone builds
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;87-real;89-real;90-real;100-real;120-real;120-virtual")
endif()

project(warp_source_include CUDA CXX)

# Enable testing support
enable_testing()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Check that generated kernel exists (must be compiled before configuration)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/wp___main__.cu")
    message(FATAL_ERROR 
        "Generated kernel not found in ${CMAKE_CURRENT_SOURCE_DIR}/generated/\n"
        "Run the following command first to generate the kernel:\n"
        "  python compile_kernel.py\n"
        "or:\n"
        "  uv run compile_kernel.py")
endif()

message(STATUS "Target architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Add line information for profilers/debuggers
# Suppress warnings from generated code (550: set but never used, 177: declared but never referenced)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -diag-suppress 550 -diag-suppress 177")

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

# Add executable (generated .cu is included in main.cu, not compiled separately)
# Named after example directory for consistency with Makefile
add_executable(01_source_include main.cu)

# Set C++ and CUDA standards
target_compile_features(01_source_include PRIVATE cxx_std_17 cuda_std_17)

# Add includes
# Use SYSTEM for Warp headers to suppress library warnings
target_include_directories(01_source_include SYSTEM PRIVATE ${WARP_INCLUDE})
target_include_directories(01_source_include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

message(STATUS "Warp include: ${WARP_INCLUDE}")
message(STATUS "Target architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Add test that runs the example executable
add_test(NAME source_include_test COMMAND $<TARGET_FILE:01_source_include>)
set_tests_properties(source_include_test PROPERTIES
    LABELS "cpp_example"
)
