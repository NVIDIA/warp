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

# Top-level CMakeLists.txt for Warp C++ Examples Testing
#
# This file orchestrates testing of all C++ examples using CTest.
# Each example subdirectory is added as a separate CMake project.

cmake_minimum_required(VERSION 3.20)

# Set CUDA architectures BEFORE project() to prevent CMake auto-detection
# Generate CUBIN binaries for sm_75 through sm_120, plus PTX for forward compatibility
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;87-real;89-real;90-real;100-real;120-real;120-virtual")
    message(STATUS "Setting default CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

project(warp_cpp_examples CUDA CXX)

# Enable testing at the top level
enable_testing()

# Find all example subdirectories that contain a CMakeLists.txt
file(GLOB EXAMPLE_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*")

foreach(DIR ${EXAMPLE_DIRS})
    # Check if it's a directory and contains CMakeLists.txt
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${DIR}")
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${DIR}/CMakeLists.txt")
            message(STATUS "Adding example: ${DIR}")
            add_subdirectory(${DIR})
        endif()
    endif()
endforeach()

message(STATUS "")
message(STATUS "===============================================")
message(STATUS "Warp C++ Examples Test Configuration Complete")
message(STATUS "===============================================")
message(STATUS "")
message(STATUS "To run all example tests:")
message(STATUS "  ctest --output-on-failure --verbose")
message(STATUS "")
message(STATUS "To run a specific test:")
message(STATUS "  ctest -R <test_name> --output-on-failure --verbose")
message(STATUS "")
message(STATUS "To list all tests:")
message(STATUS "  ctest -N")
message(STATUS "")
message(STATUS "===============================================")
