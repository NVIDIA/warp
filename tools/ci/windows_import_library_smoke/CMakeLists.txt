# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(windows_import_library_smoke LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "The Windows import-library smoke test requires Windows")
endif()

set(WARP_PACKAGE_ROOT "" CACHE PATH "Path to the Warp Python package directory")
if(NOT WARP_PACKAGE_ROOT)
    message(FATAL_ERROR "Set WARP_PACKAGE_ROOT to the directory containing native/ and bin/")
endif()

get_filename_component(WARP_PACKAGE_ROOT "${WARP_PACKAGE_ROOT}" ABSOLUTE)
set(WARP_NATIVE_DIR "${WARP_PACKAGE_ROOT}/native")
set(WARP_BIN_DIR "${WARP_PACKAGE_ROOT}/bin")

foreach(required_file
    "${WARP_NATIVE_DIR}/warp.h"
    "${WARP_NATIVE_DIR}/warp_clang.h"
    "${WARP_BIN_DIR}/warp.dll"
    "${WARP_BIN_DIR}/warp.lib"
    "${WARP_BIN_DIR}/warp-clang.dll"
    "${WARP_BIN_DIR}/warp-clang.lib"
)
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "Required Warp package file not found: ${required_file}")
    endif()
endforeach()

add_library(Warp::runtime SHARED IMPORTED)
set_target_properties(Warp::runtime PROPERTIES
    IMPORTED_LOCATION "${WARP_BIN_DIR}/warp.dll"
    IMPORTED_IMPLIB "${WARP_BIN_DIR}/warp.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${WARP_NATIVE_DIR}"
)

add_library(Warp::clang SHARED IMPORTED)
set_target_properties(Warp::clang PROPERTIES
    IMPORTED_LOCATION "${WARP_BIN_DIR}/warp-clang.dll"
    IMPORTED_IMPLIB "${WARP_BIN_DIR}/warp-clang.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${WARP_NATIVE_DIR}"
)

add_executable(windows_import_library_smoke main.cpp)
target_compile_features(windows_import_library_smoke PRIVATE cxx_std_17)
target_link_libraries(windows_import_library_smoke PRIVATE Warp::runtime Warp::clang)

add_custom_command(TARGET windows_import_library_smoke POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:Warp::runtime>"
        "$<TARGET_FILE_DIR:windows_import_library_smoke>"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:Warp::clang>"
        "$<TARGET_FILE_DIR:windows_import_library_smoke>"
)

enable_testing()
add_test(NAME windows_import_library_smoke COMMAND windows_import_library_smoke)
