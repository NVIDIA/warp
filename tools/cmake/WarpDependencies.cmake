# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

function(warp_get_host_arch out_var)
    string(TOLOWER "${CMAKE_HOST_SYSTEM_PROCESSOR}" _processor)
    if(_processor MATCHES "^(x86_64|amd64)$")
        set(_arch "x86_64")
    elseif(_processor MATCHES "^(aarch64|arm64)$")
        set(_arch "aarch64")
    else()
        message(FATAL_ERROR "Unsupported host architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
    endif()
    set(${out_var} "${_arch}" PARENT_SCOPE)
endfunction()

function(warp_get_packman_command out_var)
    if(WIN32)
        set(_packman "${PROJECT_SOURCE_DIR}/tools/packman/packman.cmd")
    else()
        set(_packman "${PROJECT_SOURCE_DIR}/tools/packman/packman")
    endif()

    if(NOT EXISTS "${_packman}")
        message(FATAL_ERROR "Packman was not found at ${_packman}")
    endif()

    set(${out_var} "${_packman}" PARENT_SCOPE)
endfunction()

function(warp_get_packman_platform out_var)
    warp_get_host_arch(_arch)
    if(WIN32)
        set(_platform "windows-${_arch}")
    elseif(APPLE)
        set(_platform "darwin-${_arch}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(_platform "linux-${_arch}")
    else()
        message(FATAL_ERROR "Unsupported Packman platform: ${CMAKE_SYSTEM_NAME}")
    endif()
    set(${out_var} "${_platform}" PARENT_SCOPE)
endfunction()

function(warp_get_packman_python out_var)
    if(NOT WARP_PYTHON_EXECUTABLE)
        message(FATAL_ERROR "warp_find_python() must be called before fetching Packman dependencies")
    endif()
    set(${out_var} "${WARP_PYTHON_EXECUTABLE}" PARENT_SCOPE)
endfunction()

function(warp_configure_cuda_root)
    if(WARP_CUDA_PATH)
        set(CUDAToolkit_ROOT "${WARP_CUDA_PATH}" CACHE PATH "CUDA Toolkit root" FORCE)
        if(EXISTS "${WARP_CUDA_PATH}/bin/nvcc${CMAKE_EXECUTABLE_SUFFIX}")
            set(CMAKE_CUDA_COMPILER "${WARP_CUDA_PATH}/bin/nvcc${CMAKE_EXECUTABLE_SUFFIX}" CACHE FILEPATH "CUDA compiler" FORCE)
        endif()
        return()
    endif()

    if((DEFINED CUDAToolkit_ROOT AND NOT CUDAToolkit_ROOT STREQUAL "")
       OR (DEFINED CMAKE_CUDA_COMPILER AND NOT CMAKE_CUDA_COMPILER STREQUAL ""))
        return()
    endif()

    if(DEFINED ENV{WARP_CUDA_PATH})
        set(_cuda_root "$ENV{WARP_CUDA_PATH}")
    elseif(DEFINED ENV{CUDA_HOME})
        set(_cuda_root "$ENV{CUDA_HOME}")
    elseif(DEFINED ENV{CUDA_PATH})
        set(_cuda_root "$ENV{CUDA_PATH}")
    else()
        set(_cuda_root "")
    endif()

    if(_cuda_root)
        set(CUDAToolkit_ROOT "${_cuda_root}" CACHE PATH "CUDA Toolkit root" FORCE)
        if((NOT DEFINED CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER STREQUAL "")
           AND EXISTS "${_cuda_root}/bin/nvcc${CMAKE_EXECUTABLE_SUFFIX}")
            set(CMAKE_CUDA_COMPILER "${_cuda_root}/bin/nvcc${CMAKE_EXECUTABLE_SUFFIX}" CACHE FILEPATH "CUDA compiler" FORCE)
        endif()
    endif()
endfunction()

function(warp_set_default_cuda_architectures)
    if((DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
       OR (DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES} AND NOT "$CACHE{CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
       OR (DEFINED ENV{CUDAARCHS} AND NOT "$ENV{CUDAARCHS}" STREQUAL ""))
        return()
    endif()

    set(CMAKE_CUDA_ARCHITECTURES "75-virtual" CACHE STRING "CUDA architectures" FORCE)
endfunction()

function(warp_find_python)
    find_package(Python3 3.10 COMPONENTS Interpreter REQUIRED)
    set(WARP_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" PARENT_SCOPE)
endfunction()

function(warp_get_python_script_command out_var script_path)
    if(NOT WARP_PYTHON_EXECUTABLE)
        message(FATAL_ERROR "warp_find_python() must be called before generating headers")
    endif()

    find_program(WARP_UV_EXECUTABLE uv)
    if(WARP_UV_EXECUTABLE)
        set(_command "${WARP_UV_EXECUTABLE};run;--no-project;--script;${script_path}")
    else()
        set(_command "${WARP_PYTHON_EXECUTABLE};${script_path}")
    endif()

    set("${out_var}" "${_command}" PARENT_SCOPE)
endfunction()

function(warp_add_generated_headers_target)
    set(_exports_header "${PROJECT_SOURCE_DIR}/warp/native/exports.h")
    set(_exports_header_script "${PROJECT_SOURCE_DIR}/tools/build/generate_exports_header.py")
    warp_get_python_script_command(_exports_header_command "${_exports_header_script}")

    add_custom_command(
        OUTPUT "${_exports_header}"
        COMMAND ${_exports_header_command} "${PROJECT_SOURCE_DIR}"
        DEPENDS
            "${_exports_header_script}"
            "${PROJECT_SOURCE_DIR}/warp/_src/generated_files.py"
            "${PROJECT_SOURCE_DIR}/warp/_src/context.py"
            "${PROJECT_SOURCE_DIR}/warp/_src/builtins.py"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Generating warp/native/exports.h"
        VERBATIM
    )

    add_custom_target(warp_generated_headers DEPENDS "${_exports_header}")
endfunction()

function(warp_find_libmathdx)
    if(NOT WARP_ENABLE_CUDA OR NOT WARP_USE_LIBMATHDX)
        set(WARP_LIBMATHDX_INCLUDE_DIR "" PARENT_SCOPE)
        set(WARP_LIBMATHDX_LIBRARY "" PARENT_SCOPE)
        return()
    endif()

    if(NOT CUDAToolkit_VERSION)
        message(FATAL_ERROR "CUDAToolkit_VERSION is required to find libmathdx")
    endif()

    if(WARP_LIBMATHDX_PATH)
        set(_mathdx_root "${WARP_LIBMATHDX_PATH}")
    elseif(DEFINED ENV{LIBMATHDX_HOME})
        set(_mathdx_root "$ENV{LIBMATHDX_HOME}")
    else()
        warp_get_packman_python(_packman_python)
        warp_get_packman_command(_packman)
        warp_get_packman_platform(_platform)
        string(REGEX MATCH "^[0-9]+" _cuda_major "${CUDAToolkit_VERSION}")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env "PM_PYTHON_EXT=${_packman_python}"
                    "${_packman}" pull --verbose --platform "${_platform}" --include-tag "cu${_cuda_major}"
                    "${PROJECT_SOURCE_DIR}/deps/libmathdx-deps.packman.xml"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            COMMAND_ERROR_IS_FATAL ANY
        )
        set(_mathdx_root "${PROJECT_SOURCE_DIR}/_build/target-deps/libmathdx")
    endif()

    if(WIN32)
        set(_mathdx_lib_dir "${_mathdx_root}/lib/x64")
    else()
        set(_mathdx_lib_dir "${_mathdx_root}/lib")
    endif()

    if(WARP_USE_DYNAMIC_CUDA)
        set(_mathdx_names mathdx)
    else()
        set(_mathdx_names mathdx_static mathdx)
    endif()

    unset(_warp_libmathdx_library CACHE)
    find_library(_warp_libmathdx_library NAMES ${_mathdx_names} PATHS "${_mathdx_lib_dir}" NO_DEFAULT_PATH)
    if(NOT EXISTS "${_mathdx_root}/include")
        warp_invalid_libmathdx_path("libmathdx include directory was not found at ${_mathdx_root}/include")
    endif()
    if(NOT _warp_libmathdx_library)
        warp_invalid_libmathdx_path("libmathdx library was not found under ${_mathdx_lib_dir}")
    endif()

    set(WARP_LIBMATHDX_INCLUDE_DIR "${_mathdx_root}/include" PARENT_SCOPE)
    set(WARP_LIBMATHDX_LIBRARY "${_warp_libmathdx_library}" PARENT_SCOPE)
endfunction()

# Explicit dependency paths are cached by CMake. Clear invalid cache entries so
# a rerun without the option can fall back to Packman instead of failing forever.
function(warp_invalid_libmathdx_path message)
    if(WARP_LIBMATHDX_PATH AND DEFINED CACHE{WARP_LIBMATHDX_PATH})
        unset(WARP_LIBMATHDX_PATH CACHE)
        message(FATAL_ERROR
            "${message}. The invalid WARP_LIBMATHDX_PATH cache entry was cleared; "
            "rerun configure without -DWARP_LIBMATHDX_PATH to use Packman discovery, "
            "set LIBMATHDX_HOME, or pass -DWARP_USE_LIBMATHDX=OFF."
        )
    endif()

    message(FATAL_ERROR "${message}")
endfunction()

# Keep invalid explicit LLVM paths recoverable for the same reason as libmathdx.
function(warp_invalid_llvm_path message)
    if(WARP_LLVM_PATH AND DEFINED CACHE{WARP_LLVM_PATH})
        unset(WARP_LLVM_PATH CACHE)
        message(FATAL_ERROR
            "${message}. The invalid WARP_LLVM_PATH cache entry was cleared; "
            "rerun configure without -DWARP_LLVM_PATH to use Packman discovery."
        )
    endif()

    message(FATAL_ERROR "${message}")
endfunction()

function(warp_find_llvm)
    if(NOT WARP_BUILD_CLANG)
        set(WARP_LLVM_INCLUDE_DIR "" PARENT_SCOPE)
        set(WARP_LLVM_LIBRARIES "" PARENT_SCOPE)
        set(WARP_LLVM_LIBRARY_DIR "" PARENT_SCOPE)
        return()
    endif()

    warp_get_host_arch(_arch)

    if(WARP_LLVM_PATH)
        set(_llvm_root "${WARP_LLVM_PATH}")
    else()
        warp_get_packman_python(_packman_python)
        warp_get_packman_command(_packman)
        if(WIN32)
            set(_llvm_package "15.0.7-windows-x86_64-ptx-vs142")
        elseif(APPLE)
            set(_llvm_package "15.0.7-darwin-aarch64-macos11")
        elseif(_arch STREQUAL "aarch64")
            set(_llvm_package "15.0.7-linux-aarch64-gcc7.5")
        else()
            set(_llvm_package "18.1.3-linux-x86_64-gcc9.4")
        endif()
        set(_llvm_root "${PROJECT_SOURCE_DIR}/_build/host-deps/llvm-project/release-${_arch}")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env "PM_PYTHON_EXT=${_packman_python}"
                    "${_packman}" install -l "${_llvm_root}" clang+llvm-warp "${_llvm_package}"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            COMMAND_ERROR_IS_FATAL ANY
        )
    endif()

    if(NOT EXISTS "${_llvm_root}/include")
        warp_invalid_llvm_path("LLVM include directory was not found at ${_llvm_root}/include")
    endif()
    if(NOT EXISTS "${_llvm_root}/lib")
        warp_invalid_llvm_path("LLVM library directory was not found at ${_llvm_root}/lib")
    endif()

    if(WIN32)
        file(GLOB _llvm_libs CONFIGURE_DEPENDS "${_llvm_root}/lib/*.lib")
    else()
        file(GLOB _llvm_libs CONFIGURE_DEPENDS "${_llvm_root}/lib/*.a")
    endif()
    if(NOT _llvm_libs)
        warp_invalid_llvm_path("No LLVM static libraries were found under ${_llvm_root}/lib")
    endif()

    # Consumers still need platform system libraries expected by LLVM. Linux
    # consumers should use link grouping for these static archives, while Darwin
    # consumers may need to duplicate LLVM archives to resolve circular symbols.
    set(WARP_LLVM_INCLUDE_DIR "${_llvm_root}/include" PARENT_SCOPE)
    set(WARP_LLVM_LIBRARIES "${_llvm_libs}" PARENT_SCOPE)
    set(WARP_LLVM_LIBRARY_DIR "${_llvm_root}/lib" PARENT_SCOPE)
endfunction()
