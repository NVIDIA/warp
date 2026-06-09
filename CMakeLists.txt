# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

if(APPLE)
    set(_warp_enable_cuda_default OFF)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Apple Silicon only" FORCE)
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "macOS deployment target" FORCE)
    endif()
else()
    set(_warp_enable_cuda_default ON)
endif()

option(WARP_ENABLE_CUDA "Build Warp with CUDA support" ${_warp_enable_cuda_default})
if(APPLE AND WARP_ENABLE_CUDA)
    message(FATAL_ERROR "CUDA builds are not supported on macOS")
endif()

option(WARP_BUILD_CLANG "Build warp-clang" ON)
option(WARP_VERIFY_FP "Enable floating point verification" OFF)
option(WARP_FAST_MATH "Enable fast math compiler options" OFF)
set(WARP_LIBMATHDX_PATH "" CACHE PATH "")
set(WARP_LLVM_PATH "" CACHE PATH "")
if(WARP_ENABLE_CUDA)
    option(WARP_USE_LIBMATHDX "Build Warp with NVIDIA libmathdx support" ON)
    option(WARP_USE_DYNAMIC_CUDA "Link CUDA runtime and compiler libraries dynamically" OFF)
    set(WARP_CUDA_PATH "" CACHE PATH "")
else()
    set(WARP_USE_LIBMATHDX OFF)
    set(WARP_USE_DYNAMIC_CUDA OFF)
    set(WARP_CUDA_PATH "")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake")
include(WarpDependencies)

if(WARP_ENABLE_CUDA)
    warp_configure_cuda_root()
    warp_set_default_cuda_architectures()
    project(warp LANGUAGES CXX CUDA)
else()
    project(warp LANGUAGES CXX)
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "CMAKE_CONFIGURATION_TYPES: ${CMAKE_CONFIGURATION_TYPES}")
else()
    message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WARP_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    find_package(CUDAToolkit 12.0 REQUIRED)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

warp_find_python()
warp_add_generated_headers_target()
warp_find_libmathdx()
warp_find_llvm()

set(WARP_NATIVE_DIR "${PROJECT_SOURCE_DIR}/warp/native")
set(WARP_BIN_DIR "${PROJECT_SOURCE_DIR}/warp/bin")

function(warp_set_shared_target_output target output_name)
    set_target_properties(
        "${target}"
        PROPERTIES
            OUTPUT_NAME "${output_name}"
            ARCHIVE_OUTPUT_DIRECTORY "${WARP_BIN_DIR}"
            LIBRARY_OUTPUT_DIRECTORY "${WARP_BIN_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${WARP_BIN_DIR}"
    )

    if(NOT APPLE)
        set_target_properties("${target}" PROPERTIES PREFIX "")
    endif()

    foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(
            "${target}"
            PROPERTIES
                ARCHIVE_OUTPUT_DIRECTORY_${config} "${WARP_BIN_DIR}"
                LIBRARY_OUTPUT_DIRECTORY_${config} "${WARP_BIN_DIR}"
                RUNTIME_OUTPUT_DIRECTORY_${config} "${WARP_BIN_DIR}"
        )
    endforeach()
endfunction()

function(warp_add_common_cxx_options target)
    target_compile_options(
        "${target}"
        PRIVATE
            "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/nologo>"
            "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/GR->"
            "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>"
            "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4624>"
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>"
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wuninitialized>"
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fno-rtti>"
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fvisibility=hidden>"
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fvisibility-inlines-hidden>"
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-Og>"
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-g>"
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<NOT:$<CONFIG:Debug>>>:-O3>"
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<BOOL:${WARP_FAST_MATH}>>:/fp:fast>"
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<BOOL:${WARP_FAST_MATH}>>:-ffast-math>"
    )
endfunction()

function(warp_add_linux_link_options target)
    target_link_libraries("${target}" PRIVATE pthread dl rt)
    target_link_options(
        "${target}"
        PRIVATE
            -static-libstdc++
            -static-libgcc
            "LINKER:--version-script,${WARP_NATIVE_DIR}/warp.map"
            "LINKER:-z,lazy"
            "LINKER:--exclude-libs,ALL"
            "LINKER:-rpath,$ORIGIN"
    )
endfunction()

function(warp_find_optional_cuda_library out_var)
    cmake_parse_arguments(ARG "" "" "NAMES" ${ARGN})

    set(_cuda_library_hints)
    foreach(_dir
            "${CUDAToolkit_LIBRARY_DIR}"
            "${CUDAToolkit_LIBRARY_ROOT}"
            "${CUDAToolkit_TARGET_DIR}/lib64"
            "${CUDAToolkit_TARGET_DIR}/lib"
            "${CUDAToolkit_ROOT}/lib64"
            "${CUDAToolkit_ROOT}/lib"
            "${CUDAToolkit_ROOT}/lib/x64"
    )
        if(_dir)
            list(APPEND _cuda_library_hints "${_dir}")
        endif()
    endforeach()

    unset(_warp_cuda_library CACHE)
    find_library(_warp_cuda_library NAMES ${ARG_NAMES} HINTS ${_cuda_library_hints})
    set("${out_var}" "${_warp_cuda_library}" PARENT_SCOPE)
    unset(_warp_cuda_library CACHE)
endfunction()

function(warp_find_required_cuda_library out_var)
    cmake_parse_arguments(ARG "" "" "NAMES" ${ARGN})

    warp_find_optional_cuda_library("${out_var}" NAMES ${ARG_NAMES})
    if(NOT "${${out_var}}")
        string(REPLACE ";" ", " _cuda_library_names "${ARG_NAMES}")
        message(FATAL_ERROR "Required CUDA library ${_cuda_library_names} was not found by CUDAToolkit")
    endif()

    set("${out_var}" "${${out_var}}" PARENT_SCOPE)
endfunction()

function(warp_require_cuda_target target)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "Required CUDA target ${target} was not found by CUDAToolkit")
    endif()
endfunction()

function(warp_link_nvptxcompiler target)
    if(WARP_USE_DYNAMIC_CUDA)
        if(TARGET CUDA::nvptxcompiler)
            target_link_libraries("${target}" PRIVATE CUDA::nvptxcompiler)
        else()
            warp_find_required_cuda_library(WARP_NVPTXCOMPILER_LIBRARY NAMES nvptxcompiler)
            target_link_libraries("${target}" PRIVATE "${WARP_NVPTXCOMPILER_LIBRARY}")
        endif()
    else()
        if(TARGET CUDA::nvptxcompiler_static)
            target_link_libraries("${target}" PRIVATE CUDA::nvptxcompiler_static)
        else()
            warp_find_required_cuda_library(WARP_NVPTXCOMPILER_STATIC_LIBRARY NAMES nvptxcompiler_static)
            target_link_libraries("${target}" PRIVATE "${WARP_NVPTXCOMPILER_STATIC_LIBRARY}")
        endif()
    endif()
endfunction()

set(WARP_CPP_SOURCES
    warp/native/warp.cpp
    warp/native/bvh.cpp
    warp/native/scan.cpp
    warp/native/apic.cpp
    warp/native/alloc_tracker.cpp
    warp/native/crt.cpp
    warp/native/error.cpp
    warp/native/cuda_util.cpp
    warp/native/mesh.cpp
    warp/native/hashgrid.cpp
    warp/native/reduce.cpp
    warp/native/runlength_encode.cpp
    warp/native/sort.cpp
    warp/native/sparse.cpp
    warp/native/volume.cpp
    warp/native/texture.cpp
    warp/native/mathdx.cpp
    warp/native/coloring.cpp
    warp/native/deterministic.cpp
    warp/native/fastcall.cpp
)

set(WARP_SOURCES ${WARP_CPP_SOURCES})

if(WARP_ENABLE_CUDA)
    list(
        APPEND WARP_SOURCES
        warp/native/bvh.cu
        warp/native/deterministic.cu
        warp/native/mesh.cu
        warp/native/sort.cu
        warp/native/hashgrid.cu
        warp/native/reduce.cu
        warp/native/runlength_encode.cu
        warp/native/scan.cu
        warp/native/sparse.cu
        warp/native/volume.cu
        warp/native/volume_builder.cu
        warp/native/warp.cu
    )
endif()

add_library(warp SHARED ${WARP_SOURCES})
add_dependencies(warp warp_generated_headers)
warp_set_shared_target_output(warp warp)
warp_add_common_cxx_options(warp)
set_target_properties(
    warp
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        MSVC_RUNTIME_LIBRARY "MultiThreaded"
)
target_include_directories(warp PRIVATE "${WARP_NATIVE_DIR}" ${WARP_PYTHON_INCLUDE_DIRS})
target_compile_features(warp PRIVATE cxx_std_17)

if(WARP_ENABLE_CUDA)
    target_compile_features(warp PRIVATE cuda_std_17)
    set_target_properties(
        warp
        PROPERTIES CUDA_RUNTIME_LIBRARY "$<IF:$<BOOL:${WARP_USE_DYNAMIC_CUDA}>,Shared,Static>"
    )
endif()

if(WARP_ENABLE_CUDA AND WARP_LIBMATHDX_LIBRARY)
    set(WARP_ENABLE_MATHDX_VALUE 1)
else()
    set(WARP_ENABLE_MATHDX_VALUE 0)
endif()

# CMake builds are intended for local development. Use build_lib.py for release
# builds that need the full CUDA compatibility policy.
set(WARP_ENABLE_CUDA_COMPATIBILITY_VALUE 0)

target_compile_definitions(
    warp
    PRIVATE
        WP_ENABLE_CUDA=$<IF:$<BOOL:${WARP_ENABLE_CUDA}>,1,0>
        WP_ENABLE_MATHDX=${WARP_ENABLE_MATHDX_VALUE}
        WP_ENABLE_CUDA_COMPATIBILITY=${WARP_ENABLE_CUDA_COMPATIBILITY_VALUE}
        WP_ENABLE_DEBUG=$<IF:$<CONFIG:Debug>,1,0>
        "$<$<CXX_COMPILER_ID:MSVC>:NDEBUG>"
        "$<$<CXX_COMPILER_ID:MSVC>:_ITERATOR_DEBUG_LEVEL=0>"
        "$<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:_DEBUG>"
        "$<$<BOOL:${WARP_VERIFY_FP}>:WP_VERIFY_FP>"
        "$<$<BOOL:${WARP_FAST_MATH}>:WP_FAST_MATH>"
        "$<$<PLATFORM_ID:Linux,Darwin>:_GLIBCXX_USE_CXX11_ABI=0>"
)
set_source_files_properties(warp/native/fastcall.cpp PROPERTIES COMPILE_DEFINITIONS Py_LIMITED_API=0x030a0000)

if(WARP_ENABLE_CUDA)
    target_compile_options(
        warp
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=221>"
            # Intentional: -t0 lets nvcc use all CPUs; use -t1 to constrain threading.
            "$<$<COMPILE_LANGUAGE:CUDA>:-t0>"
            "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-g>"
            "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>"
            "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0>"
            "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-lineinfo>"
            "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-O3>"
            "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${WARP_FAST_MATH}>>:--use_fast_math>"
    )

    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
        foreach(_cuda_include_dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
            if(EXISTS "${_cuda_include_dir}/cccl")
                target_include_directories(warp PRIVATE "${_cuda_include_dir}/cccl")
            endif()
        endforeach()
    endif()

    # CUDA imported targets differ across Toolkit versions and static/dynamic
    # modes. Prefer official CMake targets, then fall back to direct library
    # lookup for components that older CMake/CUDA combinations do not expose.
    if(WARP_USE_DYNAMIC_CUDA)
        warp_require_cuda_target(CUDA::cudart)
        warp_require_cuda_target(CUDA::nvrtc)
        target_link_libraries(warp PRIVATE CUDA::cudart CUDA::nvrtc)
        warp_link_nvptxcompiler(warp)
        if(TARGET CUDA::nvJitLink)
            target_link_libraries(warp PRIVATE CUDA::nvJitLink)
        else()
            warp_find_optional_cuda_library(WARP_NVJITLINK_LIBRARY NAMES nvJitLink)
            if(WARP_NVJITLINK_LIBRARY)
                target_link_libraries(warp PRIVATE "${WARP_NVJITLINK_LIBRARY}")
            endif()
        endif()
    else()
        warp_require_cuda_target(CUDA::cudart_static)
        warp_require_cuda_target(CUDA::nvrtc_static)
        target_link_libraries(warp PRIVATE CUDA::cudart_static CUDA::nvrtc_static)
        warp_link_nvptxcompiler(warp)
        if(TARGET CUDA::nvrtc_builtins_static)
            target_link_libraries(warp PRIVATE CUDA::nvrtc_builtins_static)
        elseif(TARGET CUDA::nvrtc-builtins_static)
            target_link_libraries(warp PRIVATE CUDA::nvrtc-builtins_static)
        else()
            warp_find_optional_cuda_library(WARP_NVRTC_BUILTINS_LIBRARY NAMES nvrtc-builtins_static nvrtc_builtins_static)
            if(WARP_NVRTC_BUILTINS_LIBRARY)
                target_link_libraries(warp PRIVATE "${WARP_NVRTC_BUILTINS_LIBRARY}")
            endif()
        endif()
        if(TARGET CUDA::nvJitLink_static)
            target_link_libraries(warp PRIVATE CUDA::nvJitLink_static)
        elseif(TARGET CUDA::nvJitLink)
            target_link_libraries(warp PRIVATE CUDA::nvJitLink)
        else()
            warp_find_optional_cuda_library(WARP_NVJITLINK_LIBRARY NAMES nvJitLink_static nvJitLink)
            if(WARP_NVJITLINK_LIBRARY)
                target_link_libraries(warp PRIVATE "${WARP_NVJITLINK_LIBRARY}")
            endif()
        endif()
    endif()

    if(WARP_LIBMATHDX_LIBRARY)
        target_include_directories(warp PRIVATE "${WARP_LIBMATHDX_INCLUDE_DIR}")
        target_link_libraries(warp PRIVATE "${WARP_LIBMATHDX_LIBRARY}")
    endif()
endif()

if(WIN32)
    target_link_libraries(warp PRIVATE ws2_32 user32 delayimp "${WARP_PYTHON3_LIBRARY}")
    target_link_options(warp PRIVATE /DELAYLOAD:python3.dll)
elseif(APPLE)
    target_link_options(warp PRIVATE "LINKER:-undefined,dynamic_lookup" "LINKER:-rpath,@loader_path")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    warp_add_linux_link_options(warp)
endif()

if(WARP_BUILD_CLANG)
    add_library(
        warp-clang
        SHARED
        warp/native/clang/clang.cpp
        warp/native/crt.cpp
    )
    add_dependencies(warp-clang warp_generated_headers)
    warp_set_shared_target_output(warp-clang warp-clang)
    warp_add_common_cxx_options(warp-clang)
    set_target_properties(
        warp-clang
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            MSVC_RUNTIME_LIBRARY "MultiThreaded"
    )
    target_compile_features(warp-clang PRIVATE cxx_std_17)
    target_include_directories(warp-clang PRIVATE "${WARP_NATIVE_DIR}" "${WARP_LLVM_INCLUDE_DIR}")
    # Match the GCC ABI version used by Warp's LLVM archives.
    target_compile_options(
        warp-clang
        PRIVATE "$<$<AND:$<PLATFORM_ID:Linux>,$<COMPILE_LANG_AND_ID:CXX,GNU>>:-fabi-version=13>"
    )
    target_compile_definitions(
        warp-clang
        PRIVATE
            WP_ENABLE_CUDA=0
            WP_ENABLE_MATHDX=0
            WP_ENABLE_CUDA_COMPATIBILITY=0
            WP_ENABLE_DEBUG=$<IF:$<CONFIG:Debug>,1,0>
            "$<$<CXX_COMPILER_ID:MSVC>:NDEBUG>"
            "$<$<CXX_COMPILER_ID:MSVC>:_ITERATOR_DEBUG_LEVEL=0>"
            "$<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:_DEBUG>"
            "$<$<BOOL:${WARP_FAST_MATH}>:WP_FAST_MATH>"
            "$<$<PLATFORM_ID:Linux,Darwin>:_GLIBCXX_USE_CXX11_ABI=0>"
    )

    if(WIN32)
        target_link_libraries(
            warp-clang
            PRIVATE
                ${WARP_LLVM_LIBRARIES}
                Version
                Ws2_32
                ntdll
                delayimp
                "${WARP_PYTHON3_LIBRARY}"
        )
        target_link_options(warp-clang PRIVATE /DELAYLOAD:python3.dll)
    elseif(APPLE)
        # Repeat LLVM static archives to resolve circular dependencies on macOS.
        target_link_libraries(warp-clang PRIVATE ${WARP_LLVM_LIBRARIES} ${WARP_LLVM_LIBRARIES} pthread dl)
        target_link_options(warp-clang PRIVATE "LINKER:-undefined,dynamic_lookup")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(
            warp-clang
            PRIVATE
                -Wl,--start-group
                ${WARP_LLVM_LIBRARIES}
                -Wl,--end-group
        )
        warp_add_linux_link_options(warp-clang)
    endif()
endif()
