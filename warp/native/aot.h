/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Warp AOT (Ahead-Of-Time) utilities for C++ integration
// This header provides common boilerplate for C++ programs that use
// Warp-generated kernels via PTX loading or source inclusion.

// Auto-detect CUDA compilation
#if defined(__CUDACC__) || defined(__CUDA__) || defined(__NVCC__)
#define WP_ENABLE_CUDA 1
#else
#define WP_ENABLE_CUDA 0
#endif

// Default tile block dimension (can be overridden before including aot.h)
#ifndef WP_TILE_BLOCK_DIM
#define WP_TILE_BLOCK_DIM 256
#endif

// CUDA-specific includes and error checking macros
#if WP_ENABLE_CUDA
#include <cstdlib>  // For exit() in error checking macros
#include <iostream>  // For std::cerr in error checking macros

#include <cuda.h>  // CUDA Driver API
#include <cuda_runtime.h>  // CUDA Runtime API

#ifndef CHECK_CU
// CUDA Driver API error checking
#define CHECK_CU(call) \
        do { \
            CUresult err = call; \
            if (err != CUDA_SUCCESS) { \
                const char* errStr; \
                cuGetErrorString(err, &errStr); \
                std::cerr << "CUDA Driver API error at " << __FILE__ << ":" << __LINE__ \
                          << " - " << errStr << std::endl; \
                exit(1); \
            } \
        } while(0)
#endif

#ifndef CHECK_CUDA
// CUDA Runtime API error checking
#define CHECK_CUDA(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                std::cerr << "CUDA Runtime API error at " << __FILE__ << ":" << __LINE__ \
                          << " - " << cudaGetErrorString(err) << std::endl; \
                exit(1); \
            } \
        } while(0)
#endif

#endif  // WP_ENABLE_CUDA

// Include Warp builtin types and functions
#include "builtin.h"
