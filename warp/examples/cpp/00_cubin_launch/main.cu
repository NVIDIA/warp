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

/*
 * Example: Loading and launching Warp-compiled kernels from C++
 *
 * This program demonstrates how to:
 * 1. Load a CUBIN file compiled by Warp
 * 2. Launch kernels using CUDA Driver API
 * 3. Pass scalar and array parameters using Warp's data structures
 *
 * This example uses one module with one kernel for simplicity.
 * The same patterns extend to multiple modules and kernels.
 *
 * SAXPY (Single-Precision A·X Plus Y) computes: y = alpha * x + y
 *
 * See README.md for build instructions.
 */

#include "aot.h"  // Warp AOT utilities (includes CUDA headers, builtin.h)

#include <cmath>
#include <fstream>
#include <vector>

// CUBIN_FILE and KERNEL_NAME are defined by the build system via -D flags
#ifndef CUBIN_FILE
#error "CUBIN_FILE not defined. This should be set by the build system."
#endif

#ifndef KERNEL_NAME
#error "KERNEL_NAME not defined. This should be set by the build system."
#endif
std::string read_file(const char* filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv)
{
    std::cout << "=== Warp C++ SAXPY Example ===" << std::endl;

    constexpr int ARRAY_SIZE = 1024 * 1024;
    constexpr int BLOCK_SIZE = 256;
    constexpr float TOLERANCE = 1e-5f;
    constexpr float ALPHA = 2.5f;

    std::cout << "\nInitializing CUDA..." << std::endl;
    CHECK_CU(cuInit(0));

    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));

    char device_name[256];
    CHECK_CU(cuDeviceGetName(device_name, sizeof(device_name), device));
    std::cout << "Using device: " << device_name << std::endl;

    // Use primary context for Driver/Runtime API interoperability
    CUcontext context;
    CHECK_CU(cuDevicePrimaryCtxRetain(&context, device));
    CHECK_CU(cuCtxSetCurrent(context));

    std::cout << "\nLoading CUBIN file..." << std::endl;
    const std::string cubin_data = read_file(CUBIN_FILE);
    std::cout << "Loaded " << cubin_data.size() << " bytes from " << CUBIN_FILE << std::endl;

    CUmodule module;
    CHECK_CU(cuModuleLoadData(&module, cubin_data.c_str()));

    CUfunction kernel;
    CHECK_CU(cuModuleGetFunction(&kernel, module, KERNEL_NAME));
    std::cout << "Kernel: " << KERNEL_NAME << std::endl;

    std::cout << "\nAllocating and initializing data..." << std::endl;
    const size_t size_bytes = ARRAY_SIZE * sizeof(float);

    std::vector<float> h_x(ARRAY_SIZE);
    std::vector<float> h_y(ARRAY_SIZE);
    std::vector<float> h_y_original(ARRAY_SIZE);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = 1.0f;
        h_y_original[i] = h_y[i];
    }

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, size_bytes));
    CHECK_CUDA(cudaMalloc(&d_y, size_bytes));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y.data(), size_bytes, cudaMemcpyHostToDevice));

    std::cout << "Allocated " << (2 * size_bytes) / (1024 * 1024) << " MB on device" << std::endl;
    std::cout << "Array size: " << ARRAY_SIZE << " elements, Alpha: " << ALPHA << std::endl;

    // Prepare kernel parameters
    // CRITICAL: Using positional initialization for MSVC compatibility
    wp::launch_bounds_t dim = { { ARRAY_SIZE, 0, 0, 0 }, 1, size_t(ARRAY_SIZE) };
    wp::array_t<wp::float32> arr_x(d_x, ARRAY_SIZE);
    wp::array_t<wp::float32> arr_y(d_y, ARRAY_SIZE);
    wp::float32 alpha_scalar = ALPHA;

    std::cout << "\nLaunching kernel..." << std::endl;

    // CRITICAL: Parameter order matches kernel signature: saxpy_cuda_kernel_forward(dim, alpha, x, y)
    void* params[] = { &dim, &alpha_scalar, &arr_x, &arr_y };

    const int grid_dim = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::cout << "Grid: " << grid_dim << " blocks, Block: " << BLOCK_SIZE << " threads" << std::endl;
    std::cout << "Computing: y = " << ALPHA << " * x + y" << std::endl;

    CHECK_CU(cuLaunchKernel(kernel, grid_dim, 1, 1, BLOCK_SIZE, 1, 1, 0, nullptr, params, nullptr));

    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "Kernel completed successfully" << std::endl;

    // Verify results: y_new = alpha * x + y_original
    std::cout << "\nVerifying results..." << std::endl;
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, size_bytes, cudaMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        const float expected = ALPHA * h_x[i] + h_y_original[i];
        if (std::fabs(h_y[i] - expected) > TOLERANCE) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "✓ All results correct!" << std::endl;
        const int idx = 10;
        std::cout << "Example: y[" << idx << "] = " << ALPHA << " * " << h_x[idx] << " + " << h_y_original[idx] << " = "
                  << h_y[idx] << std::endl;
    } else {
        std::cerr << "✗ Verification failed!" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CU(cuModuleUnload(module));
    CHECK_CU(cuDevicePrimaryCtxRelease(device));

    return success ? 0 : 1;
}
