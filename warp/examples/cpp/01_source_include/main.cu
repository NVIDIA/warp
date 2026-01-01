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

/**
 * Warp C++ Integration: Source Inclusion with Automatic Differentiation
 *
 * Demonstrates statically including Warp-generated CUDA source (.cu) in C++.
 * Implements gradient descent for linear regression using auto-generated
 * forward and backward kernels.
 *
 * Workflow:
 * 1. Python: Define differentiable kernel, compile to .cu source
 * 2. C++: Include generated source, launch with <<<>>> syntax
 * 3. Training: Forward pass computes loss, backward pass computes gradients
 *
 * Note: This example uses a single module with one differentiable kernel.
 */

#include "aot.h"  // Warp AOT utilities (includes CUDA headers, builtin.h)

#include <cmath>
#include <random>
#include <vector>

// Warp-generated kernels (forward and backward)
#include "generated/wp___main__.cu"

int main()
{
    std::cout << "=== Warp C++ Source Inclusion: Autodiff Example ===" << std::endl;

    constexpr int N_SAMPLES = 1000;
    constexpr int BLOCK_SIZE = 256;
    constexpr float LEARNING_RATE = 0.01f;
    constexpr int N_ITERATIONS = 100;

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;

    // Generate synthetic data: y = 3.5*x + 1.2 + noise
    std::cout << "\nGenerating synthetic data..." << std::endl;

    const float true_a = 3.5f;
    const float true_b = 1.2f;

    std::vector<float> h_x(N_SAMPLES);
    std::vector<float> h_y_true(N_SAMPLES);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise_dist(-0.1f, 0.1f);

    for (int i = 0; i < N_SAMPLES; i++) {
        h_x[i] = i / 100.0f;
        h_y_true[i] = true_a * h_x[i] + true_b + noise_dist(rng);
    }

    std::cout << "Generated " << N_SAMPLES << " samples" << std::endl;
    std::cout << "True model: y = " << true_a << "*x + " << true_b << std::endl;

    std::cout << "\nAllocating device memory..." << std::endl;

    float *d_params, *d_x, *d_y_true, *d_loss;
    CHECK_CUDA(cudaMalloc(&d_params, 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, N_SAMPLES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y_true, N_SAMPLES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_loss, sizeof(float)));

    float *d_adj_params, *d_adj_x, *d_adj_y_true, *d_adj_loss;
    CHECK_CUDA(cudaMalloc(&d_adj_params, 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_adj_x, N_SAMPLES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_adj_y_true, N_SAMPLES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_adj_loss, sizeof(float)));

    std::cout << "Initializing data..." << std::endl;

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), N_SAMPLES * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_true, h_y_true.data(), N_SAMPLES * sizeof(float), cudaMemcpyHostToDevice));

    float h_params[2] = { 0.5f, 0.5f };
    CHECK_CUDA(cudaMemcpy(d_params, h_params, 2 * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Initial guess: y = " << h_params[0] << "*x + " << h_params[1] << std::endl;

    // Prepare Warp structures
    // CRITICAL: Using positional initialization for MSVC compatibility
    wp::launch_bounds_t dim = { { N_SAMPLES, 0, 0, 0 }, 1, size_t(N_SAMPLES) };

    wp::array_t<wp::float32> arr_params(d_params, 2);
    wp::array_t<wp::float32> arr_x(d_x, N_SAMPLES);
    wp::array_t<wp::float32> arr_y_true(d_y_true, N_SAMPLES);
    wp::array_t<wp::float32> arr_loss(d_loss, 1);

    wp::array_t<wp::float32> adj_params(d_adj_params, 2);
    wp::array_t<wp::float32> adj_x(d_adj_x, N_SAMPLES);
    wp::array_t<wp::float32> adj_y_true(d_adj_y_true, N_SAMPLES);
    wp::array_t<wp::float32> adj_loss(d_adj_loss, 1);

    int grid_dim = (N_SAMPLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Training loop
    std::cout << "\nRunning gradient descent (learning rate: " << LEARNING_RATE << ")..." << std::endl;

    // Debug: Check initial data
    float debug_x0, debug_y0, debug_p0, debug_p1;
    CHECK_CUDA(cudaMemcpy(&debug_x0, d_x, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&debug_y0, d_y_true, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&debug_p0, d_params, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&debug_p1, d_params + 1, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "DEBUG: Initial data - x[0]=" << debug_x0 << ", y_true[0]=" << debug_y0 << ", params=[" << debug_p0
              << ", " << debug_p1 << "]" << std::endl;

    for (int iter = 0; iter < N_ITERATIONS; iter++) {
        CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));
        CHECK_CUDA(cudaMemset(d_adj_params, 0, 2 * sizeof(float)));

        // CRITICAL: Seed the loss gradient (∂L/∂L = 1) for backpropagation
        float adj_loss_val = 1.0f;
        CHECK_CUDA(cudaMemcpy(d_adj_loss, &adj_loss_val, sizeof(float), cudaMemcpyHostToDevice));

        // Forward pass
        compute_loss_cuda_kernel_forward<<<grid_dim, BLOCK_SIZE>>>(dim, arr_params, arr_x, arr_y_true, arr_loss);
        CHECK_CUDA(cudaGetLastError());

        // Debug: Check forward pass result on first iteration
        if (iter == 0) {
            float debug_loss;
            CHECK_CUDA(cudaMemcpy(&debug_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
            std::cout << "DEBUG: Iter 0 - Forward pass loss=" << debug_loss
                      << " (expected ~321000 for untrained params)" << std::endl;
        }

        // Backward pass (auto-generated by Warp)
        compute_loss_cuda_kernel_backward<<<grid_dim, BLOCK_SIZE>>>(
            dim, arr_params, arr_x, arr_y_true, arr_loss, adj_params, adj_x, adj_y_true, adj_loss
        );
        CHECK_CUDA(cudaGetLastError());

        // Debug: Check gradients on first iteration
        if (iter == 0) {
            float debug_grads[2];
            CHECK_CUDA(cudaMemcpy(debug_grads, d_adj_params, 2 * sizeof(float), cudaMemcpyDeviceToHost));
            std::cout << "DEBUG: Iter 0 - Gradients=[" << debug_grads[0] << ", " << debug_grads[1]
                      << "] (should be non-zero)" << std::endl;
        }

        // Update parameters on GPU: params -= (learning_rate / N_SAMPLES) * gradients
        float normalized_lr = LEARNING_RATE / N_SAMPLES;
        update_params_cuda_kernel_forward<<<1, 2>>>({ { 2, 0, 0, 0 }, 1, 2 }, normalized_lr, adj_params, arr_params);
        CHECK_CUDA(cudaGetLastError());

        if (iter % 10 == 0) {
            float h_loss;
            CHECK_CUDA(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_params, d_params, 2 * sizeof(float), cudaMemcpyDeviceToHost));
            std::cout << "Iter " << iter << ": MSE=" << h_loss / N_SAMPLES << ", params=[" << h_params[0] << ", "
                      << h_params[1] << "]" << std::endl;
        }
    }

    std::cout << "\nTraining results:" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_params, d_params, 2 * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Learned: y = " << h_params[0] << "*x + " << h_params[1] << std::endl;
    std::cout << "True:    y = " << true_a << "*x + " << true_b << std::endl;
    std::cout << "Error:   Δa=" << std::abs(h_params[0] - true_a) << ", Δb=" << std::abs(h_params[1] - true_b)
              << std::endl;

    // With noisy data (±0.1), perfect recovery is impossible
    float error_threshold = 0.2f;
    bool converged
        = (std::abs(h_params[0] - true_a) < error_threshold) && (std::abs(h_params[1] - true_b) < error_threshold);

    if (converged) {
        std::cout << "✓ Training converged successfully!" << std::endl;
    } else {
        std::cout << "✗ Training did not converge" << std::endl;
    }

    cudaFree(d_params);
    cudaFree(d_x);
    cudaFree(d_y_true);
    cudaFree(d_loss);
    cudaFree(d_adj_params);
    cudaFree(d_adj_x);
    cudaFree(d_adj_y_true);
    cudaFree(d_adj_loss);

    return converged ? 0 : 1;
}
