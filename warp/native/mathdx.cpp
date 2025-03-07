/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "builtin.h"

// stubs for platforms where there is no CUDA
#if !WP_ENABLE_CUDA || !WP_ENABLE_MATHDX

extern "C"
{

WP_API
bool cuda_compile_fft(
                      const char* ltoir_output_path,
                      const char* symbol_name, int num_include_dirs,
                      const char** include_dirs,
                      const char* mathdx_include_dir,
                      int arch,
                      int size,
                      int elements_per_thread,
                      int direction,
                      int precision,
                      int* shared_memory_size)
{
    printf("CUDA is disabled and/or Warp was not compiled with MathDx support.\n");
    return false;
}

WP_API bool cuda_compile_dot(
                             const char* fatbin_output_path,
                             const char* ltoir_output_path,
                             const char* symbol_name,
                             int num_include_dirs,
                             const char** include_dirs,
                             const char* mathdx_include_dir,
                             int arch,
                             int M,
                             int N,
                             int K,
                             int precision_A,
                             int precision_B,
                             int precision_C,
                             int type,
                             int a_arrangement,
                             int b_arrangement,
                             int c_arrangement,
                             int num_threads)
{
    printf("CUDA is disabled and/or Warp was not compiled with MathDx support.\n");
    return false;
}

WP_API bool cuda_compile_solver(
                                const char* ltoir_output_path,
                                const char* symbol_name,
                                int num_include_dirs,
                                const char** include_dirs,
                                const char* mathdx_include_dir,
                                int arch,
                                int M,
                                int N,
                                int function,
                                int precision,
                                int fill_mode,
                                int num_threads)
{
    printf("CUDA is disabled and/or Warp was not compiled with MathDx support.\n");
    return false;
}

} // extern "C"

#endif // !WP_ENABLE_CUDA || !WP_ENABLE_MATHDX
