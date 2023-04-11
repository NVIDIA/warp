/** Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "builtin.h"

// stubs for platforms where there is no CUDA
#if !WP_ENABLE_CUDA || !WP_ENABLE_CUTLASS

extern "C"
{

WP_API
bool cutlass_gemm(
                  int compute_capability,
                  int m, int n, int k,
                  const char* datatype_str,
                  const void* a, const void* b, const void* c, void* d,
                  float alpha, float beta,
                  bool row_major_a, bool row_major_b,
                  bool allow_tf32x3_arith,
                  int batch_count)
{
    printf("CUDA is disabled and/or CUTLASS is disabled.\n");
    return false;
}

} // extern "C"

#endif // !WP_ENABLE_CUDA || !WP_ENABLE_CUTLASS
