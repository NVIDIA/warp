/** Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "../native/builtin.h"

namespace wp {

extern "C" {

WP_API size_t compile_cpp(const char* cpp_src, const char* include_dir, bool debug, bool verbose, bool verify_fp, bool fast_math, const char* output_file)
{
    return -2;
}

}  // extern "C"

}  // namespace wp