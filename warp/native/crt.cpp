/** Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "crt.h"

#include <cmath>
#include <cstdio>
#include <cassert>

extern "C" WP_API int _wp_isfinite(double x)
{
    return std::isfinite(x);
}

extern "C" WP_API void _wp_assert(const char* expression, const char* file, unsigned int line)
{
    fflush(stdout);
    fprintf(stderr,
        "Assertion failed: '%s'\n"
        "At '%s:%d'\n",
        expression, file, line);
    fflush(stderr);

    // Now invoke the standard assert(), which may abort the program or break
    // into the debugger as decided by the runtime environment.
    assert(false && "assert() failed");
}