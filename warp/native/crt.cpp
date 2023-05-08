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

// Export CRT symbols from warp.dll for use by compute kernel DLLs
// These are declared in crt.h
#if defined(_MSC_VER)

#pragma comment(linker,"/export:printf")

#pragma comment(linker,"/export:abs")
#pragma comment(linker,"/export:llabs")

#pragma comment(linker,"/export:fmodf")
#pragma comment(linker,"/export:fmod")
#pragma comment(linker,"/export:logf")
#pragma comment(linker,"/export:log")
#pragma comment(linker,"/export:log2f")
#pragma comment(linker,"/export:log2")
#pragma comment(linker,"/export:log10f")
#pragma comment(linker,"/export:log10")
#pragma comment(linker,"/export:expf")
#pragma comment(linker,"/export:exp")
#pragma comment(linker,"/export:sqrtf")
#pragma comment(linker,"/export:sqrt")
#pragma comment(linker,"/export:powf")
#pragma comment(linker,"/export:pow")
#pragma comment(linker,"/export:floorf")
#pragma comment(linker,"/export:floor")
#pragma comment(linker,"/export:ceilf")
#pragma comment(linker,"/export:ceil")
#pragma comment(linker,"/export:fabsf")
#pragma comment(linker,"/export:fabs")
#pragma comment(linker,"/export:roundf")
#pragma comment(linker,"/export:round")
#pragma comment(linker,"/export:truncf")
#pragma comment(linker,"/export:trunc")
#pragma comment(linker,"/export:rintf")
#pragma comment(linker,"/export:rint")
#pragma comment(linker,"/export:acosf")
#pragma comment(linker,"/export:acos")
#pragma comment(linker,"/export:asinf")
#pragma comment(linker,"/export:asin")
#pragma comment(linker,"/export:atanf")
#pragma comment(linker,"/export:atan")
#pragma comment(linker,"/export:atan2f")
#pragma comment(linker,"/export:atan2")
#pragma comment(linker,"/export:cosf")
#pragma comment(linker,"/export:cos")
#pragma comment(linker,"/export:sinf")
#pragma comment(linker,"/export:sin")
#pragma comment(linker,"/export:tanf")
#pragma comment(linker,"/export:tan")
#pragma comment(linker,"/export:sinhf")
#pragma comment(linker,"/export:sinh")
#pragma comment(linker,"/export:coshf")
#pragma comment(linker,"/export:cosh")
#pragma comment(linker,"/export:tanhf")
#pragma comment(linker,"/export:tanh")
#pragma comment(linker,"/export:fmaf")

#pragma comment(linker,"/export:memset")
#pragma comment(linker,"/export:memcpy")

#pragma comment(linker,"/export:_wp_isfinite")
#pragma comment(linker,"/export:_wp_assert")

// For functions with large stack frames the MSVC compiler will emit a call to
// __chkstk() to linearly touch each memory page. This grows the stack without
// triggering the stack overflow guards.
#pragma comment(linker,"/export:__chkstk")

// The MSVC linker checks for the _fltused symbol if any floating-point
// functionality is used. It's defined by the Microsoft CRT to indicate that
// the x87 FPU control word was properly initialized.
#pragma comment(linker,"/export:_fltused")

#endif  // _MSC_VER