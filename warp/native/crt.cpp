/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "crt.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#include <intrin.h>  // for __debugbreak()
#elif defined(__has_builtin)
#if __has_builtin(__builtin_debugtrap)
#define WP_HAS_DEBUGTRAP 1
#endif
#endif

extern "C" WP_API int _wp_isfinite(double x) { return std::isfinite(x); }

extern "C" WP_API int _wp_isnan(double x) { return std::isnan(x); }

extern "C" WP_API int _wp_isinf(double x) { return std::isinf(x); }

extern "C" WP_API void _wp_assert(const char* expression, const char* file, unsigned int line)
{
    fflush(stdout);
    fprintf(
        stderr,
        "Assertion failed: '%s'\n"
        "At '%s:%u'\n",
        expression, file, line
    );
    fflush(stderr);

    // Break into debugger if attached, then abort.
    // We don't use assert() because crt.cpp may be compiled with NDEBUG defined,
    // which would make assert() a no-op.
#if defined(_MSC_VER)
    __debugbreak();
#elif defined(WP_HAS_DEBUGTRAP)
    __builtin_debugtrap();  // Clang - breakpoint, can continue
#elif defined(__GNUC__)
    __builtin_trap();  // GCC - causes SIGILL, does not return
#endif

    abort();
}
