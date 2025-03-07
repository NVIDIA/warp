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
#include <cassert>

extern "C" WP_API int _wp_isfinite(double x)
{
    return std::isfinite(x);
}

extern "C" WP_API int _wp_isnan(double x)
{
    return std::isnan(x);
}

extern "C" WP_API int _wp_isinf(double x)
{
    return std::isinf(x);
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
