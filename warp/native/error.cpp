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

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

namespace wp
{
static char g_error_buffer[4096] = "";
static bool g_error_output_enabled = true;
static FILE* g_error_stream = stderr;

const char* get_error_string()
{
    return g_error_buffer;
}

void set_error_string(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_error_buffer, sizeof(g_error_buffer), fmt, args);
    if (g_error_output_enabled)
    {
        // note: we deliberately avoid vfprintf() due to problems with runtime glibc mismatch
        fputs(g_error_buffer, g_error_stream);
        fputc('\n', g_error_stream);
        fflush(g_error_stream);
    }
    va_end(args);
}

void append_error_string(const char* fmt, ...)
{
    size_t offset = strlen(g_error_buffer);
    if (offset + 2 > sizeof(g_error_buffer))
        return;
    g_error_buffer[offset++] = '\n';
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_error_buffer + offset, sizeof(g_error_buffer) - offset, fmt, args);
    if (g_error_output_enabled)
    {
        // note: we deliberately avoid vfprintf() due to problems with runtime glibc mismatch
        fputs(g_error_buffer + offset, g_error_stream);
        fputc('\n', g_error_stream);
        fflush(g_error_stream);
    }
    va_end(args);
}

void set_error_output_enabled(bool enable)
{
    g_error_output_enabled = enable;
}

bool is_error_output_enabled()
{
    return g_error_output_enabled;
}

} // end of namespace wp
