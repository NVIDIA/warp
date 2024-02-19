/** Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
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
        vfprintf(g_error_stream, fmt, args);
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
        vfprintf(g_error_stream, fmt, args);
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
