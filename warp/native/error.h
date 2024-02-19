/** Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

namespace wp
{
// functions related to error reporting

// get error string from Python
const char* get_error_string();

// set error message for Python
// these functions also print the error message if error output is enabled
void set_error_string(const char* fmt, ...);
void append_error_string(const char* fmt, ...);

// allow disabling printing errors, which is handy during tests that expect failure
void set_error_output_enabled(bool enable);
bool is_error_output_enabled();

}
