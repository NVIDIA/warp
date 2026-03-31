// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace wp {
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
