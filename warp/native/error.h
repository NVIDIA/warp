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
