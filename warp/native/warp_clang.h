// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "version.h"

#include <cstdint>

// This native C ABI is experimental. Use this header only with warp-clang from
// the exact same Warp release.
extern "C" {

WP_API int wp_load_obj(const char* object_file, const char* module_name, bool use_legacy_linker);
WP_API int wp_unload_obj(const char* module_name);
WP_API uint64_t wp_lookup(const char* module_name, const char* function_name);
WP_API const char* wp_warp_clang_version();

}  // extern "C"
