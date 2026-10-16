// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "cpu_block_runtime.h"
#include "version.h"

#include <cstddef>
#include <cstdint>

// This native C ABI is experimental. Use this header only with warp-clang from
// the exact same Warp release.
extern "C" {

WP_API int wp_load_obj(const char* object_file, const char* module_name, bool use_legacy_linker);
// Load a CPU object from host bytes, copying them before returning.
// Returns 0 on success or -1 on failure. `module_name` must be unique among
// loaded objects; duplicate names make symbol lookups ambiguous.
WP_API int wp_load_obj_from_bytes(const void* data, size_t size, const char* module_name, bool use_legacy_linker);
WP_API int wp_unload_obj(const char* module_name);
WP_API uint64_t wp_lookup(const char* module_name, const char* function_name);
WP_API int wp_llvm_set_cpu_block_runtime(const wp_cpu_block_runtime_api* api);
WP_API const char* wp_warp_clang_version();

}  // extern "C"
