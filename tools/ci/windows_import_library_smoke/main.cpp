// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "apic.h"
#include "warp.h"
#include "warp_clang.h"

#include <cstring>
#include <iostream>

int main()
{
    if (wp_init(WP_VERSION_STRING) != 0) {
        const char* error = wp_get_error_string();
        std::cerr << "Failed to initialize Warp: " << (error == nullptr ? "<null>" : error) << '\n';
        return 1;
    }

    const char* runtime_version = wp_version();
    if (runtime_version == nullptr || std::strcmp(runtime_version, WP_VERSION_STRING) != 0) {
        std::cerr << "warp version mismatch: expected " << WP_VERSION_STRING << ", got "
                  << (runtime_version == nullptr ? "<null>" : runtime_version) << '\n';
        return 2;
    }

    const char* clang_version = wp_warp_clang_version();
    if (clang_version == nullptr || std::strcmp(clang_version, WP_VERSION_STRING) != 0) {
        std::cerr << "warp-clang version mismatch: expected " << WP_VERSION_STRING << ", got "
                  << (clang_version == nullptr ? "<null>" : clang_version) << '\n';
        return 3;
    }

    APICState* state = wp_apic_create_state();
    if (state == nullptr) {
        std::cerr << "Failed to create an APIC state\n";
        return 4;
    }
    wp_apic_destroy_state(state);

    // Volatile reads keep optimized builds linked to every function declared
    // by warp_clang.h without requiring an object-file fixture.
    volatile auto load_obj = &wp_load_obj;
    volatile auto unload_obj = &wp_unload_obj;
    volatile auto lookup = &wp_lookup;
    if (load_obj == nullptr || unload_obj == nullptr || lookup == nullptr) {
        std::cerr << "A warp-clang import resolved to null\n";
        return 5;
    }

    return 0;
}
