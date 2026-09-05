// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"
#include "warp_clang.h"

#include <cstring>
#include <iostream>

int main()
{
    const char* runtime_version = wp_version();
    if (runtime_version == nullptr || std::strcmp(runtime_version, WP_VERSION_STRING) != 0) {
        std::cerr << "warp version mismatch: expected " << WP_VERSION_STRING << ", got "
                  << (runtime_version == nullptr ? "<null>" : runtime_version) << '\n';
        return 1;
    }

    const char* clang_version = wp_warp_clang_version();
    if (clang_version == nullptr || std::strcmp(clang_version, WP_VERSION_STRING) != 0) {
        std::cerr << "warp-clang version mismatch: expected " << WP_VERSION_STRING << ", got "
                  << (clang_version == nullptr ? "<null>" : clang_version) << '\n';
        return 2;
    }

    return 0;
}
