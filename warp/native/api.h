// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Define WP_BUILD_DLL only when building Warp's native shared libraries.
// Windows consumers leave it undefined so WP_API imports Warp's symbols.

#ifndef WP_API
#if defined(__CUDA_ARCH__)
#define WP_API
#elif defined(_WIN32)
#if defined(WP_BUILD_DLL) || defined(WP_NO_CRT)
#define WP_API __declspec(dllexport)
#else
#define WP_API __declspec(dllimport)
#endif
#else
#define WP_API __attribute__((visibility("default")))
#endif
#endif  // WP_API
