/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "builtin.h"

namespace wp {

// Texture descriptor passed to kernels
// Contains the CUDA texture object handle and resolution info
struct texture2d_t {
    uint64 tex;  // CUtexObject handle
    int32 width;
    int32 height;
    int32 num_channels;

    CUDA_CALLABLE inline texture2d_t()
        : tex(0)
        , width(0)
        , height(0)
        , num_channels(0)
    {
    }

    CUDA_CALLABLE inline texture2d_t(uint64 tex, int32 width, int32 height, int32 num_channels)
        : tex(tex)
        , width(width)
        , height(height)
        , num_channels(num_channels)
    {
    }
};

struct texture3d_t {
    uint64 tex;  // CUtexObject handle
    int32 width;
    int32 height;
    int32 depth;
    int32 num_channels;

    CUDA_CALLABLE inline texture3d_t()
        : tex(0)
        , width(0)
        , height(0)
        , depth(0)
        , num_channels(0)
    {
    }

    CUDA_CALLABLE inline texture3d_t(uint64 tex, int32 width, int32 height, int32 depth, int32 num_channels)
        : tex(tex)
        , width(width)
        , height(height)
        , depth(depth)
        , num_channels(num_channels)
    {
    }
};

// Texture sampling functions
// These use CUDA texture fetch intrinsics on device, return 0 on host

// Helper to convert CUDA types to Warp types
template <typename T>
struct texture_sample_helper;

template <>
struct texture_sample_helper<float> {
    static CUDA_CALLABLE float sample_2d(const texture2d_t& tex, float u, float v)
    {
#if defined(__CUDA_ARCH__)
        return tex2D<float>(tex.tex, u, v);
#else
        return 0.0f;
#endif
    }

    static CUDA_CALLABLE float sample_3d(const texture3d_t& tex, float u, float v, float w)
    {
#if defined(__CUDA_ARCH__)
        return tex3D<float>(tex.tex, u, v, w);
#else
        return 0.0f;
#endif
    }

    static CUDA_CALLABLE float zero() { return 0.0f; }
};

template <>
struct texture_sample_helper<vec2f> {
    static CUDA_CALLABLE vec2f sample_2d(const texture2d_t& tex, float u, float v)
    {
#if defined(__CUDA_ARCH__)
        float2 val = tex2D<float2>(tex.tex, u, v);
        return vec2f(val.x, val.y);
#else
        return vec2f(0.0f, 0.0f);
#endif
    }

    static CUDA_CALLABLE vec2f sample_3d(const texture3d_t& tex, float u, float v, float w)
    {
#if defined(__CUDA_ARCH__)
        float2 val = tex3D<float2>(tex.tex, u, v, w);
        return vec2f(val.x, val.y);
#else
        return vec2f(0.0f, 0.0f);
#endif
    }

    static CUDA_CALLABLE vec2f zero() { return vec2f(0.0f, 0.0f); }
};

template <>
struct texture_sample_helper<vec4f> {
    static CUDA_CALLABLE vec4f sample_2d(const texture2d_t& tex, float u, float v)
    {
#if defined(__CUDA_ARCH__)
        float4 val = tex2D<float4>(tex.tex, u, v);
        return vec4f(val.x, val.y, val.z, val.w);
#else
        return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
#endif
    }

    static CUDA_CALLABLE vec4f sample_3d(const texture3d_t& tex, float u, float v, float w)
    {
#if defined(__CUDA_ARCH__)
        float4 val = tex3D<float4>(tex.tex, u, v, w);
        return vec4f(val.x, val.y, val.z, val.w);
#else
        return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
#endif
    }

    static CUDA_CALLABLE vec4f zero() { return vec4f(0.0f, 0.0f, 0.0f, 0.0f); }
};

// 2D texture sampling with vec2 coordinates
template <typename T>
CUDA_CALLABLE T texture_sample(const texture2d_t& tex, const vec2f& uv)
{
    return texture_sample_helper<T>::sample_2d(tex, uv[0], uv[1]);
}

// 2D texture sampling with separate u, v coordinates
template <typename T>
CUDA_CALLABLE T texture_sample(const texture2d_t& tex, float u, float v)
{
    return texture_sample_helper<T>::sample_2d(tex, u, v);
}

// 3D texture sampling with vec3 coordinates
template <typename T>
CUDA_CALLABLE T texture_sample(const texture3d_t& tex, const vec3f& uvw)
{
    return texture_sample_helper<T>::sample_3d(tex, uvw[0], uvw[1], uvw[2]);
}

// 3D texture sampling with separate u, v, w coordinates
template <typename T>
CUDA_CALLABLE T texture_sample(const texture3d_t& tex, float u, float v, float w)
{
    return texture_sample_helper<T>::sample_3d(tex, u, v, w);
}

// Adjoint stubs for texture sampling (non-differentiable for now)
template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture2d_t& tex,
    const vec2f& uv,
    texture2d_t& adj_tex,
    vec2f& adj_uv,
    const T& adj_ret
)
{
    // Texture sampling is not differentiable in this implementation
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture2d_t& tex,
    float u,
    float v,
    texture2d_t& adj_tex,
    float& adj_u,
    float& adj_v,
    const T& adj_ret
)
{
    // Texture sampling is not differentiable in this implementation
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture3d_t& tex,
    const vec3f& uvw,
    texture3d_t& adj_tex,
    vec3f& adj_uvw,
    const T& adj_ret
)
{
    // Texture sampling is not differentiable in this implementation
}

template <typename T>
CUDA_CALLABLE void adj_texture_sample(
    const texture3d_t& tex,
    float u,
    float v,
    float w,
    texture3d_t& adj_tex,
    float& adj_u,
    float& adj_v,
    float& adj_w,
    const T& adj_ret
)
{
    // Texture sampling is not differentiable in this implementation
}

// Type aliases for code generation
using Texture2D = texture2d_t;
using Texture3D = texture3d_t;

}  // namespace wp
