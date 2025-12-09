/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    uint64 tex;      // CUtexObject handle
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
    uint64 tex;      // CUtexObject handle
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

// Resolution query functions
CUDA_CALLABLE inline int32 texture_width(const texture2d_t& tex)
{
    return tex.width;
}

CUDA_CALLABLE inline int32 texture_height(const texture2d_t& tex)
{
    return tex.height;
}

CUDA_CALLABLE inline int32 texture_width(const texture3d_t& tex)
{
    return tex.width;
}

CUDA_CALLABLE inline int32 texture_height(const texture3d_t& tex)
{
    return tex.height;
}

CUDA_CALLABLE inline int32 texture_depth(const texture3d_t& tex)
{
    return tex.depth;
}

// Adjoint stubs for resolution queries (non-differentiable)
CUDA_CALLABLE inline void adj_texture_width(const texture2d_t& tex, texture2d_t& adj_tex, int32 adj_ret) {}
CUDA_CALLABLE inline void adj_texture_height(const texture2d_t& tex, texture2d_t& adj_tex, int32 adj_ret) {}
CUDA_CALLABLE inline void adj_texture_width(const texture3d_t& tex, texture3d_t& adj_tex, int32 adj_ret) {}
CUDA_CALLABLE inline void adj_texture_height(const texture3d_t& tex, texture3d_t& adj_tex, int32 adj_ret) {}
CUDA_CALLABLE inline void adj_texture_depth(const texture3d_t& tex, texture3d_t& adj_tex, int32 adj_ret) {}

// Texture sampling functions
// These use CUDA texture fetch intrinsics on device, return 0 on host

// 2D texture sampling with normalized coordinates [0,1]
CUDA_CALLABLE inline float tex2d_f(const texture2d_t& tex, float u, float v)
{
#if defined(__CUDA_ARCH__)
    return tex2D<float>(tex.tex, u, v);
#else
    return 0.0f;
#endif
}

CUDA_CALLABLE inline vec2f tex2d_v2(const texture2d_t& tex, float u, float v)
{
#if defined(__CUDA_ARCH__)
    float2 val = tex2D<float2>(tex.tex, u, v);
    return vec2f(val.x, val.y);
#else
    return vec2f(0.0f, 0.0f);
#endif
}

CUDA_CALLABLE inline vec3f tex2d_v3(const texture2d_t& tex, float u, float v)
{
#if defined(__CUDA_ARCH__)
    float4 val = tex2D<float4>(tex.tex, u, v);
    return vec3f(val.x, val.y, val.z);
#else
    return vec3f(0.0f, 0.0f, 0.0f);
#endif
}

CUDA_CALLABLE inline vec4f tex2d_v4(const texture2d_t& tex, float u, float v)
{
#if defined(__CUDA_ARCH__)
    float4 val = tex2D<float4>(tex.tex, u, v);
    return vec4f(val.x, val.y, val.z, val.w);
#else
    return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
#endif
}

// 3D texture sampling with normalized coordinates [0,1]
CUDA_CALLABLE inline float tex3d_f(const texture3d_t& tex, float u, float v, float w)
{
#if defined(__CUDA_ARCH__)
    return tex3D<float>(tex.tex, u, v, w);
#else
    return 0.0f;
#endif
}

CUDA_CALLABLE inline vec2f tex3d_v2(const texture3d_t& tex, float u, float v, float w)
{
#if defined(__CUDA_ARCH__)
    float2 val = tex3D<float2>(tex.tex, u, v, w);
    return vec2f(val.x, val.y);
#else
    return vec2f(0.0f, 0.0f);
#endif
}

CUDA_CALLABLE inline vec3f tex3d_v3(const texture3d_t& tex, float u, float v, float w)
{
#if defined(__CUDA_ARCH__)
    float4 val = tex3D<float4>(tex.tex, u, v, w);
    return vec3f(val.x, val.y, val.z);
#else
    return vec3f(0.0f, 0.0f, 0.0f);
#endif
}

CUDA_CALLABLE inline vec4f tex3d_v4(const texture3d_t& tex, float u, float v, float w)
{
#if defined(__CUDA_ARCH__)
    float4 val = tex3D<float4>(tex.tex, u, v, w);
    return vec4f(val.x, val.y, val.z, val.w);
#else
    return vec4f(0.0f, 0.0f, 0.0f, 0.0f);
#endif
}

// Adjoint stubs for texture sampling (non-differentiable for now)
CUDA_CALLABLE inline void adj_tex2d_f(
    const texture2d_t& tex, float u, float v,
    texture2d_t& adj_tex, float& adj_u, float& adj_v,
    float adj_ret)
{
    // Texture sampling is not differentiable in this implementation
}

CUDA_CALLABLE inline void adj_tex2d_v2(
    const texture2d_t& tex, float u, float v,
    texture2d_t& adj_tex, float& adj_u, float& adj_v,
    const vec2f& adj_ret)
{
}

CUDA_CALLABLE inline void adj_tex2d_v3(
    const texture2d_t& tex, float u, float v,
    texture2d_t& adj_tex, float& adj_u, float& adj_v,
    const vec3f& adj_ret)
{
}

CUDA_CALLABLE inline void adj_tex2d_v4(
    const texture2d_t& tex, float u, float v,
    texture2d_t& adj_tex, float& adj_u, float& adj_v,
    const vec4f& adj_ret)
{
}

CUDA_CALLABLE inline void adj_tex3d_f(
    const texture3d_t& tex, float u, float v, float w,
    texture3d_t& adj_tex, float& adj_u, float& adj_v, float& adj_w,
    float adj_ret)
{
}

CUDA_CALLABLE inline void adj_tex3d_v2(
    const texture3d_t& tex, float u, float v, float w,
    texture3d_t& adj_tex, float& adj_u, float& adj_v, float& adj_w,
    const vec2f& adj_ret)
{
}

CUDA_CALLABLE inline void adj_tex3d_v3(
    const texture3d_t& tex, float u, float v, float w,
    texture3d_t& adj_tex, float& adj_u, float& adj_v, float& adj_w,
    const vec3f& adj_ret)
{
}

CUDA_CALLABLE inline void adj_tex3d_v4(
    const texture3d_t& tex, float u, float v, float w,
    texture3d_t& adj_tex, float& adj_u, float& adj_v, float& adj_w,
    const vec4f& adj_ret)
{
}

}  // namespace wp

