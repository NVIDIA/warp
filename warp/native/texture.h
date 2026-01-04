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

// Do not include CUDA headers here; this header is included in host-only
// translation units via builtin.h. Use uint64 handles and cast in device code.

namespace wp {
enum TextureFilterMode {
    TEX_FILTER_POINT = 0,
    TEX_FILTER_LINEAR = 1,
};
enum TextureAddressMode {
    TEX_ADDRESS_CLAMP = 0,
    TEX_ADDRESS_WRAP = 1,
    TEX_ADDRESS_BORDER = 2,
};
struct Texture2D {
    int width;
    int height;
    int num_channels;
    void* context;
    unsigned long long handle;  // cudaTextureObject_t stored as uint64
    uint64 array_handle;  // cudaArray_t stored as uint64
};

CUDA_CALLABLE inline Texture2D texture_get(uint64 id) { return *(Texture2D*)(id); }

// Device-side sampler using normalized UVs, bilinear filtering
CUDA_CALLABLE inline vec4 texture_sample_v4(uint64 id, vec2 uv)
{
#if defined(__CUDA_ARCH__)

    // Avoid including CUDA headers in kernels: rely on intrinsic availability
    typedef unsigned long long cudaTextureObject_t;
    const Texture2D* t = (const Texture2D*)(id);
    if (!t)
        return { 0.0f, 0.0f, 0.0f, 0.0f };
    cudaTextureObject_t tex = (cudaTextureObject_t)(t->handle);

    // sample according to channel count
    if (t->num_channels <= 1) {
        float c = tex2D<float>(tex, uv[0], uv[1]);
        return { c, c, c, 1.0f };
    } else if (t->num_channels == 2) {
        float2 c = tex2D<float2>(tex, uv[0], uv[1]);
        return { c.x, c.y, 0.0f, 1.0f };
    } else {
        float4 c = tex2D<float4>(tex, uv[0], uv[1]);
        return { c.x, c.y, c.z, 1.0f };
    }
#else
    (void)id;
    (void)uv;
    return { 0.0f, 0.0f, 0.0f, 0.0f };
#endif
}

CUDA_CALLABLE inline vec3 texture_sample_v3(uint64 id, vec2 uv)
{
    const vec4 c = texture_sample_v4(id, uv);
    return { c[0], c[1], c[2] };
}

CUDA_CALLABLE inline float texture_sample_f(uint64 id, vec2 uv)
{
    const vec4 c = texture_sample_v4(id, uv);
    return c[0];
}

CUDA_CALLABLE inline void adj_texture_sample_v4(uint64 id, vec2 uv, uint64& adj_id, vec2& adj_uv, vec4& adj_ret) { }

CUDA_CALLABLE inline void adj_texture_sample_v3(uint64 id, vec2 uv, uint64& adj_id, vec2& adj_uv, vec3& adj_ret) { }

CUDA_CALLABLE inline void adj_texture_sample_f(uint64 id, vec2 uv, uint64& adj_id, vec2& adj_uv, float& adj_ret) { }

CUDA_CALLABLE bool texture_get_descriptor(uint64 id, Texture2D& desc);
CUDA_CALLABLE bool texture_set_descriptor(uint64 id, const Texture2D& desc);
CUDA_CALLABLE void texture_add_descriptor(uint64 id, const Texture2D& desc);
CUDA_CALLABLE void texture_rem_descriptor(uint64 id);

}  // namespace wp