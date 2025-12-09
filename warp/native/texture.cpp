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

#include "warp.h"

#include "cuda_util.h"

#include <cstring>

// CUDA texture API implementation for Warp
// This file implements texture creation and destruction for 2D and 3D CUDA textures

// Data type constants (must match Python side)
#define WP_TEXTURE_DTYPE_UINT8  0
#define WP_TEXTURE_DTYPE_UINT16 1
#define WP_TEXTURE_DTYPE_FLOAT32 2

#if WP_ENABLE_CUDA

// Helper function to get CUDA array format from dtype
static CUarray_format get_cuda_format(int dtype)
{
    switch (dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
        return CU_AD_FORMAT_UNSIGNED_INT8;
    case WP_TEXTURE_DTYPE_UINT16:
        return CU_AD_FORMAT_UNSIGNED_INT16;
    case WP_TEXTURE_DTYPE_FLOAT32:
    default:
        return CU_AD_FORMAT_FLOAT;
    }
}

// Helper function to get bytes per channel from dtype
static int get_bytes_per_channel(int dtype)
{
    switch (dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
        return 1;
    case WP_TEXTURE_DTYPE_UINT16:
        return 2;
    case WP_TEXTURE_DTYPE_FLOAT32:
    default:
        return 4;
    }
}

bool wp_texture2d_create(
    void* context,
    int width,
    int height,
    int num_channels,
    int dtype,
    int filter_mode,
    int address_mode,
    const void* data,
    uint64_t* tex_handle_out,
    uint64_t* array_handle_out
)
{
    if (width <= 0 || height <= 0 || data == nullptr) {
        return false;
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        return false;
    }

    ContextGuard guard(context);

    // Determine the CUDA array format
    CUarray_format format = get_cuda_format(dtype);
    int bytes_per_channel = get_bytes_per_channel(dtype);

    // Create CUDA array descriptor
    CUDA_ARRAY_DESCRIPTOR arr_desc = {};
    arr_desc.Width = width;
    arr_desc.Height = height;
    arr_desc.Format = format;
    arr_desc.NumChannels = num_channels;

    // Create the array
    CUarray cuda_array;
    CUresult result = cuArrayCreate_f(&cuda_array, &arr_desc);
    if (result != CUDA_SUCCESS) {
        return false;
    }

    // Copy data to the array
    int bytes_per_texel = num_channels * bytes_per_channel;
    CUDA_MEMCPY2D copy_params = {};
    copy_params.srcXInBytes = 0;
    copy_params.srcY = 0;
    copy_params.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy_params.srcHost = data;
    copy_params.srcPitch = width * bytes_per_texel;

    copy_params.dstXInBytes = 0;
    copy_params.dstY = 0;
    copy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy_params.dstArray = cuda_array;

    copy_params.WidthInBytes = width * bytes_per_texel;
    copy_params.Height = height;

    result = cuMemcpy2D_f(&copy_params);
    if (result != CUDA_SUCCESS) {
        cuArrayDestroy_f(cuda_array);
        return false;
    }

    // Create resource descriptor
    CUDA_RESOURCE_DESC res_desc = {};
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = cuda_array;
    res_desc.flags = 0;

    // Create texture descriptor
    CUDA_TEXTURE_DESC tex_desc = {};

    // Address mode: 0=wrap, 1=clamp, 2=mirror, 3=border
    CUaddress_mode cuda_address_mode;
    switch (address_mode) {
    case 0:
        cuda_address_mode = CU_TR_ADDRESS_MODE_WRAP;
        break;
    case 1:
        cuda_address_mode = CU_TR_ADDRESS_MODE_CLAMP;
        break;
    case 2:
        cuda_address_mode = CU_TR_ADDRESS_MODE_MIRROR;
        break;
    case 3:
        cuda_address_mode = CU_TR_ADDRESS_MODE_BORDER;
        break;
    default:
        cuda_address_mode = CU_TR_ADDRESS_MODE_CLAMP;
        break;
    }

    tex_desc.addressMode[0] = cuda_address_mode;
    tex_desc.addressMode[1] = cuda_address_mode;
    tex_desc.addressMode[2] = cuda_address_mode;

    // Filter mode: 0=nearest, 1=linear
    tex_desc.filterMode = (filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;

    // Use normalized coordinates [0,1]
    // For integer textures (uint8/uint16), CUDA returns normalized floats [0,1] by default
    // (CU_TRSF_READ_AS_INTEGER is NOT set, so reads are normalized automatically)
    tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;

    tex_desc.maxAnisotropy = 0;
    tex_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    tex_desc.mipmapLevelBias = 0;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.maxMipmapLevelClamp = 0;

    // Create texture object
    CUtexObject tex_object;
    result = cuTexObjectCreate_f(&tex_object, &res_desc, &tex_desc, nullptr);
    if (result != CUDA_SUCCESS) {
        cuArrayDestroy_f(cuda_array);
        return false;
    }

    *tex_handle_out = (uint64_t)tex_object;
    *array_handle_out = (uint64_t)cuda_array;

    return true;
}

void wp_texture2d_destroy(void* context, uint64_t tex_handle, uint64_t array_handle)
{
    if (tex_handle == 0 && array_handle == 0)
        return;

    ContextGuard guard(context);

    if (tex_handle != 0) {
        cuTexObjectDestroy_f((CUtexObject)tex_handle);
    }

    if (array_handle != 0) {
        cuArrayDestroy_f((CUarray)array_handle);
    }
}

bool wp_texture3d_create(
    void* context,
    int width,
    int height,
    int depth,
    int num_channels,
    int dtype,
    int filter_mode,
    int address_mode,
    const void* data,
    uint64_t* tex_handle_out,
    uint64_t* array_handle_out
)
{
    if (width <= 0 || height <= 0 || depth <= 0 || data == nullptr) {
        return false;
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        return false;
    }

    ContextGuard guard(context);

    // Determine the CUDA array format
    CUarray_format format = get_cuda_format(dtype);
    int bytes_per_channel = get_bytes_per_channel(dtype);

    // Create CUDA 3D array descriptor
    CUDA_ARRAY3D_DESCRIPTOR arr_desc = {};
    arr_desc.Width = width;
    arr_desc.Height = height;
    arr_desc.Depth = depth;
    arr_desc.Format = format;
    arr_desc.NumChannels = num_channels;
    arr_desc.Flags = 0;

    // Create the 3D array
    CUarray cuda_array;
    CUresult result = cuArray3DCreate_f(&cuda_array, &arr_desc);
    if (result != CUDA_SUCCESS) {
        return false;
    }

    // Copy data to the 3D array
    int bytes_per_texel = num_channels * bytes_per_channel;
    CUDA_MEMCPY3D copy_params = {};
    copy_params.srcXInBytes = 0;
    copy_params.srcY = 0;
    copy_params.srcZ = 0;
    copy_params.srcLOAD = 0;
    copy_params.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy_params.srcHost = data;
    copy_params.srcPitch = width * bytes_per_texel;
    copy_params.srcHeight = height;

    copy_params.dstXInBytes = 0;
    copy_params.dstY = 0;
    copy_params.dstZ = 0;
    copy_params.dstLOAD = 0;
    copy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy_params.dstArray = cuda_array;

    copy_params.WidthInBytes = width * bytes_per_texel;
    copy_params.Height = height;
    copy_params.Depth = depth;

    result = cuMemcpy3D_f(&copy_params);
    if (result != CUDA_SUCCESS) {
        cuArrayDestroy_f(cuda_array);
        return false;
    }

    // Create resource descriptor
    CUDA_RESOURCE_DESC res_desc = {};
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = cuda_array;
    res_desc.flags = 0;

    // Create texture descriptor
    CUDA_TEXTURE_DESC tex_desc = {};

    // Address mode: 0=wrap, 1=clamp, 2=mirror, 3=border
    CUaddress_mode cuda_address_mode;
    switch (address_mode) {
    case 0:
        cuda_address_mode = CU_TR_ADDRESS_MODE_WRAP;
        break;
    case 1:
        cuda_address_mode = CU_TR_ADDRESS_MODE_CLAMP;
        break;
    case 2:
        cuda_address_mode = CU_TR_ADDRESS_MODE_MIRROR;
        break;
    case 3:
        cuda_address_mode = CU_TR_ADDRESS_MODE_BORDER;
        break;
    default:
        cuda_address_mode = CU_TR_ADDRESS_MODE_CLAMP;
        break;
    }

    tex_desc.addressMode[0] = cuda_address_mode;
    tex_desc.addressMode[1] = cuda_address_mode;
    tex_desc.addressMode[2] = cuda_address_mode;

    // Filter mode: 0=nearest, 1=linear
    tex_desc.filterMode = (filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;

    // Use normalized coordinates [0,1]
    tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;

    tex_desc.maxAnisotropy = 0;
    tex_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    tex_desc.mipmapLevelBias = 0;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.maxMipmapLevelClamp = 0;

    // Create texture object
    CUtexObject tex_object;
    result = cuTexObjectCreate_f(&tex_object, &res_desc, &tex_desc, nullptr);
    if (result != CUDA_SUCCESS) {
        cuArrayDestroy_f(cuda_array);
        return false;
    }

    *tex_handle_out = (uint64_t)tex_object;
    *array_handle_out = (uint64_t)cuda_array;

    return true;
}

void wp_texture3d_destroy(void* context, uint64_t tex_handle, uint64_t array_handle)
{
    if (tex_handle == 0 && array_handle == 0)
        return;

    ContextGuard guard(context);

    if (tex_handle != 0) {
        cuTexObjectDestroy_f((CUtexObject)tex_handle);
    }

    if (array_handle != 0) {
        cuArrayDestroy_f((CUarray)array_handle);
    }
}

#else  // WP_ENABLE_CUDA

// Stub implementations for non-CUDA builds
bool wp_texture2d_create(
    void* context,
    int width,
    int height,
    int num_channels,
    int dtype,
    int filter_mode,
    int address_mode,
    const void* data,
    uint64_t* tex_handle_out,
    uint64_t* array_handle_out
)
{
    return false;
}

void wp_texture2d_destroy(void* context, uint64_t tex_handle, uint64_t array_handle) { }

bool wp_texture3d_create(
    void* context,
    int width,
    int height,
    int depth,
    int num_channels,
    int dtype,
    int filter_mode,
    int address_mode,
    const void* data,
    uint64_t* tex_handle_out,
    uint64_t* array_handle_out
)
{
    return false;
}

void wp_texture3d_destroy(void* context, uint64_t tex_handle, uint64_t array_handle) { }

#endif  // WP_ENABLE_CUDA
