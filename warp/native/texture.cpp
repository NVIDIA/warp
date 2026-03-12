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

#include "error.h"
#include "texture.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// CPU Host Texture Implementation
// ============================================================================

using wp::CpuTexture;
using wp::get_texture_bytes_per_channel;

uint64_t wp_texture_create_host(
    int ndim,
    int* shape,
    int num_channels,
    int dtype,
    int filter_mode,
    int* address_modes,
    bool use_normalized_coords,
    void** data_ptr_out
)
{
    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Warp error: Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture dimensions must be positive integers, got %d at dimension %d", shape[i], i
            );
            return 0;
        }
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        wp::set_error_string("Warp error: Textures support 1, 2, or 4 channels, got (%d)", num_channels);
        return 0;
    }

    CpuTexture* tex = nullptr;
    if (ndim == 1) {
        tex = new CpuTexture(shape[0], num_channels, dtype, filter_mode, address_modes[0], use_normalized_coords);
    } else if (ndim == 2) {
        tex = new CpuTexture(
            shape[0], shape[1], num_channels, dtype, filter_mode, address_modes[0], address_modes[1],
            use_normalized_coords
        );
    } else {
        tex = new CpuTexture(
            shape[0], shape[1], shape[2], num_channels, dtype, filter_mode, address_modes[0], address_modes[1],
            address_modes[2], use_normalized_coords
        );
    }

    if (data_ptr_out) {
        *data_ptr_out = tex->data;
    }

    return reinterpret_cast<uint64_t>(tex);
}

void wp_texture_destroy_host(uint64_t tex_handle) { delete (CpuTexture*)tex_handle; }

#if WP_ENABLE_CUDA

// ============================================================================
// CUDA Device Texture Implementation
// ============================================================================

#include "cuda_util.h"

// Helper function to get CUDA array format from dtype
static CUarray_format get_cuda_format(int dtype)
{
    switch (dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
        return CU_AD_FORMAT_UNSIGNED_INT8;
    case WP_TEXTURE_DTYPE_UINT16:
        return CU_AD_FORMAT_UNSIGNED_INT16;
    case WP_TEXTURE_DTYPE_UINT32:
        return CU_AD_FORMAT_UNSIGNED_INT32;
    case WP_TEXTURE_DTYPE_INT8:
        return CU_AD_FORMAT_SIGNED_INT8;
    case WP_TEXTURE_DTYPE_INT16:
        return CU_AD_FORMAT_SIGNED_INT16;
    case WP_TEXTURE_DTYPE_INT32:
        return CU_AD_FORMAT_SIGNED_INT32;
    case WP_TEXTURE_DTYPE_FLOAT16:
        return CU_AD_FORMAT_HALF;
    case WP_TEXTURE_DTYPE_FLOAT32:
    default:
        return CU_AD_FORMAT_FLOAT;
    }
}

// Helper function to convert address mode int to CUDA address mode
static CUaddress_mode get_cuda_address_mode(int address_mode)
{
    switch (address_mode) {
    case 0:
        return CU_TR_ADDRESS_MODE_WRAP;
    case 1:
        return CU_TR_ADDRESS_MODE_CLAMP;
    case 2:
        return CU_TR_ADDRESS_MODE_MIRROR;
    case 3:
        return CU_TR_ADDRESS_MODE_BORDER;
    default:
        return CU_TR_ADDRESS_MODE_CLAMP;
    }
}

uint64_t wp_texture_create_device(void* context, int ndim, int* shape, int num_channels, int dtype, bool surface_access)
{
    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Warp error: Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture dimensions must be positive integers, got %d at dimension %d", shape[i], i
            );
            return 0;
        }
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        wp::set_error_string("Warp error: Textures support 1, 2, or 4 channels, got (%d)", num_channels);
        return 0;
    }

    ContextGuard guard(context);

    size_t width = shape[0];
    size_t height = ndim > 1 ? shape[1] : 0;
    size_t depth = ndim > 2 ? shape[2] : 0;

    // determine the CUDA array format
    CUarray_format format = get_cuda_format(dtype);

    // create array descriptor
    CUDA_ARRAY3D_DESCRIPTOR arr_desc = {};
    arr_desc.Width = width;
    arr_desc.Height = height;
    arr_desc.Depth = depth;
    arr_desc.Format = format;
    arr_desc.NumChannels = num_channels;
    arr_desc.Flags = surface_access ? CUDA_ARRAY3D_SURFACE_LDST : 0;

    // create the CUDA array
    CUarray cuda_array = NULL;
    check_cu(cuArray3DCreate_f(&cuda_array, &arr_desc));

    return reinterpret_cast<uint64_t>(cuda_array);
}

void wp_texture_destroy_device(void* context, uint64_t array_handle)
{
    ContextGuard guard(context);
    if (array_handle != 0) {
        check_cu(cuArrayDestroy_f((CUarray)array_handle));
    }
}

bool wp_texture_copy_device(
    void* context,
    unsigned width_bytes,
    unsigned height,
    unsigned depth,
    int dst_memory_type,
    uint64_t dst_handle,
    unsigned dst_pitch,
    unsigned dst_height,
    int src_memory_type,
    uint64_t src_handle,
    unsigned src_pitch,
    unsigned src_height,
    void* stream
)
{
    ContextGuard guard(context);

    CUstream cuda_stream = static_cast<CUstream>(stream);

    CUDA_MEMCPY3D copy_params = {};
    copy_params.WidthInBytes = width_bytes;
    copy_params.Height = height;
    copy_params.Depth = depth;

    copy_params.dstMemoryType = static_cast<CUmemorytype>(dst_memory_type);
    if (dst_memory_type == CU_MEMORYTYPE_HOST) {
        copy_params.dstHost = reinterpret_cast<void*>(dst_handle);
    } else if (dst_memory_type == CU_MEMORYTYPE_DEVICE) {
        copy_params.dstDevice = static_cast<CUdeviceptr>(dst_handle);
    } else if (dst_memory_type == CU_MEMORYTYPE_ARRAY) {
        copy_params.dstArray = reinterpret_cast<CUarray>(dst_handle);
    } else {
        wp::set_error_string("Invalid destination memory type %d", dst_memory_type);
        return false;
    }
    copy_params.dstPitch = dst_pitch;
    copy_params.dstHeight = dst_height;

    copy_params.srcMemoryType = static_cast<CUmemorytype>(src_memory_type);
    if (src_memory_type == CU_MEMORYTYPE_HOST) {
        copy_params.srcHost = reinterpret_cast<void*>(src_handle);
    } else if (src_memory_type == CU_MEMORYTYPE_DEVICE) {
        copy_params.srcDevice = static_cast<CUdeviceptr>(src_handle);
    } else if (src_memory_type == CU_MEMORYTYPE_ARRAY) {
        copy_params.srcArray = reinterpret_cast<CUarray>(src_handle);
    } else {
        wp::set_error_string("Invalid source memory type %d", src_memory_type);
        return false;
    }
    copy_params.srcPitch = src_pitch;
    copy_params.srcHeight = src_height;

    return check_cu(cuMemcpy3DAsync_f(&copy_params, cuda_stream));
}

bool wp_texture_descriptor_from_cuda_array(void* context, uint64_t array_handle, wp::cuda_array_desc_t* desc_out)
{
    if (!array_handle || !desc_out) {
        wp::set_error_string("Warp error: NULL array handle");
        return false;
    }

    ContextGuard guard(context);

    CUarray cuda_array = reinterpret_cast<CUarray>(array_handle);

    CUDA_ARRAY3D_DESCRIPTOR desc;
    if (!check_cu(cuArray3DGetDescriptor_f(&desc, cuda_array)))
        return false;

    switch (desc.Format) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT8;
        break;
    case CU_AD_FORMAT_UNSIGNED_INT16:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT16;
        break;
    case CU_AD_FORMAT_UNSIGNED_INT32:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT32;
        break;
    case CU_AD_FORMAT_SIGNED_INT8:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT8;
        break;
    case CU_AD_FORMAT_SIGNED_INT16:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT16;
        break;
    case CU_AD_FORMAT_SIGNED_INT32:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT32;
        break;
    case CU_AD_FORMAT_HALF:
        desc_out->dtype = WP_TEXTURE_DTYPE_FLOAT16;
        break;
    case CU_AD_FORMAT_FLOAT:
        desc_out->dtype = WP_TEXTURE_DTYPE_FLOAT32;
        break;
    default:
        wp::set_error_string("Warp error: Unsupported texture format");
        return false;
    }

    desc_out->shape[0] = int32_t(desc.Width);
    desc_out->shape[1] = int32_t(desc.Height);
    desc_out->shape[2] = int32_t(desc.Depth);

    if (desc.Depth == 0) {
        if (desc.Height == 0)
            desc_out->ndim = 1;
        else
            desc_out->ndim = 2;
    } else {
        desc_out->ndim = 3;
    }

    // TODO: desc.Flags?

    desc_out->num_channels = int32_t(desc.NumChannels);

    return true;
}

uint64_t wp_texture_object_create_device(
    void* context, uint64_t array_handle, int ndim, int filter_mode, int* address_modes, bool use_normalized_coords
)
{
    if (!array_handle) {
        wp::set_error_string("Null texture array handle");
        return 0;
    }

    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    ContextGuard guard(context);

    // Create resource descriptor
    CUDA_RESOURCE_DESC res_desc = {};
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = reinterpret_cast<CUarray>(array_handle);
    res_desc.flags = 0;

    // Create texture descriptor
    CUDA_TEXTURE_DESC tex_desc = {};

    // Per-axis address modes
    for (int i = 0; i < 3; i++) {
        if (i < ndim)
            tex_desc.addressMode[i] = get_cuda_address_mode(address_modes[i]);
        else
            tex_desc.addressMode[i] = CU_TR_ADDRESS_MODE_CLAMP;
    }

    // Filter mode: 0=nearest, 1=linear
    tex_desc.filterMode = (filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;

    // Coordinate mode: normalized [0,1] or texel space [0,width/height]
    // For uint8/uint16 textures, CUDA automatically normalizes values to [0,1] when sampled
    // (since CU_TRSF_READ_AS_INTEGER is NOT set). Float32 textures return values as-is.
    tex_desc.flags = use_normalized_coords ? CU_TRSF_NORMALIZED_COORDINATES : 0;

    tex_desc.maxAnisotropy = 0;
    tex_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    tex_desc.mipmapLevelBias = 0;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.maxMipmapLevelClamp = 0;

    // Create texture object
    CUtexObject tex_object = 0;
    check_cu(cuTexObjectCreate_f(&tex_object, &res_desc, &tex_desc, nullptr));

    return tex_object;
}

void wp_texture_object_destroy_device(void* context, uint64_t tex_handle)
{
    ContextGuard guard(context);

    if (tex_handle != 0) {
        check_cu(cuTexObjectDestroy_f((CUtexObject)tex_handle));
    }
}

uint64_t wp_surface_object_create_device(void* context, uint64_t array_handle)
{
    ContextGuard guard(context);

    cudaResourceDesc desc = {};
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = reinterpret_cast<cudaArray_t>(array_handle);

    cudaSurfaceObject_t surface = 0;
    check_cuda(cudaCreateSurfaceObject(&surface, &desc));

    return static_cast<uint64_t>(surface);
}

void wp_surface_object_destroy_device(void* context, uint64_t surface_handle)
{
    if (!surface_handle)
        return;

    ContextGuard guard(context);
    check_cuda(cudaDestroySurfaceObject(static_cast<cudaSurfaceObject_t>(surface_handle)));
}

#else  // WP_ENABLE_CUDA

// Stub implementations for non-CUDA builds

uint64_t wp_texture_create_device(void* context, int ndim, int* shape, int num_channels, int dtype, bool surface_access)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

void wp_texture_destroy_device(void* context, uint64_t array_handle) { }

bool wp_texture_copy_device(
    void* context,
    unsigned width_bytes,
    unsigned height,
    unsigned depth,
    int dst_memory_type,
    uint64_t dst_handle,
    unsigned dst_pitch,
    unsigned dst_height,
    int src_memory_type,
    uint64_t src_handle,
    unsigned src_pitch,
    unsigned src_height,
    void* stream
)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return false;
}

bool wp_texture_descriptor_from_cuda_array(void* context, uint64_t array_handle, wp::cuda_array_desc_t* desc_out)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return false;
}

uint64_t wp_texture_object_create_device(
    void* context, uint64_t array_handle, int ndim, int filter_mode, int* address_modes, bool use_normalized_coords
)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

void wp_texture_object_destroy_device(void* context, uint64_t tex_handle) { }

WP_API uint64_t wp_surface_object_create_device(void* context, uint64_t array_handle)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

WP_API void wp_surface_object_destroy_device(void* context, uint64_t surface_handle) { }

#endif  // WP_ENABLE_CUDA
