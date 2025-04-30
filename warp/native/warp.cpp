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

#include "warp.h"
#include "scan.h"
#include "array.h"
#include "exports.h"
#include "error.h"

#include <stdlib.h>
#include <string.h>

uint16_t float_to_half_bits(float x)
{
    // adapted from Fabien Giesen's post: https://gist.github.com/rygorous/2156668
    union fp32
    {
        uint32_t u;
        float f;

        struct
        {
            unsigned int mantissa : 23;
            unsigned int exponent : 8;
            unsigned int sign : 1;
        };
    };

    fp32 f;
    f.f = x;

    fp32 f32infty = { 255 << 23 };
    fp32 f16infty = { 31 << 23 };
    fp32 magic = { 15 << 23 };
    uint32_t sign_mask = 0x80000000u;
    uint32_t round_mask = ~0xfffu; 
    uint16_t u;

    uint32_t sign = f.u & sign_mask;
    f.u ^= sign;

    // NOTE all the integer compares in this function can be safely
    // compiled into signed compares since all operands are below
    // 0x80000000. Important if you want fast straight SSE2 code
    // (since there's no unsigned PCMPGTD).

    if (f.u >= f32infty.u) // Inf or NaN (all exponent bits set)
        u = (f.u > f32infty.u) ? 0x7e00 : 0x7c00; // NaN->qNaN and Inf->Inf
    else // (De)normalized number or zero
    {
        f.u &= round_mask;
        f.f *= magic.f;
        f.u -= round_mask;
        if (f.u > f16infty.u) f.u = f16infty.u; // Clamp to signed infinity if overflowed

        u = f.u >> 13; // Take the bits!
    }

    u |= sign >> 16;
    return u;
}

float half_bits_to_float(uint16_t u)
{
    // adapted from Fabien Giesen's post: https://gist.github.com/rygorous/2156668
    union fp32
    {
        uint32_t u;
        float f;

        struct
        {
            unsigned int mantissa : 23;
            unsigned int exponent : 8;
            unsigned int sign : 1;
        };
    };

    static const fp32 magic = { 113 << 23 };
    static const uint32_t shifted_exp = 0x7c00 << 13; // exponent mask after shift
    fp32 o;

    o.u = (u & 0x7fff) << 13;     // exponent/mantissa bits
    uint32_t exp = shifted_exp & o.u;   // just the exponent
    o.u += (127 - 15) << 23;        // exponent adjust

    // handle exponent special cases
    if (exp == shifted_exp) // Inf/NaN?
        o.u += (128 - 16) << 23;    // extra exp adjust
    else if (exp == 0) // Zero/Denormal?
    {
        o.u += 1 << 23;             // extra exp adjust
        o.f -= magic.f;             // renormalize
    }

    o.u |= (u & 0x8000) << 16;    // sign bit
    return o.f;
}

int init()
{
#if WP_ENABLE_CUDA
    int cuda_init(void);
    // note: it's safe to proceed even if CUDA initialization failed
    cuda_init();
#endif

    return 0;
}

void shutdown()
{
}

const char* get_error_string()
{
    return wp::get_error_string();
}

void set_error_output_enabled(int enable)
{
    wp::set_error_output_enabled(bool(enable));
}

int is_error_output_enabled()
{
    return int(wp::is_error_output_enabled());
}

int is_cuda_enabled()
{
    return int(WP_ENABLE_CUDA);
}

int is_cuda_compatibility_enabled()
{
    return int(WP_ENABLE_CUDA_COMPATIBILITY);
}

int is_mathdx_enabled()
{
    return int(WP_ENABLE_MATHDX);
}

int is_debug_enabled()
{
    return int(WP_ENABLE_DEBUG);
}

void* alloc_host(size_t s)
{
    return malloc(s);
}

void free_host(void* ptr)
{
    free(ptr);
}

bool memcpy_h2h(void* dest, void* src, size_t n)
{
    memcpy(dest, src, n);
    return true;
}

void memset_host(void* dest, int value, size_t n)
{
    if ((n%4) > 0)
    {
        memset(dest, value, n);
    }
    else
    {
        const size_t num_words = n/4;
        for (size_t i=0; i < num_words; ++i)
            ((int*)dest)[i] = value;
    }
}

// fill memory buffer with a value: this is a faster memtile variant
// for types bigger than one byte, but requires proper alignment of dst
template <typename T>
void memtile_value_host(T* dst, T value, size_t n)
{
    while (n--)
        *dst++ = value;
}

void memtile_host(void* dst, const void* src, size_t srcsize, size_t n)
{
    size_t dst_addr = reinterpret_cast<size_t>(dst);
    size_t src_addr = reinterpret_cast<size_t>(src);

    // try memtile_value first because it should be faster, but we need to ensure proper alignment
    if (srcsize == 8 && (dst_addr & 7) == 0 && (src_addr & 7) == 0)
        memtile_value_host(reinterpret_cast<int64_t*>(dst), *reinterpret_cast<const int64_t*>(src), n);
    else if (srcsize == 4 && (dst_addr & 3) == 0 && (src_addr & 3) == 0)
        memtile_value_host(reinterpret_cast<int32_t*>(dst), *reinterpret_cast<const int32_t*>(src), n);
    else if (srcsize == 2 && (dst_addr & 1) == 0 && (src_addr & 1) == 0)
        memtile_value_host(reinterpret_cast<int16_t*>(dst), *reinterpret_cast<const int16_t*>(src), n);
    else if (srcsize == 1)
        memset(dst, *reinterpret_cast<const int8_t*>(src), n);
    else
    {
        // generic version
        while (n--)
        {
            memcpy(dst, src, srcsize);
            dst = (int8_t*)dst + srcsize;
        }
    }
}

void array_scan_int_host(uint64_t in, uint64_t out, int len, bool inclusive)
{
    scan_host((const int*)in, (int*)out, len, inclusive);
}

void array_scan_float_host(uint64_t in, uint64_t out, int len, bool inclusive)
{
    scan_host((const float*)in, (float*)out, len, inclusive);
}


static void array_copy_nd(void* dst, const void* src,
                      const int* dst_strides, const int* src_strides,
                      const int*const* dst_indices, const int*const* src_indices,
                      const int* shape, int ndim, int elem_size)
{
    if (ndim == 1)
    {
        for (int i = 0; i < shape[0]; i++)
        {
            int src_idx = src_indices[0] ? src_indices[0][i] : i;
            int dst_idx = dst_indices[0] ? dst_indices[0][i] : i;
            const char* p = (const char*)src + src_idx * src_strides[0];
            char* q = (char*)dst + dst_idx * dst_strides[0];
            // copy element
            memcpy(q, p, elem_size);
        }
    }
    else
    {
        for (int i = 0; i < shape[0]; i++)
        {
            int src_idx = src_indices[0] ? src_indices[0][i] : i;
            int dst_idx = dst_indices[0] ? dst_indices[0][i] : i;
            const char* p = (const char*)src + src_idx * src_strides[0];
            char* q = (char*)dst + dst_idx * dst_strides[0];
            // recurse on next inner dimension
            array_copy_nd(q, p, dst_strides + 1, src_strides + 1, dst_indices + 1, src_indices + 1, shape + 1, ndim - 1, elem_size);
        }
    }
}


static void array_copy_to_fabric(wp::fabricarray_t<void>& dst, const void* src_data,
                                 int src_stride, const int* src_indices, int elem_size)
{
    const int8_t* src_ptr = static_cast<const int8_t*>(src_data);

    if (src_indices)
    {
        // copy from indexed array
        for (size_t i = 0; i < dst.nbuckets; i++)
        {
            const wp::fabricbucket_t& bucket = dst.buckets[i];
            int8_t* dst_ptr = static_cast<int8_t*>(bucket.ptr);
            size_t bucket_size = bucket.index_end - bucket.index_start;
            for (size_t j = 0; j < bucket_size; j++)
            {
                int idx = *src_indices;
                memcpy(dst_ptr, src_ptr + idx * elem_size, elem_size);
                dst_ptr += elem_size;
                ++src_indices;
            }
        }
    }
    else
    {
        if (src_stride == elem_size)
        {
            // copy from contiguous array
            for (size_t i = 0; i < dst.nbuckets; i++)
            {
                const wp::fabricbucket_t& bucket = dst.buckets[i];
                size_t num_bytes = (bucket.index_end - bucket.index_start) * elem_size;
                memcpy(bucket.ptr, src_ptr, num_bytes);
                src_ptr += num_bytes;
            }
        }
        else
        {
            // copy from strided array
            for (size_t i = 0; i < dst.nbuckets; i++)
            {
                const wp::fabricbucket_t& bucket = dst.buckets[i];
                int8_t* dst_ptr = static_cast<int8_t*>(bucket.ptr);
                size_t bucket_size = bucket.index_end - bucket.index_start;
                for (size_t j = 0; j < bucket_size; j++)
                {
                    memcpy(dst_ptr, src_ptr, elem_size);
                    src_ptr += src_stride;
                    dst_ptr += elem_size;
                }
            }
        }
    }
}

static void array_copy_from_fabric(const wp::fabricarray_t<void>& src, void* dst_data,
                                   int dst_stride, const int* dst_indices, int elem_size)
{
    int8_t* dst_ptr = static_cast<int8_t*>(dst_data);

    if (dst_indices)
    {
        // copy to indexed array
        for (size_t i = 0; i < src.nbuckets; i++)
        {
            const wp::fabricbucket_t& bucket = src.buckets[i];
            const int8_t* src_ptr = static_cast<const int8_t*>(bucket.ptr);
            size_t bucket_size = bucket.index_end - bucket.index_start;
            for (size_t j = 0; j < bucket_size; j++)
            {
                int idx = *dst_indices;
                memcpy(dst_ptr + idx * elem_size, src_ptr, elem_size);
                src_ptr += elem_size;
                ++dst_indices;
            }
        }
    }
    else
    {
        if (dst_stride == elem_size)
        {
            // copy to contiguous array
            for (size_t i = 0; i < src.nbuckets; i++)
            {
                const wp::fabricbucket_t& bucket = src.buckets[i];
                size_t num_bytes = (bucket.index_end - bucket.index_start) * elem_size;
                memcpy(dst_ptr, bucket.ptr, num_bytes);
                dst_ptr += num_bytes;
            }
        }
        else
        {
            // copy to strided array
            for (size_t i = 0; i < src.nbuckets; i++)
            {
                const wp::fabricbucket_t& bucket = src.buckets[i];
                const int8_t* src_ptr = static_cast<const int8_t*>(bucket.ptr);
                size_t bucket_size = bucket.index_end - bucket.index_start;
                for (size_t j = 0; j < bucket_size; j++)
                {
                    memcpy(dst_ptr, src_ptr, elem_size);
                    dst_ptr += dst_stride;
                    src_ptr += elem_size;
                }
            }
        }
    }
}

static void array_copy_fabric_to_fabric(wp::fabricarray_t<void>& dst, const wp::fabricarray_t<void>& src, int elem_size)
{
    wp::fabricbucket_t* dst_bucket = dst.buckets;
    const wp::fabricbucket_t* src_bucket = src.buckets;
    int8_t* dst_ptr = static_cast<int8_t*>(dst_bucket->ptr);
    const int8_t* src_ptr = static_cast<const int8_t*>(src_bucket->ptr);
    size_t dst_remaining = dst_bucket->index_end - dst_bucket->index_start;
    size_t src_remaining = src_bucket->index_end - src_bucket->index_start;
    size_t total_copied = 0;

    while (total_copied < dst.size)
    {
        if (dst_remaining <= src_remaining)
        {
            // copy to destination bucket
            size_t num_elems = dst_remaining;
            size_t num_bytes = num_elems * elem_size;
            memcpy(dst_ptr, src_ptr, num_bytes);

            // advance to next destination bucket
            ++dst_bucket;
            dst_ptr = static_cast<int8_t*>(dst_bucket->ptr);
            dst_remaining = dst_bucket->index_end - dst_bucket->index_start;

            // advance source offset
            src_ptr += num_bytes;
            src_remaining -= num_elems;

            total_copied += num_elems;
        }
        else
        {
            // copy to destination bucket
            size_t num_elems = src_remaining;
            size_t num_bytes = num_elems * elem_size;
            memcpy(dst_ptr, src_ptr, num_bytes);

            // advance to next source bucket
            ++src_bucket;
            src_ptr = static_cast<const int8_t*>(src_bucket->ptr);
            src_remaining = src_bucket->index_end - src_bucket->index_start;

            // advance destination offset
            dst_ptr += num_bytes;
            dst_remaining -= num_elems;

            total_copied += num_elems;
        }
    }
}


static void array_copy_to_fabric_indexed(wp::indexedfabricarray_t<void>& dst, const void* src_data,
                                         int src_stride, const int* src_indices, int elem_size)
{
    const int8_t* src_ptr = static_cast<const int8_t*>(src_data);

    if (src_indices)
    {
        // copy from indexed array
        for (size_t i = 0; i < dst.size; i++)
        {
            size_t src_idx = src_indices[i];
            size_t dst_idx = dst.indices[i];
            void* dst_ptr = fabricarray_element_ptr(dst.fa, dst_idx, elem_size);
            memcpy(dst_ptr, src_ptr + dst_idx * elem_size, elem_size);
        }
    }
    else
    {
        // copy from contiguous/strided array
        for (size_t i = 0; i < dst.size; i++)
        {
            size_t dst_idx = dst.indices[i];
            void* dst_ptr = fabricarray_element_ptr(dst.fa, dst_idx, elem_size);
            if (dst_ptr)
            {
                memcpy(dst_ptr, src_ptr, elem_size);
                src_ptr += src_stride;
            }
        }
    }
}


static void array_copy_fabric_indexed_to_fabric(wp::fabricarray_t<void>& dst, const wp::indexedfabricarray_t<void>& src, int elem_size)
{
    wp::fabricbucket_t* dst_bucket = dst.buckets;
    int8_t* dst_ptr = static_cast<int8_t*>(dst_bucket->ptr);
    int8_t* dst_end = dst_ptr + elem_size * (dst_bucket->index_end - dst_bucket->index_start);

    for (size_t i = 0; i < src.size; i++)
    {
        size_t src_idx = src.indices[i];
        const void* src_ptr = fabricarray_element_ptr(src.fa, src_idx, elem_size);

        if (dst_ptr >= dst_end)
        {
            // advance to next destination bucket
            ++dst_bucket;
            dst_ptr = static_cast<int8_t*>(dst_bucket->ptr);
            dst_end = dst_ptr + elem_size * (dst_bucket->index_end - dst_bucket->index_start);
        }

        memcpy(dst_ptr, src_ptr, elem_size);

        dst_ptr += elem_size;
    }
}


static void array_copy_fabric_indexed_to_fabric_indexed(wp::indexedfabricarray_t<void>& dst, const wp::indexedfabricarray_t<void>& src, int elem_size)
{
    for (size_t i = 0; i < src.size; i++)
    {
        size_t src_idx = src.indices[i];
        size_t dst_idx = dst.indices[i];

        const void* src_ptr = fabricarray_element_ptr(src.fa, src_idx, elem_size);
        void* dst_ptr = fabricarray_element_ptr(dst.fa, dst_idx, elem_size);

        memcpy(dst_ptr, src_ptr, elem_size);
    }
}


static void array_copy_fabric_to_fabric_indexed(wp::indexedfabricarray_t<void>& dst, const wp::fabricarray_t<void>& src, int elem_size)
{
    wp::fabricbucket_t* src_bucket = src.buckets;
    const int8_t* src_ptr = static_cast<const int8_t*>(src_bucket->ptr);
    const int8_t* src_end = src_ptr + elem_size * (src_bucket->index_end - src_bucket->index_start);

    for (size_t i = 0; i < dst.size; i++)
    {
        size_t dst_idx = dst.indices[i];
        void* dst_ptr = fabricarray_element_ptr(dst.fa, dst_idx, elem_size);

        if (src_ptr >= src_end)
        {
            // advance to next source bucket
            ++src_bucket;
            src_ptr = static_cast<int8_t*>(src_bucket->ptr);
            src_end = src_ptr + elem_size * (src_bucket->index_end - src_bucket->index_start);
        }

        memcpy(dst_ptr, src_ptr, elem_size);

        src_ptr += elem_size;
    }
}


static void array_copy_from_fabric_indexed(const wp::indexedfabricarray_t<void>& src, void* dst_data,
                                           int dst_stride, const int* dst_indices, int elem_size)
{
    int8_t* dst_ptr = static_cast<int8_t*>(dst_data);

    if (dst_indices)
    {
        // copy to indexed array
        for (size_t i = 0; i < src.size; i++)
        {
            size_t idx = src.indices[i];
            if (idx < src.fa.size)
            {
                const void* src_ptr = fabricarray_element_ptr(src.fa, idx, elem_size);
                int dst_idx = dst_indices[i];
                memcpy(dst_ptr + dst_idx * elem_size, src_ptr, elem_size);
            }
            else
            {
                fprintf(stderr, "Warp copy error: Source index %llu is out of bounds for fabric array of size %llu",
                        (unsigned long long)idx, (unsigned long long)src.fa.size);
            }
        }
    }
    else
    {
        // copy to contiguous/strided array
        for (size_t i = 0; i < src.size; i++)
        {
            size_t idx = src.indices[i];
            if (idx < src.fa.size)
            {
                const void* src_ptr = fabricarray_element_ptr(src.fa, idx, elem_size);
                memcpy(dst_ptr, src_ptr, elem_size);
                dst_ptr += dst_stride;
            }
            else
            {
                fprintf(stderr, "Warp copy error: Source index %llu is out of bounds for fabric array of size %llu",
                        (unsigned long long)idx, (unsigned long long)src.fa.size);
            }
        }
    }
}


WP_API bool array_copy_host(void* dst, void* src, int dst_type, int src_type, int elem_size)
{
    if (!src || !dst)
        return false;

    const void* src_data = NULL;
    void* dst_data = NULL;
    int src_ndim = 0;
    int dst_ndim = 0;
    const int* src_shape = NULL;
    const int* dst_shape = NULL;
    const int* src_strides = NULL;
    const int* dst_strides = NULL;
    const int*const* src_indices = NULL;
    const int*const* dst_indices = NULL;

    const wp::fabricarray_t<void>* src_fabricarray = NULL;
    wp::fabricarray_t<void>* dst_fabricarray = NULL;

    const wp::indexedfabricarray_t<void>* src_indexedfabricarray = NULL;
    wp::indexedfabricarray_t<void>* dst_indexedfabricarray = NULL;

    const int* null_indices[wp::ARRAY_MAX_DIMS] = { NULL };

    if (src_type == wp::ARRAY_TYPE_REGULAR)
    {
        const wp::array_t<void>& src_arr = *static_cast<const wp::array_t<void>*>(src);
        src_data = src_arr.data;
        src_ndim = src_arr.ndim;
        src_shape = src_arr.shape.dims;
        src_strides = src_arr.strides;
        src_indices = null_indices;
    }
    else if (src_type == wp::ARRAY_TYPE_INDEXED)
    {
        const wp::indexedarray_t<void>& src_arr = *static_cast<const wp::indexedarray_t<void>*>(src);
        src_data = src_arr.arr.data;
        src_ndim = src_arr.arr.ndim;
        src_shape = src_arr.shape.dims;
        src_strides = src_arr.arr.strides;
        src_indices = src_arr.indices;
    }
    else if (src_type == wp::ARRAY_TYPE_FABRIC)
    {
        src_fabricarray = static_cast<const wp::fabricarray_t<void>*>(src);
        src_ndim = 1;
    }
    else if (src_type == wp::ARRAY_TYPE_FABRIC_INDEXED)
    {
        src_indexedfabricarray = static_cast<const wp::indexedfabricarray_t<void>*>(src);
        src_ndim = 1;
    }
    else
    {
        fprintf(stderr, "Warp copy error: Invalid source array type (%d)\n", src_type);
        return false;
    }

    if (dst_type == wp::ARRAY_TYPE_REGULAR)
    {
        const wp::array_t<void>& dst_arr = *static_cast<const wp::array_t<void>*>(dst);
        dst_data = dst_arr.data;
        dst_ndim = dst_arr.ndim;
        dst_shape = dst_arr.shape.dims;
        dst_strides = dst_arr.strides;
        dst_indices = null_indices;
    }
    else if (dst_type == wp::ARRAY_TYPE_INDEXED)
    {
        const wp::indexedarray_t<void>& dst_arr = *static_cast<const wp::indexedarray_t<void>*>(dst);
        dst_data = dst_arr.arr.data;
        dst_ndim = dst_arr.arr.ndim;
        dst_shape = dst_arr.shape.dims;
        dst_strides = dst_arr.arr.strides;
        dst_indices = dst_arr.indices;
    }
    else if (dst_type == wp::ARRAY_TYPE_FABRIC)
    {
        dst_fabricarray = static_cast<wp::fabricarray_t<void>*>(dst);
        dst_ndim = 1;
    }
    else if (dst_type == wp::ARRAY_TYPE_FABRIC_INDEXED)
    {
        dst_indexedfabricarray = static_cast<wp::indexedfabricarray_t<void>*>(dst);
        dst_ndim = 1;
    }
    else
    {
        fprintf(stderr, "Warp copy error: Invalid destination array type (%d)\n", dst_type);
        return false;
    }

    if (src_ndim != dst_ndim)
    {
        fprintf(stderr, "Warp copy error: Incompatible array dimensionalities (%d and %d)\n", src_ndim, dst_ndim);
        return false;
    }

    // handle fabric arrays
    if (dst_fabricarray)
    {
        size_t n = dst_fabricarray->size;
        if (src_fabricarray)
        {
            // copy from fabric to fabric
            if (src_fabricarray->size != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            array_copy_fabric_to_fabric(*dst_fabricarray, *src_fabricarray, elem_size);
            return true;
        }
        else if (src_indexedfabricarray)
        {
            // copy from fabric indexed to fabric
            if (src_indexedfabricarray->size != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            array_copy_fabric_indexed_to_fabric(*dst_fabricarray, *src_indexedfabricarray, elem_size);
            return true;
        }
        else
        {
            // copy to fabric
            if (size_t(src_shape[0]) != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            array_copy_to_fabric(*dst_fabricarray, src_data, src_strides[0], src_indices[0], elem_size);
            return true;
        }
    }
    else if (dst_indexedfabricarray)
    {
        size_t n = dst_indexedfabricarray->size;
        if (src_fabricarray)
        {
            // copy from fabric to fabric indexed
            if (src_fabricarray->size != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            array_copy_fabric_to_fabric_indexed(*dst_indexedfabricarray, *src_fabricarray, elem_size);
            return true;
        }
        else if (src_indexedfabricarray)
        {
            // copy from fabric indexed to fabric indexed
            if (src_indexedfabricarray->size != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            array_copy_fabric_indexed_to_fabric_indexed(*dst_indexedfabricarray, *src_indexedfabricarray, elem_size);
            return true;
        }
        else
        {
            // copy to fabric indexed
            if (size_t(src_shape[0]) != n)
            {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return false;
            }
            array_copy_to_fabric_indexed(*dst_indexedfabricarray, src_data, src_strides[0], src_indices[0], elem_size);
            return true;
        }
    }
    else if (src_fabricarray)
    {
        // copy from fabric
        size_t n = src_fabricarray->size;
        if (size_t(dst_shape[0]) != n)
        {
            fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
            return false;
        }
        array_copy_from_fabric(*src_fabricarray, dst_data, dst_strides[0], dst_indices[0], elem_size);
        return true;
    }
    else if (src_indexedfabricarray)
    {
        // copy from fabric indexed
        size_t n = src_indexedfabricarray->size;
        if (size_t(dst_shape[0]) != n)
        {
            fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
            return false;
        }
        array_copy_from_fabric_indexed(*src_indexedfabricarray, dst_data, dst_strides[0], dst_indices[0], elem_size);
        return true;
    }

    for (int i = 0; i < src_ndim; i++)
    {
        if (src_shape[i] != dst_shape[i])
        {
            fprintf(stderr, "Warp copy error: Incompatible array shapes\n");
            return 0;
        }
    }

    array_copy_nd(dst_data, src_data,
              dst_strides, src_strides,
              dst_indices, src_indices,
              src_shape, src_ndim, elem_size);

    return true;
}


static void array_fill_strided(void* data, const int* shape, const int* strides, int ndim, const void* value, int value_size)
{
    if (ndim == 1)
    {
        char* p = (char*)data;
        for (int i = 0; i < shape[0]; i++)
        {
            memcpy(p, value, value_size);
            p += strides[0];
        }
    }
    else
    {
        for (int i = 0; i < shape[0]; i++)
        {
            char* p = (char*)data + i * strides[0];
            // recurse on next inner dimension
            array_fill_strided(p, shape + 1, strides + 1, ndim - 1, value, value_size);
        }
    }
}


static void array_fill_indexed(void* data, const int* shape, const int* strides, const int*const* indices, int ndim, const void* value, int value_size)
{
    if (ndim == 1)
    {
        for (int i = 0; i < shape[0]; i++)
        {
            int idx = indices[0] ? indices[0][i] : i;
            char* p = (char*)data + idx * strides[0];
            memcpy(p, value, value_size);
        }
    }
    else
    {
        for (int i = 0; i < shape[0]; i++)
        {
            int idx = indices[0] ? indices[0][i] : i;
            char* p = (char*)data + idx * strides[0];
            // recurse on next inner dimension
            array_fill_indexed(p, shape + 1, strides + 1, indices + 1, ndim - 1, value, value_size);
        }
    }
}


static void array_fill_fabric(wp::fabricarray_t<void>& fa, const void* value_ptr, int value_size)
{
    for (size_t i = 0; i < fa.nbuckets; i++)
    {
        const wp::fabricbucket_t& bucket = fa.buckets[i];
        size_t bucket_size = bucket.index_end - bucket.index_start;
        memtile_host(bucket.ptr, value_ptr, value_size, bucket_size);
    }
}


static void array_fill_fabric_indexed(wp::indexedfabricarray_t<void>& ifa, const void* value_ptr, int value_size)
{
    for (size_t i = 0; i < ifa.size; i++)
    {
        size_t idx = size_t(ifa.indices[i]);
        if (idx < ifa.fa.size)
        {
            void* p = fabricarray_element_ptr(ifa.fa, idx, value_size);
            memcpy(p, value_ptr, value_size);
        }
    }
}


WP_API void array_fill_host(void* arr_ptr, int arr_type, const void* value_ptr, int value_size)
{
    if (!arr_ptr || !value_ptr)
        return;

    if (arr_type == wp::ARRAY_TYPE_REGULAR)
    {
        wp::array_t<void>& arr = *static_cast<wp::array_t<void>*>(arr_ptr);
        array_fill_strided(arr.data, arr.shape.dims, arr.strides, arr.ndim, value_ptr, value_size);
    }
    else if (arr_type == wp::ARRAY_TYPE_INDEXED)
    {
        wp::indexedarray_t<void>& ia = *static_cast<wp::indexedarray_t<void>*>(arr_ptr);
        array_fill_indexed(ia.arr.data, ia.shape.dims, ia.arr.strides, ia.indices, ia.arr.ndim, value_ptr, value_size);
    }
    else if (arr_type == wp::ARRAY_TYPE_FABRIC)
    {
        wp::fabricarray_t<void>& fa = *static_cast<wp::fabricarray_t<void>*>(arr_ptr);
        array_fill_fabric(fa, value_ptr, value_size);
    }
    else if (arr_type == wp::ARRAY_TYPE_FABRIC_INDEXED)
    {
        wp::indexedfabricarray_t<void>& ifa = *static_cast<wp::indexedfabricarray_t<void>*>(arr_ptr);
        array_fill_fabric_indexed(ifa, value_ptr, value_size);
    }
    else
    {
        fprintf(stderr, "Warp fill error: Invalid array type id %d\n", arr_type);
    }
}


// impl. files
// TODO: compile as separate translation units
#include "bvh.cpp"
#include "scan.cpp"


// stubs for platforms where there is no CUDA
#if !WP_ENABLE_CUDA

void* alloc_pinned(size_t s)
{
    // CUDA is not available, fall back on system allocator
    return alloc_host(s);
}

void free_pinned(void* ptr)
{
    // CUDA is not available, fall back on system allocator
    free_host(ptr);
}

void* alloc_device(void* context, size_t s)
{
    return NULL;
}

void* alloc_device_default(void* context, size_t s)
{
    return NULL;
}

void* alloc_device_async(void* context, size_t s)
{
    return NULL;
}

void free_device(void* context, void* ptr)
{
}

void free_device_default(void* context, void* ptr)
{
}

void free_device_async(void* context, void* ptr)
{
}

bool memcpy_h2d(void* context, void* dest, void* src, size_t n, void* stream)
{
    return false;
}

bool memcpy_d2h(void* context, void* dest, void* src, size_t n, void* stream)
{
    return false;
}

bool memcpy_d2d(void* context, void* dest, void* src, size_t n, void* stream)
{
    return false;
}

bool memcpy_p2p(void* dst_context, void* dst, void* src_context, void* src, size_t n, void* stream)
{    
    return false;
}

void memset_device(void* context, void* dest, int value, size_t n)
{
}

void memtile_device(void* context, void* dest, const void* src, size_t srcsize, size_t n)
{
}

bool array_copy_device(void* context, void* dst, void* src, int dst_type, int src_type, int elem_size)
{
    return false;
}

void array_fill_device(void* context, void* arr, int arr_type, const void* value, int value_size)
{
}

WP_API int cuda_driver_version() { return 0; }
WP_API int cuda_toolkit_version() { return 0; }
WP_API bool cuda_driver_is_initialized() { return false; }

WP_API int nvrtc_supported_arch_count() { return 0; }
WP_API void nvrtc_supported_archs(int* archs) {}

WP_API int cuda_device_get_count() { return 0; }
WP_API void* cuda_device_get_primary_context(int ordinal) { return NULL; }
WP_API const char* cuda_device_get_name(int ordinal) { return NULL; }
WP_API int cuda_device_get_arch(int ordinal) { return 0; }
WP_API int cuda_device_get_sm_count(int ordinal) { return 0; }
WP_API void cuda_device_get_uuid(int ordinal, char uuid[16]) {}
WP_API int cuda_device_get_pci_domain_id(int ordinal) { return -1; }
WP_API int cuda_device_get_pci_bus_id(int ordinal) { return -1; }
WP_API int cuda_device_get_pci_device_id(int ordinal) { return -1; }
WP_API int cuda_device_is_uva(int ordinal) { return 0; }
WP_API int cuda_device_is_mempool_supported(int ordinal) { return 0; }
WP_API int cuda_device_is_ipc_supported(int ordinal) { return 0; }
WP_API int cuda_device_set_mempool_release_threshold(int ordinal, uint64_t threshold) { return 0; }
WP_API uint64_t cuda_device_get_mempool_release_threshold(int ordinal) { return 0; }
WP_API uint64_t cuda_device_get_mempool_used_mem_current(int ordinal) { return 0; }
WP_API uint64_t cuda_device_get_mempool_used_mem_high(int ordinal) { return 0; }
WP_API void cuda_device_get_memory_info(int ordinal, size_t* free_mem, size_t* total_mem) {}

WP_API void* cuda_context_get_current() { return NULL; }
WP_API void cuda_context_set_current(void* ctx) {}
WP_API void cuda_context_push_current(void* context) {}
WP_API void cuda_context_pop_current() {}
WP_API void* cuda_context_create(int device_ordinal) { return NULL; }
WP_API void cuda_context_destroy(void* context) {}
WP_API void cuda_context_synchronize(void* context) {}
WP_API uint64_t cuda_context_check(void* context) { return 0; }
WP_API int cuda_context_get_device_ordinal(void* context) { return -1; }
WP_API int cuda_context_is_primary(void* context) { return 0; }
WP_API void* cuda_context_get_stream(void* context) { return NULL; }
WP_API void cuda_context_set_stream(void* context, void* stream, int sync) {}

WP_API int cuda_is_peer_access_supported(int target_ordinal, int peer_ordinal) { return 0; }
WP_API int cuda_is_peer_access_enabled(void* target_context, void* peer_context) { return 0; }
WP_API int cuda_set_peer_access_enabled(void* target_context, void* peer_context, int enable) { return 0; }
WP_API int cuda_is_mempool_access_enabled(int target_ordinal, int peer_ordinal) { return 0; }
WP_API int cuda_set_mempool_access_enabled(int target_ordinal, int peer_ordinal, int enable) { return 0; }

WP_API void cuda_ipc_get_mem_handle(void* ptr, char* out_buffer) {}
WP_API void* cuda_ipc_open_mem_handle(void* context, char* handle) { return NULL; }
WP_API void cuda_ipc_close_mem_handle(void* ptr) {}
WP_API void cuda_ipc_get_event_handle(void* context, void* event, char* out_buffer) {}
WP_API void* cuda_ipc_open_event_handle(void* context, char* handle) { return NULL; }

WP_API void* cuda_stream_create(void* context, int priority) { return NULL; }
WP_API void cuda_stream_destroy(void* context, void* stream) {}
WP_API int cuda_stream_query(void* stream) { return 0; }
WP_API void cuda_stream_register(void* context, void* stream) {}
WP_API void cuda_stream_unregister(void* context, void* stream) {}
WP_API void* cuda_stream_get_current() { return NULL; }
WP_API void cuda_stream_synchronize(void* stream) {}
WP_API void cuda_stream_wait_event(void* stream, void* event) {}
WP_API void cuda_stream_wait_stream(void* stream, void* other_stream, void* event) {}
WP_API int cuda_stream_is_capturing(void* stream) { return 0; }
WP_API uint64_t cuda_stream_get_capture_id(void* stream) { return 0; }
WP_API int cuda_stream_get_priority(void* stream) { return 0; }

WP_API void* cuda_event_create(void* context, unsigned flags) { return NULL; }
WP_API void cuda_event_destroy(void* event) {}
WP_API int cuda_event_query(void* event) { return 0; }
WP_API void cuda_event_record(void* event, void* stream, bool timing) {}
WP_API void cuda_event_synchronize(void* event) {}
WP_API float cuda_event_elapsed_time(void* start_event, void* end_event) { return 0.0f; }

WP_API bool cuda_graph_begin_capture(void* context, void* stream, int external) { return false; }
WP_API bool cuda_graph_end_capture(void* context, void* stream, void** graph_ret) { return false; }
WP_API bool cuda_graph_create_exec(void* context, void* graph, void** graph_exec_ret) { return false; }
WP_API bool cuda_graph_launch(void* graph, void* stream) { return false; }
WP_API bool cuda_graph_destroy(void* context, void* graph) { return false; }
WP_API bool cuda_graph_exec_destroy(void* context, void* graph_exec) { return false; }

WP_API bool cuda_graph_insert_if_else(void* context, void* stream, int* condition, void** if_graph_ret, void** else_graph_ret) { return false; }
WP_API bool cuda_graph_insert_while(void* context, void* stream, int* condition, void** body_graph_ret, uint64_t* handle_ret) { return false; }
WP_API bool cuda_graph_set_condition(void* context, void* stream, int* condition, uint64_t handle) { return false; }
WP_API bool cuda_graph_pause_capture(void* context, void* stream, void** graph_ret) { return false; }
WP_API bool cuda_graph_resume_capture(void* context, void* stream, void* graph) { return false; }
WP_API bool cuda_graph_insert_child_graph(void* context, void* stream, void* child_graph) { return false; }

WP_API size_t cuda_compile_program(const char* cuda_src, const char* program_name, int arch, const char* include_dir, int num_cuda_include_dirs, const char** cuda_include_dirs, bool debug, bool verbose, bool verify_fp, bool fast_math, bool fuse_fp, bool lineinfo, bool compile_time_trace, const char* output_path, size_t num_ltoirs, char** ltoirs, size_t* ltoir_sizes, int* ltoir_input_types) { return 0; }

WP_API void* cuda_load_module(void* context, const char* ptx) { return NULL; }
WP_API void cuda_unload_module(void* context, void* module) {}
WP_API void* cuda_get_kernel(void* context, void* module, const char* name) { return NULL; }
WP_API size_t cuda_launch_kernel(void* context, void* kernel, size_t dim, int max_blocks, int block_dim, int shared_memory_bytes, void** args, void* stream) { return 0; }

WP_API int cuda_get_max_shared_memory(void* context) { return 0; }
WP_API bool cuda_configure_kernel_shared_memory(void* kernel, int size) { return false; }

WP_API void cuda_set_context_restore_policy(bool always_restore) {}
WP_API int cuda_get_context_restore_policy() { return false; }

WP_API void array_scan_int_device(uint64_t in, uint64_t out, int len, bool inclusive) {}
WP_API void array_scan_float_device(uint64_t in, uint64_t out, int len, bool inclusive) {}

WP_API void cuda_graphics_map(void* context, void* resource) {}
WP_API void cuda_graphics_unmap(void* context, void* resource) {}
WP_API void cuda_graphics_device_ptr_and_size(void* context, void* resource, uint64_t* ptr, size_t* size) {}
WP_API void* cuda_graphics_register_gl_buffer(void* context, uint32_t gl_buffer, unsigned int flags) { return NULL; }
WP_API void cuda_graphics_unregister_resource(void* context, void* resource) {}

WP_API void cuda_timing_begin(int flags) {}
WP_API int cuda_timing_get_result_count() { return 0; }
WP_API void cuda_timing_end(timing_result_t* results, int size) {}

#endif // !WP_ENABLE_CUDA
