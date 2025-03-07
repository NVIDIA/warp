/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace wp
{

struct fabricbucket_t
{
    size_t index_start;
    size_t index_end;
    void* ptr;
    size_t* lengths;
};


template <typename T>
struct fabricarray_t
{
    CUDA_CALLABLE inline fabricarray_t()
        : nbuckets(0),
          size(0)
    {}

    CUDA_CALLABLE inline bool empty() const { return !size; }

    fabricbucket_t* buckets;  // array of fabricbucket_t on the correct device

    size_t nbuckets;
    size_t size;
};


template <typename T>
struct indexedfabricarray_t
{
    CUDA_CALLABLE inline indexedfabricarray_t()
        : indices(),
          size(0)
    {}

    CUDA_CALLABLE inline bool empty() const { return !size; }

    fabricarray_t<T> fa;

    // TODO: we use 32-bit indices for consistency with other Warp indexed arrays,
    // but Fabric uses 64-bit indexing.
    int* indices;
    size_t size;
};


#ifndef FABRICARRAY_USE_BINARY_SEARCH
#define FABRICARRAY_USE_BINARY_SEARCH 1
#endif

template <typename T>
CUDA_CALLABLE inline const fabricbucket_t* fabricarray_find_bucket(const fabricarray_t<T>& fa, size_t i)
{
#if FABRICARRAY_USE_BINARY_SEARCH
    // use binary search to find the right bucket
    const fabricbucket_t* bucket = nullptr;
    size_t lo = 0;
    size_t hi = fa.nbuckets - 1;
    while (hi >= lo)
    {
        size_t mid = (lo + hi) >> 1;
        bucket = fa.buckets + mid;
        if (i >= bucket->index_end)
            lo = mid + 1;
        else if (i < bucket->index_start)
            hi = mid - 1;
        else
            return bucket;
    }
    return nullptr;
#else
    // use linear search to find the right bucket
    const fabricbucket_t* bucket = fa.buckets;
    const fabricbucket_t* bucket_end = bucket + fa.nbuckets;
    for (; bucket < bucket_end; ++bucket)
    {
        if (i < bucket->index_end)
            return bucket;
    }
    return nullptr;
#endif
}


// Compute the pointer to a fabricarray element at index i.
// This function is similar to wp::index(), but the array data type doesn't need to be known at compile time.
CUDA_CALLABLE inline void* fabricarray_element_ptr(const fabricarray_t<void>& fa, size_t i, size_t elem_size)
{
    const fabricbucket_t* bucket = fabricarray_find_bucket(fa, i);

    size_t index_in_bucket = i - bucket->index_start;

    return (char*)bucket->ptr + index_in_bucket * elem_size;
}


template <typename T>
CUDA_CALLABLE inline T& index(const fabricarray_t<T>& fa, size_t i)
{
    const fabricbucket_t* bucket = fabricarray_find_bucket(fa, i);
    assert(bucket && "Fabric array index out of range");

    size_t index_in_bucket = i - bucket->index_start;

    T& result = *((T*)bucket->ptr + index_in_bucket);

    FP_VERIFY_FWD_1(result)

    return result;
}


// indexing for fabric array of arrays
template <typename T>
CUDA_CALLABLE inline T& index(const fabricarray_t<T>& fa, size_t i, size_t j)
{
    const fabricbucket_t* bucket = fabricarray_find_bucket(fa, i);
    assert(bucket && "Fabric array index out of range");

    assert(bucket->lengths && "Missing inner array lengths");

    size_t index_in_bucket = i - bucket->index_start;

    void* ptr = *((void**)bucket->ptr + index_in_bucket);
    size_t length = *((size_t*)bucket->lengths + index_in_bucket);

    assert(j < length && "Fabric array inner index out of range");

    T& result = *((T*)ptr + j);

    FP_VERIFY_FWD_1(result)

    return result;
}


template <typename T>
CUDA_CALLABLE inline array_t<T> view(fabricarray_t<T>& fa, size_t i)
{
    const fabricbucket_t* bucket = fabricarray_find_bucket(fa, i);
    assert(bucket && "Fabric array index out of range");

    assert(bucket->lengths && "Missing inner array lengths");

    size_t index_in_bucket = i - bucket->index_start;

    void* ptr = *((void**)bucket->ptr + index_in_bucket);
    size_t length = *((size_t*)bucket->lengths + index_in_bucket);

    return array_t<T>((T*)ptr, int(length));
}


template <typename T>
CUDA_CALLABLE inline T& index(const indexedfabricarray_t<T>& ifa, size_t i)
{
    // index lookup
    assert(i < ifa.size);
    i = size_t(ifa.indices[i]);

    const fabricbucket_t* bucket = fabricarray_find_bucket(ifa.fa, i);
    assert(bucket && "Fabric array index out of range");

    size_t index_in_bucket = i - bucket->index_start;

    T& result = *((T*)bucket->ptr + index_in_bucket);

    FP_VERIFY_FWD_1(result)

    return result;
}


// indexing for fabric array of arrays
template <typename T>
CUDA_CALLABLE inline T& index(const indexedfabricarray_t<T>& ifa, size_t i, size_t j)
{
    // index lookup
    assert(i < ifa.size);
    i = size_t(ifa.indices[i]);

    const fabricbucket_t* bucket = fabricarray_find_bucket(ifa.fa, i);
    assert(bucket && "Fabric array index out of range");

    assert(bucket->lengths && "Missing inner array lengths");

    size_t index_in_bucket = i - bucket->index_start;

    void* ptr = *((void**)bucket->ptr + index_in_bucket);
    size_t length = *((size_t*)bucket->lengths + index_in_bucket);

    assert(j < length && "Fabric array inner index out of range");

    T& result = *((T*)ptr + j);

    FP_VERIFY_FWD_1(result)

    return result;
}


template <typename T>
CUDA_CALLABLE inline array_t<T> view(indexedfabricarray_t<T>& ifa, size_t i)
{
    // index lookup
    assert(i < ifa.size);
    i = size_t(ifa.indices[i]);

    const fabricbucket_t* bucket = fabricarray_find_bucket(ifa.fa, i);
    assert(bucket && "Fabric array index out of range");

    assert(bucket->lengths && "Missing inner array lengths");

    size_t index_in_bucket = i - bucket->index_start;

    void* ptr = *((void**)bucket->ptr + index_in_bucket);
    size_t length = *((size_t*)bucket->lengths + index_in_bucket);

    return array_t<T>((T*)ptr, int(length));
}

} // namespace wp
