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

namespace wp
{

#if FP_CHECK

#define FP_ASSERT_FWD(value) \
    print(value); \
    printf(")\n"); \
    assert(0); \

#define FP_ASSERT_ADJ(value, adj_value) \
    print(value); \
    printf(", "); \
    print(adj_value); \
    printf(")\n"); \
    assert(0); \

#define FP_VERIFY_FWD(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(addr", __FILE__, __LINE__, __FUNCTION__); \
        FP_ASSERT_FWD(value) \
    } \

#define FP_VERIFY_FWD_1(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(arr, %d) ", __FILE__, __LINE__, __FUNCTION__, i); \
        FP_ASSERT_FWD(value) \
    } \

#define FP_VERIFY_FWD_2(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(arr, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j); \
        FP_ASSERT_FWD(value) \
    } \

#define FP_VERIFY_FWD_3(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(arr, %d, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j, k); \
        FP_ASSERT_FWD(value) \
    } \

#define FP_VERIFY_FWD_4(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(arr, %d, %d, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j, k, l); \
        FP_ASSERT_FWD(value) \
    } \

#define FP_VERIFY_ADJ(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(addr",  __FILE__, __LINE__, __FUNCTION__); \
        FP_ASSERT_ADJ(value, adj_value); \
    } \

#define FP_VERIFY_ADJ_1(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(arr, %d) ",  __FILE__, __LINE__, __FUNCTION__, i); \
        FP_ASSERT_ADJ(value, adj_value); \
    } \

#define FP_VERIFY_ADJ_2(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(arr, %d, %d) ",  __FILE__, __LINE__, __FUNCTION__, i, j); \
        FP_ASSERT_ADJ(value, adj_value); \
    } \

#define FP_VERIFY_ADJ_3(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(arr, %d, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j, k); \
        FP_ASSERT_ADJ(value, adj_value); \
    } \

#define FP_VERIFY_ADJ_4(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(arr, %d, %d, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j, k, l); \
        FP_ASSERT_ADJ(value, adj_value); \
    } \


#else

#define FP_VERIFY_FWD(value) {}
#define FP_VERIFY_FWD_1(value) {}
#define FP_VERIFY_FWD_2(value) {}
#define FP_VERIFY_FWD_3(value) {}
#define FP_VERIFY_FWD_4(value) {}

#define FP_VERIFY_ADJ(value, adj_value) {}
#define FP_VERIFY_ADJ_1(value, adj_value) {}
#define FP_VERIFY_ADJ_2(value, adj_value) {}
#define FP_VERIFY_ADJ_3(value, adj_value) {}
#define FP_VERIFY_ADJ_4(value, adj_value) {}

#endif  // WP_FP_CHECK

const int ARRAY_MAX_DIMS = 4;       // must match constant in types.py

// must match constants in types.py
const int ARRAY_TYPE_REGULAR = 0;
const int ARRAY_TYPE_INDEXED = 1;
const int ARRAY_TYPE_FABRIC = 2;
const int ARRAY_TYPE_FABRIC_INDEXED = 3;

struct shape_t
{
    int dims[ARRAY_MAX_DIMS];

    CUDA_CALLABLE inline shape_t()
        : dims()
    {}

    CUDA_CALLABLE inline int operator[](int i) const
    {
        assert(i < ARRAY_MAX_DIMS);
        return dims[i];
    }

    CUDA_CALLABLE inline int& operator[](int i)
    {
        assert(i < ARRAY_MAX_DIMS);
        return dims[i];
    }    
};

CUDA_CALLABLE inline int extract(const shape_t& s, int i)
{
    return s.dims[i];
}

CUDA_CALLABLE inline void adj_extract(const shape_t& s, int i, const shape_t& adj_s, int adj_i, int adj_ret) {}

inline CUDA_CALLABLE void print(shape_t s)
{
    // todo: only print valid dims, currently shape has a fixed size
    // but we don't know how many dims are valid (e.g.: 1d, 2d, etc)
    // should probably store ndim with shape
    printf("(%d, %d, %d, %d)\n", s.dims[0], s.dims[1], s.dims[2], s.dims[3]);
}
inline CUDA_CALLABLE void adj_print(shape_t s, shape_t& shape_t) {}


template <typename T>
struct array_t
{
    CUDA_CALLABLE inline array_t()
        : data(nullptr),
          grad(nullptr),
          shape(),
          strides(),
          ndim(0)
    {}

    CUDA_CALLABLE array_t(T* data, int size, T* grad=nullptr) : data(data), grad(grad) {
        // constructor for 1d array
        shape.dims[0] = size;
        shape.dims[1] = 0;
        shape.dims[2] = 0;
        shape.dims[3] = 0;
        ndim = 1;
        strides[0] = sizeof(T);
        strides[1] = 0;
        strides[2] = 0;
        strides[3] = 0;
    }
    CUDA_CALLABLE array_t(T* data, int dim0, int dim1, T* grad=nullptr) : data(data), grad(grad) {
        // constructor for 2d array
        shape.dims[0] = dim0;
        shape.dims[1] = dim1;
        shape.dims[2] = 0;
        shape.dims[3] = 0;
        ndim = 2;
        strides[0] = dim1 * sizeof(T);
        strides[1] = sizeof(T);
        strides[2] = 0;
        strides[3] = 0;
    }
    CUDA_CALLABLE array_t(T* data, int dim0, int dim1, int dim2, T* grad=nullptr) : data(data), grad(grad) {
        // constructor for 3d array
        shape.dims[0] = dim0;
        shape.dims[1] = dim1;
        shape.dims[2] = dim2;
        shape.dims[3] = 0;
        ndim = 3;
        strides[0] = dim1 * dim2 * sizeof(T);
        strides[1] = dim2 * sizeof(T);
        strides[2] = sizeof(T);
        strides[3] = 0;
    }
    CUDA_CALLABLE array_t(T* data, int dim0, int dim1, int dim2, int dim3, T* grad=nullptr) : data(data), grad(grad) {
        // constructor for 4d array
        shape.dims[0] = dim0;
        shape.dims[1] = dim1;
        shape.dims[2] = dim2;
        shape.dims[3] = dim3;
        ndim = 4;
        strides[0] = dim1 * dim2 * dim3 * sizeof(T);
        strides[1] = dim2 * dim3 * sizeof(T);
        strides[2] = dim3 * sizeof(T);
        strides[3] = sizeof(T);
    }

    CUDA_CALLABLE array_t(uint64 data, int size, uint64 grad=0)
        : array_t((T*)(data), size, (T*)(grad))
    {}

    CUDA_CALLABLE array_t(uint64 data, int dim0, int dim1, uint64 grad=0)
        : array_t((T*)(data), dim0, dim1, (T*)(grad))
    {}

    CUDA_CALLABLE array_t(uint64 data, int dim0, int dim1, int dim2, uint64 grad=0)
        : array_t((T*)(data), dim0, dim1, dim2, (T*)(grad))
    {}

    CUDA_CALLABLE array_t(uint64 data, int dim0, int dim1, int dim2, int dim3, uint64 grad=0)
        : array_t((T*)(data), dim0, dim1, dim2, dim3, (T*)(grad))
    {}

    CUDA_CALLABLE inline bool empty() const { return !data; }

    T* data;
    T* grad;
    shape_t shape;
    int strides[ARRAY_MAX_DIMS];
    int ndim;

    CUDA_CALLABLE inline operator T*() const { return data; }
};


// TODO:
// - templated index type?
// - templated dimensionality? (also for array_t to save space when passing arrays to kernels)
template <typename T>
struct indexedarray_t
{
    CUDA_CALLABLE inline indexedarray_t()
        : arr(),
          indices(),
          shape()
    {}

    CUDA_CALLABLE inline bool empty() const { return !arr.data; }

    array_t<T> arr;
    int* indices[ARRAY_MAX_DIMS];  // index array per dimension (can be NULL)
    shape_t shape;  // element count per dimension (num. indices if indexed, array dim if not)
};


// return stride (in bytes) of the given index
template <typename T>
CUDA_CALLABLE inline size_t stride(const array_t<T>& a, int dim)
{
    return size_t(a.strides[dim]);
}

template <typename T>
CUDA_CALLABLE inline T* data_at_byte_offset(const array_t<T>& a, size_t byte_offset)
{
    return reinterpret_cast<T*>(reinterpret_cast<char*>(a.data) + byte_offset);
}

template <typename T>
CUDA_CALLABLE inline T* grad_at_byte_offset(const array_t<T>& a, size_t byte_offset)
{   
    return reinterpret_cast<T*>(reinterpret_cast<char*>(a.grad) + byte_offset);
}

template <typename T>
CUDA_CALLABLE inline size_t byte_offset(const array_t<T>& arr, int i)
{
    assert(i >= 0 && i < arr.shape[0]);
    
    return i*stride(arr, 0);
}

template <typename T>
CUDA_CALLABLE inline size_t byte_offset(const array_t<T>& arr, int i, int j)
{
    // if (i < 0 || i >= arr.shape[0])
    //     printf("i: %d > arr.shape[0]: %d\n", i, arr.shape[0]);

    // if (j < 0 || j >= arr.shape[1])
    //     printf("j: %d > arr.shape[1]: %d\n", j, arr.shape[1]);


    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);
    
    return i*stride(arr, 0) + j*stride(arr, 1);
}

template <typename T>
CUDA_CALLABLE inline size_t byte_offset(const array_t<T>& arr, int i, int j, int k)
{
    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);
    assert(k >= 0 && k < arr.shape[2]);

    return i*stride(arr, 0) + j*stride(arr, 1) + k*stride(arr, 2);
}

template <typename T>
CUDA_CALLABLE inline size_t byte_offset(const array_t<T>& arr, int i, int j, int k, int l)
{
    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);
    assert(k >= 0 && k < arr.shape[2]);
    assert(l >= 0 && l < arr.shape[3]);

    return i*stride(arr, 0) + j*stride(arr, 1) + k*stride(arr, 2) + l*stride(arr, 3);
}

template <typename T>
CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i)
{
    assert(arr.ndim == 1);
    T& result = *data_at_byte_offset(arr, byte_offset(arr, i));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i, int j)
{
    assert(arr.ndim == 2);
    T& result = *data_at_byte_offset(arr, byte_offset(arr, i, j));
    FP_VERIFY_FWD_2(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i, int j, int k)
{
    assert(arr.ndim == 3);
    T& result = *data_at_byte_offset(arr, byte_offset(arr, i, j, k));
    FP_VERIFY_FWD_3(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i, int j, int k, int l)
{
    assert(arr.ndim == 4);
    T& result = *data_at_byte_offset(arr, byte_offset(arr, i, j, k, l));
    FP_VERIFY_FWD_4(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index_grad(const array_t<T>& arr, int i)
{
    T& result = *grad_at_byte_offset(arr, byte_offset(arr, i));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index_grad(const array_t<T>& arr, int i, int j)
{
    T& result = *grad_at_byte_offset(arr, byte_offset(arr, i, j));
    FP_VERIFY_FWD_2(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index_grad(const array_t<T>& arr, int i, int j, int k)
{
    T& result = *grad_at_byte_offset(arr, byte_offset(arr, i, j, k));
    FP_VERIFY_FWD_3(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index_grad(const array_t<T>& arr, int i, int j, int k, int l)
{
    T& result = *grad_at_byte_offset(arr, byte_offset(arr, i, j, k, l));
    FP_VERIFY_FWD_4(result)

    return result;
}


template <typename T>
CUDA_CALLABLE inline T& index(const indexedarray_t<T>& iarr, int i)
{
    assert(iarr.arr.ndim == 1);
    assert(i >= 0 && i < iarr.shape[0]);

    if (iarr.indices[0])
    {
        i = iarr.indices[0][i];
        assert(i >= 0 && i < iarr.arr.shape[0]);
    }

    T& result = *data_at_byte_offset(iarr.arr, byte_offset(iarr.arr, i));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index(const indexedarray_t<T>& iarr, int i, int j)
{
    assert(iarr.arr.ndim == 2);
    assert(i >= 0 && i < iarr.shape[0]);
    assert(j >= 0 && j < iarr.shape[1]);

    if (iarr.indices[0])
    {
        i = iarr.indices[0][i];
        assert(i >= 0 && i < iarr.arr.shape[0]);
    }
    if (iarr.indices[1])
    {
        j = iarr.indices[1][j];
        assert(j >= 0 && j < iarr.arr.shape[1]);
    }

    T& result = *data_at_byte_offset(iarr.arr, byte_offset(iarr.arr, i, j));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index(const indexedarray_t<T>& iarr, int i, int j, int k)
{
    assert(iarr.arr.ndim == 3);
    assert(i >= 0 && i < iarr.shape[0]);
    assert(j >= 0 && j < iarr.shape[1]);
    assert(k >= 0 && k < iarr.shape[2]);

    if (iarr.indices[0])
    {
        i = iarr.indices[0][i];
        assert(i >= 0 && i < iarr.arr.shape[0]);
    }
    if (iarr.indices[1])
    {
        j = iarr.indices[1][j];
        assert(j >= 0 && j < iarr.arr.shape[1]);
    }
    if (iarr.indices[2])
    {
        k = iarr.indices[2][k];
        assert(k >= 0 && k < iarr.arr.shape[2]);
    }

    T& result = *data_at_byte_offset(iarr.arr, byte_offset(iarr.arr, i, j, k));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index(const indexedarray_t<T>& iarr, int i, int j, int k, int l)
{
    assert(iarr.arr.ndim == 4);
    assert(i >= 0 && i < iarr.shape[0]);
    assert(j >= 0 && j < iarr.shape[1]);
    assert(k >= 0 && k < iarr.shape[2]);
    assert(l >= 0 && l < iarr.shape[3]);

    if (iarr.indices[0])
    {
        i = iarr.indices[0][i];
        assert(i >= 0 && i < iarr.arr.shape[0]);
    }
    if (iarr.indices[1])
    {
        j = iarr.indices[1][j];
        assert(j >= 0 && j < iarr.arr.shape[1]);
    }
    if (iarr.indices[2])
    {
        k = iarr.indices[2][k];
        assert(k >= 0 && k < iarr.arr.shape[2]);
    }
    if (iarr.indices[3])
    {
        l = iarr.indices[3][l];
        assert(l >= 0 && l < iarr.arr.shape[3]);
    }

    T& result = *data_at_byte_offset(iarr.arr, byte_offset(iarr.arr, i, j, k, l));
    FP_VERIFY_FWD_1(result)

    return result;
}


template <typename T>
CUDA_CALLABLE inline array_t<T> view(array_t<T>& src, int i)
{
    assert(src.ndim > 1);
    assert(i >= 0 && i < src.shape[0]);

    array_t<T> a;
    size_t offset = byte_offset(src, i);
    a.data = data_at_byte_offset(src, offset);
    if (src.grad)
        a.grad = grad_at_byte_offset(src, offset);
    a.shape[0] = src.shape[1];
    a.shape[1] = src.shape[2];
    a.shape[2] = src.shape[3];
    a.strides[0] = src.strides[1];
    a.strides[1] = src.strides[2];
    a.strides[2] = src.strides[3];
    a.ndim = src.ndim-1; 

    return a;
}

template <typename T>
CUDA_CALLABLE inline array_t<T> view(array_t<T>& src, int i, int j)
{
    assert(src.ndim > 2);
    assert(i >= 0 && i < src.shape[0]);
    assert(j >= 0 && j < src.shape[1]);

    array_t<T> a;
    size_t offset = byte_offset(src, i, j);
    a.data = data_at_byte_offset(src, offset);
    if (src.grad)
        a.grad = grad_at_byte_offset(src, offset);
    a.shape[0] = src.shape[2];
    a.shape[1] = src.shape[3];
    a.strides[0] = src.strides[2];
    a.strides[1] = src.strides[3];
    a.ndim = src.ndim-2;
    
    return a;
}

template <typename T>
CUDA_CALLABLE inline array_t<T> view(array_t<T>& src, int i, int j, int k)
{
    assert(src.ndim > 3);
    assert(i >= 0 && i < src.shape[0]);
    assert(j >= 0 && j < src.shape[1]);
    assert(k >= 0 && k < src.shape[2]);

    array_t<T> a;
    size_t offset = byte_offset(src, i, j, k);
    a.data = data_at_byte_offset(src, offset);
    if (src.grad)
        a.grad = grad_at_byte_offset(src, offset);
    a.shape[0] = src.shape[3];
    a.strides[0] = src.strides[3];
    a.ndim = src.ndim-3;
    
    return a;
}


template <typename T>
CUDA_CALLABLE inline indexedarray_t<T> view(indexedarray_t<T>& src, int i)
{
    assert(src.arr.ndim > 1);

    if (src.indices[0])
    {
        assert(i >= 0 && i < src.shape[0]);
        i = src.indices[0][i];
    }

    indexedarray_t<T> a;
    a.arr = view(src.arr, i);
    a.indices[0] = src.indices[1];
    a.indices[1] = src.indices[2];
    a.indices[2] = src.indices[3];
    a.shape[0] = src.shape[1];
    a.shape[1] = src.shape[2];
    a.shape[2] = src.shape[3];

    return a;
}

template <typename T>
CUDA_CALLABLE inline indexedarray_t<T> view(indexedarray_t<T>& src, int i, int j)
{
    assert(src.arr.ndim > 2);

    if (src.indices[0])
    {
        assert(i >= 0 && i < src.shape[0]);
        i = src.indices[0][i];
    }
    if (src.indices[1])
    {
        assert(j >= 0 && j < src.shape[1]);
        j = src.indices[1][j];
    }

    indexedarray_t<T> a;
    a.arr = view(src.arr, i, j);
    a.indices[0] = src.indices[2];
    a.indices[1] = src.indices[3];
    a.shape[0] = src.shape[2];
    a.shape[1] = src.shape[3];
    
    return a;
}

template <typename T>
CUDA_CALLABLE inline indexedarray_t<T> view(indexedarray_t<T>& src, int i, int j, int k)
{
    assert(src.arr.ndim > 3);

    if (src.indices[0])
    {
        assert(i >= 0 && i < src.shape[0]);
        i = src.indices[0][i];
    }
    if (src.indices[1])
    {
        assert(j >= 0 && j < src.shape[1]);
        j = src.indices[1][j];
    }
    if (src.indices[2])
    {
        assert(k >= 0 && k < src.shape[2]);
        k = src.indices[2][k];
    }

    indexedarray_t<T> a;
    a.arr = view(src.arr, i, j, k);
    a.indices[0] = src.indices[3];
    a.shape[0] = src.shape[3];
    
    return a;
}

template<template<typename> class A1, template<typename> class A2, template<typename> class A3, typename T>
inline CUDA_CALLABLE void adj_view(A1<T>& src, int i, A2<T>& adj_src, int adj_i, A3<T> adj_ret) {}
template<template<typename> class A1, template<typename> class A2, template<typename> class A3, typename T>
inline CUDA_CALLABLE void adj_view(A1<T>& src, int i, int j, A2<T>& adj_src, int adj_i, int adj_j, A3<T> adj_ret) {}
template<template<typename> class A1, template<typename> class A2, template<typename> class A3, typename T>
inline CUDA_CALLABLE void adj_view(A1<T>& src, int i, int j, int k, A2<T>& adj_src, int adj_i, int adj_j, int adj_k, A3<T> adj_ret) {}

// TODO: lower_bound() for indexed arrays?

template <typename T>
CUDA_CALLABLE inline int lower_bound(const array_t<T>& arr, int arr_begin, int arr_end, T value)
{
    assert(arr.ndim == 1);

    int lower = arr_begin;
    int upper = arr_end - 1;

    while(lower < upper)
    {
        int mid = lower + (upper - lower) / 2;
        
        if (arr[mid] < value)
        {
            lower = mid + 1;
        }
        else
        {
            upper = mid;
        }
    }

    return lower;
}

template <typename T>
CUDA_CALLABLE inline int lower_bound(const array_t<T>& arr, T value)
{
    return lower_bound(arr, 0, arr.shape[0], value);
}

template <typename T> inline CUDA_CALLABLE void adj_lower_bound(const array_t<T>& arr, T value, array_t<T> adj_arr, T adj_value, int adj_ret) {}
template <typename T> inline CUDA_CALLABLE void adj_lower_bound(const array_t<T>& arr, int arr_begin, int arr_end, T value, array_t<T> adj_arr, int adj_arr_begin, int adj_arr_end, T adj_value, int adj_ret) {}

template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_add(const A<T>& buf, int i, T value) { return atomic_add(&index(buf, i), value); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_add(const A<T>& buf, int i, int j, T value) { return atomic_add(&index(buf, i, j), value); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_add(const A<T>& buf, int i, int j, int k, T value) { return atomic_add(&index(buf, i, j, k), value); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_add(const A<T>& buf, int i, int j, int k, int l, T value) { return atomic_add(&index(buf, i, j, k, l), value); }

template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_sub(const A<T>& buf, int i, T value) { return atomic_add(&index(buf, i), -value); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_sub(const A<T>& buf, int i, int j, T value) { return atomic_add(&index(buf, i, j), -value); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_sub(const A<T>& buf, int i, int j, int k, T value) { return atomic_add(&index(buf, i, j, k), -value); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_sub(const A<T>& buf, int i, int j, int k, int l, T value) { return atomic_add(&index(buf, i, j, k, l), -value); }

template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_min(const A<T>& buf, int i, T value) { return atomic_min(&index(buf, i), value); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_min(const A<T>& buf, int i, int j, T value) { return atomic_min(&index(buf, i, j), value); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_min(const A<T>& buf, int i, int j, int k, T value) { return atomic_min(&index(buf, i, j, k), value); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_min(const A<T>& buf, int i, int j, int k, int l, T value) { return atomic_min(&index(buf, i, j, k, l), value); }

template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_max(const A<T>& buf, int i, T value) { return atomic_max(&index(buf, i), value); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_max(const A<T>& buf, int i, int j, T value) { return atomic_max(&index(buf, i, j), value); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_max(const A<T>& buf, int i, int j, int k, T value) { return atomic_max(&index(buf, i, j, k), value); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T atomic_max(const A<T>& buf, int i, int j, int k, int l, T value) { return atomic_max(&index(buf, i, j, k, l), value); }

template<template<typename> class A, typename T>
inline CUDA_CALLABLE T* address(const A<T>& buf, int i) { return &index(buf, i); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T* address(const A<T>& buf, int i, int j) { return &index(buf, i, j); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T* address(const A<T>& buf, int i, int j, int k) { return &index(buf, i, j, k); }
template<template<typename> class A, typename T>
inline CUDA_CALLABLE T* address(const A<T>& buf, int i, int j, int k, int l) { return &index(buf, i, j, k, l); }

template<template<typename> class A, typename T>
inline CUDA_CALLABLE void array_store(const A<T>& buf, int i, T value)
{
    FP_VERIFY_FWD_1(value)

    index(buf, i) = value;
}
template<template<typename> class A, typename T>
inline CUDA_CALLABLE void array_store(const A<T>& buf, int i, int j, T value)
{
    FP_VERIFY_FWD_2(value)

    index(buf, i, j) = value;
}
template<template<typename> class A, typename T>
inline CUDA_CALLABLE void array_store(const A<T>& buf, int i, int j, int k, T value)
{
    FP_VERIFY_FWD_3(value)

    index(buf, i, j, k) = value;
}
template<template<typename> class A, typename T>
inline CUDA_CALLABLE void array_store(const A<T>& buf, int i, int j, int k, int l, T value)
{
    FP_VERIFY_FWD_4(value)

    index(buf, i, j, k, l) = value;
}

template<typename T>
inline CUDA_CALLABLE void store(T* address, T value)
{
    FP_VERIFY_FWD(value)

    *address = value;
}

template<typename T>
inline CUDA_CALLABLE T load(T* address)
{
    T value = *address;
    FP_VERIFY_FWD(value)

    return value;
}

// select operator to check for array being null
template <typename T1, typename T2>
CUDA_CALLABLE inline T2 select(const array_t<T1>& arr, const T2& a, const T2& b) { return arr.data?b:a; }

template <typename T1, typename T2>
CUDA_CALLABLE inline void adj_select(const array_t<T1>& arr, const T2& a, const T2& b, const array_t<T1>& adj_cond, T2& adj_a, T2& adj_b, const T2& adj_ret)
{
    if (arr.data)
        adj_b += adj_ret;
    else
        adj_a += adj_ret;
}

// where operator to check for array being null, opposite convention compared to select
template <typename T1, typename T2>
CUDA_CALLABLE inline T2 where(const array_t<T1>& arr, const T2& a, const T2& b) { return arr.data?a:b; }

template <typename T1, typename T2>
CUDA_CALLABLE inline void adj_where(const array_t<T1>& arr, const T2& a, const T2& b, const array_t<T1>& adj_cond, T2& adj_a, T2& adj_b, const T2& adj_ret)
{
    if (arr.data)
        adj_a += adj_ret;
    else
        adj_b += adj_ret;
}

// stub for the case where we have an nested array inside a struct and
// atomic add the whole struct onto an array (e.g.: during backwards pass)
template <typename T>
CUDA_CALLABLE inline void atomic_add(array_t<T>*, array_t<T>) {}

// for float and vector types this is just an alias for an atomic add
template <typename T>
CUDA_CALLABLE inline void adj_atomic_add(T* buf, T value) { atomic_add(buf, value); }


// for integral types we do not accumulate gradients
CUDA_CALLABLE inline void adj_atomic_add(int8* buf, int8 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint8* buf, uint8 value) { }
CUDA_CALLABLE inline void adj_atomic_add(int16* buf, int16 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint16* buf, uint16 value) { }
CUDA_CALLABLE inline void adj_atomic_add(int32* buf, int32 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint32* buf, uint32 value) { }
CUDA_CALLABLE inline void adj_atomic_add(int64* buf, int64 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint64* buf, uint64 value) { }

CUDA_CALLABLE inline void adj_atomic_add(bool* buf, bool value) { }

// only generate gradients for T types
template<typename T>
inline CUDA_CALLABLE void adj_address(const array_t<T>& buf, int i, const array_t<T>& adj_buf, int adj_i, const T& adj_output)
{
    if (adj_buf.data)
        adj_atomic_add(&index(adj_buf, i), adj_output);
    else if (buf.grad)
        adj_atomic_add(&index_grad(buf, i), adj_output);
}
template<typename T>
inline CUDA_CALLABLE void adj_address(const array_t<T>& buf, int i, int j, const array_t<T>& adj_buf, int adj_i, int adj_j, const T& adj_output)
{
    if (adj_buf.data)
        adj_atomic_add(&index(adj_buf, i, j), adj_output);
    else if (buf.grad)
        adj_atomic_add(&index_grad(buf, i, j), adj_output);
}
template<typename T>
inline CUDA_CALLABLE void adj_address(const array_t<T>& buf, int i, int j, int k, const array_t<T>& adj_buf, int adj_i, int adj_j, int adj_k, const T& adj_output)
{
    if (adj_buf.data)
        adj_atomic_add(&index(adj_buf, i, j, k), adj_output);
    else if (buf.grad)
        adj_atomic_add(&index_grad(buf, i, j, k), adj_output);
}
template<typename T>
inline CUDA_CALLABLE void adj_address(const array_t<T>& buf, int i, int j, int k, int l, const array_t<T>& adj_buf, int adj_i, int adj_j, int adj_k, int adj_l, const T& adj_output)
{
    if (adj_buf.data)
        adj_atomic_add(&index(adj_buf, i, j, k, l), adj_output);
    else if (buf.grad)
        adj_atomic_add(&index_grad(buf, i, j, k, l), adj_output);
}

template<typename T>
inline CUDA_CALLABLE void adj_array_store(const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int adj_i, T& adj_value)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i);
    else if (buf.grad)
        adj_value += index_grad(buf, i);

    FP_VERIFY_ADJ_1(value, adj_value)
}
template<typename T>
inline CUDA_CALLABLE void adj_array_store(const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int adj_i, int adj_j, T& adj_value)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j);
    else if (buf.grad)
        adj_value += index_grad(buf, i, j);

    FP_VERIFY_ADJ_2(value, adj_value)
}
template<typename T>
inline CUDA_CALLABLE void adj_array_store(const array_t<T>& buf, int i, int j, int k, T value, const array_t<T>& adj_buf, int adj_i, int adj_j, int adj_k, T& adj_value)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j, k);
    else if (buf.grad)
        adj_value += index_grad(buf, i, j, k);

    FP_VERIFY_ADJ_3(value, adj_value)
}
template<typename T>
inline CUDA_CALLABLE void adj_array_store(const array_t<T>& buf, int i, int j, int k, int l, T value, const array_t<T>& adj_buf, int adj_i, int adj_j, int adj_k, int adj_l, T& adj_value)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j, k, l);
    else if (buf.grad)
        adj_value += index_grad(buf, i, j, k, l);

    FP_VERIFY_ADJ_4(value, adj_value)
}

template<typename T>
inline CUDA_CALLABLE void adj_store(const T* address, T value, const T& adj_address, T& adj_value)
{
	// nop; generic store() operations are not differentiable, only array_store() is
    FP_VERIFY_ADJ(value, adj_value)
}

template<typename T>
inline CUDA_CALLABLE void adj_load(const T* address, const T& adj_address, T& adj_value)
{
    // nop; generic load() operations are not differentiable
}

template<typename T>
inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i);
    else if (buf.grad)
        adj_value += index_grad(buf, i);

    FP_VERIFY_ADJ_1(value, adj_value)
}
template<typename T>
inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int adj_i, int adj_j, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j);
    else if (buf.grad)
        adj_value += index_grad(buf, i, j);

    FP_VERIFY_ADJ_2(value, adj_value)
}
template<typename T>
inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, int j, int k, T value, const array_t<T>& adj_buf, int adj_i, int adj_j, int adj_k, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j, k);
    else if (buf.grad)
        adj_value += index_grad(buf, i, j, k);

    FP_VERIFY_ADJ_3(value, adj_value)
}
template<typename T>
inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, int j, int k, int l, T value, const array_t<T>& adj_buf, int adj_i, int adj_j, int adj_k, int adj_l, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j, k, l);
    else if (buf.grad)
        adj_value += index_grad(buf, i, j, k, l);

    FP_VERIFY_ADJ_4(value, adj_value)
}

template<typename T>
inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i);
    else if (buf.grad)
        adj_value -= index_grad(buf, i);

    FP_VERIFY_ADJ_1(value, adj_value)
}
template<typename T>
inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int adj_i, int adj_j, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i, j);
    else if (buf.grad)
        adj_value -= index_grad(buf, i, j);

    FP_VERIFY_ADJ_2(value, adj_value)
}
template<typename T>
inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, int j, int k, T value, const array_t<T>& adj_buf, int adj_i, int adj_j, int adj_k, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i, j, k);
    else if (buf.grad)
        adj_value -= index_grad(buf, i, j, k);

    FP_VERIFY_ADJ_3(value, adj_value)
}
template<typename T>
inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, int j, int k, int l, T value, const array_t<T>& adj_buf, int adj_i, int adj_j, int adj_k, int adj_l, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i, j, k, l);
    else if (buf.grad)
        adj_value -= index_grad(buf, i, j, k, l);

    FP_VERIFY_ADJ_4(value, adj_value)
}

// generic array types that do not support gradient computation (indexedarray, etc.)
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_address(const A1<T>& buf, int i, const A2<T>& adj_buf, int adj_i, const T& adj_output) {}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_address(const A1<T>& buf, int i, int j, const A2<T>& adj_buf, int adj_i, int adj_j, const T& adj_output) {}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_address(const A1<T>& buf, int i, int j, int k, const A2<T>& adj_buf, int adj_i, int adj_j, int adj_k, const T& adj_output) {}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_address(const A1<T>& buf, int i, int j, int k, int l, const A2<T>& adj_buf, int adj_i, int adj_j, int adj_k, int adj_l, const T& adj_output) {}

template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_array_store(const A1<T>& buf, int i, T value, const A2<T>& adj_buf, int adj_i, T& adj_value) {}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_array_store(const A1<T>& buf, int i, int j, T value, const A2<T>& adj_buf, int adj_i, int adj_j, T& adj_value) {}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_array_store(const A1<T>& buf, int i, int j, int k, T value, const A2<T>& adj_buf, int adj_i, int adj_j, int adj_k, T& adj_value) {}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_array_store(const A1<T>& buf, int i, int j, int k, int l, T value, const A2<T>& adj_buf, int adj_i, int adj_j, int adj_k, int adj_l, T& adj_value) {}

template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_add(const A1<T>& buf, int i, T value, const A2<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret) {}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_add(const A1<T>& buf, int i, int j, T value, const A2<T>& adj_buf, int adj_i, int adj_j, T& adj_value, const T& adj_ret) {}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_add(const A1<T>& buf, int i, int j, int k, T value, const A2<T>& adj_buf, int adj_i, int adj_j, int adj_k, T& adj_value, const T& adj_ret) {}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_add(const A1<T>& buf, int i, int j, int k, int l, T value, const A2<T>& adj_buf, int adj_i, int adj_j, int adj_k, int adj_l, T& adj_value, const T& adj_ret) {}

template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_sub(const A1<T>& buf, int i, T value, const A2<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret) {}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_sub(const A1<T>& buf, int i, int j, T value, const A2<T>& adj_buf, int adj_i, int adj_j, T& adj_value, const T& adj_ret) {}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_sub(const A1<T>& buf, int i, int j, int k, T value, const A2<T>& adj_buf, int adj_i, int adj_j, int adj_k, T& adj_value, const T& adj_ret) {}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_sub(const A1<T>& buf, int i, int j, int k, int l, T value, const A2<T>& adj_buf, int adj_i, int adj_j, int adj_k, int adj_l, T& adj_value, const T& adj_ret) {}

// generic handler for scalar values
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_min(const A1<T>& buf, int i, T value, const A2<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret) {
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i), &index(adj_buf, i), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i), &index_grad(buf, i), value, adj_value);

    FP_VERIFY_ADJ_1(value, adj_value)
}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_min(const A1<T>& buf, int i, int j, T value, const A2<T>& adj_buf, int adj_i, int adj_j, T& adj_value, const T& adj_ret) {
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j), &index(adj_buf, i, j), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j), &index_grad(buf, i, j), value, adj_value);

    FP_VERIFY_ADJ_2(value, adj_value)
}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_min(const A1<T>& buf, int i, int j, int k, T value, const A2<T>& adj_buf, int adj_i, int adj_j, int adj_k, T& adj_value, const T& adj_ret) {
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j, k), &index(adj_buf, i, j, k), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j, k), &index_grad(buf, i, j, k), value, adj_value);

    FP_VERIFY_ADJ_3(value, adj_value)
}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_min(const A1<T>& buf, int i, int j, int k, int l, T value, const A2<T>& adj_buf, int adj_i, int adj_j, int adj_k, int adj_l, T& adj_value, const T& adj_ret) {
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j, k, l), &index(adj_buf, i, j, k, l), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j, k, l), &index_grad(buf, i, j, k, l), value, adj_value);

    FP_VERIFY_ADJ_4(value, adj_value)
}

template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_max(const A1<T>& buf, int i, T value, const A2<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret) {
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i), &index(adj_buf, i), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i), &index_grad(buf, i), value, adj_value);

    FP_VERIFY_ADJ_1(value, adj_value)
}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_max(const A1<T>& buf, int i, int j, T value, const A2<T>& adj_buf, int adj_i, int adj_j, T& adj_value, const T& adj_ret) {
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j), &index(adj_buf, i, j), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j), &index_grad(buf, i, j), value, adj_value);

    FP_VERIFY_ADJ_2(value, adj_value)
}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_max(const A1<T>& buf, int i, int j, int k, T value, const A2<T>& adj_buf, int adj_i, int adj_j, int adj_k, T& adj_value, const T& adj_ret) {
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j, k), &index(adj_buf, i, j, k), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j, k), &index_grad(buf, i, j, k), value, adj_value);

    FP_VERIFY_ADJ_3(value, adj_value)
}
template<template<typename> class A1, template<typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_max(const A1<T>& buf, int i, int j, int k, int l, T value, const A2<T>& adj_buf, int adj_i, int adj_j, int adj_k, int adj_l, T& adj_value, const T& adj_ret) {
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j, k, l), &index(adj_buf, i, j, k, l), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j, k, l), &index_grad(buf, i, j, k, l), value, adj_value);

    FP_VERIFY_ADJ_4(value, adj_value)
}

template<template<typename> class A, typename T>
CUDA_CALLABLE inline int len(const A<T>& a)
{
    return a.shape[0];
}

template<template<typename> class A, typename T>
CUDA_CALLABLE inline void adj_len(const A<T>& a, A<T>& adj_a, int& adj_ret)
{
}


} // namespace wp

#include "fabric.h"
