#pragma once

#include "builtin.h"

namespace wp
{

const int ARRAY_MAX_DIMS = 4;    // must match constant in types.py

template <typename T>
struct array_t
{
    array_t() {}    
    array_t(int) {} // for backward a = 0 initialization syntax

    T* data;
    int shape[ARRAY_MAX_DIMS];
    int ndim;

    CUDA_CALLABLE inline operator T*() const { return data; }
};

// return stride (in elements) of the given index
template <typename T>
CUDA_CALLABLE inline int stride(const array_t<T>& a, int dim)
{
    int stride = 1;
    for (int i=dim+1; i < a.ndim; ++i)
        stride = stride*a.shape[i];

    return stride;
}

template <typename T>
CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i)
{
    assert(arr.ndim == 1);
    assert(i >= 0 && i < arr.shape[0]);
    
    const int idx = i;

    T& result = arr.data[idx];
    #if FP_CHECK
    assert(isfinite(result));
    if (!isfinite(result)) {
        printf("%s:%d - index(arr, %d) = ", __FILE__, __LINE__, i);
        print(result);
        printf(")\n");
    }
    #endif
    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i, int j)
{
    assert(arr.ndim == 2);
    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);

    const int idx = i*arr.shape[1] + j;

    T& result = arr.data[idx];
    #if FP_CHECK
    assert(isfinite(result));
    if (!isfinite(result)) {
        printf("%s:%d - index(arr, %d, %d) = ", __FILE__, __LINE__, i, j);
        print(result);
        printf(")\n");
    }
    #endif
    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i, int j, int k)
{
    assert(arr.ndim == 3);
    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);
    assert(k >= 0 && k < arr.shape[2]);

    const int idx = i*arr.shape[1]*arr.shape[2] + 
                    j*arr.shape[2] +
                    k;
       
    T& result = arr.data[idx];
    #if FP_CHECK
    assert(isfinite(result));
    if (!isfinite(result)) {
        printf("%s:%d - index(arr, %d, %d, %d) = ", __FILE__, __LINE__, i, j, k);
        print(result);
        printf(")\n");
    }
    #endif
    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i, int j, int k, int l)
{
    assert(arr.ndim == 3);
    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);
    assert(k >= 0 && k < arr.shape[2]);
    assert(l >= 0 && l < arr.shape[3]);

    const int idx = i*arr.shape[1]*arr.shape[2]*arr.shape[3] + 
                    j*arr.shape[2]*arr.shape[3] + 
                    k*arr.shape[3] + 
                    l;

    T& result = arr.data[idx];
    #if FP_CHECK
    assert(isfinite(result));
    if (!isfinite(result)) {
        printf("%s:%d - index(arr, %d, %d, %d, %d) = ", __FILE__, __LINE__, i, j, k, l);
        print(result);
        printf(")\n");
    }
    #endif
    return result;
}

template <typename T>
CUDA_CALLABLE inline array_t<T> view(array_t<T>& src, int i)
{
    array_t<T> a;
    a.data = src.data + i*stride(src, 0);
    a.shape[0] = src.shape[1];
    a.shape[1] = src.shape[2];
    a.shape[2] = src.shape[3];
    a.ndim = src.ndim-1; 

    return a;
}

template <typename T>
CUDA_CALLABLE inline array_t<T> view(array_t<T>& src, int i, int j)
{
    array_t<T> a;
    a.data = src.data + i*stride(src, 0) + j*stride(src,1);
    a.shape[0] = src.shape[2];
    a.shape[1] = src.shape[3];
    a.ndim = src.ndim-2;
    
    return a;
}

template <typename T>
CUDA_CALLABLE inline array_t<T> view(array_t<T>& src, int i, int j, int k)
{
    array_t<T> a;
    a.data = src.data + i*stride(src, 0) + j*stride(src,1) + k*stride(src,2);
    a.shape[0] = src.shape[3];
    a.ndim = src.ndim-3;
    
    return a;
}

template <typename T> inline CUDA_CALLABLE void adj_view(array_t<T>& src, int i, array_t<T>& adj_src, int adj_i, array_t<T> adj_ret) {}
template <typename T> inline CUDA_CALLABLE void adj_view(array_t<T>& src, int i, int j, array_t<T>& adj_src, int adj_i, int adj_j, array_t<T> adj_ret) {}
template <typename T> inline CUDA_CALLABLE void adj_view(array_t<T>& src, int i, int j, int k, array_t<T>& adj_src, int adj_i, int adj_j, int adj_k, array_t<T> adj_ret) {}


template<typename T> inline CUDA_CALLABLE T atomic_add(const array_t<T>& buf, int i, T value) { return atomic_add(&index(buf, i), value); }
template<typename T> inline CUDA_CALLABLE T atomic_add(const array_t<T>& buf, int i, int j, T value) { return atomic_add(&index(buf, i, j), value); }
template<typename T> inline CUDA_CALLABLE T atomic_add(const array_t<T>& buf, int i, int j, int k, T value) { return atomic_add(&index(buf, i, j, k), value); }
template<typename T> inline CUDA_CALLABLE T atomic_add(const array_t<T>& buf, int i, int j, int k, int l, T value) { return atomic_add(&index(buf, i, j, k, l), value); }

template<typename T> inline CUDA_CALLABLE T atomic_sub(const array_t<T>& buf, int i, T value) { return atomic_add(&index(buf, i), -value); }
template<typename T> inline CUDA_CALLABLE T atomic_sub(const array_t<T>& buf, int i, int j, T value) { return atomic_add(&index(buf, i, j), -value); }
template<typename T> inline CUDA_CALLABLE T atomic_sub(const array_t<T>& buf, int i, int j, int k, T value) { return atomic_add(&index(buf, i, j, k), -value); }
template<typename T> inline CUDA_CALLABLE T atomic_sub(const array_t<T>& buf, int i, int j, int k, int l, T value) { return atomic_add(&index(buf, i, j, k, l), -value); }


template<typename T> inline CUDA_CALLABLE T load(const array_t<T>& buf, int i) { return index(buf, i); }
template<typename T> inline CUDA_CALLABLE T load(const array_t<T>& buf, int i, int j) { return index(buf, i, j); }
template<typename T> inline CUDA_CALLABLE T load(const array_t<T>& buf, int i, int j, int k) { return index(buf, i, j, k); }
template<typename T> inline CUDA_CALLABLE T load(const array_t<T>& buf, int i, int j, int k, int l) { return index(buf, i, j, k, l); }


template<typename T> inline CUDA_CALLABLE void store(const array_t<T>& buf, int i, T value)
{
    #if FP_CHECK
    assert(isfinite(value));
    if (!isfinite(value))
    {
        printf("%s:%d - store(arr, %d, ", __FILE__, __LINE__, i);
        print(value);
        printf(")\n");
    }
    #endif
    index(buf, i) = value;
}
template<typename T> inline CUDA_CALLABLE void store(const array_t<T>& buf, int i, int j, T value)
{
    #if FP_CHECK
    assert(isfinite(value));
    if (!isfinite(value))
    {
        printf("%s:%d - store(arr, %d, %d, ", __FILE__, __LINE__, i, j);
        print(value);
        printf(")\n");
    }
    #endif
    index(buf, i, j) = value;
}
template<typename T> inline CUDA_CALLABLE void store(const array_t<T>& buf, int i, int j, int k, T value)
{
    #if FP_CHECK
    assert(isfinite(value));
    if (!isfinite(value))
    {
        printf("%s:%d - store(arr, %d, %d, %d, ", __FILE__, __LINE__, i, j, k);
        print(value);
        printf(")\n");
    }
    #endif
    index(buf, i, j, k) = value;
}
template<typename T> inline CUDA_CALLABLE void store(const array_t<T>& buf, int i, int j, int k, int l, T value)
{
    #if FP_CHECK
    assert(isfinite(value));
    if (!isfinite(value))
    {
        printf("%s:%d - store(arr, %d, %d, %d, %d, ", __FILE__, __LINE__, i, j, k, l);
        print(value);
        printf(")\n");
    }
    #endif
    index(buf, i, j, k, l) = value;
}


// for float and vector types this is just an alias for an atomic add
template <typename T>
CUDA_CALLABLE inline void adj_atomic_add(T* buf, T value) { atomic_add(buf, value); }

// for integral types (and doubles) we do not accumulate gradients
CUDA_CALLABLE inline void adj_atomic_add(int8* buf, int8 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint8* buf, uint8 value) { }
CUDA_CALLABLE inline void adj_atomic_add(int16* buf, int16 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint16* buf, uint16 value) { }
CUDA_CALLABLE inline void adj_atomic_add(int32* buf, int32 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint32* buf, uint32 value) { }
CUDA_CALLABLE inline void adj_atomic_add(int64* buf, int64 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint64* buf, uint64 value) { }
CUDA_CALLABLE inline void adj_atomic_add(float64* buf, float64 value) { }

// only generate gradients for T types
template<typename T> inline CUDA_CALLABLE void adj_load(const array_t<T>& buf, int i, const array_t<T>& adj_buf, int& adj_i, const T& adj_output) { if (adj_buf.data) { adj_atomic_add(&index(adj_buf, i), adj_output); } }
template<typename T> inline CUDA_CALLABLE void adj_load(const array_t<T>& buf, int i, int j, const array_t<T>& adj_buf, int& adj_i, int& adj_j, const T& adj_output) { if (adj_buf.data) { adj_atomic_add(&index(adj_buf, i, j), adj_output); } }
template<typename T> inline CUDA_CALLABLE void adj_load(const array_t<T>& buf, int i, int j, int k, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, const T& adj_output) { if (adj_buf.data) { adj_atomic_add(&index(adj_buf, i, j, k), adj_output); } }
template<typename T> inline CUDA_CALLABLE void adj_load(const array_t<T>& buf, int i, int j, int k, int l, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, int & adj_l, const T& adj_output) { if (adj_buf.data) { adj_atomic_add(&index(adj_buf, i, j, k, l), adj_output); } }

template<typename T> inline CUDA_CALLABLE void adj_store(const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int& adj_i, T& adj_value)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i);
    #if FP_CHECK
        assert(isfinite(value) && isfinite(adj_value));
        if (!isfinite(value) || !isfinite(adj_value))
        {
            printf("%s:%d - adj_store(arr, %d, ", __FILE__, __LINE__, i);
            print(value);
            printf(", ");
            print(adj_value);
            printf(")\n");
        }
    #endif
}
template<typename T> inline CUDA_CALLABLE void adj_store(const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, T& adj_value)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j);
    #if FP_CHECK
        assert(isfinite(value) && isfinite(adj_value));
        if (!isfinite(value) || !isfinite(adj_value))
        {
            printf("%s:%d - adj_store(arr, %d, %d, ", __FILE__, __LINE__, i, j);
            print(value);
            printf(", ");
            print(adj_value);
            printf(")\n");
        }
    #endif
}
template<typename T> inline CUDA_CALLABLE void adj_store(const array_t<T>& buf, int i, int j, int k, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, T& adj_value)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j, k);
    #if FP_CHECK
        assert(isfinite(value) && isfinite(adj_value));
        if (!isfinite(value) || !isfinite(adj_value))
        {
            printf("%s:%d - adj_store(arr, %d, %d, %d, ", __FILE__, __LINE__, i, j, k);
            print(value);
            printf(", ");
            print(adj_value);
            printf(")\n");
        }
    #endif
}
template<typename T> inline CUDA_CALLABLE void adj_store(const array_t<T>& buf, int i, int j, int k, int l, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, int& adj_l, T& adj_value)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j, k, l);
    #if FP_CHECK
        assert(isfinite(value) && isfinite(adj_value));
        if (!isfinite(value) || !isfinite(adj_value))
        {
            printf("%s:%d - adj_store(arr, %d, %d, %d, %d, ", __FILE__, __LINE__, i, j, k, l);
            print(value);
            printf(", ");
            print(adj_value);
            printf(")\n");
        }
    #endif
}

template<typename T> inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int& adj_i, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i);
    #if FP_CHECK
        assert(isfinite(value) && isfinite(adj_value) && isfinite(adj_ret));
        if (!isfinite(adj_value) || !isfinite(adj_ret))
        {
            printf("%s:%d - adj_atomic_add(arr, %d, ", __FILE__, __LINE__, i);
            print(value);
            printf(", ");
            print(adj_value);
            printf(", ");
            print(adj_ret);
            printf(")\n");
        }
    #endif
}
template<typename T> inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j);
    #if FP_CHECK
        assert(isfinite(value) && isfinite(adj_value) && isfinite(adj_ret));
        if (!isfinite(adj_value) || !isfinite(adj_ret))
        {
            printf("%s:%d - adj_atomic_add(arr, %d, %d, ", __FILE__, __LINE__, i, j);
            print(value);
            printf(", ");
            print(adj_value);
            printf(", ");
            print(adj_ret);
            printf(")\n");
        }
    #endif
}
template<typename T> inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, int j, int k, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j, k);
    #if FP_CHECK
        assert(isfinite(value) && isfinite(adj_value) && isfinite(adj_ret));
        if (!isfinite(adj_value) || !isfinite(adj_ret))
        {
            printf("%s:%d - adj_atomic_add(arr, %d, %d, %d, ", __FILE__, __LINE__, i, j, k);
            print(value);
            printf(", ");
            print(adj_value);
            printf(", ");
            print(adj_ret);
            printf(")\n");
        }
    #endif
}
template<typename T> inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, int j, int k, int l, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, int& adj_l, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j, k, l);
    #if FP_CHECK
        assert(isfinite(value) && isfinite(adj_value) && isfinite(adj_ret));
        if (!isfinite(adj_value) || !isfinite(adj_ret))
        {
            printf("%s:%d - adj_atomic_add(arr, %d, %d, %d, %d, ", __FILE__, __LINE__, i, j, k, l);
            print(value);
            printf(", ");
            print(adj_value);
            printf(", ");
            print(adj_ret);
            printf(")\n");
        }
    #endif
}


template<typename T> inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int& adj_i, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i);
    #if FP_CHECK
        assert(isfinite(value) && isfinite(adj_value) && isfinite(adj_ret));
        if (!isfinite(adj_value) || !isfinite(adj_ret))
        {
            printf("%s:%d - adj_atomic_sub(arr, %d, ", __FILE__, __LINE__, i);
            print(value);
            printf(", ");
            print(adj_value);
            printf(", ");
            print(adj_ret);
            printf(")\n");
        }
    #endif
}
template<typename T> inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i, j);
    #if FP_CHECK
        assert(isfinite(value) && isfinite(adj_value) && isfinite(adj_ret));
        if (!isfinite(adj_value) || !isfinite(adj_ret))
        {
            printf("%s:%d - adj_atomic_sub(arr, %d, %d, ", __FILE__, __LINE__, i, j);
            print(value);
            printf(", ");
            print(adj_value);
            printf(", ");
            print(adj_ret);
            printf(")\n");
        }
    #endif
}
template<typename T> inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, int j, int k, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i, j, k);
    #if FP_CHECK
        assert(isfinite(value) && isfinite(adj_value) && isfinite(adj_ret));
        if (!isfinite(adj_value) || !isfinite(adj_ret))
        {
            printf("%s:%d - adj_atomic_sub(arr, %d, %d, %d, ", __FILE__, __LINE__, i, j, k);
            print(value);
            printf(", ");
            print(adj_value);
            printf(", ");
            print(adj_ret);
            printf(")\n");
        }
    #endif
}
template<typename T> inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, int j, int k, int l, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, int& adj_l, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i, j, k, l);
    #if FP_CHECK
        assert(isfinite(value) && isfinite(adj_value) && isfinite(adj_ret));
        if (!isfinite(adj_value) || !isfinite(adj_ret))
        {
            printf("%s:%d - adj_atomic_sub(arr, %d, %d, %d, %d, ", __FILE__, __LINE__, i, j, k, l);
            print(value);
            printf(", ");
            print(adj_value);
            printf(", ");
            print(adj_ret);
            printf(")\n");
        }
    #endif
}

} // namespace wp