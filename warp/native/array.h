#pragma once

#include "builtin.h"

//#include <assert.h>

namespace wp
{


const int kMaxArrayDims = 4;

template <typename T>
struct array_t
{
    T* data;
    int shape[kMaxArrayDims];
    //int stride[kMaxArrayDims];
    int ndims;
};

template <typename T>
T& index(const array_t<T>& arr, int i)
{
    assert(arr.ndims == 1);
    assert(arr.shape[0] > i);
    
    const int idx = i;

    return arr.data[idx];
}

template <typename T>
T& index(const array_t<T>& arr, int i, int j)
{
    assert(arr.ndims == 2);
    assert(arr.shape[0] > i);
    assert(arr.shape[1] > j);

    const int idx = i*arr.shape[1] + j;

    return arr.data[idx];
}

template <typename T>
T& index(const array_t<T>& arr, int i, int j, int k)
{
    assert(arr.ndims == 3);
    assert(arr.shape[0] > i);
    assert(arr.shape[1] > j);
    assert(arr.shape[2] > k);

    const int idx = i*arr.shape[1]*arr.shape[2] + 
                      j*arr.shape[2] +
                      k;
       
    return arr.data[idx];
}

template <typename T>
T& index(const array_t<T>& arr, int i, int j, int k, int l)
{
    assert(arr.ndims == 3);
    assert(arr.shape[0] > i);
    assert(arr.shape[1] > j);
    assert(arr.shape[2] > k);
    assert(arr.shape[3] > l);

    const int idx = i*arr.shape[1]*arr.shape[2]*arr.shape[3] + 
                      j*arr.shape[2]*arr.shape[3] + 
                      k*arr.shape[3] + 
                      l;

    // int i = 0;
    // for i < in range(0, kMaxArrayDims):
    //     i += coord[i]
    //     i *= arr.shape[i]

    return arr.data[idx];
}


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


template<typename T> inline CUDA_CALLABLE void store(const array_t<T>& buf, int i, T value) { index(buf, i) = value; }
template<typename T> inline CUDA_CALLABLE void store(const array_t<T>& buf, int i, int j, T value) { index(buf, i, j) = value; }
template<typename T> inline CUDA_CALLABLE void store(const array_t<T>& buf, int i, int j, int k, T value) { index(buf, i, j, k) = value; }
template<typename T> inline CUDA_CALLABLE void store(const array_t<T>& buf, int i, int j, int k, int l, T value) { index(buf, i, j, k, l) = value; }


// for float and vector types this is just an alias for an atomic add
template <typename T>
void adj_atomic_add(T* buf, T value) { atomic_add(buf, value); }

// for integral types (and doubles) we do not accumulate gradients
void adj_atomic_add(int8* buf, int8 value) { }
void adj_atomic_add(uint8* buf, uint8 value) { }
void adj_atomic_add(int16* buf, int16 value) { }
void adj_atomic_add(uint16* buf, uint16 value) { }
void adj_atomic_add(int32* buf, int32 value) { }
void adj_atomic_add(uint32* buf, uint32 value) { }
void adj_atomic_add(int64* buf, int64 value) { }
void adj_atomic_add(uint64* buf, uint64 value) { }
void adj_atomic_add(float64* buf, float64 value) { }

// only generate gradients for T types
template<typename T> inline CUDA_CALLABLE void adj_load(const array_t<T>& buf, int i, const array_t<T>& adj_buf, int& adj_i, const T& adj_output) { if (adj_buf.data) { adj_atomic_add(&index(adj_buf, i), adj_output); } }
template<typename T> inline CUDA_CALLABLE void adj_load(const array_t<T>& buf, int i, int j, const array_t<T>& adj_buf, int& adj_i, int& adj_j, const T& adj_output) { if (adj_buf.data) { adj_atomic_add(&index(adj_buf, i, j), adj_output); } }
template<typename T> inline CUDA_CALLABLE void adj_load(const array_t<T>& buf, int i, int j, int k, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, const T& adj_output) { if (adj_buf.data) { adj_atomic_add(&index(adj_buf, i, j, k), adj_output); } }
template<typename T> inline CUDA_CALLABLE void adj_load(const array_t<T>& buf, int i, int j, int k, int l, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, int & adj_l, const T& adj_output) { if (adj_buf.data) { adj_atomic_add(&index(adj_buf, i, j, k, l), adj_output); } }

template<typename T> inline CUDA_CALLABLE void adj_store(const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int& adj_i, T& adj_value) { if(adj_buf.data) adj_value += index(adj_buf, i); }
template<typename T> inline CUDA_CALLABLE void adj_store(const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, T& adj_value) { if(adj_buf.data) adj_value += index(adj_buf, i, j); }
template<typename T> inline CUDA_CALLABLE void adj_store(const array_t<T>& buf, int i, int j, int k, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, T& adj_value) { if(adj_buf.data) adj_value += index(adj_buf, i, j, k); }
template<typename T> inline CUDA_CALLABLE void adj_store(const array_t<T>& buf, int i, int j, int k, int l, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, int& adj_l, T& adj_value) { if(adj_buf.data) adj_value += index(adj_buf, i, j, k, l); }

template<typename T> inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int& adj_i, T& adj_value, const T& adj_ret) { if(adj_buf.data) adj_value += index(adj_buf, i); }
template<typename T> inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, T& adj_value, const T& adj_ret) { if(adj_buf.data) adj_value += index(adj_buf, i, j); }
template<typename T> inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, int j, int k, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, T& adj_value, const T& adj_ret) { if(adj_buf.data) adj_value += index(adj_buf, i, j, k); }
template<typename T> inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, int j, int k, int l, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, int& adj_l, T& adj_value, const T& adj_ret) { if(adj_buf.data) adj_value += index(adj_buf, i, j, k, l); }

template<typename T> inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int& adj_i, T& adj_value, const T& adj_ret) { if(adj_buf.data) adj_value -= index(adj_buf, i); }
template<typename T> inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, T& adj_value, const T& adj_ret) { if(adj_buf.data) adj_value -= index(adj_buf, i, j); }
template<typename T> inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, int j, int k, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, T& adj_value, const T& adj_ret) { if(adj_buf.data) adj_value -= index(adj_buf, i, j, k); }
template<typename T> inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, int j, int k, int l, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, int& adj_l, T& adj_value, const T& adj_ret) { if(adj_buf.data) adj_value -= index(adj_buf, i, j, k, l); }

} // namespace wp