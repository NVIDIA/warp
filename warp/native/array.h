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

#define FP_VERIFY_FWD_1(value) {}
#define FP_VERIFY_FWD_2(value) {}
#define FP_VERIFY_FWD_3(value) {}
#define FP_VERIFY_FWD_4(value) {}

#define FP_VERIFY_ADJ_1(value, adj_value) {}
#define FP_VERIFY_ADJ_2(value, adj_value) {}
#define FP_VERIFY_ADJ_3(value, adj_value) {}
#define FP_VERIFY_ADJ_4(value, adj_value) {}

#endif  // WP_FP_CHECK

const int ARRAY_MAX_DIMS = 4;    // must match constant in types.py


struct shape_t
{
    int dims[ARRAY_MAX_DIMS];

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

CUDA_CALLABLE inline int index(const shape_t& s, int i)
{
    return s.dims[i];
}

CUDA_CALLABLE inline void adj_index(const shape_t& s, int i, const shape_t& adj_s, int adj_i, int adj_ret) {}

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
    CUDA_CALLABLE inline array_t() {}    
    CUDA_CALLABLE inline array_t(int) {} // for backward a = 0 initialization syntax

    array_t(T* data, int size) : data(data) {
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
    array_t(T* data, int dim0, int dim1) : data(data) {
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
    array_t(T* data, int dim0, int dim1, int dim2) : data(data) {
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
    array_t(T* data, int dim0, int dim1, int dim2, int dim3) : data(data) {
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

    T* data;
    shape_t shape;
    int strides[ARRAY_MAX_DIMS];
    int ndim;

    CUDA_CALLABLE inline operator T*() const { return data; }
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
CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i)
{
    assert(arr.ndim == 1);
    assert(i >= 0 && i < arr.shape[0]);
    
    const size_t byte_offset = i*stride(arr, 0);

    T& result = *data_at_byte_offset(arr, byte_offset);
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i, int j)
{
    assert(arr.ndim == 2);
    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);

    const size_t byte_offset = i*stride(arr,0) + j*stride(arr,1);

    T& result = *data_at_byte_offset(arr, byte_offset);
    FP_VERIFY_FWD_2(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i, int j, int k)
{
    assert(arr.ndim == 3);
    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);
    assert(k >= 0 && k < arr.shape[2]);

    const size_t byte_offset = i*stride(arr,0) + 
                            j*stride(arr,1) +
                            k*stride(arr,2);
       
    T& result = *data_at_byte_offset(arr, byte_offset);
    FP_VERIFY_FWD_3(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i, int j, int k, int l)
{
    assert(arr.ndim == 4);
    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);
    assert(k >= 0 && k < arr.shape[2]);
    assert(l >= 0 && l < arr.shape[3]);

    const size_t byte_offset = i*stride(arr,0) + 
                            j*stride(arr,1) + 
                            k*stride(arr,2) + 
                            l*stride(arr,3);

    T& result = *data_at_byte_offset(arr, byte_offset);
    FP_VERIFY_FWD_4(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline array_t<T> view(array_t<T>& src, int i)
{
    array_t<T> a;
    a.data = data_at_byte_offset(src, i*stride(src, 0));
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
    array_t<T> a;
    a.data = data_at_byte_offset(src, i*stride(src, 0) + j*stride(src,1));
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
    array_t<T> a;
    a.data = data_at_byte_offset(src, i*stride(src, 0) + j*stride(src,1) + k*stride(src,2));
    a.shape[0] = src.shape[3];
    a.strides[0] = src.strides[3];
    a.ndim = src.ndim-3;
    
    return a;
}

template <typename T>
CUDA_CALLABLE inline int lower_bound(const array_t<T>& arr, T value)
{
    assert(arr.ndim == 1);
    int n = arr.shape[0];

    int lower = 0;
    int upper = n - 1;

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

template <typename T> inline CUDA_CALLABLE void adj_lower_bound(const array_t<T>& arr, T value, array_t<T> adj_arr, T adj_value, int adj_ret) {}

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

template<typename T> inline CUDA_CALLABLE T atomic_min(const array_t<T>& buf, int i, T value) { return atomic_min(&index(buf, i), value); }
template<typename T> inline CUDA_CALLABLE T atomic_min(const array_t<T>& buf, int i, int j, T value) { return atomic_min(&index(buf, i, j), value); }
template<typename T> inline CUDA_CALLABLE T atomic_min(const array_t<T>& buf, int i, int j, int k, T value) { return atomic_min(&index(buf, i, j, k), value); }
template<typename T> inline CUDA_CALLABLE T atomic_min(const array_t<T>& buf, int i, int j, int k, int l, T value) { return atomic_min(&index(buf, i, j, k, l), value); }

template<typename T> inline CUDA_CALLABLE T atomic_max(const array_t<T>& buf, int i, T value) { return atomic_max(&index(buf, i), value); }
template<typename T> inline CUDA_CALLABLE T atomic_max(const array_t<T>& buf, int i, int j, T value) { return atomic_max(&index(buf, i, j), value); }
template<typename T> inline CUDA_CALLABLE T atomic_max(const array_t<T>& buf, int i, int j, int k, T value) { return atomic_max(&index(buf, i, j, k), value); }
template<typename T> inline CUDA_CALLABLE T atomic_max(const array_t<T>& buf, int i, int j, int k, int l, T value) { return atomic_max(&index(buf, i, j, k, l), value); }


template<typename T> inline CUDA_CALLABLE T load(const array_t<T>& buf, int i) { return index(buf, i); }
template<typename T> inline CUDA_CALLABLE T load(const array_t<T>& buf, int i, int j) { return index(buf, i, j); }
template<typename T> inline CUDA_CALLABLE T load(const array_t<T>& buf, int i, int j, int k) { return index(buf, i, j, k); }
template<typename T> inline CUDA_CALLABLE T load(const array_t<T>& buf, int i, int j, int k, int l) { return index(buf, i, j, k, l); }


template<typename T> inline CUDA_CALLABLE void store(const array_t<T>& buf, int i, T value)
{
    FP_VERIFY_FWD_1(value)

    index(buf, i) = value;
}
template<typename T> inline CUDA_CALLABLE void store(const array_t<T>& buf, int i, int j, T value)
{
    FP_VERIFY_FWD_2(value)

    index(buf, i, j) = value;
}
template<typename T> inline CUDA_CALLABLE void store(const array_t<T>& buf, int i, int j, int k, T value)
{
    FP_VERIFY_FWD_3(value)

    index(buf, i, j, k) = value;
}
template<typename T> inline CUDA_CALLABLE void store(const array_t<T>& buf, int i, int j, int k, int l, T value)
{
    FP_VERIFY_FWD_4(value)

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

    FP_VERIFY_ADJ_1(value, adj_value)
}
template<typename T> inline CUDA_CALLABLE void adj_store(const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, T& adj_value)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j);

    FP_VERIFY_ADJ_2(value, adj_value)

}
template<typename T> inline CUDA_CALLABLE void adj_store(const array_t<T>& buf, int i, int j, int k, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, T& adj_value)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j, k);

    FP_VERIFY_ADJ_3(value, adj_value)
}
template<typename T> inline CUDA_CALLABLE void adj_store(const array_t<T>& buf, int i, int j, int k, int l, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, int& adj_l, T& adj_value)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j, k, l);

    FP_VERIFY_ADJ_4(value, adj_value)
}

template<typename T> inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int& adj_i, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i);

    FP_VERIFY_ADJ_1(value, adj_value)
}
template<typename T> inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j);

    FP_VERIFY_ADJ_2(value, adj_value)
}
template<typename T> inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, int j, int k, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j, k);

    FP_VERIFY_ADJ_3(value, adj_value)
}
template<typename T> inline CUDA_CALLABLE void adj_atomic_add(const array_t<T>& buf, int i, int j, int k, int l, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, int& adj_l, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j, k, l);

    FP_VERIFY_ADJ_4(value, adj_value)
}


template<typename T> inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int& adj_i, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i);

    FP_VERIFY_ADJ_1(value, adj_value)
}
template<typename T> inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i, j);

    FP_VERIFY_ADJ_2(value, adj_value)
}
template<typename T> inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, int j, int k, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i, j, k);

    FP_VERIFY_ADJ_3(value, adj_value)
}
template<typename T> inline CUDA_CALLABLE void adj_atomic_sub(const array_t<T>& buf, int i, int j, int k, int l, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, int& adj_l, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i, j, k, l);

    FP_VERIFY_ADJ_4(value, adj_value)
}

template<typename T> inline CUDA_CALLABLE void adj_atomic_min(const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int& adj_i, T& adj_value, const T& adj_ret) {}
template<typename T> inline CUDA_CALLABLE void adj_atomic_min(const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, T& adj_value, const T& adj_ret) {}
template<typename T> inline CUDA_CALLABLE void adj_atomic_min(const array_t<T>& buf, int i, int j, int k, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, T& adj_value, const T& adj_ret) {}
template<typename T> inline CUDA_CALLABLE void adj_atomic_min(const array_t<T>& buf, int i, int j, int k, int l, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, int& adj_l, T& adj_value, const T& adj_ret) {}

template<typename T> inline CUDA_CALLABLE void adj_atomic_max(const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int& adj_i, T& adj_value, const T& adj_ret) {}
template<typename T> inline CUDA_CALLABLE void adj_atomic_max(const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, T& adj_value, const T& adj_ret) {}
template<typename T> inline CUDA_CALLABLE void adj_atomic_max(const array_t<T>& buf, int i, int j, int k, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, T& adj_value, const T& adj_ret) {}
template<typename T> inline CUDA_CALLABLE void adj_atomic_max(const array_t<T>& buf, int i, int j, int k, int l, T value, const array_t<T>& adj_buf, int& adj_i, int& adj_j, int& adj_k, int& adj_l, T& adj_value, const T& adj_ret) {}


} // namespace wp