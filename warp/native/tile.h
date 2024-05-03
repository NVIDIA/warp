#pragma once

#include "builtin.h"

// #define WP_CONCAT(x, y) x ## y
// #define WP_SHARED_MEM(name, id) WP_CONCAT(name, id)

// #define zero(a) memset(a, 0, sizeof(a));

// #define tile_zeros(a, b, dtype) [](){\
// static dtype WP_SHARED_MEM(data_, __LINE__)[a][b]; \
// zero(WP_SHARED_MEM(data_, __LINE__)); \
// return array_t<dtype>WP_SHARED_MEM(data_, __LINE__; )}()

#if !defined(__CUDA_ARCH__)
#define __shared__ static
#endif

namespace wp
{

// 2D tile zero
template <typename T, int M, int N, int Index>
inline CUDA_CALLABLE array_t<T> tile_zeros()
{
    __shared__ T data[M*N];
    
    return array_t<T>(data, M, N, nullptr);
}

// 2D tile load
template <typename T, int M, int N, int Index>
inline CUDA_CALLABLE array_t<T> tile_load(const array_t<T>& src, int i, int j)
{
    const int length = M*N;

    __shared__ T data[length];
    
    // cooperatively load the tile, using a block-stride iterator
    // todo: use cub::BlockLoad or cg::memcpy_async()?
    for (int t=threadIdx.y; t < length; t += blockDim.y)
    {  
        data[t] = index(src, i*M + t/N, j*N + t%N);
    }
        
    return array_t<T>(data, M, N, nullptr);
}

// 2D tile store
template <typename T>
inline CUDA_CALLABLE array_t<T> tile_store(const array_t<T>& dest, const array_t<T>& src, int i, int j)
{
    const int length = src.shape[0]*src.shape[1];

    // cooperatively store the tile, using a block-stride iterator
    // todo: use cub::BlockStore or cg::memcpy_async()?
    for (int t=threadIdx.y; t < length; t += blockDim.y)
    {  
        index(dest, i*M + t/N, j*N + t%N, i) = src.data[t];
    }
        
    return array_t<T>(data, M, N, nullptr);
}


// 2D gemm accumulate out += A*B
template <typename T>
inline CUDA_CALLABLE void tile_matmul(const array_t<T>& A, const array_t<T>& B, const array_t<T>& out)
{    
    const int length = out.shape[0]*out.shape[1];

    for (int t=threadIdx.y; t < length; t += blockDim.y)
    {  
        // compute output index
        const int i = t%out.shape[0];
        const int j = t/out.shape[1];

        T sum = T(0.0);

        for (int k=0; k < A.shape[1]; ++k)
        {
            sum += index(A, i, k)*index(B, k, j);
        }

        index(out, i, j) += sum;
    }
}




} // namespace wp