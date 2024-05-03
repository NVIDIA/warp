#pragma once

#include "builtin.h"

// #define WP_CONCAT(x, y) x ## y
// #define WP_TILE_SHARED_MEM(name, id) WP_CONCAT(name, id)

// #define zero(a) memset(a, 0, sizeof(a));

// #define tile_zeros(a, b, dtype) [](){\
// static dtype WP_TILE_SHARED_MEM(data_, __LINE__)[a][b]; \
// zero(WP_TILE_SHARED_MEM(data_, __LINE__)); \
// return array_t<dtype>WP_TILE_SHARED_MEM(data_, __LINE__; )}()

#if !defined(__CUDA_ARCH__)
#define WP_TILE_SHARED static
#define WP_TILE_SYNC void
#else
#define WP_TILE_SHARED __shared__
#define WP_TILE_SYNC __syncthreads
#endif

namespace wp
{

// 2D tile zero
template <typename T, int M, int N, int Index>
inline CUDA_CALLABLE array_t<T> tile_zeros()
{
    const int length = M*N;

    WP_TILE_SHARED T data[length];
    
    for (int t=threadIdx.x; t < length; t += blockDim.x)
    {  
        data[t] = T(0.0);
    }

    return array_t<T>(data, M, N, nullptr);
}

// 2D tile load
template <typename T, int M, int N, int Index>
inline CUDA_CALLABLE array_t<T> tile_load(const array_t<T>& src, int i, int j)
{
    const int length = M*N;

    WP_TILE_SHARED T data[length];
    
    // cooperatively load the tile, using a block-stride iterator
    // todo: use cub::BlockLoad or cg::memcpy_async()?
    for (int t=threadIdx.x; t < length; t += blockDim.x)
    {  
        data[t] = index(src, i*M + t/N, j*N + t%N);
    }
        
    return array_t<T>(data, M, N, nullptr);
}

// 2D tile store
template <typename T>
inline CUDA_CALLABLE void tile_store(array_t<T>& dest, int i, int j, const array_t<T>& src)
{
    const int M = src.shape[0];
    const int N = src.shape[1];
    
    const int length = M*N;

    // cooperatively store the tile, using a block-stride iterator
    // todo: use cub::BlockStore or cg::memcpy_async()?
    for (int t=threadIdx.x; t < length; t += blockDim.x)
    {  
        index(dest, i*M + t/N, j*N + t%N) = src.data[t];
    }
}


// 2D gemm accumulate out += A*B
template <typename T>
inline CUDA_CALLABLE void tile_matmul(const array_t<T>& A, const array_t<T>& B, const array_t<T>& out)
{    
    const int length = out.shape[0]*out.shape[1];

    WP_TILE_SYNC();

    for (int t=threadIdx.x; t < length; t += blockDim.x)
    {  
        // compute output index
        const int i = t/out.shape[1];
        const int j = t%out.shape[1];

        T sum = T(0.0);

        for (int k=0; k < A.shape[1]; ++k)
        {
            sum += index(A, i, k)*index(B, k, j);
        }

        index(out, i, j) += sum;
    }

    WP_TILE_SYNC();
}




} // namespace wp