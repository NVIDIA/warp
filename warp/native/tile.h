#pragma once

#include "builtin.h"

// todo: requires CTK, replace with inline ptx
#include "cuda_pipeline_primitives.h"

#if !defined(__CUDA_ARCH__)
#define WP_TILE_SHARED static
#define WP_TILE_SYNC void
#else
#define WP_TILE_SHARED __shared__
#define WP_TILE_SYNC __syncthreads
#endif


namespace wp
{

// CUTLASS_PRAGMA_(UNROLL|NO_UNROLL) optimization directives for the CUDA compiler.
#if defined(__CUDA_ARCH__) && !defined(__INTELLISENSE__)
  #if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
    #define WP_PRAGMA_UNROLL _Pragma("unroll")
    #define WP_PRAGMA_NO_UNROLL _Pragma("unroll 1")
  #else
    #define WP_PRAGMA_UNROLL #pragma unroll
    #define WP_PRAGMA_NO_UNROLL #pragma unroll 1
  #endif

#else

    #define WP_PRAGMA_UNROLL
    #define WP_PRAGMA_NO_UNROLL

#endif


// 2D tile zero
template <typename T, int M, int N, int Index>
inline CUDA_CALLABLE array_t<T> tile_zeros()
{
    const int length = M*N;

    WP_TILE_SHARED __align__(16) T data[length];
    
    WP_PRAGMA_UNROLL
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

    WP_TILE_SHARED __align__(16) T data[length];
    
    //---------------
    // naive-synchronous load
    //
    // WP_PRAGMA_UNROLL
    // for (int t=threadIdx.x; t < length; t += blockDim.x)
    // {  
    //     data[t] = index(src, i*M + t/N, j*N + t%N);
    // }

    //---------------
    // async 128 bit loads (assumes row-major i.e.: stride 1 on y axis and 4-element alignment on dimension)
    const int s = 4;

    WP_PRAGMA_UNROLL
    for (int t=threadIdx.x*s; t < length; t += blockDim.x*s)
    {  
        __pipeline_memcpy_async(&data[t],
                                &index(src, i*M + t/N, j*N + t%N),
                                sizeof(T)*s);
    }

    __pipeline_commit();


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
    WP_PRAGMA_UNROLL
    for (int t=threadIdx.x; t < length; t += blockDim.x)
    {  
        index(dest, i*M + t/N, j*N + t%N) = src.data[t];
    }
}

template <typename T>
inline CUDA_CALLABLE const T& index(const T* __restrict__ p, int i, int j, int stride)
{
    return p[i*stride + j];
}

template <typename T>
inline CUDA_CALLABLE T& index(T* __restrict__ p, int i, int j, int stride)
{
    return p[i*stride + j];
}

template <unsigned M, unsigned N, typename T>
struct partition_t
{
    inline partition_t(array_t<T> A)
    {
        data = A;
        
        // todo: do ceil div for non-multiples of M,N
        shape[0] = A.shape[0]/M;
        shape[1] = A.shape[1]/N;
    }

    // underlying data
    array_t<T> data;
    
    // partition dimensions
    int shape[2];
};

template <unsigned M, unsigned N, typename T>
inline int partition_size(const partition_t<M, N, T>& tile)
{
    return tile.shape[0]*tile.shape[1];
}

// returns the x, y coordinates of a tile given a linear index
template <unsigned M, unsigned N, typename T>
inline void partition_coord(const partition_t<M, N, T>& tile, const int t, int& i, int& j)
{
    i = t/tile.shape[1];
    j = t%tile.shape[1];
}

template <unsigned M, unsigned N, typename T>
inline mat_t<M, N, T> partition_load(const partition_t<M, N, T>& tile, int i, int j)
{
    mat_t<M, N, T> out;
    
    const int tile_i = i*M;
    const int tile_j = j*N;

    WP_PRAGMA_UNROLL
    for (int i=0; i < M; ++i)
    {
        WP_PRAGMA_UNROLL
        for (int j=0; j < N; ++j)
        {
            out.data[i][j] = index(tile.data, tile_i + i, tile_j + j);
        }
    }

    return out;
}

template <unsigned M, unsigned N, typename T>
inline void partition_store(const partition_t<M, N, T>& tile, int i, int j, const mat_t<M, N, T>& value)
{
    mat_t<M, N, T> out;

    const int tile_i = M*i;
    const int tile_j = N*j;

    WP_PRAGMA_UNROLL
    for (int i=0; i < M; ++i)
    {	
        WP_PRAGMA_UNROLL
        for (int j=0; j < N; ++j)
        {
            index(tile.data, tile_i + i, tile_j + j) = value.data[i][j];
        }
    }
}


template <typename T>
inline CUDA_CALLABLE void tile_matmul(const array_t<T>& A, const array_t<T>& B, const array_t<T>& out)
{   
    const int TILE_M = 4;
    const int TILE_N = 4;
    const int TILE_K = 4;

    partition_t A_tile = partition_t<TILE_M, TILE_K, T>(A);
    partition_t B_tile = partition_t<TILE_K, TILE_N, T>(B);
    partition_t C_tile = partition_t<TILE_M, TILE_N, T>(out);

    const int length = partition_size(C_tile);

    __pipeline_wait_prior(0);

    WP_TILE_SYNC();

    for (int t=threadIdx.x; t < length; t += blockDim.x)
    {  
        int i, j;
        partition_coord(C_tile, t, i, j);

        // accumulator
        mat_t<TILE_M, TILE_N, T> sum = partition_load(C_tile, i, j);

        WP_PRAGMA_UNROLL
        for (int k=0; k < A_tile.shape[1]; k++)
        {
            const mat_t<TILE_M, TILE_K, T> a = partition_load(A_tile, i, k);
            const mat_t<TILE_K, TILE_N, T> b = partition_load(B_tile, k, j);

            sum += mul(a, b);
        }
        
        partition_store(C_tile, i, j, sum);
    }

    WP_TILE_SYNC();
}



// 2D gemm accumulate out += A*B
template <typename T>
inline CUDA_CALLABLE void tile_matmul_scalar(const array_t<T>& A, const array_t<T>& B, const array_t<T>& out)
{    
    const int length = out.shape[0]*out.shape[1];

    WP_TILE_SYNC();

    const T* __restrict__ A_ptr = A.data;
    const T* __restrict__ B_ptr = B.data;
    T* __restrict__ C_ptr = out.data;

    WP_PRAGMA_UNROLL
    for (int t=threadIdx.x; t < length; t += blockDim.x)
    {  
        // compute output index
        const int i = t/out.shape[1];
        const int j = t%out.shape[1];

        T sum(0.0);

        WP_PRAGMA_UNROLL
        for (int k=0; k < A.shape[1]; ++k)
        {
            T a = index(A_ptr, i, k, A.shape[1]);
            T b = index(B_ptr, k, j, B.shape[1]);

            sum = fmaf(a, b, sum);
        }
        
        index(C_ptr, i, j, out.shape[1]) += sum;
    }

    WP_TILE_SYNC();
}



} // namespace wp