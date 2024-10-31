/** Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "builtin.h"

#define USE_CUTE 0

#if USE_CUTE
#include "cutlass/include/cute/tensor.hpp"
#include "cutlass/include/cute/algorithm/cooperative_gemm.hpp"
#endif // USE_CUTE

namespace wp
{

/*
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
*/

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


#if !USE_CUTE

template <typename T>
inline CUDA_CALLABLE void gemm(const array_t<T>& A, const array_t<T>& B, const array_t<T>& out)
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
template <typename TileA, typename TileB, typename TileC>
inline CUDA_CALLABLE void tile_matmul_scalar(const TileA& A,
                                             const TileB& B,
                                             TileC& out)
{    
    const int length = tile_size(out);

    WP_TILE_SYNC();

    using T = typename TileA::Type;

    WP_PRAGMA_UNROLL
    for (int t=threadIdx.x; t < length; t += WP_TILE_BLOCK_DIM)
    {  
        // compute output index
        const int i = t/out.N;
        const int j = t%out.N;

        T sum(0.0);

        WP_PRAGMA_UNROLL
        for (int k=0; k < A.N; ++k)
        {
            T a = A(i,k);
            T b = B(k,j);

            sum += a*b; // todo: use fmaf() 
        }
        
        out(i,j) += sum;
    }

    WP_TILE_SYNC();
}

#else


template <typename T>
inline CUDA_CALLABLE void tile_matmul(const array_t<T>& A, const array_t<T>& B, const array_t<T>& out)
{
	using namespace cute;

    __pipeline_wait_prior(0);

    // ensure smem tile is ready
 	WP_TILE_SYNC();

	// Define CTA matrix size (static)
	auto bM = Int<64>{};
	auto bN = Int<64>{};
	auto bK = Int<8>{};

	// Define the smem layouts (static)
	auto sA = make_layout(make_shape(bM, bK), LayoutRight{});  
	auto sB = make_layout(make_shape(bN, bK));   
	auto sC = make_layout(make_shape(bM, bN), LayoutRight{});

    Tensor s_a_tensor = make_tensor(make_smem_ptr<float>(A.data), sA);
    Tensor s_b_tensor = make_tensor(make_smem_ptr<float>(B.data), sB);
    Tensor s_c_tensor = make_tensor(make_smem_ptr<float>(out.data), sC);


    // TiledMMA tiled_mma = make_tiled_mma(UniversalFMA<float,float,float>{},
    //                              Layout<Shape<_16,_8,_1>>{});  // 16x8x1 UniversalFMA, assumes blockDim=128


    // TiledMMA tiled_mma = make_tiled_mma(UniversalFMA<float,float,float>{},
    //                                     Layout<Shape<_8,_16>,Stride<_16,_1>>{});  // 8x16x1 UniversalFMA, assumes blockDim=128



    TiledMMA tiled_mma = make_tiled_mma(UniversalFMA<float,float,float>{},
                                        Layout<Shape<_2,_64>,Stride<_64,_1>>{});  // 8x16x1 UniversalFMA, assumes blockDim=128


    cooperative_gemm< AutoVectorizingCopyWithAssumedAlignment<sizeof_bits_v<float>>,
                      AutoVectorizingCopyWithAssumedAlignment<sizeof_bits_v<float>>, 
                      AutoVectorizingCopyWithAssumedAlignment<sizeof_bits_v<float>>
                    >(
      threadIdx.x, tiled_mma,
      1.0f, s_a_tensor, s_b_tensor, 1.0f, s_c_tensor,
      cute::identity(), cute::identity(), cute::identity(), cute::identity()
    );

    WP_TILE_SYNC();

}

#endif // USE_CUTE


#if 0

template <typename TileA, typename TileB, typename TileC>
void tile_matmul(TileA& a, TileB& b, TileC& c)
{
    static_assert(wp::is_same<typename TileA::Type, typename TileB::Type>::value, "Error, tile datatypes must match");
    static_assert(TileA::N == TileB::M, "Error, inner dimensions must match");
    static_assert(TileC::M == TileA::M, "Error, first output dimension must match");
    static_assert(TileC::N == TileB::N, "Error, second output dimension must match");
   
    tile_matmul_scalar(a, b, c);
}


template <typename TileA, typename TileB, typename TileC,
          typename AdjTileA, typename AdjTileB, typename AdjTileC>
void adj_tile_matmul(TileA& a, TileB& b, TileC& c,
                     AdjTileA& adj_a, AdjTileB& adj_b, AdjTileC& adj_c)
{
    tile_matmul_scalar(adj_c, wp::tile_transpose(b), adj_a);
    tile_matmul_scalar(wp::tile_transpose(a), adj_c, adj_b);
}

#endif // 0

} // namespace wp
