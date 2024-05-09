#pragma once

#include "builtin.h"

#include "cuda_pipeline_primitives.h"

//#include "cutlass/include/cute/tensor.hpp"

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

#if 0
template <class TA, class ASmemLayout, class AThreadLayout,
          class TB, class BSmemLayout, class BThreadLayout,
          class TC, class CSmemLayout, class CThreadLayout>

CUDA_CALLABLE inline void
gemm_device(TA const* smemA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* smemB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * smemC, CSmemLayout sC_layout, CThreadLayout tC)
{
	using namespace cute;

	static_assert(is_static<AThreadLayout>::value);
	static_assert(is_static<BThreadLayout>::value);
	static_assert(is_static<CThreadLayout>::value);


	static_assert(is_static<ASmemLayout>::value);
	static_assert(is_static<BSmemLayout>::value);
	static_assert(is_static<CSmemLayout>::value);


	Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
	Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)
	Tensor sC = make_tensor(make_smem_ptr(smemC), sC_layout);            // (BLK_M,BLK_K)

	
	Tensor tAsA = local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K)
	Tensor tBsB = local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)


	// Partition sA (M,K) by the rows of tC
	Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<X,_1>{});   // (THR_M,BLK_K)
	// Partition sB (K,M) by the rows of tC
	Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<_1, X>{});   // (THR_N,BLK_K)

	// Partition gC (M,N) by the tile of tC
	Tensor tCsC = local_partition(sC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

	// Allocate the accumulators -- same shape/layout as the partitioned data
	Tensor tCrC = make_tensor_like(tCsC);                                // (THR_M,THR_N)

	//*******************
	// MM-QUESTION: this is not quite right, we need a 3d shape, but should we use local_partition or local_tile?
	auto K_TILE_MAX = 1;//size<2>(tAsA);

	// ensure smem is ready
	__syncthreads();

	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		print(sA); printf("\n");
		print(sB); printf("\n");
		print(sC); printf("\n");

		print(tCsA); printf("\n");
		print(tCsB); printf("\n");
		print(tCsC); printf("\n");
	}

	for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
	{
		// Copy gmem to smem with tA|tB thread-partitioned tensors
		// copy(tAgA(_,_,k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
		// copy(tBgB(_,_,k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)

		//*******************
		// MM-QUESTION: how to 'advance' tCsA and tCsB to next tile in smem instead of above copy from global?
		gemm(tCsA, tCsB, tCrC);
	}

	CUTE_UNROLL
	for (int i = 0; i < size(tCsA); ++i) {
		tCsC(i) += tCrC(i);
	}

	// ensure writes to shared are visible
    __syncthreads();         
}

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
    
    // cooperatively load the tile, using a block-stride iterator
    // todo: use cub::BlockLoad or cg::memcpy_async()?

    // WP_PRAGMA_UNROLL
    // for (int t=threadIdx.x; t < length; t += blockDim.x)
    // {  
    //     data[t] = index(src, i*M + t/N, j*N + t%N);
    // }

    // // async copies
    WP_PRAGMA_UNROLL
    for (int t=threadIdx.x*4; t < length; t += blockDim.x*4)
    {  
        //data[t] = index(src, i*M + t/N, j*N + t%N);
        __pipeline_memcpy_async(&data[t],
                                &index(src, i*M + t/N, j*N + t%N),
                                sizeof(T)*4);
    }

    __pipeline_commit();
    __pipeline_wait_prior(0);

        
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
    WP_PRAGMA_UNROLL
    for (int t=threadIdx.x; t < length; t += blockDim.x)
    {  
        index(dest, i*M + t/N, j*N + t%N) = src.data[t];
    }
}

// template <typename T>
// inline CUDA_CALLABLE void tile_matmul_cute(const array_t<T>& A, const array_t<T>& B, const array_t<T>& out)
// {
// 	using namespace cute;

// 	// Define CTA matrix size (static)

// 	auto bM = Int<64>{};
// 	auto bN = Int<64>{};
// 	auto bK = Int<8>{};

// 	auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

// 	// Define the smem layouts (static)
// 	auto sA = make_layout(make_shape(bM,bK), LayoutRight{});  
// 	auto sB = make_layout(make_shape(bN,bK));   
// 	auto sC = make_layout(make_shape(bM, bN), LayoutRight{});

// 	// Define the thread layouts (static)
// 	auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});  
// 	auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});  
// 	auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}), LayoutRight{});  

//   gemm_device
//       (A.data, sA, tA,
//        B.data, sB, tB,
//        out.data,sC, tC);
// }


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
	partition_t(array_t<T> A)
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
int partition_size(const partition_t<M, N, T>& tile)
{
	return tile.shape[0]*tile.shape[1];
}

// returns the x, y coordinates of a tile given a linear index
template <unsigned M, unsigned N, typename T>
void partition_coord(const partition_t<M, N, T>& tile, const int t, int& i, int& j)
{
	i = t/tile.shape[1];
	j = t%tile.shape[1];
}

template <unsigned M, unsigned N, typename T>
mat_t<M, N, T> partition_load(const partition_t<M, N, T>& tile, int i, int j)
{
	mat_t<M, N, T> out;
	
	const int tile_i = i*M;
	const int tile_j = j*N;

	// WP_PRAGMA_UNROLL
	// for (int i=0; i < M; ++i)
	// {
	// 	WP_PRAGMA_UNROLL
	// 	for (int j=0; j < N; ++j)
	// 	{
	// 		out.data[i][j] = index(tile.data, tile_i + i, tile_j + j);
	// 	}
	// }
	

	return out;
}

template <unsigned M, unsigned N, typename T>
void partition_store(const partition_t<M, N, T>& tile, int i, int j, const mat_t<M, N, T>& value)
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

    WP_TILE_SYNC();

    WP_PRAGMA_UNROLL
    for (int t=threadIdx.x; t < length; t += blockDim.x)
    {  
		int i, j;
		partition_coord(C_tile, t, i, j);

		// accumulator
		mat_t<TILE_M, TILE_N, T> sum = partition_load(C_tile, i, j);

        WP_PRAGMA_UNROLL
        for (int k=0; k < A_tile.shape[1]; ++k)
        {
			mat_t<TILE_M, TILE_K, T> a = partition_load(A_tile, i, k);
			mat_t<TILE_K, TILE_M, T> b = partition_load(B_tile, k, j);

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