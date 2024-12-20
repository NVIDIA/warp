/** Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "builtin.h"

#if !defined(__CUDA_ARCH__)
#define WP_TILE_SHARED static
#define WP_TILE_SYNC void
#else
#define WP_TILE_SHARED __shared__
#define WP_TILE_SYNC __syncthreads
#endif

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

#define WP_USE_ASYNC_PIPELINE 0
#if WP_USE_ASYNC_PIPELINE
#include "cuda_pipeline_primitives.h"
#endif // WP_USE_ASYNC_PIPELINE

#define WP_USE_REGISTER_GEMM 0

/* Tile Expressions

[ ] Tiles
    [x] Register, Shared, Global
    [ ] Layouts
        [x] Simple
        [ ] Cute
    [x] Remove Alloc type from tile_shared_t
    [x] wp.launch_tiled() helper
[ ] Creation
    [x] zeros
    [x] ones
    [x] arange
    [x] tile()
    [x] untile()
    [ ] fromfunction()
    [ ] explicit storage
[ ] Load/Store
    [ ] 1D load/store variants
    [ ] max_coord option for non-aligned loads
    [ ] Indexed load
    [x] wp.tile_atomic_add()
[ ] Maps
    [x] Support user functions
    [x] Support built-in functions
    [ ] Support for lambda functions
    [ ] Infer tile_map() output from operator type (e.g.: dot for each element)
[ ] Reductions
    [x] Sum
        [x] Forward
        [x] Reverse
    [x] Min
    [x] Max
    [x] Custom
[x] MatMul
    [x] Forward
    [x] Reverse
[ ] Operators
    [ ] +, -, *, /, @?
    [ ] += for matmul, e.g.: c += a@b, or c = a@b
[ ] Reshape
    [ ] Broadcasting
    [ ] Transpose
        [x] Shared
        [ ] Register
    [ ] Slice
[ ] Runtime
    [x] Compile-time block dimensions
    [x] Switch between SIMT / Tile based execution if `block_dim` not provided to wp.launch()
[ ] Examples
    [ ] Point registration
    [ ] GEMM
    [ ] MLP
    [ ] LayerNorm  
    [ ] SoftMax
    [ ] GEMM
    [ ] warp.sim (CRBA)
    [ ] Batched MLP
    [ ] Layer norm
    [ ] FNO + Burgers equation
    [ ] Stochastic financial modeling
    [ ] Convolution: https://github.com/NVIDIA/MinkowskiEngine/blob/master/src/convolution_kernel.cu#L123
    [ ] MeshCNN (Modulus, Oliver)
    [ ] BioNemo (Ali)
    [ ] Skinning (David/Or/Vismay)
    [ ] warp.sim (VBD)
[ ] Error checking
    [ ] Ensure functions passed to tile_map() are compatible with tile type
    [ ] Ensure that args passed to tile ops are compatible
    [ ] Ensure tile load/store operations don't go out of bounds of arrays in debug mode

*/

/*
Notes on shared memory synchronization
======================================

Currently operations that write to shared memory tiles (e.g.: tile_load())
must synchronize before they return through WP_TILE_SYNC(), this
ensures subsequent read operations from the tile do not cause a race condition.

For tile_shared_t adjoints, the gradient accumulation is done through shared
memory atomics, i.e.: atomic_add(), since for broadcast tiles multiple threads
may map to the same location. Synchronization is still required after these 
updates, since subsequent operations e.g.: adj_tile_load() will store the
gradients to memory, and all updates must be visible at that point, e.g.:

    a = wp.tile_load(...)
    b = wp.tile_load(...)
    c = wp.tile_matmul(a, b)
    wp.tile_store(c)

    // loads incoming adjoints from global -> shared
    wp.adj_tile_store(c, adj_c)
    // consumes adj_c, requires synchronization
    wp.adj_tile_matmul(a, b, adj_a, adj_b, adj_c)
    // consumes adj_b, requires synchronization
    wp.adj_tile_load(..., adj_b)
    // consumes adj_b, requires synchronization
    wp.adj_tile_load(..., adj_a)

Generally synchronization to adjoint tiles will happen through the
tile_shared_t::add() and tile_shared_t::assign() function automatically,
but in some cases e.g.: tile_matmul() it is done manually.

The current synchronization strategy is conservative, and can lead to more
synchronization than necessary. A more sophisticated strategy would be
to track the 'dirty' state of shared tiles, and synchronize only when
necessary. In addition, custom synchronization for e.g.: tile_load()
operations could be added through a SyncProvider template parameter on
the tile_shared_t type, for example to support barrier synchronization
for asynchronous global to shared loads.
*/

namespace wp
{

// Primary template
template <typename T, typename U>
struct is_same {
    static constexpr bool value = false;
};

// Specialization for the case when T and U are the same type
template <typename T>
struct is_same<T, T> {
    static constexpr bool value = true;
};


template <typename Tile>
constexpr int tile_size(Tile& t) { return Tile::M*Tile::N; }

constexpr int tile_regcount(int m, int n) {
    return (m*n + WP_TILE_BLOCK_DIM - 1) / WP_TILE_BLOCK_DIM;
}

struct coord_t
{
    int i;
    int j;
};


// represents a tile stored in global memory with dynamic strides
// only used to represent the source for tile loads to register/shared
template <typename T>
struct tile_global_t
{
    using Type = T;

    array_t<T> data;
    int x;
    int y;

    tile_global_t(array_t<T>& a, int x, int y) : data(a), x(x), y(y)
    {
    }
};

// represents a tile stored in registers across a block
template <typename T, int M_, int N_>
struct tile_register_t
{
    using Type = T;
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int Size = M*N;

    static constexpr int NumRegs = tile_regcount(M, N);

    static constexpr bool Aligned = Size%WP_TILE_BLOCK_DIM == 0;

    T data[NumRegs];
   
    inline CUDA_CALLABLE tile_register_t(T value=T(0.0)) 
    {
        // zero-initialize by default necessary for tile adjoints
        // need to check if this results in worse codegen
        // than doing adj_var = tile_zeros() explicitly
        // in backwards pass and letting default constructor
        // avoid initialization
        
        for (int i=0; i < NumRegs; ++i)
            data[i] = value;
    }

    inline CUDA_CALLABLE auto& operator=(const tile_global_t<T>& t)
    {
        if (t.data.ndim == 1)
            copy_from_global(t.data, t.x); // 1d load
        else
            copy_from_global(t.data, t.x, t.y); // 2d load
       
        return *this;

    }

    // define the += operator which is used during backward pass codegen
    // when returning a register tile from a user defined function
    inline CUDA_CALLABLE auto& operator += (tile_register_t<T, M, N>& rhs) 
    {
        this->grad_add(rhs);
        return *this;
    }

    inline CUDA_CALLABLE T& operator()(int index)
    {
        assert(index < NumRegs);
        return data[index];
    }    

    inline CUDA_CALLABLE const T& operator()(int index) const
    {
        assert(index < NumRegs);
        return data[index];
    }


    // compute linear tile index from a local register index
    inline CUDA_CALLABLE int index(int reg) const
    {
        return threadIdx.x + reg*WP_TILE_BLOCK_DIM;
    }

    // compute tile coordinate from linear index
    inline CUDA_CALLABLE coord_t coord(int index) const
    {
        return {index/N, index%N};
    }

    // Returns the number of valid registers for this tile
    // i.e.: how many registers map to a valid coordinate.
    // When a tile's size is not aligned to the block dimension
    // some of the trailing registers may lie outside the valid range
    inline CUDA_CALLABLE int valid() const
    {
        return (Size - threadIdx.x)/WP_TILE_BLOCK_DIM;
    }    

    inline CUDA_CALLABLE void assign(const tile_register_t<T, M, N>& tile) 
    { 
        for (int i=0; i < NumRegs; ++i)
            data[i] = tile.data[i];
    }

    inline CUDA_CALLABLE void zero()
    {
        for (int i=0; i < NumRegs; ++i)
            data[i] = T(0);        
    }

    // extract a single tile element to a native type
    inline CUDA_CALLABLE Type extract(int i, int j)
    {
        // map from logical coords (i, j) -> (thread, reg)
        const int linear = i*N + j;

        const int thread = linear/NumRegs;
        const int reg = linear%NumRegs;

        WP_TILE_SHARED Type scratch;

        // ensure any previously scheduled threads have finished reading from scratch
        WP_TILE_SYNC();

        if (threadIdx.x == thread)
        {
            scratch = data[reg];
        }

        // ensure extraction thread has updated smem
        WP_TILE_SYNC();

        return scratch;
    }
    

    // backward version of scalar extract
    inline CUDA_CALLABLE void adj_extract(int i, int j, Type adj_ret)
    {
        // map from logical coords (i, j) -> (thread, reg)
        const int linear = i*N + j;

        const int thread = linear/NumRegs;
        const int reg = linear%NumRegs;

        if (threadIdx.x == thread)
        {
            data[reg] += adj_ret;
        }
    }

    inline CUDA_CALLABLE void print() const;


    // return the in-register version of this tile (nop)
    inline CUDA_CALLABLE auto& copy_to_register() 
    {
        return *this; 
    }

    inline CUDA_CALLABLE const auto& copy_to_register() const 
    {
        return *this; 
    }

    // in-place gradient zero
    inline CUDA_CALLABLE void grad_zero()
    {
        zero();
    }

    // accumulate gradients onto this tile
    inline CUDA_CALLABLE void grad_add(const tile_register_t<T, M, N>& tile) 
    { 
        for (int i=0; i < NumRegs; ++i)
            data[i] += tile.data[i];
    }

    // copy shared tile to register
    inline CUDA_CALLABLE auto& grad_to_register() 
    {
        return *this;
    }

    void copy_to_global(array_t<T> dest, int x)
    {
        assert(dest.ndim == 1);

        const int tile_i = x*N;

        WP_PRAGMA_UNROLL
        for (int i=0; i < NumRegs; ++i)
        {
            // handle case where tile size is not 
            // aligned to block dimensions
            int linear = index(i);
            if (!Aligned && linear >= Size)
                break;

            wp::index(dest, tile_i + linear) = data[i];
        }
    }

    void copy_to_global(array_t<T> dest, int x, int y)
    {
        assert(dest.ndim == 2);

        const int tile_i = x*M;
        const int tile_j = y*N;

        // wp.array() indexing generates poor code due to char* casting
        // here we unroll some of the ops, note this assumes byte strides are 
        // aligned to the element size
        T* ptr = &wp::index(dest, tile_i, tile_j);
        const int stride_i = dest.strides[0]/sizeof(T);
        const int stride_j = dest.strides[1]/sizeof(T);

        WP_PRAGMA_UNROLL
        for (int i=0; i < NumRegs; ++i)
        {
            // handle case where tile size is not 
            // aligned to block dimensions
            int linear = index(i);
            if (!Aligned && linear >= Size)
                break;

            coord_t c = coord(linear);
            ptr[c.i*stride_i + c.j*stride_j] = data[i]; 
        }
    }

    inline CUDA_CALLABLE void copy_from_global(const array_t<T>& src, int x)
    {
        // todo: use async pipelines or TMA here
        const int tile_i = x*N;

        WP_PRAGMA_UNROLL
        for (int i=0; i < NumRegs; ++i)
        {  
            int linear = index(i);
            if (!Aligned && linear >= Size)
                break;

            data[i] = wp::index(src, tile_i + linear);
        }
    }

    inline CUDA_CALLABLE void copy_from_global(const array_t<T>& src, int x, int y)
    {
        // todo: use async pipelines or TMA here
        const int tile_i = x*M;
        const int tile_j = y*N;

        // wp.array() indexing generates poor code due to char* casting
        // here we unroll some of the ops, note this assumes array byte strides are 
        // aligned to the element size
        const T* ptr = &wp::index(src, tile_i, tile_j);

        assert(src.strides[0]%sizeof(T) == 0);
        assert(src.strides[1]%sizeof(T) == 0);

        const int stride_i = src.strides[0]/sizeof(T);
        const int stride_j = src.strides[1]/sizeof(T);

        WP_PRAGMA_UNROLL
        for (int i=0; i < NumRegs; ++i)
        {  
            int linear = index(i);
            if (!Aligned && linear >= Size)
                break;

            coord_t c = coord(linear);
            data[i] = ptr[c.i*stride_i + c.j*stride_j];
        }
    }
};

// helper to allocate a register tile like another tile
template<typename Tile>
auto tile_register_like()
{
    using T = typename Tile::Type;

    return tile_register_t<T, Tile::M, Tile::N>(T(0.0));
}

inline CUDA_CALLABLE int tile_align(int num_bytes)
{
    // note this much match value in Python types.py
    const int alignment = 16;

    return ((num_bytes + alignment - 1) / alignment) * alignment;
}

inline CUDA_CALLABLE void* tile_alloc_shared(int num_bytes, bool init=false)
{
    // we maintain a per-thread offset into dynamic
    // shared memory that allows us to keep track of 
    // current use across dynamic function calls
    __shared__ int smem_base[WP_TILE_BLOCK_DIM];

    if (init)
    {
        smem_base[threadIdx.x] = 0;
        return NULL;
    }
    else
    {
        const int offset = smem_base[threadIdx.x];
        
        // one entry per-thread so no need for synchronization
        smem_base[threadIdx.x] += tile_align(num_bytes);

        extern __shared__ char dynamic_smem_base[];
        return &(dynamic_smem_base[offset]);
    }
}



template <typename T, int M_, int N_, int StrideM_=N_, int StrideN_=1, bool Owner_=true>
struct tile_shared_t
{
    using Type = T;
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int Size = M*N;
    
    static constexpr int StrideM = StrideM_;
    static constexpr int StrideN = StrideN_;

    static constexpr bool Aligned = Size%WP_TILE_BLOCK_DIM == 0;
    static constexpr bool Unique = (StrideM >= N) && (StrideN >= 1);
    static constexpr bool Owner = Owner_;

    struct Storage
    {
        T* ptr;

        Storage(T* p) : ptr(p) {}

        inline CUDA_CALLABLE T& operator()(int i, int j)
        {
            assert(i < M);
            assert(j < N);

            return ptr[i*StrideM + j*StrideN];
        }

        inline CUDA_CALLABLE const T& operator()(int i, int j) const
        {
            assert(i < M);
            assert(j < N);

            return ptr[i*StrideM + j*StrideN];
        }

        inline CUDA_CALLABLE T& operator()(int index)
        {
            assert(index < M*N);

            // unravel
            int i = index/N;
            int j = index%N;

            return (*this)(i,j);
        }    

        inline CUDA_CALLABLE const T& operator()(int index) const
        {
            assert(index < M*N);

            // unravel
            int i = index/N;
            int j = index%N;

            return (*this)(i,j);
        }            
    };

    Storage data;
    Storage grad;

    // default initialization (non-initialized)
    inline CUDA_CALLABLE tile_shared_t() : data(NULL), grad(NULL) 
    {
    }

    // initialize from an existing tile's memory
    inline CUDA_CALLABLE tile_shared_t(T* data, T* grad=NULL) : data(data), grad(grad)
    {
    }

    inline CUDA_CALLABLE ~tile_shared_t()
    {
        if (Owner)
        {
            // update our per-thread shared memory allocator
            if (data.ptr)
                tile_alloc_shared(-M*N*int(sizeof(T)));

            if (grad.ptr)
                tile_alloc_shared(-M*N*int(sizeof(T)));
        }
    }

    // assign from a register tile
    template <typename Tile>
    inline CUDA_CALLABLE auto& operator=(const Tile& t)
    {
        assign(t);
        return *this;
    }

    // construct from another shared tile, this constructor
    // is invoked for reshape operations like `wp.tile_transpose()`
    template <typename OtherT, int OtherM, int OtherN, int OtherStrideM, int OtherStrideN>
    inline CUDA_CALLABLE auto& operator=(const tile_shared_t<OtherT, OtherM, OtherN, OtherStrideM, OtherStrideN>& rhs) 
    {
        using OtherTile = tile_shared_t<OtherT, OtherM, OtherN, OtherStrideM, OtherStrideN>;

        // check dimensions are compatible
        static_assert(Size == OtherTile::Size);

        // alias tile directly
        data = rhs.data;
        grad = rhs.grad;

        return *this;
    }    

    // assign from a global tile (load)
    inline CUDA_CALLABLE auto& operator=(const tile_global_t<T>& t)
    {        
        if (t.data.ndim == 1)
            copy_from_global(t.data, t.x);  // 1d load
        else
            copy_from_global(t.data, t.x, t.y); // 2d load
        
        // synchronization happens in copy functions above

        return *this;
    }

    // assign from a constant value
    inline CUDA_CALLABLE auto& operator=(const T& x)
    {
        for (int i=threadIdx.x; i < M*N; i+= WP_TILE_BLOCK_DIM)
            data(i) = x;

        WP_TILE_SYNC();
        return *this;
    }

    
    // compute tile coordinate from linear index
    inline CUDA_CALLABLE coord_t coord(int index) const
    {
        return {index/N, index%N};
    }

    // in-place zero
    inline CUDA_CALLABLE void zero()
    {
        for (int i=threadIdx.x; i < M*N; i+= WP_TILE_BLOCK_DIM)
            data(i) = T(0);

        WP_TILE_SYNC();
    }

    // extract a single tile element to a native type
    inline CUDA_CALLABLE Type extract(int i, int j)
    {
        return data(i, j);
    }
        
    // backward of scalar extraction
    inline CUDA_CALLABLE void adj_extract(int i, int j, Type adj_ret)
    {
        if (threadIdx.x == 0)
            data(i, j) += adj_ret;

        WP_TILE_SYNC();
    }


    // copy register tile to shared
    inline CUDA_CALLABLE void assign(const tile_register_t<T, M, N>& tile)
    { 
        WP_PRAGMA_UNROLL
        for (int i=0; i < tile.NumRegs; ++i)
        {
            const int linear = tile.index(i);

            // handle case where tile size is not
            // aligned to block dimensions
            if (!Aligned && linear >= Size)
                break;

            data(linear) = tile.data[i];
        }

        WP_TILE_SYNC();
    }

    // in-place gradient zero
    inline CUDA_CALLABLE void grad_zero()
    {
        // todo: make this subtile (stride aware)
        for (int i=threadIdx.x; i < M*N; i+= WP_TILE_BLOCK_DIM)
            grad(i) = T(0);

        WP_TILE_SYNC();
    }


    // accumulate gradients onto this tile
    inline CUDA_CALLABLE void grad_add(const tile_register_t<T, M, N>& tile) 
    { 
        WP_PRAGMA_UNROLL
        for (int i=0; i < tile.NumRegs; ++i)
        {
            const int linear = tile.index(i);

            // handle case where tile size is not
            // aligned to block dimensions
            if (!Aligned && linear >= Size)
                break;

            if (Unique)
                grad(linear) += tile.data[i];
            else
                // use shared memory atomics to accumulate gradients
                // since for broadcast tiles (e.g.: a bias vector) multiple incoming threads 
                // may map to a single location in shared memory
                atomic_add(&grad(linear), tile.data[i]);
            
        }

        WP_TILE_SYNC();
    }

    // copy shared tile to register
    inline CUDA_CALLABLE tile_register_t<T, M, N> grad_to_register() 
    { 
        tile_register_t<T, M, N> out;

        WP_PRAGMA_UNROLL
        for (int i=0; i < out.NumRegs; ++i)
        {
            const int linear = out.index(i);

            // handle case where tile size is not
            // aligned to block dimensions
            if (!Aligned && linear >= Size)
                break;

            out(i) = grad(linear);
        }

        return out;
    }

    inline CUDA_CALLABLE void print() const
    {
        if (threadIdx.x == 0)
        {
            printf("tile(m=%d, n=%d, storage=shared) = [", M, N);
            for (int i=0; i < M; ++i)
            {
                printf("%*s[", i>0, "");
                for (int j=0; j < N; ++j)
                {
                    printf("%g ", double(data(i, j)));
                }

                if (i == M-1)
                    printf("]]\n");
                else
                    printf("]\n");
            }
        }
    }

    // copy shared tile to register
    inline CUDA_CALLABLE tile_register_t<T, M, N> copy_to_register() const
    { 
        tile_register_t<T, M, N> out;

        WP_PRAGMA_UNROLL
        for (int i=0; i < out.NumRegs; ++i)
        {
            const int linear = out.index(i);

            // handle case where tile size is not
            // aligned to block dimensions
            if (!Aligned && linear >= Size)
                break;

            out(i) = data(linear);
        }

        return out;
    }

    inline CUDA_CALLABLE void copy_to_global(array_t<T> dest, int x) const
    {
        assert(dest.ndim == 1);

        // todo: use TMA here
        const int tile_i = x*N;

        WP_PRAGMA_UNROLL
        for (int i=threadIdx.x; i < Size; i += WP_TILE_BLOCK_DIM)
        {
            wp::index(dest, tile_i + i) = data(i);
        }
    }

    inline CUDA_CALLABLE void copy_to_global(array_t<T> dest, int x, int y)
    {
        // todo: use TMA here
        const int tile_i = x*M;
        const int tile_j = y*N;

        // check each row is contiguous and 128bit aligned
        if (StrideN == 1 && dest.strides[1] == sizeof(T) && (N*sizeof(T))%sizeof(float4) == 0)
        {            
            constexpr int num_rows = M;
            constexpr int num_cols = (N*sizeof(T))/sizeof(float4);

            tile_shared_t<float4, num_rows, num_cols> src128((float4*)data.ptr);

            // alias of shared tile with 128bit type
            float4* ptr = (float4*)&wp::index(dest, tile_i, tile_j);

            assert(((uint64_t)(data.ptr))%sizeof(float4) == 0);
            assert(((uint64_t)(ptr))%sizeof(float4) == 0);

            const int stride_i = dest.strides[0]/sizeof(float4);
            const int stride_j = 1;

            WP_PRAGMA_UNROLL
            for (int i=threadIdx.x; i < src128.Size; i += WP_TILE_BLOCK_DIM)
            {  
                coord_t c = src128.coord(i);
                ptr[c.i*stride_i + c.j*stride_j] = src128.data(i);
            }
        }
        else
        {
            // wp.array() indexing generates poor code due to char* casting
            // here we unroll some of the ops, note this assumes byte strides are 
            // aligned to the element size
            T* ptr = &wp::index(dest, tile_i, tile_j);
            const int stride_i = dest.strides[0]/sizeof(T);
            const int stride_j = dest.strides[1]/sizeof(T);    

            WP_PRAGMA_UNROLL
            for (int i=threadIdx.x; i < Size; i += WP_TILE_BLOCK_DIM)
            {
                coord_t c = coord(i);
                ptr[c.i*stride_i + c.j*stride_j] = data(c.i, c.j);
            }
        }
    }

    inline CUDA_CALLABLE void copy_from_global(const array_t<T>& src, int x)
    {
        // todo: use async pipelines or TMA here
        const int tile_i = x*N;

        WP_PRAGMA_UNROLL
        for (int i=threadIdx.x; i < Size; i += WP_TILE_BLOCK_DIM)
        {  
            data(i) = wp::index(src, tile_i + i);
        }

        WP_TILE_SYNC();
    }

    inline CUDA_CALLABLE void copy_from_global(const array_t<T>& src, int x, int y)
    {
        // todo: use async pipelines or TMA here
        const int tile_i = x*M;
        const int tile_j = y*N;

        // check each row is contiguous and 128bit aligned
        if (StrideN == 1 && src.strides[1] == sizeof(T) && (N*sizeof(T))%sizeof(float4) == 0)
        {            
            constexpr int num_rows = M;
            constexpr int num_cols = (N*sizeof(T))/sizeof(float4);

            // alias of shared tile with 128bit type
            tile_shared_t<float4, num_rows, num_cols> dest128((float4*)data.ptr);

            const float4* ptr = (const float4*)&wp::index(src, tile_i, tile_j);

            assert(((uint64_t)(data.ptr))%sizeof(float4) == 0);
            assert(((uint64_t)(ptr))%sizeof(float4) == 0);

            const int stride_i = src.strides[0]/sizeof(float4);
            //const int stride_j = 1;

            WP_PRAGMA_UNROLL
            for (int i=threadIdx.x; i < dest128.Size; i += WP_TILE_BLOCK_DIM)
            {  
                coord_t c = dest128.coord(i);
                
#if WP_USE_ASYNC_PIPELINE
                __pipeline_memcpy_async(&dest128.data(i),
                        &ptr[c.i*stride_i + c.j],
                        sizeof(float4));
#else
                dest128.data(i) = ptr[c.i*stride_i + c.j];
#endif // WP_USE_ASYNC_PIPELINE
            }

#if WP_USE_ASYNC_PIPELINE
            __pipeline_commit();
#endif // WP_USE_ASYNC_PIPELINE

        }
        else
        {
            // wp.array() indexing generates poor code due to char* casting
            // here we unroll some of the ops, note this assumes array byte strides are 
            // aligned to the element size
            const T* ptr = &wp::index(src, tile_i, tile_j);

            assert(src.strides[0]%sizeof(T) == 0);
            assert(src.strides[1]%sizeof(T) == 0);

            const int stride_i = src.strides[0]/sizeof(T);
            const int stride_j = src.strides[1]/sizeof(T);

            WP_PRAGMA_UNROLL
            for (int i=threadIdx.x; i < Size; i += WP_TILE_BLOCK_DIM)
            {  
                coord_t c = coord(i);
                data(c.i, c.j) = ptr[c.i*stride_i + c.j*stride_j];
            }
        }

#if !WP_USE_ASYNC_PIPELINE
    WP_TILE_SYNC();
#endif

    }
};

template <typename T, int M, int N>
void tile_register_t<T, M, N>::print() const
{
    // create a temporary shared tile so that
    // we can print it deterministically
    WP_TILE_SHARED T smem[M*N];
    
    tile_shared_t<T, M, N> scratch(smem, NULL);
    scratch.assign(*this);

    WP_TILE_SYNC();

    if (threadIdx.x == 0)
    {
        printf("tile(m=%d, n=%d, storage=register) = [", M, N);
        for (int i=0; i < M; ++i)
        {
            printf("%*s[", i>0, "");
            for (int j=0; j < N; ++j)
            {
                printf("%g ", double(scratch.data(i, j)));
            }

            if (i == M-1)
                printf("]]\n");
            else
                printf("]\n");
        }
    }

    WP_TILE_SYNC();
}

template <typename T, int M, int N>
inline CUDA_CALLABLE void print(const tile_register_t<T, M, N>& t)
{
    t.print();
}

template <typename T, int M, int N>
inline CUDA_CALLABLE void adj_print(const tile_register_t<T, M, N>& t, const tile_register_t<T, M, N>& a)
{
    a.print();
}

template <typename T, int M, int N, int StrideM, int StrideN, bool Owner>
inline CUDA_CALLABLE void print(const tile_shared_t<T, M, N, StrideM, StrideN, Owner>& t)
{
    t.print();
}

template <typename T, int M, int N, int StrideM, int StrideN, bool Owner>
inline CUDA_CALLABLE void adj_print(const tile_shared_t<T, M, N, StrideM, StrideN, Owner>& t, const tile_shared_t<T, M, N, StrideM, StrideN, Owner>& a)
{
    a.print();
}

// helpers to allocate shared tiles
template <typename T, int M, int N, bool RequiresGrad>
inline CUDA_CALLABLE auto tile_alloc_empty()

{   constexpr int Len = M*N;
    T* data = (T*)tile_alloc_shared(Len*sizeof(T));
    T* grad = NULL;

#if FP_CHECK

    for (int i=threadIdx.x; i < Len; i+= WP_TILE_BLOCK_DIM)
        data[i] = T(nanf(""));

    WP_TILE_SYNC();

#endif // FP_CHECK


    if (RequiresGrad)
    {
        grad = (T*)tile_alloc_shared(Len*sizeof(T));

        for (int i=threadIdx.x; i < Len; i+= WP_TILE_BLOCK_DIM)
            grad[i] = T(0);

        WP_TILE_SYNC();
    }
       
    return tile_shared_t<T, M, N>(data, grad);
}

template <typename T, int M, int N, bool RequiresGrad>
inline CUDA_CALLABLE auto tile_alloc_zeros()
{
    // compute the total storage required for the tile (may be different from M*N) for broadcast tiles
    constexpr int Len = M*N;
    T* data = (T*)tile_alloc_shared(Len*sizeof(T));
    T* grad = NULL;

    for (int i=threadIdx.x; i < Len; i+= WP_TILE_BLOCK_DIM)
        data[i] = T(0);

    if (RequiresGrad)
    {
        grad = (T*)tile_alloc_shared(Len*sizeof(T));

        for (int i=threadIdx.x; i < Len; i+= WP_TILE_BLOCK_DIM)
            grad[i] = T(0);
    }

    WP_TILE_SYNC();

    return tile_shared_t<T, M, N, StrideM, StrideN>(data, grad);
}


//-----------------------------------------------------------------------------------------------------
// High level entry points for each op (correspond to one Warp builtin)

// construct a tile from a local SIMT value (one per-thread)
template <typename T>
inline CUDA_CALLABLE auto tile(const T& x)
{
    tile_register_t<T, 1, WP_TILE_BLOCK_DIM> result;
    
    static_assert(result.NumRegs == 1);

    result.data[0] = x;
    return result;
}

// overload for constructing a tile from a per-thread vector
template <typename T, unsigned Length>
inline CUDA_CALLABLE auto tile(const wp::vec_t<Length, T>& x)
{
    tile_register_t<T, Length, WP_TILE_BLOCK_DIM> result;
    
    static_assert(result.NumRegs == Length);

    for (int i=0; i < Length; ++i)
        result.data[i] = x[i]; 

    return result;
}

// construct a tile from a local SIMT value (one per-thread)
template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile(const T& x, T& adj_x, AdjTile& adj_ret)
{
    static_assert(AdjTile::M == 1);
    static_assert(AdjTile::N == WP_TILE_BLOCK_DIM);
    
    auto adj_reg = adj_ret.copy_to_register();

    adj_x += adj_reg.data[0];
}

template <typename T, unsigned Length, typename AdjTile>
inline CUDA_CALLABLE void adj_tile(const wp::vec_t<Length, T>& x, wp::vec_t<Length, T>& adj_x, AdjTile& adj_ret)
{
    static_assert(AdjTile::M == Length);
    static_assert(AdjTile::N == WP_TILE_BLOCK_DIM);

    auto adj_reg = adj_ret.copy_to_register();

    for (int i=0; i < Length; ++i)
        adj_x[i] += adj_reg.data[i];
}

template <typename Tile>
inline CUDA_CALLABLE auto untile(Tile& tile)
{    
    // code-gen should have set the tile to 
    // have exactly the block dimension so 
    // there is exactly one value per-thread
    auto reg = tile.copy_to_register();

    // scalar case
    if constexpr(Tile::M == 1)
    {
        return reg.data[0];
    }
        
    // vector case
    if constexpr(Tile::M > 1)
    {
        wp::vec_t<Tile::M, typename Tile::Type> v;
        for (int i=0; i < Tile::M; ++i)
            v[i] = reg.data[i];

        return v;
    }
}

template <typename Tile, typename Value>
inline CUDA_CALLABLE void adj_untile(Tile& tile, Tile& adj_tile, Value& adj_ret)
{    
    auto adj = adj_tile.copy_to_register();   
    
    // scalar case
    if constexpr(Tile::M == 1)
    {
        adj.data[0] += adj_ret;
    }

    // vector case
    if constexpr(Tile::M > 1)
    {
        for (int i=0; i < Tile::M; ++i)
            adj.data[i] = adj_ret[i];
    }

    adj_tile.assign(adj);
}

// zero initialized tile
template <typename T, int M, int N>
inline CUDA_CALLABLE auto tile_zeros()
{
    // tile variable assignment operator will handle initialization (since lhs could be shared/register tile)
    return T(0);
}

// one-initialized tile
template <typename T, int M, int N>
inline CUDA_CALLABLE auto tile_ones()
{
    // tile variable assignment operator will handle initialization (since lhs could be shared/register tile)
    return T(1);
}

// tile with evenly spaced values
template <typename T, int M, int N>
inline CUDA_CALLABLE auto tile_arange(T start, T stop, T step)
{
    tile_register_t<T, M, N> out;
    
    WP_PRAGMA_UNROLL
    for (int i=0; i < out.NumRegs; ++i)
    {
        const int linear = out.index(i);

        // handle case where tile size is not
        // aligned to block dimensions
        if (!out.Aligned && linear >= out.Size)
            break;

        out.data[i] = start + linear*step;
    }
    
    return out;
}

template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_arange(T start, T stop, T step,
                                          T& adj_start, T& adj_stop, T& adj_step, AdjTile& adj_ret) {}

// entry point for 1d load
template <typename T, int N>
inline CUDA_CALLABLE auto tile_load(array_t<T>& src, int x)
{
    return tile_global_t<T>(src, x, 0);
}

// entry point for 2d load
template <typename T, int M, int N>
inline CUDA_CALLABLE auto tile_load(array_t<T>& src, int x, int y)
{
    return tile_global_t<T>(src, x, y);
}

// entry point for 1d store
template <typename T, typename Tile>
inline CUDA_CALLABLE void tile_store(array_t<T>& dest, int x, Tile& src)
{
    // dispatch to tile type
    src.copy_to_global(dest, x);
}

// entry point for 2d store
template <typename T, typename Tile>
inline CUDA_CALLABLE void tile_store(array_t<T>& dest, int x, int y, Tile& src)
{
    // dispatch to tile type
    src.copy_to_global(dest, x, y);
}

template <typename T, typename Tile>
inline CUDA_CALLABLE auto tile_atomic_add(array_t<T>& dest, int x, int y, Tile& src)
{
    auto src_reg = src.copy_to_register();

    const int tile_i = x*src_reg.M;
    const int tile_j = y*src_reg.N;

    tile_register_t<T, src_reg.M, src_reg.N> previous;

    WP_PRAGMA_UNROLL
    for (int i=0; i < src_reg.NumRegs; ++i)
    {
        // handle case where tile size is not 
        // aligned to block dimensions
        int linear = src_reg.index(i);
        if (!src_reg.Aligned && linear >= src_reg.Size)
            break;

        coord_t c = src_reg.coord(linear);
        previous.data[i] = atomic_add(dest, tile_i + c.i, tile_j + c.j, src_reg.data[i]);
    }

    return previous;
}



//-------------------------------------
// Adjoints

template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load(array_t<T>& src, int x,
                                        array_t<T>& adj_src, int adj_x,
                                        AdjTile& adj_ret)
{
    // early out
    // if (!src.grad)
    //     return;

    auto adj_reg = adj_ret.grad_to_register();

    const int tile_i = x*adj_reg.N;

    // add gradients to src array
    WP_PRAGMA_UNROLL
    for (int i=0; i < adj_reg.NumRegs; ++i)
    {  
        int linear = adj_reg.index(i);
        if (!adj_reg.Aligned && linear >= adj_reg.Size)
            break;

        auto grad = adj_reg.data[i];

        if (adj_src.data)
            adj_atomic_add(&index(adj_src, tile_i + linear), grad);
        else if (src.grad)
            adj_atomic_add(&index_grad(src, tile_i + linear), grad);
    }
}

template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load(array_t<T>& src, int x, int y,
                                        array_t<T>& adj_src, int adj_x, int adj_y,
                                        AdjTile& adj_ret)
{
    // early out
    // if (!src.grad)
    //     return;

    auto adj_reg = adj_ret.grad_to_register();

    const int tile_i = x*adj_reg.M;
    const int tile_j = y*adj_reg.N;

    // add gradients to src array
    WP_PRAGMA_UNROLL
    for (int i=0; i < adj_reg.NumRegs; ++i)
    {  
        int linear = adj_reg.index(i);
        if (!adj_reg.Aligned && linear >= adj_reg.Size)
            break;

        coord_t coord = adj_reg.coord(linear);

        auto grad = adj_reg.data[i];

        if (adj_src.data)
            adj_atomic_add(&index(adj_src, tile_i + coord.i, tile_j + coord.j), grad);
        else if (src.grad)
            adj_atomic_add(&index_grad(src, tile_i + coord.i, tile_j + coord.j), grad);
    }
}


template <typename T, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_store(array_t<T>& dest, int x, Tile& t, array_t<T>& adj_dest, int adj_x, AdjTile& adj_t)
{  
    // convert to register if necessary
    tile_register_t<T, AdjTile::M, AdjTile::N> adj_reg;

    const int tile_i = x*adj_reg.N;

    // load gradients from output
    WP_PRAGMA_UNROLL
    for (int i=0; i < adj_reg.NumRegs; ++i)
    {  
        int linear = adj_reg.index(i);
        if (!adj_reg.Aligned && linear >= adj_reg.Size)
            break;

         if (adj_dest.data)
            adj_reg.data[i] = index(adj_dest, tile_i + linear);
        else if (dest.grad)
            adj_reg.data[i] = index_grad(dest, tile_i + linear);
    }

    // store adjoint back to tile
    adj_t.grad_add(adj_reg);
}

template <typename T, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_store(array_t<T>& dest, int x, int y, Tile& t, array_t<T>& adj_dest, int adj_x, int adj_y, AdjTile& adj_t)
{  
    // allocate register tile to load grads into
    tile_register_t<T, AdjTile::M, AdjTile::N> adj_reg;

    const int tile_i = x*adj_reg.M;
    const int tile_j = y*adj_reg.N;

    // load gradients from output
    WP_PRAGMA_UNROLL
    for (int i=0; i < adj_reg.NumRegs; ++i)
    {  
        int linear = adj_reg.index(i);
        if (!adj_reg.Aligned && linear >= adj_reg.Size)
            break;

        coord_t coord = adj_reg.coord(linear);

         if (adj_dest.data)
            adj_reg.data[i] = index(adj_dest, tile_i + coord.i, tile_j + coord.j);
        else if (dest.grad)
            adj_reg.data[i] = index_grad(dest, tile_i + coord.i, tile_j + coord.j);
    }

    // store adjoint back to tile
    adj_t.grad_add(adj_reg);
}

template <typename T, typename Tile, typename AdjTile, typename AdjRet>
inline CUDA_CALLABLE void adj_tile_atomic_add(array_t<T>& dest, int x, int y, Tile& t, array_t<T>& adj_dest, int adj_x, int adj_y, AdjTile& adj_t, AdjRet& adj_ret)
{  
    adj_tile_store(dest, x, y, t, adj_dest, adj_x, adj_y, adj_t);
}


// unary map
template <typename Tile, typename Fwd>
inline CUDA_CALLABLE auto tile_map(Fwd op,
                                   Tile &a)
{
    auto out = tile_register_t<typename Tile::Type, Tile::M, Tile::N>();
    auto a_reg = a.copy_to_register();
    
    WP_PRAGMA_UNROLL
    for (int i=0; i < out.NumRegs; ++i)
    {
        out.data[i] = op(a_reg.data[i]);
    }

    return out;
}


template <typename Tile, typename AdjTile, typename Fwd, typename Adj>
inline CUDA_CALLABLE void adj_tile_map(Fwd op,
                                       Tile& a,
                                       Adj adj_op,
                                       Tile& adj_a,
                                       AdjTile& adj_ret)
{
    auto a_reg = a.copy_to_register();   
    auto adj_a_reg = tile_register_like<Tile>();
    auto adj_ret_reg = adj_ret.grad_to_register();

    WP_PRAGMA_UNROLL
    for (int i=0; i < a_reg.NumRegs; ++i)
    {        
        adj_op(a_reg.data[i], adj_a_reg.data[i], adj_ret_reg.data[i]);
    }

    // write adjoints back
    adj_a.grad_add(adj_a_reg);
}

// binary map
template <typename TileA, typename TileB, typename Fwd>
inline CUDA_CALLABLE auto tile_map(Fwd op,
                                   TileA& a,
                                   TileB& b)
{
    auto out = tile_register_t<typename TileA::Type, TileA::M, TileA::N>();

    auto a_reg = a.copy_to_register();
    auto b_reg = b.copy_to_register();

    WP_PRAGMA_UNROLL
    for (int i=0; i < out.NumRegs; ++i)
        out.data[i] = op(a_reg.data[i], b_reg.data[i]);

    return out;
}


template <typename TileA, typename TileB, typename Fwd, typename Adj, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_map(Fwd op,
                                       TileA &a,
                                       TileB &b,
                                       Adj adj_op,
                                       TileA &adj_a,
                                       TileB &adj_b,
                                       AdjTile &adj_ret)
{
    auto a_reg = a.copy_to_register();   
    auto b_reg = b.copy_to_register();

    // allocate storage for adjoints
    auto adj_a_reg = tile_register_like<TileA>();
    auto adj_b_reg = tile_register_like<TileB>();

    auto adj_ret_reg = adj_ret.grad_to_register();

    WP_PRAGMA_UNROLL
    for (int i=0; i < a_reg.NumRegs; ++i)
    {
        adj_op(a_reg.data[i], b_reg.data[i], adj_a_reg.data[i], adj_b_reg.data[i], adj_ret_reg.data[i]);
    }

    adj_a.grad_add(adj_a_reg);
    adj_b.grad_add(adj_b_reg);
}

// wrap the operator in a lambda so that we don't have to do overload resolution for things like e.g.: wp.sin()
// this is important because many of the builtin operators don't follow particular conventions on references for 
// the `adj_ret` parameter, which means it's not possible to figure out the overload we need using simple casting
#define tile_unary_map(op, a) tile_map([](auto x) { return op(x);}, a)
#define adj_tile_unary_map(op, a, adj_op, adj_a, adj_ret) adj_tile_map([](auto x) { return op(x);}, a, [](auto x, auto& adj_x, auto adj_ret) { adj_op(x, adj_x, adj_ret);}, adj_a, adj_ret)

#define tile_binary_map(op, a, b) tile_map([](auto x, auto y) { return op(x, y);}, a, b)
#define adj_tile_binary_map(op, a, b, adj_op, adj_a, adj_b, adj_ret) adj_tile_map([](auto x, auto y) { return op(x, y);}, a, b, [](auto x, auto y, auto& adj_x, auto& adj_y, auto adj_ret) { adj_op(x, y, adj_x, adj_y, adj_ret);}, adj_a, adj_b, adj_ret)

// -tile (unary neg)
template <typename Tile>
inline CUDA_CALLABLE auto tile_neg(Tile& a) { return tile_unary_map(wp::neg, a); }

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_neg(Tile& a, Tile& adj_a, AdjTile& adj_ret) { adj_tile_unary_map(wp::neg, a, wp::adj_neg, adj_a, adj_ret); }


// tile + tile
template <typename TileA, typename TileB>
inline CUDA_CALLABLE auto tile_add(TileA& a, TileB& b)
{
    return tile_binary_map(add, a, b);
}

// // tile + tile, we implement this 
// template <typename TileA, typename TileB>
// inline CUDA_CALLABLE auto add(TileA& a, TileB& b)
// {
//     return tile_binary_map(add, a, b);
// }


template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_add(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b, AdjTile& adj_c)
{   
    adj_tile_binary_map(add, a, b, adj_add, adj_a, adj_b, adj_c);
}

// tile*scalar
template <typename Tile>
inline CUDA_CALLABLE auto tile_mul(Tile& a, const typename Tile::Type& s)
{
    // promote scalar to a constant tile
    auto s_tile = tile_register_t<typename Tile::Type, Tile::M, Tile::N>(s);

    return tile_binary_map(mul, a, s_tile);
}

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_mul(Tile& a, const typename Tile::Type& s,
                                       Tile& adj_a, typename Tile::Type& adj_s,
                                       AdjTile& adj_c)
{
    auto s_tile = tile_register_t<typename Tile::Type, Tile::M, Tile::N>(s);
    auto adj_s_tile = tile_register_t<typename Tile::Type, Tile::M, Tile::N>();

    adj_tile_binary_map(mul, a, s_tile, adj_mul, adj_a, adj_s_tile, adj_c);

    for (int i=0; i < adj_s_tile.NumRegs; ++i)
    {
        adj_s += adj_s_tile.data[i];
    }
}


// scalar*tile
template <typename Tile>
inline CUDA_CALLABLE auto tile_mul(const typename Tile::Type& s, Tile& a)
{
    // promote scalar to a constant tile
    auto s_tile = tile_register_t<typename Tile::Type, Tile::M, Tile::N>(s);

    return tile_binary_map(mul, s_tile, a);
}

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_mul(const typename Tile::Type& s, Tile& a,
                                       typename Tile::Type& adj_s, Tile& adj_a,
                                       AdjTile& adj_c)
{
    auto s_tile = tile_register_t<typename Tile::Type, Tile::M, Tile::N>(s);
    auto adj_s_tile = tile_register_t<typename Tile::Type, Tile::M, Tile::N>();

    adj_tile_binary_map(mul, s_tile, a, adj_mul, adj_s_tile, adj_a, adj_c);

    for (int i=0; i < adj_s_tile.NumRegs; ++i)
    {
        adj_s += adj_s_tile.data[i];
    }
}



template<typename Tile>
typename Tile::Type tile_extract(Tile& t, int i, int j)
{
    assert(i < Tile::M);
    assert(j < Tile::N);

    return t.extract(i, j);
}

template<typename Tile, typename AdjTile>
void adj_tile_extract(Tile& t, int i, int j, AdjTile& adj_t, int adj_i, int adj_j, typename Tile::Type adj_ret)
{
    assert(i < Tile::M);
    assert(j < Tile::N);

    adj_t.adj_extract(i, j, adj_ret);
}

namespace partitioned_gemm
{

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

template <int PartitionM, int PartitionN, typename Tile>
struct partition_t
{
    static constexpr int M = PartitionM;
    static constexpr int N = PartitionN;
    static constexpr int Stride = Tile::N;
    
    using T = typename Tile::Type;    

    inline partition_t(Tile& A) 
    {
        data = A.data.ptr;
        
        // todo: do ceil div for non-multiples of M,N
        shape[0] = Tile::M/PartitionM;
        shape[1] = Tile::N/PartitionN;
    }

    // underlying data
    T* data;
    
    // partition dimensions
    int shape[2];
};

template <typename Partition>
inline int partition_size(const Partition& part)
{
    return part.shape[0]*part.shape[1];
}

// returns the x, y coordinates of a tile given a linear index
template <typename Partition>
inline void partition_coord(const Partition& part, const int t, int& i, int& j)
{
    i = t/part.shape[1];
    j = t%part.shape[1];
}

template <typename Partition>
inline auto partition_load(const Partition& tile, int i, int j)
{
    mat_t<Partition::M, Partition::N, typename Partition::T> out;
    
    const int tile_i = i*Partition::M;
    const int tile_j = j*Partition::N;

    WP_PRAGMA_UNROLL
    for (int i=0; i < Partition::M; ++i)
    {
        WP_PRAGMA_UNROLL
        for (int j=0; j < Partition::N; ++j)
        {
            out.data[i][j] = index(tile.data, tile_i + i, tile_j + j, Partition::Stride);
        }
    }

    return out;
}

template <typename Partition, typename Value>
inline void partition_store(const Partition& tile, int i, int j, const Value& value)
{
    const int tile_i = Partition::M*i;
    const int tile_j = Partition::N*j;

    WP_PRAGMA_UNROLL
    for (int i=0; i < Partition::M; ++i)
    {	
        WP_PRAGMA_UNROLL
        for (int j=0; j < Partition::N; ++j)
        {
            index(tile.data, tile_i + i, tile_j + j, Partition::Stride) = value.data[i][j];
        }
    }
}

template <typename TileA, typename TileB, typename TileC>
inline CUDA_CALLABLE void matmul(TileA& A, TileB& B, TileC& out)
{   
    const int TILE_M = 4;
    const int TILE_N = 4;
    const int TILE_K = 4;

    auto A_tile = partition_t<TILE_M, TILE_K, TileA>(A);
    auto B_tile = partition_t<TILE_K, TILE_N, TileB>(B);
    auto C_tile = partition_t<TILE_M, TILE_N, TileC>(out);

    const int length = partition_size(C_tile);

    for (int t=threadIdx.x; t < length; t += blockDim.x)
    {  
        int i, j;
        partition_coord(C_tile, t, i, j);

        // accumulator
        auto sum = partition_load(C_tile, i, j);

        WP_PRAGMA_UNROLL
        for (int k=0; k < A_tile.shape[1]; k++)
        {
            const auto a = partition_load(A_tile, i, k);
            const auto b = partition_load(B_tile, k, j);

            sum += mul(a, b);
        }
        
        partition_store(C_tile, i, j, sum);
    }
}
    
} // namespace partition_gemm

template <int Add, typename Fwd, typename AdjA, typename AdjB, typename TileA, typename TileB, typename TileC>
TileC& tile_matmul(Fwd fun_forward, AdjA fun_backward_A, AdjB fun_backward_B, TileA& A, TileB& B, TileC& C)
{       
    using T = typename TileA::Type;

#if WP_USE_ASYNC_PIPELINE
    __pipeline_wait_prior(0);
    WP_TILE_SYNC();
#endif

#if WP_USE_REGISTER_GEMM
    partitioned_gemm::matmul(A, B, C);
#else
    fun_forward(T(1.0), A.data.ptr, B.data.ptr, T(Add), C.data.ptr);
#endif
    
    WP_TILE_SYNC();
    
    return C;
}

// backward for the wp.tile_matmul(a, b, out) syntax
template <typename Fwd, typename AdjA, typename AdjB, typename TileA, typename TileB, typename TileC>
void adj_tile_matmul(Fwd fun_forward, AdjA fun_backward_A, AdjB fun_backward_B, TileA& A, TileB& B, TileC& C,
                   Fwd adj_fun_forward, AdjA adj_fun_backward_A, AdjB adj_fun_backward_B, TileA& adj_A, TileB& adj_B, TileC& adj_C)
{   
    using T = typename TileA::Type;    

    fun_backward_A(T(1.0), adj_C.grad.ptr, B.data.ptr, T(1.0), adj_A.grad.ptr);
    fun_backward_B(T(1.0), A.data.ptr, adj_C.grad.ptr, T(1.0), adj_B.grad.ptr);
    WP_TILE_SYNC();
}

// backward for the out = wp.tile_matmul(a, b) syntax
template <typename Fwd, typename AdjA, typename AdjB, typename TileA, typename TileB, typename TileC>
void adj_tile_matmul(Fwd fun_forward, AdjA fun_backward_A, AdjB fun_backward_B, TileA& A, TileB& B, TileC& C, 
                   Fwd adj_fun_forward, AdjA adj_fun_backward_A, AdjB adj_fun_backward_B, TileA& adj_A, TileB& adj_B, TileC& adj_C, TileC& adj_ret)
{   
    using T = typename TileA::Type;    

    fun_backward_A(T(1.0), adj_C.grad.ptr, B.data.ptr, T(1.0), adj_A.grad.ptr);
    fun_backward_B(T(1.0), A.data.ptr, adj_C.grad.ptr, T(1.0), adj_B.grad.ptr);
    WP_TILE_SYNC();
}

// TODO(lcambier): use a properly overaligned complex type that matches cuFFTDx's expectation
// TODO(lcambier): use dynamic smem
#define tile_fft(function_name, dtype, shared_memory_size, batch_size, ept, Xinout) \
    do { \
        void function_name(dtype*, dtype*); \
        WP_TILE_SHARED __align__(16) char buffer[shared_memory_size]; \
        __align__(16) dtype data[ept]; \
        for(int b = 0; b < (int)batch_size; b++) { \
            dtype* inout = Xinout.data + (int)b * (int)ept; \
            memcpy(data, inout, sizeof(dtype) * ept); \
            function_name(data, (dtype*)buffer); \
            memcpy(inout, data, sizeof(dtype) * ept); \
            WP_TILE_SYNC(); \
        } \
    } while (0)

#define tile_ifft tile_fft

// adj_function_name, adj_dtype, adj_shared_memory_size, adj_batch_size, adj_ept are all ignored

#define adj_tile_fft(function_name, dtype, shared_memory_size, batch_size, ept, Xinout, \
                        adj_function_name, adj_dtype, adj_shared_memory_size, adj_batch_size, adj_ept, \
                        adj_Xinout) \
    do { \
        tile_ifft(function_name, dtype, shared_memory_size, batch_size, ept, adj_Xinout); \
    } while (0)

#define adj_tile_ifft(function_name, dtype, shared_memory_size, batch_size, ept, Xinout, \
                         adj_function_name, adj_dtype, adj_shared_memory_size, adj_batch_size, adj_ept, \
                         adj_Xinout) \
    do { \
        tile_fft(function_name, dtype, shared_memory_size, batch_size, ept, adj_Xinout); \
    } while (0)


template <typename Tile>
inline CUDA_CALLABLE auto tile_transpose(Tile& t)
{    
    // alias incoming tile 
    return tile_shared_t<typename Tile::Type, Tile::N, Tile::M, Tile::StrideN, Tile::StrideM, false>(t.data.ptr, t.grad.ptr);
}

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_transpose(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{    
    auto a = tile_transpose(adj_ret);
    auto b = adj_t;
    
    adj_t.assign(tile_add(a,b));
}

template <int M, int N, int StrideM, int StrideN, typename Tile>
inline CUDA_CALLABLE auto tile_broadcast(Tile& t)
{    
    // alias incoming tile with new strides
    return tile_shared_t<typename Tile::Type, M, N, StrideM, StrideN, false>(t.data.ptr, t.grad.ptr);
}

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_broadcast(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{   
    // nop, since memory is aliased grads already accumulated

}

template <int M, int N, typename Tile>
inline CUDA_CALLABLE auto tile_view(Tile& t, int i, int j)
{    
    // alias incoming tile with new strides
    return tile_shared_t<typename Tile::Type, M, N, Tile::StrideM, Tile::StrideN, false>(&t.data(i, j), &t.grad(i, j));
}

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_view(Tile& t, int i, int j, Tile& adj_t, int adj_i, int adj_j, AdjTile& adj_ret)
{   
    // nop, since memory is aliased grads already accumulated

}

template <typename TileA, typename TileB>
inline CUDA_CALLABLE void tile_assign(TileA& dest, int i, int j, TileB& src)
{   
    for (int t=threadIdx.x; t < src.Size; t += WP_TILE_BLOCK_DIM)
    {
        coord_t c = src.coord(t);
        dest.data(i + c.i, j + c.j) = src.data(c.i, c.j);
    }

    WP_TILE_SYNC();
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_assign(TileA& dest, int i, int j, TileB& src,
                                          AdjTileA& adj_dest, int adj_i, int adj_j, AdjTileB& adj_src)
{
    for (int t=threadIdx.x; t < src.Size; t += WP_TILE_BLOCK_DIM)
    {
        coord_t c = src.coord(t);
        src.grad(c.i, c.j) += dest.grad(i + c.i, j + c.j);
    } 

    WP_TILE_SYNC();
}



} // namespace wp
