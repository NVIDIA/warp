/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef __clang__
// disable warnings related to C++17 extensions on CPU JIT builds
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif // __clang__

// Check if the CUDA toolkit is available
#if WP_ENABLE_CUDA || defined(__CUDACC_RTC__)

    // If NVRTC is being used, do not include extra headers (NVRTC has built-in float4)
    #ifdef __CUDACC_RTC__
        // NVRTC: Use built-in float4 (no need for extra definitions)
    #else
        // NVCC: Include vector_types.h to get float4
        #include <cuda_runtime.h>
    #endif

#else
    // If CUDA is not available (e.g., macOS build), manually define float4
    struct alignas(16) float4 {
        float x, y, z, w;
    };
#endif 

// only used while building the warp core library
#ifndef WP_TILE_BLOCK_DIM
#define WP_TILE_BLOCK_DIM 256
#endif

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
#define WP_USE_REGISTER_GEMM 0

#if defined(__CUDACC_RTC__)
#define WP_TILE_THREAD_IDX threadIdx.x
#else
#define WP_TILE_THREAD_IDX 0 
#endif // 

 

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


template <int N>
struct tile_coord_t
{
    int indices[N];

    CUDA_CALLABLE inline int operator[](int i) const { assert(0 <= 1 && i < N); return indices[i]; }
    CUDA_CALLABLE inline int& operator[](int i) { assert(0 <= 1 && i < N); return indices[i]; }

    CUDA_CALLABLE inline tile_coord_t<N> operator + (const tile_coord_t<N>& c) const
    {
        tile_coord_t<N> out;
        for (int i=0; i < N; ++i)
        {
            out.indices[i] = indices[i] + c.indices[i];
        }
        return out;
    }    
};

// This function deduces N = sizeof...(Ints)
template <typename... Ints>
constexpr tile_coord_t<sizeof...(Ints)> tile_coord(Ints... idxs)
{
    constexpr int N = sizeof...(Ints);
    
    // Create the result
    tile_coord_t<N> result{};
    
    // Capture all arguments in a local array
    int arr[] = { static_cast<int>(idxs)... };

    // C++14 or later: 'for' is allowed in a constexpr context
    for (int i = 0; i < N; ++i)
    {
        result.indices[i] = arr[i];
    }

    return result;
}

// helpers to construct a coord from a set of indices
inline auto tile_coord(int i) 
{
    auto c = tile_coord_t<1>();
    c.indices[0] = i;
    return c;
}

inline auto tile_coord(int i, int j)
{
    auto c = tile_coord_t<2>();
    c.indices[0] = i;
    c.indices[1] = j;
    return c;
}

inline auto tile_coord(int i, int j, int k)
{
    auto c = tile_coord_t<3>();
    c.indices[0] = i;
    c.indices[1] = j;
    c.indices[2] = k;
    return c;
}    

inline auto tile_coord(int i, int j, int k, int l)
{
    auto c = tile_coord_t<4>();
    c.indices[0] = i;
    c.indices[1] = j;
    c.indices[2] = k;
    c.indices[3] = l;
    return c;
}

// represents a compile time int tuple for strides/shapes/coords
template <int... V>
struct tile_tuple_t
{
    static constexpr int N = sizeof...(V);
    static_assert(N > 0, "Expected N > 0");

    static constexpr int data[N] = { V... };

    static constexpr int dim(int i) { assert(i < N); return data[i]; }
    static constexpr int size() 
    {
        int res = data[0];
        for (int i=1; i < N; ++i)
            res *= data[i];

        return res;
    }
};

// simple helper to compute strides from a shape up to 4d
template <typename Shape>
struct compute_strides;

// 1D
template <int D0>
struct compute_strides< tile_tuple_t<D0> > { using Stride = tile_tuple_t<1>; };
// 2D
template <int D0, int D1>
struct compute_strides< tile_tuple_t<D0, D1> > { using Stride = tile_tuple_t<D1, 1>; };
// 3D
template <int D0, int D1, int D2>
struct compute_strides< tile_tuple_t<D0, D1, D2> > { using Stride = tile_tuple_t<(D1 * D2), D2, 1>; };
// 4D
template <int D0, int D1, int D2, int D3>
struct compute_strides< tile_tuple_t<D0, D1, D2, D3> > { using Stride = tile_tuple_t<(D1 * D2 * D3), (D2 * D3), D3, 1>; };


// alias of tuple to represent shapes
template <int... V>
using tile_shape_t = tile_tuple_t<V...>;

// alias of tuple to represent stride
template <int... V>
using tile_stride_t = tile_tuple_t<V...>;


// represents a tile stored in global memory with dynamic strides
// used to represent the source and offset for tile loads to register/shared
template <typename T, typename Shape_>
struct tile_global_t 
{
    using Type = T;
    using Shape = Shape_;
    using Coord = tile_coord_t<Shape::N>;

    array_t<T> data;
    Coord offset;
    
    tile_global_t(array_t<T>& a, const Coord& c) : data(a), offset(c)
    {
    }

    inline CUDA_CALLABLE int index_from_coord(const Coord& coord) const
    {
        // element index
        int index = 0;
        
        WP_PRAGMA_UNROLL
        for (int i=0; i < Shape::N; ++i)
        {
            // global = offset + coord
            int c = offset[i] + coord[i];
            index += data.strides[i]*c;
        }   

        return index/sizeof(T);
    }    
    
    inline CUDA_CALLABLE bool index(const Coord& coord, int& out) const
    {
        // element index
        int index = 0;
        
        WP_PRAGMA_UNROLL
        for (int i=0; i < Shape::N; ++i)
        {
            // global = offset + coord
            int c = offset[i] + coord[i];

            // handle out of bounds case
            if (c >= data.shape[i])
                return false;
            else
                index += data.strides[i]*c;
        }   

        // array strides are in bytes so we convert to elements
        out = index / sizeof(T);
        return true;
    }

    inline CUDA_CALLABLE T load(const Coord& coord) const
    {   
        int i;
        if (index(coord, i))
            return data.data[i];
        else
            return T(0);
    }

    inline CUDA_CALLABLE T load_grad(const Coord& coord) const
    {
        int i;
        if (index(coord, i))
            return data.grad[i];
        else
            return T(0);
    }    

    inline CUDA_CALLABLE void store(const Coord& coord, const T& x) const
    {   
        int i;
        if (index(coord, i))
            data.data[i] = x;
    }

    inline CUDA_CALLABLE T atomic_add(const Coord& coord, const T& value) const
    {
        int i;
        if (index(coord, i))
            return wp::atomic_add(&data.data[i], value);
        else
            return T(0);
    }

    inline CUDA_CALLABLE T atomic_add_grad(const Coord& coord, const T& grad) const
    {   
        int i;
        if (index(coord, i))
            return wp::atomic_add(&data.grad[i], grad);
        else
            return T(0);
    }
};

template <typename Shape_>
struct tile_layout_register_t
{
    using Shape = Shape_;
    using Coord = tile_coord_t<Shape::N>;

    static constexpr int Size = Shape::size();
    static constexpr int NumRegs = (Size + WP_TILE_BLOCK_DIM - 1) / WP_TILE_BLOCK_DIM;
    static constexpr bool Aligned = Size%WP_TILE_BLOCK_DIM == 0;

    static inline CUDA_CALLABLE int linear_from_register(int reg)
    {
        return WP_TILE_THREAD_IDX + reg*WP_TILE_BLOCK_DIM;
    }

    static inline CUDA_CALLABLE int linear_from_coord(Coord c)
    {
        int linear = 0;
        int stride = 1;
        
        WP_PRAGMA_UNROLL
        for (int i=Shape::N-1; i >= 0; --i)
        {
            linear += c[i] * stride;
            stride *= Shape::dim(i);
        }
        return linear;
    }

    static inline CUDA_CALLABLE auto coord_from_linear(int linear)
    {
        Coord c;
        
        WP_PRAGMA_UNROLL
        for (int i=Shape::N-1; i >= 0; --i)
        {
            c[i] = linear%Shape::dim(i);
            linear /= Shape::dim(i);
        }

        return c;
    }
    
    static inline CUDA_CALLABLE int thread_from_linear(int linear)
    {
        const int thread = linear%WP_TILE_BLOCK_DIM;
        return thread;
    }

    static inline CUDA_CALLABLE int register_from_linear(int linear)
    {
        const int reg = linear/WP_TILE_BLOCK_DIM;
        return reg;
    }

    static inline CUDA_CALLABLE bool valid(int linear)
    {
        if (Aligned || linear < Size)
            return true;
        else
            return false;
    }

};

// represents a tile stored in registers across a block
template <typename T, typename L>
struct tile_register_t
{
    using Type = T;
    using Layout = L;

    T data[Layout::NumRegs];
   
    inline CUDA_CALLABLE tile_register_t(T value=T(0.0)) 
    {
        // zero-initialize by default necessary for tile adjoints
        // need to check if this results in worse codegen
        // than doing adj_var = tile_zeros() explicitly
        // in backwards pass and letting default constructor
        // avoid initialization
        
        for (int i=0; i < Layout::NumRegs; ++i)
            data[i] = value;
    }

    inline CUDA_CALLABLE auto& operator=(const tile_global_t<T, typename Layout::Shape>& t)
    {
        copy_from_global(t);
        return *this;
    }

    // define the += operator which is used during backward pass codegen
    // when returning a register tile from a user defined function
    inline CUDA_CALLABLE auto& operator += (tile_register_t<T, Layout>& rhs) 
    {
        grad_add(rhs);
        return *this;
    }

    inline CUDA_CALLABLE T& operator()(int reg)
    {
        assert(reg < Layout::NumRegs);
        return data[reg];
    }    

    inline CUDA_CALLABLE const T& operator()(int reg) const
    {
        assert(reg < Layout::NumRegs);
        return data[reg];
    }

    inline CUDA_CALLABLE void assign(const tile_register_t<T, Layout>& tile) 
    { 
        for (int i=0; i < Layout::NumRegs; ++i)
            data[i] = tile.data[i];
    }

    inline CUDA_CALLABLE void zero()
    {
        for (int i=0; i < Layout::NumRegs; ++i)
            data[i] = T(0);
    }

    // extract a single tile element to a native type
    template <typename Coord>
    inline CUDA_CALLABLE Type extract(const Coord& c)
    {
        // map from logical coords (i, j) -> (thread, reg)
        const int linear = Layout::linear_from_coord(c);
        const int thread = Layout::thread_from_linear(linear);
        const int reg = Layout::register_from_linear(linear);

        WP_TILE_SHARED Type scratch;

        // ensure any previously scheduled threads have finished reading from scratch
        WP_TILE_SYNC();

        if (WP_TILE_THREAD_IDX == thread)
        {
            scratch = data[reg];
        }

        // ensure extraction thread has updated smem
        WP_TILE_SYNC();

        return scratch;
    }
    

    // backward version of scalar extract
    template <typename Coord>
    inline CUDA_CALLABLE void adj_extract(const Coord& c, Type adj_ret)
    {
        // map from logical coords (i, j) -> (thread, reg)
        const int linear = Layout::linear_from_coord(c);
        const int thread = Layout::thread_from_linear(linear);
        const int reg = Layout::register_from_linear(linear);

        if (WP_TILE_THREAD_IDX == thread)
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

    // apply a lambda to all valid entries in the tile
    // Op should be a functor that takes a register index and tile_coord_t as input
    template <typename Op>
    void apply(Op op)
    {
        WP_PRAGMA_UNROLL
        for (int i=0; i < Layout::NumRegs; ++i)
        {  
            int linear = Layout::linear_from_register(i);
            if (!Layout::valid(linear))
                break;

            auto c = Layout::coord_from_linear(linear);
            op(i, c);
        }        
    }
    

    // in-place gradient zero
    inline CUDA_CALLABLE void grad_zero()
    {
        zero();
    }

    // accumulate gradients onto this tile
    inline CUDA_CALLABLE void grad_add(const tile_register_t<T, Layout>& tile) 
    { 
        for (int i=0; i < Layout::NumRegs; ++i)
            data[i] += tile.data[i];
    }

    CUDA_CALLABLE void grad_add(const tile_global_t<T, typename Layout::Shape>& global) 
    {
        apply([&](int reg, auto c) {data[reg] = global.load_grad(c);});

    }

    inline CUDA_CALLABLE auto& grad_to_register() 
    {
        // nop for register tiles
        return *this;
    }

    template <typename Global>
    inline CUDA_CALLABLE void copy_to_global(const Global& dest)
    {
        apply([&](int reg, auto c) { dest.store(c, data[reg]); });
    }

    template <typename Global>
    inline CUDA_CALLABLE void copy_from_global(const Global& src)
    {
        apply([&](int reg, auto c) { data[reg] = src.load(c); });
    }

    // add a register tile to a global array
    template <typename Global>
    inline CUDA_CALLABLE auto atomic_add(const Global& dest)
    {
        // allocate a tile to hold previous dest value
        auto previous = *this;

        apply([&](int reg, auto c) { previous.data[reg] = dest.atomic_add(c, data[reg]); });
        return previous;
    }

    // add a register tile to the gradient of a global array
    template <typename Global>
    inline CUDA_CALLABLE auto atomic_add_grad(const Global& dest)
    {
        // allocate a tile to hold previous dest value
        auto previous = *this;

        apply([&](int reg, auto c) { previous.data[reg] = dest.atomic_add_grad(c, data[reg]); });
        return previous;
    }        
};


// helper to allocate a register tile like another tile
// users can either specify a template explicitly or
// pass in another concrete instance
template<typename Tile>
auto tile_register_like(Tile* t=nullptr)
{
    using T = typename Tile::Type;
    using L = typename Tile::Layout;

    return tile_register_t<T, tile_layout_register_t<typename L::Shape>>(T(0.0));
}

// helper to construct a register tile from a type and a list of dims
template <typename T, int... Dims>
auto tile_register()
{
    return tile_register_t<T, tile_layout_register_t<tile_shape_t<Dims...>>>();
}

inline CUDA_CALLABLE int tile_align(int num_bytes)
{
    // note this much match value in Python types.py
    const int alignment = 16;

    const int num_bytes_abs = num_bytes < 0 ? - num_bytes : num_bytes;
    const int sign = num_bytes < 0 ? - 1 : 1;

    return sign * ((num_bytes_abs + alignment - 1) / alignment) * alignment;
}

inline CUDA_CALLABLE void* tile_alloc_shared(int num_bytes, bool init=false, bool check=false)
{
    // we maintain a per-thread offset into dynamic
    // shared memory that allows us to keep track of 
    // current use across dynamic function calls
    WP_TILE_SHARED int smem_base[WP_TILE_BLOCK_DIM];

    if (init)
    {
        smem_base[WP_TILE_THREAD_IDX] = 0;
        return nullptr;
    }
    else if (check)
    {
        assert(smem_base[WP_TILE_THREAD_IDX] == 0);
        return nullptr;
    }
    else
    {
        const int offset = smem_base[WP_TILE_THREAD_IDX];
        
        // one entry per-thread so no need for synchronization
        smem_base[WP_TILE_THREAD_IDX] += tile_align(num_bytes);

#ifdef __CUDA_ARCH__
        extern __shared__ char dynamic_smem_base[];
#else
        // on CPU allocate a fixed 256k block to use for shared allocs
        static const int max_cpu_shared = 256*1024;
        static char dynamic_smem_base[max_cpu_shared];

        assert(smem_base[WP_TILE_THREAD_IDX] <= max_cpu_shared);
#endif
        return &(dynamic_smem_base[offset]);
    }
}


template <typename Shape_, typename Stride_= typename compute_strides<Shape_>::Stride>
struct tile_layout_strided_t
{
    using Shape = Shape_;
    using Stride = Stride_;
    using Coord = tile_coord_t<Shape::N>;
    
    static constexpr int Size = Shape::size();
    static constexpr bool Aligned = Size%WP_TILE_BLOCK_DIM == 0;

    static inline CUDA_CALLABLE auto coord_from_linear(int linear)
    {
        assert(linear < Size);

        Coord c;
        
        WP_PRAGMA_UNROLL
        for (int d=Shape::N-1; d >= 0; --d)
        {
            c[d] = linear%Shape::dim(d);
            linear /= Shape::dim(d);
        }

        return c;
    }

    static inline CUDA_CALLABLE int index_from_coord(Coord c)
    {
        int index = 0;

        WP_PRAGMA_UNROLL
        for (int d=0; d < Shape::N; ++d)
        {
            assert(c[d] < Shape::dim(d));

            index += c[d]*Stride::dim(d);
        }

        return index;
    }

    // checks whether a strided layout is unique, i.e.: if memory locations are only
    // ever referred to by one element in the tile, this is a basic test that only
    // checks for broadcast dimensions, it would be possible to do the full check
    // using sorted shape/strides in Python and add it as a template parameter to the type
    static constexpr bool is_unique() 
    {
        constexpr int N = Shape::N;

        // check for any broadcast dimensions
        for (int i=0; i < N; ++i)
            if (Stride::dim(i) == 0)
                return false;
        
        return true;
    }

    static constexpr bool Unique = is_unique();

    static inline CUDA_CALLABLE bool valid(int linear)
    {
        return linear < Size;
    }    

};


template <typename T, typename L, bool Owner_=true>
struct tile_shared_t 
{
    using Type = T;
    using Layout = L;
    static constexpr bool Owner = Owner_;

    struct Storage
    {
        T* ptr;

        Storage(T* p) : ptr(p) {}

        inline CUDA_CALLABLE T& operator()(typename Layout::Coord c)
        {
            assert(ptr);

            int index = Layout::index_from_coord(c);
            return ptr[index];
        }

        inline CUDA_CALLABLE const T& operator()(typename Layout::Coord c) const
        {          
            assert(ptr);

            int index = Layout::index_from_coord(c);
            return ptr[index];
        }

        inline CUDA_CALLABLE T& operator()(int linear)
        {
            assert(ptr);
            assert(Layout::valid(linear));

            auto c = Layout::coord_from_linear(linear);
            return (*this)(c);
        }    

        inline CUDA_CALLABLE const T& operator()(int linear) const
        {
            assert(ptr);
            assert(Layout::valid(linear));

            auto c = Layout::coord_from_linear(linear);
            return (*this)(c);
        }            
    };

    Storage data;
    Storage grad;

    // we need to track whether or not this tile's data has been initialized.
    // once true, any re-initialization of data that follows needs a WP_TILE_SYNC()
    // call to precede it, to allow threads that are still reading from this tile
    // to complete their work. e.g, in a dynamic loop:
    // for i in range(x):
    //     tile = wp.tile_load(arr, i, TILE_SIZE, storage="shared")
    //     # read from tile...
    bool initialized;

    // default initialization (non-initialized)
    inline CUDA_CALLABLE tile_shared_t() : data(nullptr), grad(nullptr), initialized(false)
    {
    }

    // initialize from an existing tile's memory
    inline CUDA_CALLABLE tile_shared_t(T* data, T* grad=nullptr, bool initialized=true) : data(data), grad(grad), initialized(initialized)
    {
    }

    inline CUDA_CALLABLE ~tile_shared_t()
    {
        if (Owner)
        {
            // update our per-thread shared memory allocator
            if (data.ptr)
                tile_alloc_shared(-Layout::Size*int(sizeof(T)));

            if (grad.ptr)
                tile_alloc_shared(-Layout::Size*int(sizeof(T)));
        }
    }

    // assign from a register tile
    inline CUDA_CALLABLE auto& operator=(const tile_register_t<Type, tile_layout_register_t<typename Layout::Shape>>& t)
    {
        assign(t);
        return *this;
    }

    // construct from another shared tile, this constructor
    // is invoked for reshape operations like `wp.tile_transpose()`
    template <typename OtherT, typename OtherLayout, bool OtherOwner>
    inline CUDA_CALLABLE auto& operator=(const tile_shared_t<OtherT, OtherLayout, OtherOwner>& rhs) 
    {
        // check dimensions are compatible
        static_assert(Layout::Size == OtherLayout::Size, "Expected Size == OtherLayout::Size");

        // alias tile directly
        data.ptr = rhs.data.ptr;
        grad.ptr = rhs.grad.ptr;
        initialized = rhs.initialized;

        return *this;
    }    

    // assign from a global tile (load)
    inline CUDA_CALLABLE auto& operator=(const tile_global_t<T, typename Layout::Shape>& t)
    {        
        copy_from_global(t);
        return *this;
    }

    // assign from a constant value
    inline CUDA_CALLABLE auto& operator=(const T& x)
    {
        // sync if we are re-initializing data so that any threads that are still
        // reading from this tile can complete their work, e.g.: if re-assigning
        // to a tile during a dynamic loop
        if (initialized)
            WP_TILE_SYNC();

        for (int i=WP_TILE_THREAD_IDX; i < Layout::Size; i+= WP_TILE_BLOCK_DIM)
            data(i) = x;

        initialized = true;
        WP_TILE_SYNC();
        return *this;
    }

    // in-place zero
    inline CUDA_CALLABLE void zero()
    {
        for (int i=WP_TILE_THREAD_IDX; i < Layout::Size; i+= WP_TILE_BLOCK_DIM)
            data(i) = T(0);

        WP_TILE_SYNC();
    }

    // extract a single tile element to a native type
    inline CUDA_CALLABLE Type extract(const typename Layout::Coord& c)
    {
        return data(c);
    }
        
    // backward of scalar extraction
    inline CUDA_CALLABLE void adj_extract(const typename Layout::Coord& c, Type adj_ret)
    {
        // since multiple threads may extract the same element
        // we need to accumulate using atomic operations
        wp::atomic_add(&grad(c), adj_ret);
    
        WP_TILE_SYNC();       
    }

    // add scalar value onto a single tile element
    inline CUDA_CALLABLE void add_inplace(const typename Layout::Coord& c, const Type& x)
    {
        // since multiple threads may add to the same element
        // we need to accumulate using atomic operations
        wp::atomic_add(&data(c), x);

        WP_TILE_SYNC();
    }

    // backward of inplace scalar addition
    inline CUDA_CALLABLE void adj_add_inplace(const typename Layout::Coord& c, Type& adj_x)
    {
        adj_x += grad(c);
    }

    // subtract scalar value from a single tile element
    inline CUDA_CALLABLE void sub_inplace(const typename Layout::Coord& c, const Type& x)
    {
        // since multiple threads may add to the same element
        // we need to accumulate using atomic operations
        wp::atomic_add(&data(c), -x);

        WP_TILE_SYNC();
    }

    // backward of inplace scalar subtraction
    inline CUDA_CALLABLE void adj_sub_inplace(const typename Layout::Coord& c, Type& adj_x)
    {
        adj_x -= grad(c);
    }

    // copy register tile to shared
    template <typename Tile>
    inline CUDA_CALLABLE void assign(const Tile& tile)
    { 
        if (initialized)
            WP_TILE_SYNC();

        WP_PRAGMA_UNROLL
        for (int i=0; i < Tile::Layout::NumRegs; ++i)
        {
            const int linear = Tile::Layout::linear_from_register(i);

            // handle case where tile size is not
            // aligned to block dimensions
            if (!Tile::Layout::valid(linear))
                break;         

            data(linear) = tile.data[i];
        }

        initialized = true;
        WP_TILE_SYNC();
    }

    // in-place gradient zero
    inline CUDA_CALLABLE void grad_zero()
    {
        for (int i=WP_TILE_THREAD_IDX; i < Layout::Size; i+= WP_TILE_BLOCK_DIM)
            grad(i) = T(0);

        WP_TILE_SYNC();
    }


    // accumulate gradients onto this tile
    template <typename Tile>
    inline CUDA_CALLABLE void grad_add(const Tile& tile) 
    { 
        WP_PRAGMA_UNROLL
        for (int i=0; i < Tile::Layout::NumRegs; ++i)
        {
            const int linear = Tile::Layout::linear_from_register(i);

            // handle case where tile size is not
            // aligned to block dimensions
            if (!Tile::Layout::valid(linear))
                break;

            // if the destination layout is unique (no broadcast dimensions)
            // then we can use regular non-atomic accmulation
            if (Layout::Unique)
                grad(linear) += tile.data[i];
            else
                // use shared memory atomics to accumulate gradients
                // since for broadcast tiles (e.g.: a bias vector) multiple incoming threads 
                // may map to a single location in shared memory
                wp::atomic_add(&grad(linear), tile.data[i]);
            
        }

        WP_TILE_SYNC();
    }

    // accumulate gradient onto this tile from a global array
    CUDA_CALLABLE void grad_add(const tile_global_t<T, typename Layout::Shape>& global) 
    {
        WP_PRAGMA_UNROLL
        for (int i=WP_TILE_THREAD_IDX; i < Layout::Size; i += WP_TILE_BLOCK_DIM)
        {  
            auto c = Layout::coord_from_linear(i);
            T g = global.load_grad(c);

            if (Layout::Unique)
            {
                // if the destination layout is unique (no broadcast dimensions)
                // then we can use regular non-atomic accumulation
                grad(c) += g;
            }
            else
            {
                // use shared memory atomics to accumulate gradients
                // since for broadcast tiles (e.g.: a bias vector) multiple incoming threads 
                // may map to a single location in shared memory
                wp::atomic_add(&grad(c), g);
            }
        }

        WP_TILE_SYNC();        
    }    

    // copy shared tile to register
    inline CUDA_CALLABLE auto grad_to_register() 
    { 
        using Tile = tile_register_t<T, tile_layout_register_t<typename Layout::Shape>>;
        Tile out;

        WP_PRAGMA_UNROLL
        for (int i=0; i < Tile::Layout::NumRegs; ++i)
        {
            const int linear = Tile::Layout::linear_from_register(i);

            if (!Tile::Layout::valid(linear))
                break;

            out(i) = grad(linear);
        }

        return out;
    }

    // copy shared tile to register
    inline CUDA_CALLABLE auto copy_to_register() const
    { 

        auto out = tile_register_like(this);

        using Layout = typename decltype(out)::Layout;

        WP_PRAGMA_UNROLL
        for (int i=0; i < Layout::NumRegs; ++i)
        {
            const int linear = Layout::linear_from_register(i);

            if (!Layout::valid(linear))
                break;

            out(i) = data(linear);
        }

        return out;
    }

    template <typename Global>
    inline CUDA_CALLABLE void copy_to_global(const Global& dest)
    {       

#if defined(__CUDA_ARCH__)
        // vectorized loads for specific input/output shapes
        if constexpr (Layout::Shape::N == 2)
        {
            constexpr int lastdim = Layout::Shape::N-1;
            constexpr bool contiguous_src = Layout::Stride::dim(lastdim) == 1;
            const bool contiguous_dest = dest.data.strides[lastdim] == sizeof(T);
            const int elements = min(Layout::Shape::dim(1), (dest.data.shape[lastdim] - dest.offset[lastdim]));
            const bool aligned_size = (elements*sizeof(T))%sizeof(float4) == 0;
            const bool aligned_stride = (dest.data.strides[0]/sizeof(T))%Layout::Stride::dim(0) == 0;

            float4* dest128 = (float4*)&dest.data.data[dest.index_from_coord(tile_coord(0,0))];
            const bool aligned_dst = (uint64_t)(dest128)%sizeof(float4) == 0;

            constexpr int M = Layout::Shape::dim(0);
            constexpr int N = (Layout::Shape::dim(1)*sizeof(T))/sizeof(float4);

            if (contiguous_dest && contiguous_src && aligned_size && aligned_dst && aligned_stride && N)
            {                               
                // alias of shared tile with 128bit type
                using SrcLayout = tile_layout_strided_t<tile_shape_t<M, N>>;
                tile_shared_t<float4, SrcLayout> src128((float4*)data.ptr);

                assert(((uint64_t)(data.ptr))%sizeof(float4) == 0);
                assert(((uint64_t)(dest128))%sizeof(float4) == 0);

                const int stride_i = dest.data.strides[0]/sizeof(float4);
                const int stride_j = 1;

                WP_PRAGMA_UNROLL
                for (int i=WP_TILE_THREAD_IDX; i < SrcLayout::Size; i += WP_TILE_BLOCK_DIM)
                {  
                    auto c = SrcLayout::coord_from_linear(i);
                    
                    dest128[stride_i*c[0] + stride_j*c[1]] = src128.data(i);
                }

                return;
            }
        }

#endif //defined(__CUDA_ARCH__)

        // scalar bounds checked path
        WP_PRAGMA_UNROLL
        for (int i=WP_TILE_THREAD_IDX; i < Layout::Size; i += WP_TILE_BLOCK_DIM)
        {
            auto c = Layout::coord_from_linear(i);
            dest.store(c, data(i));
        }
    }

    inline CUDA_CALLABLE void cp_async_global_to_shared_128(float4* shared_dest, const float4* global_src)
    {
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

        unsigned long long saddr = 0ULL;
        unsigned long long gaddr = 0ULL;

        asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(saddr) : "l"(shared_dest));
        asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(gaddr) : "l"(global_src));

        // Use cp.async on newer architectures
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "l"(saddr), "l"(gaddr)
        );
    #else
        // use regular load/store through register on older arches
        *shared_dest = *global_src;
    #endif
    }    

    inline CUDA_CALLABLE void cp_async_commit_and_wait_all_128()
    {
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        asm volatile(
            "cp.async.commit_group;\n"
            "cp.async.wait_group 0;\n" ::);
    #endif
    }

    template <typename Global>
    inline CUDA_CALLABLE void copy_from_global(const Global& src)
    {   
        if (initialized)
            WP_TILE_SYNC();
        
#if defined(__CUDA_ARCH__)

        // vectorized loads for specific input/output shapes
        if constexpr (Layout::Shape::N == 2)
        {
            constexpr int lastdim = Layout::Shape::N-1;
            constexpr bool contiguous_dest = Layout::Stride::dim(lastdim) == 1;
            const bool contiguous_src = src.data.strides[lastdim] == sizeof(T);
            const int elements = min(Layout::Shape::dim(1), (src.data.shape[lastdim] - src.offset[lastdim]));
            const bool aligned_size = (elements*sizeof(T))%sizeof(float4) == 0;
            const bool aligned_stride = (src.data.strides[0]/sizeof(T))%Layout::Stride::dim(0) == 0;
           
            float4* src128 = (float4*)&src.data.data[src.index_from_coord(tile_coord(0,0))];
            const bool aligned_src = (uint64_t)(src128)%sizeof(float4) == 0;

            constexpr int M = Layout::Shape::dim(0);
            constexpr int N = (Layout::Shape::dim(1)*sizeof(T))/sizeof(float4);

            if (contiguous_dest && contiguous_src && aligned_size && aligned_src && aligned_stride && N)
            {
                // alias of shared tile with 128bit type
                using DestLayout = tile_layout_strided_t<tile_shape_t<M, N>>;
                tile_shared_t<float4, DestLayout> dest128((float4*)data.ptr);                

                assert(((uint64_t)(dest128.data.ptr))%sizeof(float4) == 0);
                assert(((uint64_t)(src128))%sizeof(float4) == 0);

                const int stride_i = src.data.strides[0]/sizeof(float4);
                const int stride_j = 1;

                WP_PRAGMA_UNROLL
                for (int i=WP_TILE_THREAD_IDX; i < DestLayout::Size; i += WP_TILE_BLOCK_DIM)
                {  
                    auto c = DestLayout::coord_from_linear(i);
                    
#if WP_USE_ASYNC_PIPELINE
                    cp_async_global_to_shared_128(&dest128.data(i), &src128[stride_i*c[0] + stride_j*c[1]]);
#else
                    dest128.data(i) = src128[stride_i*c[0] + stride_j*c[1]];
#endif // WP_USE_ASYNC_PIPELINE
                }

#if WP_USE_ASYNC_PIPELINE
                cp_async_commit_and_wait_all_128();
#endif // WP_USE_ASYNC_PIPELINE

                initialized = true;
                WP_TILE_SYNC();
                return;
            }
        }

#endif //defined(__CUDA_ARCH__)

        // scalar bounds checked path
        WP_PRAGMA_UNROLL
        for (int i=WP_TILE_THREAD_IDX; i < Layout::Size; i += WP_TILE_BLOCK_DIM)
        {  
            auto c = Layout::coord_from_linear(i);
            data(i) = src.load(c);
        }

        initialized = true;
        WP_TILE_SYNC();
    }

    template <typename Global>
    inline CUDA_CALLABLE auto atomic_add(Global& dest)
    {
        return copy_to_register().atomic_add(dest);
    }

    template <typename Global>
    inline CUDA_CALLABLE auto atomic_add_grad(Global& dest)
    {
        return grad_to_register().atomic_add_grad(dest);
    }

    // overload for integral types
    inline CUDA_CALLABLE void print_value(int x) const
    {
        printf("%d", x);
    }

    // overload for floating point types
    template <typename ValueType>
    inline CUDA_CALLABLE void print_value(ValueType x) const
    {
        printf("%g", x);
    }

    template <int Level = 0>
    inline CUDA_CALLABLE void print_values(const Storage& storage, int index=0) const
    {
        using Shape = typename Layout::Shape;

        if constexpr (Level < Shape::N)
        {
            if constexpr (Level == Shape::N - 1)
            {
                // Special handling for 1D case
                printf("[");
                for (int i = 0; i < Shape::dim(Level); ++i)
                {
                    print_value(storage(index + i));

                    if (i < Shape::dim(Level) - 1)
                    {
                        printf(" ");
                    }                    
                }
                printf("]");                    
            }
            else if constexpr (Level == Shape::N - 2)
            {                   
                // Special handling for 2D case
                printf("[");
                for (int i = 0; i < Shape::dim(Level); ++i)                
                {             
                    printf("[");
                    for (int j=0; j < Shape::dim(Level+1); ++j)
                    {
                        print_value(storage(index));                        

                        if (j < Shape::dim(Level+1) - 1)
                        {
                            printf(" ");
                        }

                        ++index;
                    }    

                    printf("]");

                    // next row
                    if (i < Shape::dim(Level)-1)
                    {
                        printf("\n");

                        // indent next row
                        for (int i=0; i <= Shape::N-2; ++i)
                            printf(" ");

                    }
                }
                printf("]");
            } 
            else
            {
                printf("[");
                for (int i = 0; i < Shape::dim(Level); ++i)
                {
                    print_values<Level + 1>(storage, index + i * Shape::dim(Level));
                    if (i < Shape::dim(Level) - 1)
                    {
                        printf("\n\n");

                        // indent next row
                        for (int i=0; i <= Level; ++i)
                            printf(" ");
                    }
                }
                printf("]");
            }
        }
    }

    inline CUDA_CALLABLE void print(bool reverse=false) const
    {
        if (WP_TILE_THREAD_IDX != 0)
            return;

        if (reverse)
            print_values(grad);
        else
            print_values(data);

        printf(" = tile(shape=(");
        for (int i=0; i < Layout::Shape::N; ++i)
        {
            printf("%d", Layout::Shape::dim(i));
            if (i != Layout::Shape::N-1)
                printf(",");
        }

        printf("), storage=shared)\n");
    }    
};


template <typename T, typename L>
void tile_register_t<T, L>::print() const
{
    // create a temporary shared tile so that
    // we can print it deterministically
    WP_TILE_SHARED T smem[L::Size];
    tile_shared_t<T, tile_layout_strided_t<typename L::Shape>, false> scratch(smem, nullptr);

    scratch.assign(*this);

    WP_TILE_SYNC();

    if (WP_TILE_THREAD_IDX == 0)
    {
        scratch.print_values(scratch.data, 0);

        printf(" = tile(shape=(");
        for (int i=0; i < L::Shape::N; ++i)
        {
            printf("%d", L::Shape::dim(i));
            if (i != L::Shape::N-1)
                printf(",");
        }

        printf("), storage=register)\n");
    }

    WP_TILE_SYNC();
}

// print entry points
template <typename T, typename L>
inline CUDA_CALLABLE void print(const tile_register_t<T, L>& t) { t.print(); }
template <typename T, typename L, bool Owner>
inline CUDA_CALLABLE void print(const tile_shared_t<T, L, Owner>& t) { t.print(); }

template <typename T, typename L, bool O>
inline CUDA_CALLABLE int len(const tile_shared_t<T, L, O>& t)
{
    return L::Shape::dim(0);
}

template <typename T, typename L, bool O, typename AdjTile>
inline CUDA_CALLABLE void adj_len(const tile_shared_t<T,L,O>& t, const AdjTile& a, int& adj_ret)
{
}

template <typename T, typename L>
inline CUDA_CALLABLE int len(const tile_register_t<T, L>& t)
{
    return L::Shape::dim(0);
}

template <typename T, typename L, typename AdjTile>
inline CUDA_CALLABLE void adj_len(const tile_register_t<T,L>& t, const AdjTile& a, int& adj_ret)
{
}


template <typename T, typename L>
inline CUDA_CALLABLE void adj_print(const tile_register_t<T, L>& t, const tile_register_t<T, L>& a) { a.print(); }
template <typename T, typename L, bool Owner>
inline CUDA_CALLABLE void adj_print(const tile_shared_t<T, L, Owner>& t, const tile_shared_t<T, L, Owner>& a) { a.print(true); }



// helpers to allocate shared tiles
template <typename T, typename Shape, bool RequiresGrad>
inline CUDA_CALLABLE auto tile_alloc_empty()

{   constexpr int size = Shape::size();
    T* data = (T*)tile_alloc_shared(size*sizeof(T));
    T* grad = nullptr;

#if FP_CHECK

    // initialize tile to quiet nan
    uint32_t qnanbits = 0x7FC00000;
    float qnan = *(float*)(&qnanbits);

    for (int i=WP_TILE_THREAD_IDX; i < size; i+= WP_TILE_BLOCK_DIM)
        data[i] = T(qnan);

    WP_TILE_SYNC();

#endif // FP_CHECK


    if (RequiresGrad)
    {
        grad = (T*)tile_alloc_shared(size*sizeof(T));

        for (int i=WP_TILE_THREAD_IDX; i < size; i+= WP_TILE_BLOCK_DIM)
            grad[i] = T(0);

        WP_TILE_SYNC();
    }
       
    return tile_shared_t<T, tile_layout_strided_t<Shape>>(data, grad);
}


//-----------------------------------------------------------------------------------------------------
// High level entry points for each op (correspond to one Warp builtin)

// construct a tile from a local SIMT value (one per-thread)
template <typename T>
inline CUDA_CALLABLE auto tile(const T& x)
{
    tile_register_t<T, tile_layout_register_t<tile_shape_t<WP_TILE_BLOCK_DIM>>> result;
    
    using Layout = typename decltype(result)::Layout;
    static_assert(Layout::NumRegs == 1, "Expected Layout::NumRegs == 1");

    result.data[0] = x;
    return result;
}

// overload for constructing a tile from a per-thread vector
template <typename T, unsigned Length>
inline CUDA_CALLABLE auto tile(const wp::vec_t<Length, T>& x)
{
    tile_register_t<T, tile_layout_register_t<tile_shape_t<Length, WP_TILE_BLOCK_DIM>>> result;
    
    using Layout = typename decltype(result)::Layout;
    static_assert(Layout::NumRegs == Length, "Expected Layout::NumRegs == Length");

    for (int i=0; i < Length; ++i)
        result.data[i] = x[i]; 

    return result;
}

// construct a tile from a local SIMT value (one per-thread)
template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile(const T& x, T& adj_x, AdjTile& adj_ret)
{
    static_assert(AdjTile::Layout::Shape::N == 1, "Expected AdjTile::Layout::Shape::N == 1");
    static_assert(AdjTile::Layout::Shape::dim(0) == WP_TILE_BLOCK_DIM, "Expected AdjTile::Layout::Shape::dim(0) == WP_TILE_BLOCK_DIM");
    
    auto adj_reg = adj_ret.copy_to_register();

    adj_x += adj_reg.data[0];
}

template <typename T, unsigned Length, typename AdjTile>
inline CUDA_CALLABLE void adj_tile(const wp::vec_t<Length, T>& x, wp::vec_t<Length, T>& adj_x, AdjTile& adj_ret)
{
    static_assert(AdjTile::Layout::Shape::N == 2, "Expected AdjTile::Layout::Shape::N == 2");
    static_assert(AdjTile::Layout::Shape::dim(0) == Length, "Expected AdjTile::Layout::Shape::dim(0) == Length");
    static_assert(AdjTile::Layout::Shape::dim(1) == WP_TILE_BLOCK_DIM, "Expected AdjTile::Layout::Shape::dim(1) == WP_TILE_BLOCK_DIM");

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

    constexpr int N = Tile::Layout::Shape::N;

    // scalar case
    if constexpr(N == 1)
    {
        return reg.data[0];
    }
        
    // vector case
    if constexpr(N == 2)
    {
        constexpr int Length = Tile::Layout::Shape::dim(0);
        wp::vec_t<Length, typename Tile::Type> v;
        for (int i=0; i < Length; ++i)
            v[i] = reg.data[i];

        return v;
    }
}

template <typename Tile, typename Value>
inline CUDA_CALLABLE void adj_untile(Tile& tile, Tile& adj_tile, Value& adj_ret)
{    
    auto adj = adj_tile.copy_to_register();   
    
    constexpr int N = Tile::Layout::Shape::N;

    // scalar case
    if constexpr(N == 1)
    {
        adj.data[0] += adj_ret;
    }

    // vector case
    if constexpr(N == 2)
    {
        constexpr int Length = Tile::Layout::Shape::dim(0);
        for (int i=0; i < Length; ++i)
            adj.data[i] += adj_ret[i];
    }

    adj_tile.assign(adj);
}

// zero initialized tile
template <typename T, unsigned... Shape>
inline CUDA_CALLABLE auto tile_zeros()
{
    // tile variable assignment operator will handle initialization (since lhs could be shared/register tile)
    return T(0);
}

// one-initialized tile
template <typename T, unsigned... Shape>
inline CUDA_CALLABLE auto tile_ones()
{
    // tile variable assignment operator will handle initialization (since lhs could be shared/register tile)
    return T(1);
}

// tile with evenly spaced values
template <typename T, int Len>
inline CUDA_CALLABLE auto tile_arange(T start, T stop, T step)
{
    auto out = tile_register<T, Len>();

    using Layout = typename decltype(out)::Layout;
    
    WP_PRAGMA_UNROLL
    for (int i=0; i < Layout::NumRegs; ++i)
    {
        const int linear = Layout::linear_from_register(i);

        // handle case where tile size is not
        // aligned to block dimensions
        if (!Layout::valid(linear))
            break;

        out.data[i] = start + linear*step;
    }
    
    return out;
}

template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_arange(T start, T stop, T step,
                                          T& adj_start, T& adj_stop, T& adj_step, AdjTile& adj_ret) {}

// entry point for load operations, these just return a reference to a global memory array + coordinate
template <unsigned... Shape, typename... Indices, typename T>
inline CUDA_CALLABLE auto tile_load(array_t<T>& src, Indices... offset) 
{
    return tile_global_t<T, tile_shape_t<Shape...>>(src, tile_coord(offset...)); 
}

// // entry point for tile store operations
// template <typename... Indices, typename T, typename Tile>
// inline CUDA_CALLABLE void tile_store(array_t<T>& dest, Tile& src, Indices... x)
// {
//     src.copy_to_global(tile_global_t<T, typename Tile::Layout::Shape>(dest, tile_coord(x))); 
// }

// entry point for tile store operations
template <typename T, typename Tile>
inline CUDA_CALLABLE void tile_store(array_t<T>& dest, int x, Tile& src) { src.copy_to_global(tile_global_t<T, typename Tile::Layout::Shape>(dest, tile_coord(x))); }
template <typename T, typename Tile>
inline CUDA_CALLABLE void tile_store(array_t<T>& dest, int x, int y, Tile& src) { src.copy_to_global(tile_global_t<T, typename Tile::Layout::Shape>(dest, tile_coord(x, y))); }
template <typename T, typename Tile>
inline CUDA_CALLABLE void tile_store(array_t<T>& dest, int x, int y, int z, Tile& src) { src.copy_to_global(tile_global_t<T, typename Tile::Layout::Shape>(dest, tile_coord(x, y, z))); }
template <typename T, typename Tile>
inline CUDA_CALLABLE void tile_store(array_t<T>& dest, int x, int y, int z, int w, Tile& src) { src.copy_to_global(tile_global_t<T, typename Tile::Layout::Shape>(dest, tile_coord(x, y, z, w))); }


// compiler struggles with these if they are one line
template <typename T, typename Tile>
inline CUDA_CALLABLE auto tile_atomic_add(array_t<T>& dest, int x, Tile& src) {
    tile_global_t<T, typename Tile::Layout::Shape> global(dest, tile_coord(x));
    return src.atomic_add(global);
}
template <typename T, typename Tile>
inline CUDA_CALLABLE auto tile_atomic_add(array_t<T>& dest, int x, int y, Tile& src) {
    tile_global_t<T, typename Tile::Layout::Shape> global(dest, tile_coord(x, y));
    return src.atomic_add(global);
}
template <typename T, typename Tile>
inline CUDA_CALLABLE auto tile_atomic_add(array_t<T>& dest, int x, int y, int z, Tile& src) {
    tile_global_t<T, typename Tile::Layout::Shape> global(dest, tile_coord(x, y, z));
    return src.atomic_add(global);
}
template <typename T, typename Tile>
inline CUDA_CALLABLE auto tile_atomic_add(array_t<T>& dest, int x, int y, int z, int w, Tile& src) {
    tile_global_t<T, typename Tile::Layout::Shape> global(dest, tile_coord(x, y, z, w));
    return src.atomic_add(global);
}


//-------------------------------------
// Adjoints

template <typename T, typename AdjTile, typename Coord>
inline CUDA_CALLABLE void adj_tile_load(array_t<T>& src, Coord c,
                                        array_t<T>& adj_src, Coord adj_c,
                                        AdjTile& adj_ret)
{
    tile_global_t<T, typename AdjTile::Layout::Shape> dest(src, c);
    
    // we allow users to override grad of src
    if (adj_src.data)
        dest.data.grad = adj_src.data;

    adj_ret.atomic_add_grad(dest);   
}


template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load(array_t<T>& src, int x, array_t<T>& adj_src, int adj_x, AdjTile& adj_ret) { adj_tile_load( src, tile_coord(x), adj_src, tile_coord(0), adj_ret); }
template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load(array_t<T>& src, int x, int y, array_t<T>& adj_src, int adj_x, int adj_y, AdjTile& adj_ret) { adj_tile_load( src, tile_coord(x, y), adj_src, tile_coord(0,0), adj_ret); }
template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load(array_t<T>& src, int x, int y, int z, array_t<T>& adj_src, int adj_x, int adj_y, int adj_z, AdjTile& adj_ret) { adj_tile_load( src, tile_coord(x, y, z), adj_src, tile_coord(0,0,0), adj_ret); }
template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load(array_t<T>& src, int x, int y, int z, int w, array_t<T>& adj_src, int adj_x, int adj_y, int adj_z, int adj_w, AdjTile& adj_ret) { adj_tile_load( src, tile_coord(x, y, z, w), adj_src, tile_coord(0,0,0,0), adj_ret); }



template <typename T, typename Tile, typename AdjTile, typename Coord>
inline CUDA_CALLABLE void adj_tile_store(array_t<T>& dest, Coord c, Tile& t, array_t<T>& adj_dest, Coord adj_c, AdjTile& adj_t)
{  
    tile_global_t<T, typename AdjTile::Layout::Shape> src(dest, c);
    
    // we allow users to override grad of src
    if (adj_dest.data)
        src.data.grad = adj_dest.data;        

    if (src.data.grad == nullptr)
        return;

    adj_t.grad_add(src);
}

template <typename T, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_store(array_t<T>& dest, int x, Tile& t, array_t<T>& adj_dest, int adj_x, AdjTile& adj_t) { adj_tile_store(dest, tile_coord(x), t, adj_dest, tile_coord(0), adj_t); }
template <typename T, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_store(array_t<T>& dest, int x, int y, Tile& t, array_t<T>& adj_dest, int adj_x, int adj_y, AdjTile& adj_t) { adj_tile_store(dest, tile_coord(x, y), t, adj_dest, tile_coord(0,0), adj_t); }
template <typename T, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_store(array_t<T>& dest, int x, int y, int z, Tile& t, array_t<T>& adj_dest, int adj_x, int adj_y, int adj_z, AdjTile& adj_t) { adj_tile_store(dest, tile_coord(x, y, z), t, adj_dest, tile_coord(0,0,0), adj_t); }
template <typename T, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_store(array_t<T>& dest, int x, int y, int z, int w, Tile& t, array_t<T>& adj_dest, int adj_x, int adj_y, int adj_z, int adj_w, AdjTile& adj_t) { adj_tile_store(dest, tile_coord(x, y, z, w), t, adj_dest, tile_coord(0,0,0,0), adj_t); }



// adj_tile_atomic_add is an alias for adj_tile_store
template <typename T, typename Tile, typename AdjTile, typename AdjRet>
inline CUDA_CALLABLE void adj_tile_atomic_add(array_t<T>& dest, int x, Tile& t, array_t<T>& adj_dest, int adj_x, AdjTile& adj_t, AdjRet& adj_ret) { adj_tile_store(dest, tile_coord(x), t, adj_dest, tile_coord(adj_x), adj_t); }
template <typename T, typename Tile, typename AdjTile, typename AdjRet>
inline CUDA_CALLABLE void adj_tile_atomic_add(array_t<T>& dest, int x, int y, Tile& t, array_t<T>& adj_dest, int adj_x, int adj_y, AdjTile& adj_t, AdjRet& adj_ret) { adj_tile_store(dest, tile_coord(x, y), t, adj_dest, tile_coord(adj_x, adj_y), adj_t); }
template <typename T, typename Tile, typename AdjTile, typename AdjRet>
inline CUDA_CALLABLE void adj_tile_atomic_add(array_t<T>& dest, int x, int y, int z, Tile& t, array_t<T>& adj_dest, int adj_x, int adj_y, int adj_z, AdjTile& adj_t, AdjRet& adj_ret) { adj_tile_store(dest, tile_coord(x, y, z), t, adj_dest, tile_coord(adj_x, adj_y, adj_z), adj_t); }
template <typename T, typename Tile, typename AdjTile, typename AdjRet>
inline CUDA_CALLABLE void adj_tile_atomic_add(array_t<T>& dest, int x, int y, int z, int w, Tile& t, array_t<T>& adj_dest, int adj_x, int adj_y, int adj_z, int adj_w, AdjTile& adj_t, AdjRet& adj_ret) { adj_tile_store(dest, tile_coord(x, y, z, w), t, adj_dest, tile_coord(adj_x, adj_y, adj_z, adj_w), adj_t); }


// unary map
template <typename Tile, typename Fwd>
inline CUDA_CALLABLE auto tile_map(Fwd op,
                                   Tile &a)
{
    auto out = tile_register_like<Tile>();
    auto a_reg = a.copy_to_register();

    using Layout = typename decltype(out)::Layout;
    
    WP_PRAGMA_UNROLL
    for (int i=0; i < Layout::NumRegs; ++i)
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

    using Layout = typename decltype(a_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i=0; i < Layout::NumRegs; ++i)
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
    auto out = tile_register_like<TileA>();

    auto a_reg = a.copy_to_register();
    auto b_reg = b.copy_to_register();

    using Layout = typename decltype(out)::Layout;

    WP_PRAGMA_UNROLL
    for (int i=0; i < Layout::NumRegs; ++i)
    {
        out.data[i] = op(a_reg.data[i], b_reg.data[i]);
    }

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

    using Layout = typename decltype(a_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i=0; i < Layout::NumRegs; ++i)
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

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_add(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b, AdjTile& adj_c)
{   
    adj_tile_binary_map(add, a, b, adj_add, adj_a, adj_b, adj_c);
}

// tile - tile
template <typename TileA, typename TileB>
inline CUDA_CALLABLE auto tile_sub(TileA& a, TileB& b)
{
    return tile_binary_map(sub, a, b);
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_sub(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b, AdjTile& adj_c)
{   
    adj_tile_binary_map(sub, a, b, adj_sub, adj_a, adj_b, adj_c);
}


// tile*scalar
template <typename Tile>
inline CUDA_CALLABLE auto tile_mul(Tile& a, const typename Tile::Type& s)
{
    // promote scalar to a constant tile
    auto s_tile = tile_register_t<typename Tile::Type, tile_layout_register_t<typename Tile::Layout::Shape>>(s);

    return tile_binary_map(mul, a, s_tile);
}

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_mul(Tile& a, const typename Tile::Type& s,
                                       Tile& adj_a, typename Tile::Type& adj_s,
                                       AdjTile& adj_c)
{
    auto s_tile = tile_register_like<Tile>();
    auto adj_s_tile = tile_register_like<Tile>();

    using Layout = typename decltype(adj_s_tile)::Layout;

    // initialize to constant
    s_tile = s;

    adj_tile_binary_map(mul, a, s_tile, adj_mul, adj_a, adj_s_tile, adj_c);

    for (int i=0; i < Layout::NumRegs; ++i)
    {
        adj_s += adj_s_tile.data[i];
    }
}


// scalar*tile
template <typename Tile>
inline CUDA_CALLABLE auto tile_mul(const typename Tile::Type& s, Tile& a)
{
    return tile_mul(a, s);
}

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_mul(const typename Tile::Type& s, Tile& a,
                                       typename Tile::Type& adj_s, Tile& adj_a,
                                       AdjTile& adj_c)
{
    adj_tile_mul(a, s, adj_a, adj_s, adj_c);
}


template <typename TileA, typename TileB>
inline CUDA_CALLABLE void tile_add_inplace(TileA& a, TileB& b)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;

    // verify shapes and sizes are compatible
    static_assert(ShapeA::N == ShapeB::N, "Tile shapes must match for inplace addition");
    static_assert(ShapeA::size() == ShapeB::size(), "Tile sizes must match for inplace addition");

    auto a_reg = a.copy_to_register();
    auto b_reg = b.copy_to_register();

    using Layout = typename decltype(b_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i=0; i < Layout::NumRegs; ++i)
    {
        const int linear = Layout::linear_from_register(i);

        if(!Layout::valid(linear))
            break;

        a_reg.data[i] += b_reg.data[i];
    }

    a.assign(a_reg);
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_add_inplace(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;

    // verify shapes and sizes are compatible
    static_assert(ShapeA::N == ShapeB::N, "Tile shapes must match for inplace addition");
    static_assert(ShapeA::size() == ShapeB::size(), "Tile sizes must match for inplace addition");

    // allocate storage for adjoints
    auto adj_a_reg = adj_a.grad_to_register();
    auto adj_b_reg = tile_register_like<TileB>();

    using Layout = typename decltype(adj_a_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i=0; i < Layout::NumRegs; ++i)
    {
        const int linear = Layout::linear_from_register(i);

        if(!Layout::valid(linear))
            break;

        adj_b_reg.data[i] += adj_a_reg.data[i];
    }

    adj_b.grad_add(adj_b_reg);
}

template <typename TileA, typename TileB>
inline CUDA_CALLABLE void tile_sub_inplace(TileA& a, TileB& b)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;

    // verify shapes and sizes are compatible
    static_assert(ShapeA::N == ShapeB::N, "Tile shapes must match for inplace subtraction");
    static_assert(ShapeA::size() == ShapeB::size(), "Tile sizes must match for inplace subtraction");

    // work with register tiles for inplace operations, regardless of the storage type of the input tiles
    auto a_reg = a.copy_to_register();
    auto b_reg = b.copy_to_register();

    using Layout = typename decltype(a_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i=0; i < Layout::NumRegs; ++i)
    {
        const int linear = Layout::linear_from_register(i);

        if(!Layout::valid(linear))
            break;

        a_reg.data[i] -= b_reg.data[i];
    }

    a.assign(a_reg);
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_sub_inplace(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;

    // verify shapes and sizes are compatible
    static_assert(ShapeA::N == ShapeB::N, "Tile shapes must match for inplace subtraction");
    static_assert(ShapeA::size() == ShapeB::size(), "Tile sizes must match for inplace subtraction");

    // allocate storage for adjoints
    auto adj_a_reg = adj_a.grad_to_register();
    auto adj_b_reg = tile_register_like<TileB>();

    using Layout = typename decltype(adj_a_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i=0; i < Layout::NumRegs; ++i)
    {
        const int linear = Layout::linear_from_register(i);

        if(!Layout::valid(linear))
            break;

        adj_b_reg.data[i] -= adj_a_reg.data[i];
    }

    adj_b.grad_add(adj_b_reg);
}


template<typename Tile>
typename Tile::Type tile_extract(Tile& t, int i) { return t.extract(tile_coord(i)); }
template<typename Tile>
typename Tile::Type tile_extract(Tile& t, int i, int j) { return t.extract(tile_coord(i,j)); }
template<typename Tile>
typename Tile::Type tile_extract(Tile& t, int i, int j, int k) { return t.extract(tile_coord(i,j,k)); }
template<typename Tile>
typename Tile::Type tile_extract(Tile& t, int i, int j, int k, int l) { return t.extract(tile_coord(i,j,k,l)); }

template<typename Tile, typename AdjTile>
void adj_tile_extract(Tile& t, int i, AdjTile& adj_t, int adj_i, typename Tile::Type adj_ret) { adj_t.adj_extract(tile_coord(i), adj_ret); }
template<typename Tile, typename AdjTile>
void adj_tile_extract(Tile& t, int i, int j, AdjTile& adj_t, int adj_i, int adj_j, typename Tile::Type adj_ret) { adj_t.adj_extract(tile_coord(i, j), adj_ret); }
template<typename Tile, typename AdjTile>
void adj_tile_extract(Tile& t, int i, int j, int k, AdjTile& adj_t, int adj_i, int adj_j, int adj_k, typename Tile::Type adj_ret) { adj_t.adj_extract(tile_coord(i, j, k), adj_ret); }
template<typename Tile, typename AdjTile>
void adj_tile_extract(Tile& t, int i, int j, int k, int l, AdjTile& adj_t, int adj_i, int adj_j, int adj_k, int adj_l, typename Tile::Type adj_ret) { adj_t.adj_extract(tile_coord(i, j, k, l), adj_ret); }


template<typename Tile>
void tile_add_inplace(Tile& t, int i, typename Tile::Type value) { t.add_inplace(tile_coord(i), value); }
template<typename Tile>
void tile_add_inplace(Tile& t, int i, int j, typename Tile::Type value) { t.add_inplace(tile_coord(i,j), value); }
template<typename Tile>
void tile_add_inplace(Tile& t, int i, int j, int k, typename Tile::Type value) { t.add_inplace(tile_coord(i,j,k), value); }
template<typename Tile>
void tile_add_inplace(Tile& t, int i, int j, int k, int l, typename Tile::Type value) { t.add_inplace(tile_coord(i,j,k,l), value); }

template<typename Tile>
void tile_sub_inplace(Tile& t, int i, typename Tile::Type value) { t.sub_inplace(tile_coord(i), value); }
template<typename Tile>
void tile_sub_inplace(Tile& t, int i, int j, typename Tile::Type value) { t.sub_inplace(tile_coord(i,j), value); }
template<typename Tile>
void tile_sub_inplace(Tile& t, int i, int j, int k, typename Tile::Type value) { t.sub_inplace(tile_coord(i,j,k), value); }
template<typename Tile>
void tile_sub_inplace(Tile& t, int i, int j, int k, int l, typename Tile::Type value) { t.sub_inplace(tile_coord(i,j,k,l), value); }

template<typename Tile, typename AdjTile>
void adj_tile_add_inplace(Tile& t, int i, typename Tile::Type value, AdjTile& adj_t, int adj_i, typename Tile::Type& adj_value) { adj_t.adj_add_inplace(tile_coord(i), adj_value); }
template<typename Tile, typename AdjTile>
void adj_tile_add_inplace(Tile& t, int i, int j, typename Tile::Type value, AdjTile& adj_t, int adj_i, int adj_j, typename Tile::Type& adj_value) { adj_t.adj_add_inplace(tile_coord(i, j), adj_value); }
template<typename Tile, typename AdjTile>
void adj_tile_add_inplace(Tile& t, int i, int j, int k, typename Tile::Type value, AdjTile& adj_t, int adj_i, int adj_j, int adj_k, typename Tile::Type& adj_value) { adj_t.adj_add_inplace(tile_coord(i, j, k), adj_value); }
template<typename Tile, typename AdjTile>
void adj_tile_add_inplace(Tile& t, int i, int j, int k, int l, typename Tile::Type value, AdjTile& adj_t, int adj_i, int adj_j, int adj_k, int adj_l, typename Tile::Type& adj_value) { adj_t.adj_add_inplace(tile_coord(i, j, k, l), adj_value); }

template<typename Tile, typename AdjTile>
void adj_tile_sub_inplace(Tile& t, int i, typename Tile::Type value, AdjTile& adj_t, int adj_i, typename Tile::Type& adj_value) { adj_t.adj_sub_inplace(tile_coord(i), adj_value); }
template<typename Tile, typename AdjTile>
void adj_tile_sub_inplace(Tile& t, int i, int j, typename Tile::Type value, AdjTile& adj_t, int adj_i, int adj_j, typename Tile::Type& adj_value) { adj_t.adj_sub_inplace(tile_coord(i, j), adj_value); }
template<typename Tile, typename AdjTile>
void adj_tile_sub_inplace(Tile& t, int i, int j, int k, typename Tile::Type value, AdjTile& adj_t, int adj_i, int adj_j, int adj_k, typename Tile::Type& adj_value) { adj_t.adj_sub_inplace(tile_coord(i, j, k), adj_value); }
template<typename Tile, typename AdjTile>
void adj_tile_sub_inplace(Tile& t, int i, int j, int k, int l, typename Tile::Type value, AdjTile& adj_t, int adj_i, int adj_j, int adj_k, int adj_l, typename Tile::Type& adj_value) { adj_t.adj_sub_inplace(tile_coord(i, j, k, l), adj_value); }

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
    static constexpr int Stride = Tile::Layout::Shape::dim(1);
    
    using T = typename Tile::Type;    

    inline partition_t(Tile& A) 
    {
        data = A.data.ptr;
        
        // todo: do ceil div for non-multiples of M,N
        shape[0] = Tile::Layout::Shape::dim(0)/PartitionM;
        shape[1] = Tile::Layout::Shape::dim(1)/PartitionN;
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
            out.data[i][j] = partitioned_gemm::index(tile.data, tile_i + i, tile_j + j, Partition::Stride);
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

    //static_assert(is_same<typename TileA::Type, typename TileB::Type>::value);

    const int length = partition_size(C_tile);

    for (int t=WP_TILE_THREAD_IDX; t < length; t += WP_TILE_BLOCK_DIM)
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

template <typename LayoutA, typename LayoutB, typename LayoutC, typename StorageA, typename StorageB, typename StorageC, typename T>
inline CUDA_CALLABLE void scalar_matmul(const StorageA& A, const StorageB& B, StorageC& C, T scale)
{
    for (int t=WP_TILE_THREAD_IDX; t < LayoutC::Size; t += WP_TILE_BLOCK_DIM)
    {  
        auto coord = LayoutC::coord_from_linear(t);

        int i = coord[0];
        int j = coord[1];

        // accumulator
        auto sum = C(coord)*scale;

        WP_PRAGMA_UNROLL
        for (int k=0; k < LayoutA::Shape::dim(1); k++)
        {
            const auto a = A(tile_coord(i, k));
            const auto b = B(tile_coord(k, j));

            sum = muladd<decltype(sum)>(a, b, sum);
        }
        
        C(coord) = sum;
    }
}

template <typename TileA, typename TileL>
inline CUDA_CALLABLE void scalar_cholesky(TileA& A, TileL& L)
{
    using T = typename TileA::Type;    
    constexpr int n = TileA::Layout::Shape::dim(1);

    for (int j=0; j < n; ++j)
    {
        T s = A.data(tile_coord(j, j));

        for (int k=0; k < j; ++k) 
        {
            T r = L.data(tile_coord(j, k));
            s -= r * r;
        }

        s = wp::sqrt(s);
        T invS = 1.0 / s;

        L.data(tile_coord(j, j)) = s;

        for (int i=j+1; i < n; ++i)
        {
            s = A.data(tile_coord(i, j));

            for (int k=0; k < j; ++k)
            {
                s -= L.data(tile_coord(i, k)) * L.data(tile_coord(j, k));
            }

            L.data(tile_coord(i, j)) = s * invS;           
        }

        // zero out upper triangular portion
        for (int k=j+1; k < n; ++k)
        {
            L.data(tile_coord(j,k)) = T(0.0);
        }
    }
}

template <typename TileL, typename TileX, typename TileY>
inline CUDA_CALLABLE void scalar_cholesky_forward_substitution(TileL& L, TileX& X, TileY& Y)
{
    using T = typename TileL::Type;    
    constexpr int n = TileL::Layout::Shape::dim(1);

    for (int i=0; i < n; ++i)
    {
        T s = Y.data(tile_coord(i));

        for (int j=0; j < i; ++j)
            s -= L.data(tile_coord(i,j)) * X.data(tile_coord(j));

        X.data(tile_coord(i)) = s / L.data(tile_coord(i, i));
    }
}

template <typename TileL, typename TileX>
inline CUDA_CALLABLE void scalar_cholesky_back_substitution(TileL& L, TileX& X)
{
    using T = typename TileL::Type;    
    constexpr int n = TileL::Layout::Shape::dim(1);

    for (int i=n-1; i >= 0; --i)
    {
        T s = X.data(tile_coord(i));

        for (int j=i+1; j < n; ++j)
            s -= L.data(tile_coord(j, i)) * X.data(tile_coord(j));

        X.data(tile_coord(i)) = s / L.data(tile_coord(i, i));
    }
}

template <typename TileL, typename TileX, typename TileY>
inline CUDA_CALLABLE void scalar_cholesky_solve(TileL& L, TileX& X, TileY& Y)
{
    scalar_cholesky_forward_substitution(L, X, Y);
    scalar_cholesky_back_substitution(L, X);        
}


} // namespace partition_gemm


template <int Add, typename Fwd, typename AdjA, typename AdjB, typename TileA, typename TileB, typename TileC>
TileC& tile_matmul(Fwd fun_forward, AdjA fun_backward_A, AdjB fun_backward_B, TileA& A, TileB& B, TileC& C)
{       
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;
    using ShapeC = typename TileC::Layout::Shape;

    static_assert(ShapeA::N == 2, "Expected ShapeA::N == 2");
    static_assert(ShapeB::N == 2, "Expected ShapeB::N == 2");
    static_assert(ShapeC::N == 2, "Expected ShapeC::N == 2");

    static_assert(ShapeA::dim(1) == ShapeB::dim(0), "Expected ShapeA::dim(1) == ShapeB::dim(0)");
    static_assert(ShapeC::dim(0) == ShapeA::dim(0), "Expected ShapeC::dim(0) == ShapeA::dim(0)");
    static_assert(ShapeC::dim(1) == ShapeB::dim(1), "Expected ShapeC::dim(1) == ShapeB::dim(1)");
    

    using T = typename TileA::Type;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_matmul<typename TileA::Layout, typename TileB::Layout, typename TileC::Layout>(A.data, B.data, C.data, T(Add));
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

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    auto At = tile_transpose(A);
    auto Bt = tile_transpose(B);

    partitioned_gemm::scalar_matmul<typename TileC::Layout, typename decltype(Bt)::Layout, typename TileA::Layout>(adj_C.grad, Bt.data, adj_A.grad, T(1.0));
    partitioned_gemm::scalar_matmul<typename decltype(At)::Layout, typename TileC::Layout, typename TileB::Layout>(At.data, adj_C.grad, adj_B.grad, T(1.0));
#else
    fun_backward_A(T(1.0), adj_C.grad.ptr, B.data.ptr, T(1.0), adj_A.grad.ptr);
    fun_backward_B(T(1.0), A.data.ptr, adj_C.grad.ptr, T(1.0), adj_B.grad.ptr);
#endif

    WP_TILE_SYNC();
}

// backward for the out = wp.tile_matmul(a, b) syntax
template <typename Fwd, typename AdjA, typename AdjB, typename TileA, typename TileB, typename TileC>
void adj_tile_matmul(Fwd fun_forward, AdjA fun_backward_A, AdjB fun_backward_B, TileA& A, TileB& B, TileC& C, 
                   Fwd adj_fun_forward, AdjA adj_fun_backward_A, AdjB adj_fun_backward_B, TileA& adj_A, TileB& adj_B, TileC& adj_C, TileC& adj_ret)
{   
    using T = typename TileA::Type;    

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    auto At = tile_transpose(A);
    auto Bt = tile_transpose(B);

    partitioned_gemm::scalar_matmul<typename TileC::Layout, typename decltype(Bt)::Layout, typename TileA::Layout>(adj_C.grad, Bt.data, adj_A.grad, T(1.0));
    partitioned_gemm::scalar_matmul<typename decltype(At)::Layout, typename TileC::Layout, typename TileB::Layout>(At.data, adj_C.grad, adj_B.grad, T(1.0));
#else
    fun_backward_A(T(1.0), adj_C.grad.ptr, B.data.ptr, T(1.0), adj_A.grad.ptr);
    fun_backward_B(T(1.0), A.data.ptr, adj_C.grad.ptr, T(1.0), adj_B.grad.ptr);
#endif

    WP_TILE_SYNC();
}

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

#define tile_fft()
#define tile_ifft()

#define adj_tile_fft()
#define adj_tile_ifft()

#else

// TODO(lcambier): use a properly overaligned complex type that matches cuFFTDx's expectation
// and remove the need for __align__(16) dtypes data[...]
#define tile_fft(function_name, dtype, shared_memory_size, batch_size, ept, Xinout) \
    do { \
        void function_name(dtype*, dtype*); \
        char* buffer = (char*)wp::tile_alloc_shared(shared_memory_size); \
        __align__(16) dtype data[ept]; \
        for(int b = 0; b < (int)batch_size; b++) { \
            dtype* inout = Xinout.data + (int)b * (int)ept; \
            memcpy(data, inout, sizeof(dtype) * ept); \
            function_name(data, (dtype*)buffer); \
            memcpy(inout, data, sizeof(dtype) * ept); \
            WP_TILE_SYNC(); \
        } \
        wp::tile_alloc_shared(-shared_memory_size); \
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

#endif // !defined(__CUDA_ARCH__)

template <typename Fwd, typename TileA, typename TileL>
TileL& tile_cholesky(Fwd fun_forward, TileA& A, TileL& L)
{       
    // Copy to L
    L = A;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

    partitioned_gemm::scalar_cholesky(A, L);

#else


    // Call cholesky on L
    WP_TILE_SYNC();
    
    fun_forward(L.data.ptr, TileL::Layout::Shape::dim(0));
    
    WP_TILE_SYNC();

    // Zero-out the upper triangular part of L

    WP_PRAGMA_UNROLL
    for (int i=WP_TILE_THREAD_IDX; i < TileL::Layout::Size; i += WP_TILE_BLOCK_DIM)
    {
        auto c = TileL::Layout::coord_from_linear(i);
        
        if(c[0] < c[1]) 
            L.data(c) = 0.0;
    }

    WP_TILE_SYNC();

#endif

    return L;
}

#define adj_tile_cholesky(function_name, A, L, \
                          adj_function_name, adj_A, adj_L, adj_ret) \
    do { \
        assert(false); \
    } while (0)

template <typename Fwd, typename TileL, typename TileX, typename TileY>
TileY& tile_cholesky_solve(Fwd fun_forward, TileL& L, TileX& X, TileY& Y)
{       
    // Copy x to y

    Y = X;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

    partitioned_gemm::scalar_cholesky_solve(L, X, Y);

#else

    // Call cholesky solve on L & y

    WP_TILE_SYNC();
    
    fun_forward(L.data.ptr, Y.data.ptr); \

    WP_TILE_SYNC();
    
#endif

    return Y;
}

#define adj_tile_cholesky_solve(function_name, L, X, Y, \
                                adj_function_name, adj_L, adj_X, adj_Y, adj_ret) \
    do { \
        assert(false); \
    } while (0)






template <typename TileL, typename TileY, typename TileZ>
TileZ& tile_lower_solve(TileL& L, TileY& y, TileZ& z)
{       
    // Copy y to z
    //z = y;
	
#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

    z = y;
    partitioned_gemm::scalar_cholesky_forward_substitution(L, y, z);

#else

    // Call cholesky solve on L & z

    WP_TILE_SYNC();
    
    using T = typename TileL::Type;    
    constexpr int n = TileL::Layout::Shape::dim(1);

    if constexpr (TileY::Layout::Shape::N == 1)
    {
        // Optimize similar to the N==2 case
        constexpr int m = 1;
        constexpr int N = n * m;
        const int num_threads = WP_TILE_BLOCK_DIM;

        // Parallel copy y to z
        for (int idx = WP_TILE_THREAD_IDX; idx < N; idx += num_threads)
        {
            int row = idx; // since m == 1, col is always 0
            if (row < n)
                z.data(tile_coord(row)) = y.data(tile_coord(row));
        }
        WP_TILE_SYNC();

        // Forward substitution
        for (int i = 0; i < n; ++i)
        {
            // Divide the diagonal element (only one batch)
            if (WP_TILE_THREAD_IDX == 0)
            {
                z.data(tile_coord(i)) /= L.data(tile_coord(i, i));
            }
            WP_TILE_SYNC();

            // Update the rest of the column in parallel
            for (int idx = WP_TILE_THREAD_IDX; idx < n - i - 1; idx += num_threads)
            {
                int row = i + 1 + idx;
                if (row < n)
                {
                    z.data(tile_coord(row)) -= L.data(tile_coord(row, i)) * z.data(tile_coord(i));
                }
            }
            WP_TILE_SYNC();
        }
    }
    else if constexpr (TileY::Layout::Shape::N == 2)
    {
        // Processes multiple right hand sides in one go

        constexpr int m = TileY::Layout::Shape::dim(1);
        constexpr int N = n * m;
        const int num_threads = WP_TILE_BLOCK_DIM;

        // Parallel copy y to z
        for (int idx = WP_TILE_THREAD_IDX; idx < N; idx += num_threads)
        {
            int row = idx / m;
            int col = idx % m;
            if (row < n && col < m)
                z.data(tile_coord(row, col)) = y.data(tile_coord(row, col));
        }
        WP_TILE_SYNC();

        // Forward substitution
        for (int i = 0; i < n; ++i)
        {
            // Divide the diagonal element for all batches in parallel
            for (int batchId = WP_TILE_THREAD_IDX; batchId < m; batchId += num_threads)
            {
                z.data(tile_coord(i, batchId)) /= L.data(tile_coord(i, i));
            }
            WP_TILE_SYNC();

            // Update the rest of the column in parallel
            for (int idx = WP_TILE_THREAD_IDX; idx < (n - i - 1) * m; idx += num_threads)
            {
                int row = i + 1 + (idx / m);
                int col = idx % m;
                if (row < n && col < m)
                {
                    z.data(tile_coord(row, col)) -= L.data(tile_coord(row, i)) * z.data(tile_coord(i, col));
                }
            }
            WP_TILE_SYNC();
        }
    }

    WP_TILE_SYNC();
    
#endif

    return z;
}

#define adj_tile_lower_solve(L, y, z, \
                                adj_L, adj_y, adj_z, adj_ret) \
    do { \
        assert(false); \
    } while (0)
		
	

template <typename TileU, typename TileZ, typename TileX>
TileX& tile_upper_solve(TileU& U, TileZ& z, TileX& x)
{       
    // Copy z to x
    //x = z;
	
#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

    x = z;
    partitioned_gemm::scalar_cholesky_back_substitution(U, x);

#else

    // Call cholesky solve on U & x

    WP_TILE_SYNC();
    
    using T = typename TileU::Type;    
    constexpr int n = TileU::Layout::Shape::dim(1);

    if constexpr (TileZ::Layout::Shape::N == 1)
    {
        // Optimize similar to the N==2 case
        constexpr int m = 1;
        constexpr int N = n * m;
        const int num_threads = WP_TILE_BLOCK_DIM;

        // Parallel copy z to x
        for (int idx = WP_TILE_THREAD_IDX; idx < N; idx += num_threads)
        {
            int row = idx; // since m == 1, col is always 0
            if (row < n)
                x.data(tile_coord(row)) = z.data(tile_coord(row));
        }
        WP_TILE_SYNC();

        // Backward substitution
        for (int i = n - 1; i >= 0; --i)
        {
            // Divide the diagonal element for all batches in parallel (only one batch)
            if (WP_TILE_THREAD_IDX == 0)
            {
                x.data(tile_coord(i)) /= U.data(tile_coord(i, i));
            }
            WP_TILE_SYNC();

            // Update the rest of the column in parallel
            for (int idx = WP_TILE_THREAD_IDX; idx < i; idx += num_threads)
            {
                int row = idx;
                if (row < i)
                {
                    x.data(tile_coord(row)) -= U.data(tile_coord(row, i)) * x.data(tile_coord(i));
                }
            }
            WP_TILE_SYNC();
        }
    }
    else if constexpr (TileZ::Layout::Shape::N == 2)
    {
        // Processes multiple right hand sides in one go

        constexpr int m = TileZ::Layout::Shape::dim(1);
        constexpr int N = n * m;
        const int num_threads = WP_TILE_BLOCK_DIM;

        // Parallel copy z to x
        for (int idx = WP_TILE_THREAD_IDX; idx < N; idx += num_threads)
        {
            int row = idx / m;
            int col = idx % m;
            if (row < n && col < m)
                x.data(tile_coord(row, col)) = z.data(tile_coord(row, col));
        }
        WP_TILE_SYNC();

        // Backward substitution
        for (int i = n - 1; i >= 0; --i)
        {
            // Divide the diagonal element for all batches in parallel
            for (int batchId = WP_TILE_THREAD_IDX; batchId < m; batchId += num_threads)
            {
                x.data(tile_coord(i, batchId)) /= U.data(tile_coord(i, i));
            }
            WP_TILE_SYNC();

            // Update the rest of the column in parallel
            for (int idx = WP_TILE_THREAD_IDX; idx < i * m; idx += num_threads)
            {
                int row = idx / m;
                int col = idx % m;
                if (row < i && col < m)
                {
                    x.data(tile_coord(row, col)) -= U.data(tile_coord(row, i)) * x.data(tile_coord(i, col));
                }
            }
            WP_TILE_SYNC();
        }
    }

    WP_TILE_SYNC();
    
#endif

    return x;
}

#define adj_tile_upper_solve(U, z, x, \
                                adj_U, adj_z, adj_x, adj_ret) \
    do { \
        assert(false); \
    } while (0)




    

template <typename Tile>
inline CUDA_CALLABLE auto tile_transpose(Tile& t)
{    
    static_assert(Tile::Layout::Shape::N == 2, "Expected Tile::Layout::Shape::N == 2");

    // alias incoming tile 
    constexpr int M = Tile::Layout::Shape::dim(0);
    constexpr int N = Tile::Layout::Shape::dim(1);

    constexpr int StrideM = Tile::Layout::Stride::dim(0);
    constexpr int StrideN = Tile::Layout::Stride::dim(1);

    return tile_shared_t<typename Tile::Type, tile_layout_strided_t<tile_shape_t<N,M>, tile_stride_t<StrideN, StrideM>>, false>(t.data.ptr, t.grad.ptr);
}

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_transpose(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{    
    auto a = tile_transpose(adj_ret);
    auto b = adj_t;
    
    adj_t.assign(tile_add(a,b));
}

template <int N, int StrideN, typename Tile>
inline CUDA_CALLABLE auto tile_broadcast(Tile& t)
{
    // alias incoming tile with new strides
    return tile_shared_t<typename Tile::Type, tile_layout_strided_t<tile_shape_t<N>, tile_stride_t<StrideN>>, false>(t.data.ptr, t.grad.ptr);
}

template <int M, int N, int StrideM, int StrideN, typename Tile>
inline CUDA_CALLABLE auto tile_broadcast(Tile& t)
{
    // alias incoming tile with new strides
    return tile_shared_t<typename Tile::Type, tile_layout_strided_t<tile_shape_t<M, N>, tile_stride_t<StrideM, StrideN>>, false>(t.data.ptr, t.grad.ptr);
}

template <int M, int N, int O, int StrideM, int StrideN, int StrideO, typename Tile>
inline CUDA_CALLABLE auto tile_broadcast(Tile& t)
{
    // alias incoming tile with new strides
    return tile_shared_t<typename Tile::Type, tile_layout_strided_t<tile_shape_t<M, N, O>, tile_stride_t<StrideM, StrideN, StrideO>>, false>(t.data.ptr, t.grad.ptr);
}

template <int M, int N, int O, int P, int StrideM, int StrideN, int StrideO, int StrideP, typename Tile>
inline CUDA_CALLABLE auto tile_broadcast(Tile& t)
{
    // alias incoming tile with new strides
    return tile_shared_t<typename Tile::Type, tile_layout_strided_t<tile_shape_t<M, N, O, P>, tile_stride_t<StrideM, StrideN, StrideO, StrideP>>, false>(t.data.ptr, t.grad.ptr);
}

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_broadcast(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{   
    // nop, since memory is aliased, grads already accumulated
}


template <typename ReturnTile, typename Tile, typename... Indices>
inline CUDA_CALLABLE auto tile_view(Tile& t, Indices... indices)
{   
    auto c = tile_coord(indices...);

    // return new tile with same strides
    typename Tile::Type* data_ptr = &t.data(c);
    typename Tile::Type* grad_ptr = nullptr;
    
    if (t.grad.ptr)
        grad_ptr = &t.grad(c);

    return ReturnTile(data_ptr, grad_ptr);
}


template <typename ReturnTile, typename Tile>
inline CUDA_CALLABLE auto tile_squeeze(Tile& t)
{
    // ReturnTile layout is set in builtins.py
    typename Tile::Type* data_ptr = t.data.ptr;
    typename Tile::Type* grad_ptr = nullptr;
    
    if (t.grad.ptr)
        grad_ptr = t.grad.ptr;

    return ReturnTile(data_ptr, grad_ptr);
}

template <typename Tile, typename AdjTile, typename AdjReturnTile>
inline CUDA_CALLABLE void adj_tile_squeeze(Tile& t, AdjTile& adj_t, AdjReturnTile& adj_ret)
{
    // nop, since memory is aliased, grads already accumulated
}


template <typename ReturnTile, typename Tile>
inline CUDA_CALLABLE auto tile_reshape(Tile& t)
{
    // ReturnTile layout is set in builtins.py
    typename Tile::Type* data_ptr = t.data.ptr;
    typename Tile::Type* grad_ptr = nullptr;
    
    if (t.grad.ptr)
        grad_ptr = t.grad.ptr;

    return ReturnTile(data_ptr, grad_ptr);
}

template <typename Tile, typename AdjTile, typename AdjReturnTile>
inline CUDA_CALLABLE void adj_tile_reshape(Tile& t, AdjTile& adj_t, AdjReturnTile& adj_ret)
{
    // nop, since memory is aliased, grads already accumulated
}

template <typename TileA, typename Scalar>
inline CUDA_CALLABLE void assign(TileA& dest, int i, const Scalar& src)
{   
    dest.data(tile_coord(i)) = src;
    WP_TILE_SYNC();
}
template <typename TileA, typename Scalar>
inline CUDA_CALLABLE void assign(TileA& dest, int i, int j, const Scalar& src)
{   
    dest.data(tile_coord(i, j)) = src;
    WP_TILE_SYNC();
}
template <typename TileA, typename Scalar>
inline CUDA_CALLABLE void assign(TileA& dest, int i, int j, int k, const Scalar& src)
{   
    dest.data(tile_coord(i, j, k)) = src;
    WP_TILE_SYNC();
}
template <typename TileA, typename Scalar>
inline CUDA_CALLABLE void assign(TileA& dest, int i, int j, int k, int l, const Scalar& src)
{   
    dest.data(tile_coord(i, j, k, l)) = src;
    WP_TILE_SYNC();
}

template <typename TileA, typename AdjTileA, typename Scalar>
inline CUDA_CALLABLE void adj_assign(TileA& dest, int i, const Scalar& src, AdjTileA& adj_dest, int adj_i, Scalar& adj_src)
{
    adj_src += dest.grad(tile_coord(i));
}
template <typename TileA, typename AdjTileA, typename Scalar>
inline CUDA_CALLABLE void adj_assign(TileA& dest, int i, int j, const Scalar& src, AdjTileA& adj_dest, int adj_i, int adj_j, Scalar& adj_src)
{
    adj_src += dest.grad(tile_coord(i, j));
}
template <typename TileA, typename AdjTileA, typename Scalar>
inline CUDA_CALLABLE void adj_assign(TileA& dest, int i, int j, int k, const Scalar& src, AdjTileA& adj_dest, int adj_i, int adj_j, int adj_k, Scalar& adj_src)
{
    adj_src += dest.grad(tile_coord(i, j, k));
}
template <typename TileA, typename AdjTileA, typename Scalar>
inline CUDA_CALLABLE void adj_assign(TileA& dest, int i, int j, int k, int l, const Scalar& src, AdjTileA& adj_dest, int adj_i, int adj_j, int adj_k, int adj_l, Scalar& adj_src)
{
    adj_src += dest.grad(tile_coord(i, j, k, l));
}

template <typename TileA, typename TileB, typename Coord>
inline CUDA_CALLABLE void tile_assign(TileA& dest, TileB& src, const Coord& offset)
{   
    using Layout = typename TileB::Layout;

    for (int t=WP_TILE_THREAD_IDX; t < Layout::Size; t += WP_TILE_BLOCK_DIM)
    {
        auto c = Layout::coord_from_linear(t);
        dest.data(c + offset) = src.data(c);
    }

    WP_TILE_SYNC();
}

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB, typename Coord, typename AdjCoord>
inline CUDA_CALLABLE void adj_tile_assign(TileA& dest, TileB& src, Coord offset,
                                          AdjTileA& adj_dest, AdjTileB& adj_src, AdjCoord adj_offset)
{
    using Layout = typename TileB::Layout;

    for (int t=WP_TILE_THREAD_IDX; t < Layout::Size; t += WP_TILE_BLOCK_DIM)
    {
        auto c = Layout::coord_from_linear(t);        
        src.grad(c) += dest.grad(c + offset);
    } 

    WP_TILE_SYNC();
}


// codegen entry points, which emit calls like `tile_assign(dest, src, i, j, k)`
// a better approach here would be for codegen to just directly generate `tile_assign(dest, src, tile_coord(i, j, k))` 
// i.e.: call the above implementation methods directly, then we could remove these overloads
template <typename TileA, typename TileB>
inline CUDA_CALLABLE void tile_assign(TileA& dest, TileB& src, int i) { tile_assign(dest, src, tile_coord(i)); }
template <typename TileA, typename TileB>
inline CUDA_CALLABLE void tile_assign(TileA& dest, TileB& src, int i, int j) { tile_assign(dest, src, tile_coord(i, j)); }
template <typename TileA, typename TileB>
inline CUDA_CALLABLE void tile_assign(TileA& dest, TileB& src, int i, int j, int k) { tile_assign(dest, src, tile_coord(i, j, k)); }
template <typename TileA, typename TileB>
inline CUDA_CALLABLE void tile_assign(TileA& dest, TileB& src, int i, int j, int k, int l) { tile_assign(dest, src, tile_coord(i, j, k, l)); }

template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_assign(TileA& dest, TileB& src, int i, AdjTileA& adj_dest, AdjTileB& adj_src, int) { adj_tile_assign(dest, src, tile_coord(i), adj_dest, adj_src, tile_coord(0)); }
template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_assign(TileA& dest, TileB& src, int i, int j, AdjTileA& adj_dest, AdjTileB& adj_src, int, int) { adj_tile_assign(dest, src, tile_coord(i,j), adj_dest, adj_src, tile_coord(0)); }
template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_assign(TileA& dest, TileB& src, int i, int j, int k, AdjTileA& adj_dest, AdjTileB& adj_src, int, int, int) { adj_tile_assign(dest, src, tile_coord(i,j,k), adj_dest, adj_src, tile_coord(0)); }
template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
inline CUDA_CALLABLE void adj_tile_assign(TileA& dest, TileB& src, int i, int j, int k, int l, AdjTileA& adj_dest, AdjTileB& adj_src, int, int, int, int) { adj_tile_assign(dest, src, tile_coord(i,j,k,l), adj_dest, adj_src, tile_coord(0)); }


template <typename TileA, typename TileB, typename TileC>
inline CUDA_CALLABLE TileC& tile_diag_add(TileA& a, TileB& b, TileC& c)
{   
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;
    using ShapeC = typename TileC::Layout::Shape;

    static_assert(ShapeA::dim(0) == ShapeA::dim(1), "Expected ShapeA::dim(0) == ShapeA::dim(1)");
    static_assert(ShapeB::dim(0) == ShapeA::dim(0), "Expected ShapeB::dim(0) == ShapeA::dim(0)");
    static_assert(ShapeC::dim(0) == ShapeA::dim(0), "Expected ShapeC::dim(0) == ShapeA::dim(0)");
    static_assert(ShapeC::dim(0) == ShapeC::dim(1), "Expected ShapeC::dim(0) == ShapeC::dim(1)");

    c = a;
    
    for (int t=WP_TILE_THREAD_IDX; t < ShapeA::dim(0); t += WP_TILE_BLOCK_DIM)
    {
        c.data(tile_coord(t, t)) += b.data(tile_coord(t));
    }

    WP_TILE_SYNC();

    return c;
}

template <typename TileA, typename TileB, typename TileC, typename AdjTileA, typename AdjTileB, typename AdjTileC>
inline CUDA_CALLABLE void adj_tile_diag_add(TileA& a, TileB& b, TileC& c, AdjTileA& adj_a, AdjTileB& adj_b, AdjTileC& adj_c, AdjTileC& adj_ret)
{   
    assert(false);
}


} // namespace wp


#ifdef __clang__
#pragma clang diagnostic pop
#endif
