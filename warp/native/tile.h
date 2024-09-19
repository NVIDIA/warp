#pragma once

#include "builtin.h"

#if !defined(__CUDA_ARCH__)
#define WP_TILE_SHARED static
#define WP_TILE_SYNC void
#else
#define WP_TILE_SHARED __shared__
#define WP_TILE_SYNC __syncthreads
#endif

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



/* Tile Expressions

[ ] Tiles
    [x] Register, Shared, Global
    [ ] Layouts
        [x] Simple
        [ ] Cute
    [ ] Remove Alloc type from tile_shared_t
    
[ ] Load/Store
    [ ] 1D load/store variants
    [ ] max_coord option for non-aligned loads
    [ ] Indexed load
    [ ] wp.tile_atomic_add()
[ ] Maps
    [x] Support user functions
    [x] Support built-in functions
    [ ] Support for lambda functions
    [ ] Infer tile_map() output from operator type (e.g.: dot for each element)
[ ] Reductions
    [x] Sum
        [x] Forward
        [x] Reverse
    [ ] Min
    [ ] Max
    [ ] Custom
[x] MatMul
    [x] Forward
    [x] Reverse
[ ] Reshape
    [ ] Broadcasting
    [ ] Transpose
        [x] Shared
        [ ] Register
    [ ] Slice
[ ] Runtime
    [x] Compile-time block dimensions
    [ ] Switch between SIMT / Tile based execution if `tile_dim` not provided to wp.launch()
[ ] Examples
    [ ] GEMM
    [ ] Batched MLP
    [ ] Point cloud alignment
    [ ] Layer norm
    [ ] Convolution: https://github.com/NVIDIA/MinkowskiEngine/blob/master/src/convolution_kernel.cu#L123
    [ ] MeshCNN (Modulus, Oliver)
    [ ] BioNemo (Ali)
    [ ] Skinning (David/Or/Vismay)
    [ ] warp.sim (VBD)
    [ ] warp.sim (CRBA)
    [ ] Point clustering
    [ ] GEMM
    [ ] MLP
    [ ] LayerNorm  
    [ ] SoftMax
[ ] Error checking
    [ ] Ensure functions passed to tile_map() are compatible with tile type
    [ ] Ensure that args passed to tile ops are compatible
    [ ] Ensure tile load/store operations don't go out of bounds of arrays in debug mode

*/

// wp.tile_load(A, offset, shape)
// wp.tile_load(A, (x, y), (16, 16))
// wp.tile_load(A, (x, y, z), (3, 3, 3))

// wp.tile_load(A, index, shape)
// wp.tile_load(A, x, m)
// wp.tile_load(A, x, y, m, n)
// wp.tile_load(A, x, y, z, m, n, o)
// wp.tile_load(A, x, y, z, m, n, o, p)

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


template <typename T, int M, int N, int Alloc>
inline CUDA_CALLABLE T* tile_alloc_shared()
{
    WP_TILE_SHARED __align__(16) T data[M*N];

    for (int i=threadIdx.x; i < M*N; i+= WP_TILE_BLOCK_DIM)
        data[i] = T(0);

    return data;
}

// represents a tile stored in global memory with dynamic strides
// only used to represent the source for tile loads to register/shared
template <typename T, int M_, int N_>
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

    inline CUDA_CALLABLE tile_register_t(tile_global_t<T, M, N>& t)
    {
        // construct from a global tile
        copy_from_global(t.data, t.x, t.y);
    }


    inline CUDA_CALLABLE auto& operator=(const tile_global_t<T, M, N>& t)
    {
        // assign from a global tile
        copy_from_global(t.data, t.x, t.y);
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

    // extract a single tile element to a native type
    inline CUDA_CALLABLE Type extract(int i, int j)
    {
        // map from logical coords (i, j) -> (thread, reg)
        const int linear = i*N + j;

        const int thread = linear/NumRegs;
        const int reg = linear%NumRegs;

        WP_TILE_SHARED Type scratch;

        if (threadIdx.x == thread)
        {
            scratch = data[reg];
        }

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


    // return the in-register version of this tile (nop)
    inline CUDA_CALLABLE auto& copy_to_register() { return *this; }


    void copy_to_global(array_t<T> dest, int x, int y)
    {
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



template <typename T, int M_, int N_, int Alloc_, int StrideM_=N_, int StrideN_=1>
struct tile_shared_t
{
    using Type = T;
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int Size = M*N;
    static constexpr int Alloc = Alloc_;

    static constexpr int StrideM = StrideM_;
    static constexpr int StrideN = StrideN_;

    static constexpr bool Aligned = Size%WP_TILE_BLOCK_DIM == 0;

    T* data = NULL;

    // default initialization (non-initialized)
    inline CUDA_CALLABLE tile_shared_t() 
    {
        data = tile_alloc_shared<T, M, N, Alloc>();
    }

    // zero initialization, handles adj_tile = {0} syntax
    inline CUDA_CALLABLE tile_shared_t(int nil) 
    {
        data = tile_alloc_shared<T, M, N, Alloc>();
        zero();
    }    

    // initialize from an existing tile's memory
    inline CUDA_CALLABLE tile_shared_t(T* smem) : data(smem)
    {
    }    

    // construct from a global tile
    inline CUDA_CALLABLE tile_shared_t(tile_global_t<T, M, N>& t)
    {        
        copy_from_global(t.array, t.x, t.y);
    }

    // assign from a global tile
    inline CUDA_CALLABLE auto& operator=(const tile_global_t<T, M, N>& t)
    {
        copy_from_global(t.data, t.x, t.y);
        return *this;
    }

    // assign from a constant value
    inline CUDA_CALLABLE auto& operator=(const T& x)
    {
        // todo: make this subtile (stride aware)
        for (int i=threadIdx.x; i < M*N; i+= WP_TILE_BLOCK_DIM)
            data[i] = x;

        return *this;
    }

    
    inline CUDA_CALLABLE T& operator()(int i, int j)
    {
        assert(i < M);
        assert(j < N);

        return data[i*StrideM + j*StrideN];
    }

    inline CUDA_CALLABLE const T& operator()(int i, int j) const
    {
        assert(i < M);
        assert(j < N);

        return data[i*StrideM + j*StrideN];
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

    // compute tile coordinate from linear index
    inline CUDA_CALLABLE coord_t coord(int index) const
    {
        return {index/N, index%N};
    }

    // in-place zero
    inline CUDA_CALLABLE void zero()
    {
        // todo: make this subtile (stride aware)
        for (int i=threadIdx.x; i < M*N; i+= WP_TILE_BLOCK_DIM)
            data[i] = T(0);
    }

    // extract a single tile element to a native type
    inline CUDA_CALLABLE Type extract(int i, int j)
    {
        return (*this)(i, j);
    }
        
    // backward of scalar extraction
    inline CUDA_CALLABLE void adj_extract(int i, int j, Type adj_ret)
    {
        if (threadIdx.x == 0)
            (*this)(i, j) += adj_ret;
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

            (*this)(linear) = tile.data[i];
        }
    }

    inline CUDA_CALLABLE void print()
    {
        if (threadIdx.x == 0)
        {
            printf("[");
            for (int i=0; i < M; ++i)
            {
                printf("%*s[", i>0, "");
                for (int j=0; j < N; ++j)
                {
                    printf("%5.2f ", (*this)(i, j));
                }

                if (i == M-1)
                    printf("]]\n");
                else
                    printf("]\n");
            }
        }
    }

    // copy shared tile to register
    inline CUDA_CALLABLE tile_register_t<T, M, N> copy_to_register() 
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

            out(i) = (*this)(linear);
        }

        return out;
    }

    inline CUDA_CALLABLE void copy_to_global(array_t<T> dest, int x, int y)
    {
        // todo: use TMA here
        const int tile_i = x*M;
        const int tile_j = y*N;

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
            ptr[c.i*stride_i + c.j*stride_j] = (*this)(c.i, c.j);
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
        for (int i=threadIdx.x; i < Size; i += WP_TILE_BLOCK_DIM)
        {  
            coord_t c = coord(i);
            (*this)(c.i, c.j) = ptr[c.i*stride_i + c.j*stride_j];
        }
    }
};

template <typename Tile>
inline CUDA_CALLABLE auto tile_transpose(Tile& t)
{    
    // alias incoming tile 
    return tile_shared_t<typename Tile::Type, Tile::N, Tile::M, Tile::Alloc, Tile::StrideN, Tile::StrideM>(t.data);
}


//-----------------------------------------------------------------------------------------------------
// High level entry points for each op (correspond to one Warp builtin)

template <typename T, int M, int N, int Alloc>
inline CUDA_CALLABLE auto tile_zeros()
{
    // tile variable assignment operator will handle initialization
    return T(0.0);
}


// entry point for load
template <typename T, int M, int N, int Alloc>
inline CUDA_CALLABLE auto tile_load(array_t<T>& src, int x, int y)
{
    // just return a ref. to the global memory
    // it will be loaded to shared or registers
    // on assignment to the variable
    return tile_global_t<T, M, N>(src, x, y);
}

// entry point for store
template <typename T, typename Tile>
inline CUDA_CALLABLE void tile_store(array_t<T>& dest, int x, int y, Tile& src)
{
    // dispatch to tile type
    src.copy_to_global(dest, x, y);
}

//-------------------------------------
// Adjoints

template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load(array_t<T>& src, int x, int y,
                                        array_t<T>& adj_src, int adj_x, int adj_y,
                                        AdjTile& adj_ret)
{
    // early out
    // if (!src.grad)
    //     return;

    auto adj_reg = adj_ret.copy_to_register();

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
inline CUDA_CALLABLE void adj_tile_store(array_t<T>& dest, int x, int y, Tile& t, array_t<T>& adj_dest, int adj_x, int adj_y, AdjTile& adj_t)
{  
    // if (!dest.grad)
    //     return;

    // convert to register if necessary
    auto adj_reg = adj_t.copy_to_register();

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
            adj_reg.data[i] += index(adj_dest, tile_i + coord.i, tile_j + coord.j);
        else if (dest.grad)
            adj_reg.data[i] += index_grad(dest, tile_i + coord.i, tile_j + coord.j);
    }

    // store adjoint back to tile
    adj_t.assign(adj_reg);
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
    auto adj_a_reg = adj_a.copy_to_register();
    auto adj_ret_reg = adj_ret.copy_to_register();

    WP_PRAGMA_UNROLL
    for (int i=0; i < a_reg.NumRegs; ++i)
    {        
        adj_op(a_reg.data[i], adj_a_reg.data[i], adj_ret_reg.data[i]);
    }

    // write adjoints back
    adj_a.assign(adj_a_reg);
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
    auto adj_a_reg = adj_a.copy_to_register();
    auto adj_b_reg = adj_b.copy_to_register();    
    auto adj_ret_reg = adj_ret.copy_to_register();

    WP_PRAGMA_UNROLL
    for (int i=0; i < a_reg.NumRegs; ++i)
    {
        adj_op(a_reg.data[i], b_reg.data[i], adj_a_reg.data[i], adj_b_reg.data[i], adj_ret_reg.data[i]);
    }

    adj_a.assign(adj_a_reg);
    adj_b.assign(adj_b_reg);
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

template <typename TileA, typename TileB, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_add(TileA& a, TileB& b, TileA& adj_a, TileB& adj_b, AdjTile& adj_c)
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


} // namespace wp

