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

[x] Forward / Backward code-gen
[ ] wp.tile_map()
    [x] Support user functions
    [x] Support built-in functions
    [ ] Support for lambda functions
    [ ] Infer tile_map() output from operator type (e.g.: dot for each element)
[x] wp.tile_matmul()
    [x] Forward
    [x] Reverse
[ ] wp.tile_atomic_add()   
[ ] Support for n-d shape tiles / broadcasting / slicing / transpose?
[x] Compile-time block dimensions
[ ] Support for CUB reductions
[ ] Support for CUB sorts
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


template <typename T, int M_, int N_>
struct tile_register_t
{
    using Type = T;
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int Size = M*N;

    static constexpr int NumRegs = tile_regcount(M, N);

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

    // return the in-register version of this tile (nop)
    inline CUDA_CALLABLE auto& get() { return *this; }

    inline CUDA_CALLABLE void assign(const tile_register_t<T, M, N>& tile) 
    { 
        for (int i=0; i < NumRegs; ++i)
            data[i] = tile.data[i];
    }

    
    inline CUDA_CALLABLE void print()
    {
        printf("tid: %d ", threadIdx.x);

        for (int i=0; i < NumRegs; ++i)
        {
            printf("%f ", data[i]);
        }

        printf("\n");
    }
        
};



template <typename T, int M_, int N_, int StrideM_=N_, int StrideN_=1>
struct tile_shared_t
{
    using Type = T;
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int Size = M*N;

    static constexpr int StrideM = StrideM_;
    static constexpr int StrideN = StrideN_;

    T* data = NULL;

    inline CUDA_CALLABLE tile_shared_t() {}
    inline CUDA_CALLABLE tile_shared_t(T* smem) : data(smem)
    {
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

    // in-place zero
    inline CUDA_CALLABLE void zero()
    {
        // todo: make this subtile (stride aware)
        for (int i=threadIdx.x; i < M*N; i+= WP_TILE_BLOCK_DIM)
            data[i] = T(0);
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

    // copy shared tile to register
    inline CUDA_CALLABLE tile_register_t<T, M, N> get() 
    { 
        tile_register_t<T, M, N> out;

        WP_PRAGMA_UNROLL
        for (int i=0; i < out.NumRegs; ++i)
        {
            const int linear = out.index(i);

            // handle case where tile size is not
            // aligned to block dimensions
            if (linear > Size)
                break;

            out.data[i] = (*this)(linear);
        }

        return out;
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
            if (linear > Size)
                break;

            // todo: should use coord here to handle cases where
            // shared tile is a slice?
            data[linear] = tile.data[i];
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
                    printf("%5.2f ", data(i, j));
                }

                if (i == M-1)
                    printf("]]\n");
                else
                    printf("]\n");
            }
        }
    }
};

template <typename Tile>
inline CUDA_CALLABLE auto tile_transpose(Tile& t)
{
    // alias incoming tile 
    return tile_shared_t<typename Tile::Type, Tile::N, Tile::M, Tile::StrideN, Tile::StrideM>(t.data);
}


//-----------------------------------------------------------------------------------------------------
// High level entry points for each op (correspond to one Warp builtin)

template <typename T, int M, int N, int Index>
inline CUDA_CALLABLE auto tile_zeros()
{
    const int length = M*N;

    WP_TILE_SHARED __align__(16) T data[length];
    
    WP_PRAGMA_UNROLL
    for (int t=threadIdx.x; t < length; t += WP_TILE_BLOCK_DIM)
    {  
        data[t] = T(0.0);
    }

    return tile_shared_t<T, M, N>(data);
}


// entry point for load
template <typename T, int M, int N, int Alloc>
inline CUDA_CALLABLE auto tile_load(array_t<T>& src, int x, int y)
{
    const int length = M*N;

    WP_TILE_SHARED __align__(16) T data[length];

    tile_shared_t<T, M, N> dest(data);
    
    const int tile_i = x*M;
    const int tile_j = y*N;

    // wp.array() indexing generates poor code due to char* casting
    // here we unroll some of the ops, note this assumes byte strides are 
    // aligned to the element size
    T* ptr = &index(src, tile_i, tile_j);
    const int stride_i = src.strides[0]/sizeof(T);
    const int stride_j = src.strides[1]/sizeof(T);    

    WP_PRAGMA_UNROLL
    for (int i=threadIdx.x; i < length; i += WP_TILE_BLOCK_DIM)
    {  
        coord_t c = dest.coord(i);
        dest.data[i] = ptr[c.i*stride_i + c.j*stride_j];    //index(src, tile_i + c.i, tile_j + c.j);
    }

    return dest;
}

// entry point for store
template <typename T, typename Tile>
inline CUDA_CALLABLE void tile_store(array_t<T>& dest, int x, int y, Tile& src)
{
    auto src_reg = src.get();

    const int tile_i = x*src.M;
    const int tile_j = y*src.N;

    // wp.array() indexing generates poor code due to char* casting
    // here we unroll some of the ops, note this assumes byte strides are 
    // aligned to the element size
    T* ptr = &index(dest, tile_i, tile_j);
    const int stride_i = dest.strides[0]/sizeof(T);
    const int stride_j = dest.strides[1]/sizeof(T);
    
    WP_PRAGMA_UNROLL
    for (int i=0; i < src_reg.NumRegs; ++i)
    {
        // handle case where tile size is not 
        // aligned to block dimensions
        int index = src_reg.index(i);
        if (index > src_reg.Size)
            break;

        coord_t c = src_reg.coord(index);
        ptr[c.i*stride_i + c.j*stride_j] = src_reg.data[i]; //index(dest, tile_i + c.i, tile_j + c.j);
    }
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

    auto adj_reg = adj_ret.get();

    const int tile_i = x*adj_reg.M;
    const int tile_j = y*adj_reg.N;

    // add gradients to src array
    WP_PRAGMA_UNROLL
    for (int i=0; i < adj_reg.NumRegs; ++i)
    {  
        int linear = adj_reg.index(i);
        if (linear > adj_reg.Size)
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
    auto adj_reg = adj_t.get();

    const int tile_i = x*adj_reg.M;
    const int tile_j = y*adj_reg.N;

    // load gradients from output
    WP_PRAGMA_UNROLL
    for (int i=0; i < adj_reg.NumRegs; ++i)
    {  
        int linear = adj_reg.index(i);
        if (linear > adj_reg.Size)
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
    auto a_reg = a.get();
    
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
    auto a_reg = a.get();   
    auto adj_a_reg = adj_a.get();
    auto adj_ret_reg = adj_ret.get();

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

    auto a_reg = a.get();
    auto b_reg = b.get();

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
    auto a_reg = a.get();   
    auto b_reg = b.get();
    auto adj_a_reg = adj_a.get();
    auto adj_b_reg = adj_b.get();    
    auto adj_ret_reg = adj_ret.get();

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

// unary neg
template <typename Tile>
inline CUDA_CALLABLE auto tile_neg(Tile& a) { return tile_unary_map(wp::neg, a); }

template <typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_neg(Tile& a, Tile& adj_a, AdjTile& adj_ret) { adj_tile_unary_map(wp::neg, a, wp::adj_neg, adj_a, adj_ret); }


/*
// handle tile*scalar
template<typename Tile>
CUDA_CALLABLE inline auto tile_mul_impl(Tile& t, typename Tile::Type s,
                                        Tile& adj_t, typename Tile::Type adj_s)
{
    typedef typename Tile::Type T;
    typedef tile_constant_t<T, Tile::M, Tile::N> Constant;

    typedef tile_binary_map_t<Tile, Constant> Op;

    typename Op::FwdOp fwd = [](T a, T b) { return mul(a, b); };
    typename Op::AdjOp adj = [](T a, T b, T& adj_a, T& adj_b, T& adj_ret) { adj_mul(a, b, adj_a, adj_b, adj_ret); };

    // promote scalar to constant tile
    Constant c(s, adj_s);

    return Op(t, c, fwd, adj);
}

// handle scalar*tile
template<typename Tile>
CUDA_CALLABLE inline auto tile_mul_impl(typename Tile::Type s, Tile& t,
                                        typename Tile::Type adj_s, Tile& adj_t)
{
    typedef typename Tile::Type T;
    typedef tile_constant_t<T, Tile::M, Tile::N> Constant;

    typedef tile_binary_map_t<Constant, Tile> Op;

    typename Op::FwdOp fwd = [](T a, T b) { return mul(a, b); };
    typename Op::AdjOp adj = [](T a, T b, T& adj_a, T& adj_b, T& adj_ret) { adj_mul(a, b, adj_a, adj_b, adj_ret); };

    // promote scalar to constant tile
    Constant c(s, adj_s);

    return Op(c, t, fwd, adj);

}


#define tile_mul(a, b) tile_mul_impl(a, b adj_##a, adj_##b)
#define tile_add(a, b) tile_add_impl(a, b adj_##a, adj_##b)
*/

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
    // auto s_tile = tile_register_t<Tile::Type, Tile::M, Tile::N>(s);
    // auto adj_s_tile = tile_register_t<Tile::Type, Tile::M, Tile::N>();

    // adj_tile_binary_map(mul, a, s_tile, adj_mul, adj_a, adj_s_tile, adj_c);

    // todo: sum up contribution from all adj_s_tile onto original scalar
    //adj_tile_sum()
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
    // auto s_tile = tile_register_t<Tile::Type, Tile::M, Tile::N>(s);
    // auto adj_s_tile = tile_register_t<Tile::Type, Tile::M, Tile::N>();

    // adj_tile_binary_map(mul, a, s_tile, adj_mul, adj_a, adj_s_tile, adj_c);

    // todo: sum up contribution from all adj_s_tile onto original scalar
    //adj_tile_sum()
}


} // namespace wp

#if 0

//-----------------------------------------------------
// c = a + b

// forward
auto var_0 = wp::tile_load<wp::float32,8,4>(var_A, x, y);
auto var_1 = wp::tile_load<wp::float32,8,4>(var_B, x, y);
auto var_2 = wp::tile_add(var_0, var_1);
wp::tile_store(var_C, x, y, var_2)

// reverse
wp::adj_store(var_C, x, y, var_2, adj_C, _, _, adj_2)
wp::adj_tile_add(var_0, var_1, adj_0, adj_1, adj_2)
wp::adj_tile_load(var_B, x, y, adj_B, _, _, adj_1);
wp::adj_tile_load(var_B, x, y, adj_B, _, _, adj_0);


//-----------------------------------------------------
// x = a[0]
// c = x*2.0 + x

// forward
auto var_0 = wp::tile_load<wp::float32,8,4>(var_A, x, y);
auto var_1 = wp::tile_mul(var_0, 2.0);
auto var_2 = wp::tile_add(var_0, var_1);
wp::tile_store(var_C, x, y, var_2)

struct adj_store_t
{
    adj_store_t()
    {

    }

    float bwd(int i, float adj_ret)
    {
        return array.grad[i];
    }
};

template <typename P>
struct adj_add_t
{
    adj_add_t(P& parent)
    {
        
    }

    float bwd(int i, float& adj_a, float& adj_b)
    {
        // evaluate parent
        float adj_ret = parent.bwd(i);

        adj_a += adj_ret;
        adj_b += adj_ret;
    }
};

template <typename T>
struct adj_tile
{
    adj_tile(T& parent)
    {

    }



};

void adj_tile_load(A, x, y, adj_A, adj_x, adj_y, adj_ret)
{
    for i in A(x,y):
        adj_A[i] += adj_ret(i);
}



// reverse
wp::adj_store(var_C, x, y, var_2, adj_C, _, _, adj_2)   // adj_2->adj_C
wp::adj_tile_add(var_0, var_1, adj_0, adj_1, adj_2)     // adj_0->adj_2->adj_C, adj_1->adj_2->adj_C
wp::adj_tile_mul(var_0, 2.0, adj_0, _, adj_1);          // adj_0->adj_1->adj_2->adj_C
wp::adj_tile_load(var_A, x, y, adj_A, _, _, adj_0);     // adj_A->adj_0->adj_1->adj_2->adj_C


#endif