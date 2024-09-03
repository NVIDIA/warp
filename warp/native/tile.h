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
[ ] wp.tile_matmul()
    [x] Forward
    [ ] Reverse
[ ] Support for n-d shape tiles / broadcasting / slicing / transpose?
[x] Compile-time block dimensions
[ ] Support for CUB reductions
[ ] Support for CUB sorts
[ ] Examples
    [ ] GEMM
    [ ] Batched MLP
    [ ] Point cloud alignment
    [ ] Layer norm
[ ] Error checking
    [ ] Ensure functions passed to tile_map() are compatible with tile type
    [ ] Ensure that args passed to tile ops are compatible

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

template <typename T>
void print_tile(T& t)
{
    t.print();

    printf("[");
    for (int i=0; i < T::M; ++i)
    {
        printf("%*s[", i>0, "");
        for (int j=0; j < T::N; ++j)
        {
            printf("%5.2f ", t.data[i*T::N + j]);
        }

        if (i == T::M-1)
            printf("]]\n");
        else
            printf("]\n");
    }
}

template <typename Tile>
int tile_size(Tile& t) { return Tile::M*Tile::N; }

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
struct tile_shared_t
{
    using Type = T;
    static constexpr int M = M_;
    static constexpr int N = N_;

    T* data = NULL;

    tile_shared_t() {}
    tile_shared_t(T* smem) : data(smem)
    {
    }

    struct iterator
    {
        tile_shared_t<Type, M, N>& tile;
        int offset;
        
        inline CUDA_CALLABLE iterator(tile_shared_t<Type, M, N>& t, int i) : tile(t), offset(i) {}
        inline CUDA_CALLABLE T& operator*() const { return tile.data[offset]; }
        inline CUDA_CALLABLE iterator& operator++() { offset += WP_TILE_BLOCK_DIM; return *this; }        
        inline CUDA_CALLABLE bool valid() const { return index() < tile_size(tile); }

        // linear index into the tile's data (assuming row-major layout)
        inline CUDA_CALLABLE int index() const { return offset; }
        inline CUDA_CALLABLE coord_t coord() const
        {
            int i = index();
            return {i/N, i%N};
        }
    };    

    iterator iter() { return iterator(*this, threadIdx.x); }
};


template <typename T, int M_, int N_>
struct tile_register_t
{
    using Type = T;
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int NumRegs = tile_regcount(M, N);

    T data[NumRegs];
   
    tile_register_t() 
    {
        // zero-initialize by default
        // necessary for tile adjoints
        // need to check if this results in worse codegen
        for (int i=0; i < NumRegs; ++i)
            data[i] = T(0);
    }

    struct iterator
    {
        tile_register_t<Type, M, N>& tile;
        int offset;
       
        inline CUDA_CALLABLE iterator(tile_register_t<Type, M, N>& t, int i) : tile(t), offset(i) {}

        inline CUDA_CALLABLE T& operator*() const { return tile.data[offset]; }
        inline CUDA_CALLABLE iterator& operator++() { ++offset; return *this; }
        inline CUDA_CALLABLE bool valid() const { return offset < NumRegs && index() < tile_size(tile); }

        // linear index into the tile's data (assuming row-major layout)
        inline CUDA_CALLABLE int index() const { return threadIdx.x + offset*WP_TILE_BLOCK_DIM; }
        inline CUDA_CALLABLE coord_t coord() const
        {
            int i = index();
            return {i/N, i%N};
        }
    };    

    iterator iter() { return iterator(*this, 0); }
};



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


// entry point for store
template <typename T, int M, int N, int Alloc>
inline CUDA_CALLABLE auto tile_load(array_t<T>& src, int x, int y)
{
    const int length = M*N;

    WP_TILE_SHARED __align__(16) T data[length];

    tile_shared_t<T, M, N> dest(data);
    
    WP_PRAGMA_UNROLL
    for (auto dst_iter=dest.iter(); dst_iter.valid(); ++dst_iter)
    {  
        coord_t c = dst_iter.coord();

        *dst_iter = index(src, x*M + c.i, y*N + c.j);
    }

    return dest;
}

// entry point for store
template <typename T, typename Tile>
inline CUDA_CALLABLE void tile_store(array_t<T>& dest, int x, int y, Tile& src)
{
    const int M = src.M;
    const int N = src.N;
   
    // cooperatively store the tile, using a block-stride iterator
    WP_PRAGMA_UNROLL
    for (auto src_iter=src.iter(); src_iter.valid(); ++src_iter)
    {  
        coord_t c = src_iter.coord();

        index(dest, x*M + c.i, y*N + c.j) = *src_iter;
    }
}

//-------------------------------------
// Adjoints

template <typename T, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_load(array_t<T>& src, int x, int y,
                                        array_t<T>& adj_src, int adj_x, int adj_y,
                                        AdjTile& adj_ret)
{
    // add gradients to src array
    WP_PRAGMA_UNROLL
    for (auto adj_iter=adj_ret.iter(); adj_iter.valid(); ++adj_iter)
    {  
        coord_t c = adj_iter.coord();
        atomic_add(adj_src, x*adj_ret.M + c.i, y*adj_ret.N + c.j, *adj_iter);
    }
}

template <typename T, typename Tile, typename AdjTile>
inline CUDA_CALLABLE void adj_tile_store(array_t<T>& dest, int x, int y, Tile& t, array_t<T>& adj_dest, int adj_x, int adj_y, AdjTile& adj_t)
{
    const int M = t.M;
    const int N = t.N;

    // load gradients from output
    WP_PRAGMA_UNROLL
    for (auto adj_iter=adj_t.iter(); adj_iter.valid(); ++adj_iter)
    {  
        coord_t c = adj_iter.coord();
        *adj_iter += index(adj_dest, x*M + c.i, y*N + c.j, *adj_iter);
    }
}

// unary map
template <typename Tile, typename Fwd>
auto tile_map(Fwd op,
              Tile &a)
{
    auto out = tile_register_t<typename Tile::Type, Tile::M, Tile::N>();

    auto out_iter = out.iter();
    auto a_iter = a.iter();

    for (; out_iter.valid(); ++out_iter, ++a_iter)
    {
        *out_iter = op(*a_iter);
    }

    return out;
}

template <typename Tile, typename AdjTile, typename Fwd, typename Adj>
void adj_tile_map(Fwd op,
                  Tile &a,
                  Adj adj_op,
                  Tile &adj_a,
                  AdjTile &adj_ret)
{
    auto a_iter = a.iter();   
    auto adj_a_iter = adj_a.iter();
    auto adj_ret_iter = adj_ret.iter();

    for (; a_iter.valid(); ++a_iter, ++adj_a_iter, ++adj_ret_iter)
    {
        adj_op(*a_iter, *adj_a_iter, *adj_ret_iter);
    }
}

// binary map
template <typename TileA, typename TileB, typename Fwd>
auto tile_map(Fwd op,
              TileA &a,
              TileB &b)
{
    auto out = tile_register_t<typename TileA::Type, TileA::M, TileA::N>();

    auto out_iter = out.iter();
    auto a_iter = a.iter();
    auto b_iter = b.iter();

    for (; out_iter.valid(); ++out_iter, ++a_iter, ++b_iter)
    {
        *out_iter = op(*a_iter, *b_iter);
    }

    return out;
}

template <typename TileA, typename TileB, typename Fwd, typename Adj, typename AdjTile>
void adj_tile_map(Fwd op,
                  TileA &a,
                  TileB &b,
                  Adj adj_op,
                  TileA &adj_a,
                  TileB &adj_b,
                  AdjTile &adj_ret)
{
    auto a_iter = a.iter();   
    auto b_iter = b.iter();
    auto adj_a_iter = adj_a.iter();
    auto adj_b_iter = adj_b.iter();    
    auto adj_ret_iter = adj_ret.iter();

    for (; a_iter.valid(); ++a_iter, ++b_iter, ++adj_a_iter, ++adj_b_iter, ++adj_ret_iter)
    {
        adj_op(*a_iter, *b_iter, *adj_a_iter, *adj_b_iter, *adj_ret_iter);
    }
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
auto tile_neg(Tile& a) { return tile_unary_map(wp::neg, a); }

template <typename Tile, typename AdjTile>
void adj_tile_neg(Tile& a, Tile& adj_a, AdjTile& adj_ret) { adj_tile_unary_map(wp::neg, a, wp::adj_neg, adj_a, adj_ret); }


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