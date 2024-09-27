#pragma once

#include "tile.h"

#define WP_TILE_WARP_SIZE 32

namespace wp
{

template <typename T>
inline CUDA_CALLABLE T warp_shuffle_down(T val, int offset, int mask)
{
    typedef unsigned int Word;

    union
    {
        T output;       
        Word output_storage;
    };

    union
    {
        T input;
        Word input_storage;
    };

    input = val;

    Word* dest = reinterpret_cast<Word*>(&output);
    Word* src  = reinterpret_cast<Word*>(&input);

    unsigned int shuffle_word;

    constexpr int word_count = (sizeof(T) + sizeof(Word) - 1) / sizeof(Word);

    WP_PRAGMA_UNROLL
    for (int i=0; i < word_count; ++i)
    {
        shuffle_word = __shfl_down_sync(mask, src[i], offset, WP_TILE_WARP_SIZE);
        dest[i] = shuffle_word;
    }

  return output;
}

template <typename T>
inline CUDA_CALLABLE T warp_reduce_sum(T val)
{
    T sum = val;

    unsigned int mask = __activemask();

    if (mask == 0xFFFFFFFF)
    {
        // handle case where entire warp is active
        for (int offset=WP_TILE_WARP_SIZE/2; offset > 0; offset /= 2)
        {
            sum += warp_shuffle_down(sum, offset, mask);
        }
    }
    else
    {
        // handle partial warp case
        for (int offset=WP_TILE_WARP_SIZE/2; offset > 0; offset /= 2)
        {            
            T shfl_val = warp_shuffle_down(sum, offset, mask);
            if ((mask & (1 << ((threadIdx.x + offset)%WP_TILE_WARP_SIZE))) != 0)
                sum += shfl_val;
        }
    }

    return sum;
}

template <typename T, typename Op>
inline CUDA_CALLABLE T warp_reduce(T val, Op op)
{
    T sum = val;

    for (int offset=WP_TILE_WARP_SIZE/2; offset > 0; offset /= 2)
    {
        sum = op(sum, warp_shuffle_down(sum, offset));
    }

    return sum;
}


// non-axis version which computes sum 
// across the entire tile using the whole block
template <typename Tile>
auto tile_sum(Tile& t)
{
    using T = typename Tile::Type;

    auto input = t.copy_to_register();
    auto output = tile_register_t<T, 1, 1>();

    const int warp_count = (WP_TILE_BLOCK_DIM + WP_TILE_WARP_SIZE - 1)/WP_TILE_WARP_SIZE;
    const int warp_index = threadIdx.x/WP_TILE_WARP_SIZE;
    const int lane_index = threadIdx.x%WP_TILE_WARP_SIZE;

    T thread_sum = input.data[0];

    // thread reduction
    WP_PRAGMA_UNROLL
    for (int i=1; i < input.NumRegs; ++i)
        thread_sum += input.data[i];

    // warp reduction
    T warp_sum = warp_reduce_sum(thread_sum);

    // fixed size scratch pad for partial results in shared memory
    WP_TILE_SHARED T partials[warp_count];

    if (lane_index == 0)
    {
        partials[warp_index] = warp_sum;
    }

    // ensure partials are ready
    WP_TILE_SYNC();

    // reduce across block, todo: use warp_reduce() here
    if (threadIdx.x == 0)
    {
        T block_sum = partials[0];
        
        WP_PRAGMA_UNROLL
        for (int i=1; i < warp_count; ++i)
            block_sum += partials[i];

        output.data[0] = block_sum;
    }

    return output;
}

template <typename Tile, typename AdjTile>
void adj_tile_sum(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    using T = typename Tile::Type;

    // broadcast incoming adjoint to block
    WP_TILE_SHARED T scratch;
    if (threadIdx.x == 0)
        scratch = adj_ret.data[0];

    WP_TILE_SYNC();

    auto adj_t_reg = adj_t.copy_to_register();
    auto adj_ret_reg = tile_shared_t<T, Tile::M, Tile::N, 0, 0>(&scratch).copy_to_register();

    adj_t.assign(tile_add(adj_t_reg, adj_ret_reg));
}


template <typename Tile, typename Fwd>
auto tile_reduce(Fwd op, Tile& t, int axis)
{
    using T = typename Tile::Type;

    auto input = t.copy_to_register();
    auto output = tile_register_t<T, 1, 1>();

    const int warp_count = (WP_TILE_BLOCK_DIM + WP_TILE_WARP_SIZE - 1)/WP_TILE_WARP_SIZE;
    const int warp_index = threadIdx.x/WP_TILE_WARP_SIZE;
    const int lane_index = threadIdx.x%WP_TILE_WARP_SIZE;

    T thread_sum = input.data[0];

    // thread reduction
    WP_PRAGMA_UNROLL
    for (int i=1; i < input.NumRegs; ++i)
        thread_sum = op(thread_sum, input.data[i]);

    // warp reduction
    T warp_sum = warp_reduce(thread_sum, op);

    // fixed size scratch pad for partial results
    WP_TILE_SHARED T partials[warp_count];

    if (lane_index == 0)
    {
        partials[warp_index] = warp_sum;
    }

    WP_TILE_SYNC();

    // reduce across block, todo: use warp_reduce() here
    if (threadIdx.x == 0)
    {
        T block_sum = partials[0];
        
        WP_PRAGMA_UNROLL
        for (int i=1; i < warp_count; ++i)
            block_sum = op(block_sum, partials[i]);

        output.data[0] = block_sum;
    }

    return output;
}

template <typename Tile, typename AdjTile, typename Fwd>
void adj_tile_reduce(Tile& t, int axis, Tile& adj_t, int adj_axis, AdjTile& adj_ret)
{
    using T = typename Tile::Type;

    // broadcast incoming adjoint to block
    WP_TILE_SHARED T scratch;
    if (threadIdx.x == 0)
        scratch = adj_ret.data[0];

    WP_TILE_SYNC();

    auto adj_t_reg = adj_t.copy_to_register();
    auto adj_ret_reg = tile_shared_t<T, Tile::M, Tile::N, -1, 0, 0>(&scratch).copy_to_register();

    adj_t.assign(tile_add(adj_t_reg, adj_ret_reg));
}

} // namespace wp