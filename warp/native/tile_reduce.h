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

#include "tile.h"

#define WP_TILE_WARP_SIZE 32

namespace wp
{


template <typename T>
int argmax_tracker(T champion_value, T current_value, int champion_index, int current_index)
{
    return current_value > champion_value ? current_index : champion_index;
}

template <typename T>
int argmin_tracker(T champion_value, T current_value, int champion_index, int current_index)
{
    return current_value < champion_value ? current_index : champion_index;
}
    

#if defined(__CUDA_ARCH__)

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

template <typename T, typename Op>
inline CUDA_CALLABLE T warp_reduce(T val, Op f, unsigned int mask)
{
    T sum = val;

    if (mask == 0xFFFFFFFF)
    {
        // handle case where entire warp is active
        for (int offset=WP_TILE_WARP_SIZE/2; offset > 0; offset /= 2)
        {
            sum = f(sum, warp_shuffle_down(sum, offset, mask));
        }
    }
    else
    {
        // handle partial warp case
        for (int offset=WP_TILE_WARP_SIZE/2; offset > 0; offset /= 2)
        {            
            T shfl_val = warp_shuffle_down(sum, offset, mask);
            if ((mask & (1 << ((threadIdx.x + offset)%WP_TILE_WARP_SIZE))) != 0)
                sum = f(sum, shfl_val);
        }
    }

    return sum;
}

template <typename T>
struct ValueAndIndex
{
    T value;
    int index;
};

template <typename T, typename Op, typename OpTrack>
inline CUDA_CALLABLE ValueAndIndex<T> warp_reduce_tracked(T val, int idx, Op f, OpTrack track, unsigned int mask)
{
    T sum = val;
    int index = idx;

    if (mask == 0xFFFFFFFF)
    {
        // handle case where entire warp is active
        for (int offset=WP_TILE_WARP_SIZE/2; offset > 0; offset /= 2)
        {
            auto shfl_val = warp_shuffle_down(sum, offset, mask);
            int shfl_idx = warp_shuffle_down(index, offset, mask);            
            index = track(sum, shfl_val, index, shfl_idx);
            sum = f(sum, shfl_val);
        }
    }
    else
    {
        // handle partial warp case
        for (int offset=WP_TILE_WARP_SIZE/2; offset > 0; offset /= 2)
        {            
            T shfl_val = warp_shuffle_down(sum, offset, mask);
            int shfl_index = warp_shuffle_down(index, offset, mask);
            if ((mask & (1 << ((threadIdx.x + offset)%WP_TILE_WARP_SIZE))) != 0)
            {
                index = track(sum, shfl_val, index, shfl_index);
                sum = f(sum, shfl_val);
            }
        }
    }

    ValueAndIndex<T> result;
    result.value = sum;
    result.index = index;

    return result;
}

// non-axis version which computes sum 
// across the entire tile using the whole block
template <typename Tile, typename Op>
auto tile_reduce_impl(Op f, Tile& t)
{
    using T = typename Tile::Type;

    auto input = t.copy_to_register();
    auto output = tile_register_t<T, tile_layout_register_t<tile_shape_t<1>>>();

    const int warp_count = (WP_TILE_BLOCK_DIM + WP_TILE_WARP_SIZE - 1)/WP_TILE_WARP_SIZE;
    const int warp_index = threadIdx.x/WP_TILE_WARP_SIZE;
    const int lane_index = threadIdx.x%WP_TILE_WARP_SIZE;

    T thread_sum = input.data[0];

    using Layout = typename decltype(input)::Layout;

    // thread reduction
    WP_PRAGMA_UNROLL
    for (int i=1; i < Layout::NumRegs; ++i)
    {
        int linear = Layout::linear_from_register(i);
        if (!Layout::valid(linear))
            break;

        thread_sum = f(thread_sum, input.data[i]);
    }

    // ensure that only threads with at least one valid item participate in the reduction
    unsigned int mask = __ballot_sync(__activemask(), Layout::valid(Layout::linear_from_register(0)));

    // warp reduction
    T warp_sum = warp_reduce(thread_sum, f, mask);

    // fixed size scratch pad for partial results in shared memory
    WP_TILE_SHARED T partials[warp_count];

    // count of active warps
    WP_TILE_SHARED int active_warps;
    if (threadIdx.x == 0)
        active_warps = 0;
    
    // ensure active_warps is initialized
    WP_TILE_SYNC();

    if (lane_index == 0)
    {
        partials[warp_index] = warp_sum;
        atomicAdd(&active_warps, 1);
    }

    // ensure partials are ready
    WP_TILE_SYNC();

    // reduce across block, todo: use warp_reduce() here
    if (threadIdx.x == 0)
    {
        T block_sum = partials[0];
        
        WP_PRAGMA_UNROLL
        for (int i=1; i < active_warps; ++i)
            block_sum = f(block_sum, partials[i]);

        output.data[0] = block_sum;
    }

    return output;
}


// non-axis version which computes sum 
// across the entire tile using the whole block
template <typename Tile, typename Op, typename OpTrack>
auto tile_arg_reduce_impl(Op f, OpTrack track, Tile& t)
{
    using T = typename Tile::Type;

    auto input = t.copy_to_register();
    auto output = tile_register_t<int, tile_layout_register_t<tile_shape_t<1>>>();

    const int warp_count = (WP_TILE_BLOCK_DIM + WP_TILE_WARP_SIZE - 1)/WP_TILE_WARP_SIZE;
    const int warp_index = threadIdx.x/WP_TILE_WARP_SIZE;
    const int lane_index = threadIdx.x%WP_TILE_WARP_SIZE;

    using Layout = typename decltype(input)::Layout;

    int champion_index = Layout::NumRegs > 0 ? Layout::linear_from_register(0) : -1;
    T thread_sum = input.data[0];

    // thread reduction
    WP_PRAGMA_UNROLL
    for (int i=1; i < Layout::NumRegs; ++i)
    {
        int linear = Layout::linear_from_register(i);
        if (!Layout::valid(linear))
            break;

        champion_index = track(thread_sum, input.data[i], champion_index, linear);
        thread_sum = f(thread_sum, input.data[i]);        
    }

    // ensure that only threads with at least one valid item participate in the reduction
    unsigned int mask = __ballot_sync(__activemask(), Layout::valid(Layout::linear_from_register(0)));

    // warp reduction
    ValueAndIndex<T> warp_sum = warp_reduce_tracked(thread_sum, champion_index, f, track, mask);

    // fixed size scratch pad for partial results in shared memory
    WP_TILE_SHARED T partials[warp_count];
    WP_TILE_SHARED int partials_idx[warp_count];

    // count of active warps
    WP_TILE_SHARED int active_warps;
    if (threadIdx.x == 0)
        active_warps = 0;
    
    // ensure active_warps is initialized
    WP_TILE_SYNC();

    if (lane_index == 0)
    {
        partials[warp_index] = warp_sum.value;
        partials_idx[warp_index] = warp_sum.index;
        atomicAdd(&active_warps, 1);
    }

    // ensure partials are ready
    WP_TILE_SYNC();

    // reduce across block, todo: use warp_reduce() here
    if (threadIdx.x == 0)
    {
        T block_sum = partials[0];
        int block_champion_index = partials_idx[0];
        
        WP_PRAGMA_UNROLL
        for (int i=1; i < active_warps; ++i)
        {
            block_champion_index = track(block_sum, partials[i], block_champion_index, partials_idx[i]);
            block_sum = f(block_sum, partials[i]);
        }

        output.data[0] = block_champion_index;
    }

    return output;
}

#else

// CPU implementation

template <typename Tile, typename Op>
auto tile_reduce_impl(Op f, Tile& t)
{
   using T = typename Tile::Type;

    auto input = t.copy_to_register();
    auto output = tile_register_t<T, tile_layout_register_t<tile_shape_t<1>>>();

    using Layout = typename decltype(input)::Layout;

    T sum = input.data[0];

    WP_PRAGMA_UNROLL
    for (int i=1; i < Layout::NumRegs; ++i)
    {
        int linear = Layout::linear_from_register(i);
        if (!Layout::valid(linear))
            break;

        sum = f(sum, input.data[i]);
    }

    output.data[0] = sum;
    return output;
}

template <typename Tile, typename Op, typename OpTrack>
auto tile_arg_reduce_impl(Op f, OpTrack track, Tile& t)
{
   using T = typename Tile::Type;

    auto input = t.copy_to_register();
    auto output = tile_register_t<int, tile_layout_register_t<tile_shape_t<1>>>();

    using Layout = typename decltype(input)::Layout;

    int champion_index = Layout::NumRegs > 0 ? Layout::linear_from_register(0) : -1;
    T sum = input.data[0];

    WP_PRAGMA_UNROLL
    for (int i=1; i < Layout::NumRegs; ++i)
    {
        int linear = Layout::linear_from_register(i);
        if (!Layout::valid(linear))
            break;

        champion_index = track(sum, input.data[i], champion_index, linear);
        sum = f(sum, input.data[i]);
    }

    output.data[0] = champion_index;
    return output;
}

#endif // !defined(__CUDA_ARCH__)

inline void adj_tile_reduce_impl() 
{
    // todo: general purpose reduction gradients not implemented 
}

// entry point for Python code-gen, wraps op in a lambda to perform overload resolution
#define tile_reduce(op, t) tile_reduce_impl([](auto x, auto y) { return op(x, y);}, t)
#define adj_tile_reduce(op, a, adj_op, adj_a, adj_ret) adj_tile_reduce_impl()

#define tile_arg_reduce(op, opTrack, t) tile_arg_reduce_impl([](auto x, auto y) { return op(x, y);}, [](auto a, auto b, auto c, auto d) { return opTrack(a, b, c, d); }, t)
#define adj_tile_arg_reduce(op, a, adj_op, adj_a, adj_ret) adj_tile_arg_reduce_impl()

// convenience methods for specific reductions

template <typename Tile>
auto tile_sum(Tile& t)
{
    return tile_reduce(add, t);
}

// special case adjoint for summation
template <typename Tile, typename AdjTile>
void adj_tile_sum(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    using T = typename Tile::Type;

#if !defined(__CUDA_ARCH__)

    for (int i=0; i < Tile::Layout::Size; ++i)
    {
        adj_t(i) += adj_ret.data[0];

    }
#else
    // broadcast incoming adjoint to block
    WP_TILE_SHARED T scratch;
    if (WP_TILE_THREAD_IDX == 0)
        scratch = adj_ret.data[0];

    WP_TILE_SYNC();

    // broadcast scalar across input dimensions (note zero strides)
    auto adj_ret_reg = tile_shared_t<T, tile_layout_strided_t<typename Tile::Layout::Shape, tile_stride_t<0, 0>>, false>(&scratch, nullptr).copy_to_register();
    adj_t.grad_add(adj_ret_reg);
#endif 
}

template <typename Tile>
auto tile_max(Tile& t)
{
    return tile_reduce(max, t);
}

template <typename Tile, typename AdjTile>
void adj_tile_max(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // todo: not implemented
}

template <typename Tile>
auto tile_min(Tile& t)
{
    return tile_reduce(min, t);
}

template <typename Tile, typename AdjTile>
void adj_tile_min(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // todo: not implemented
}



template <typename Tile>
auto tile_argmax(Tile& t)
{
    return tile_arg_reduce(max, argmax_tracker, t);
}

template <typename Tile, typename AdjTile>
void adj_tile_argmax(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // todo: not implemented
}

template <typename Tile>
auto tile_argmin(Tile& t)
{
    return tile_arg_reduce(min, argmin_tracker, t);
}

template <typename Tile, typename AdjTile>
void adj_tile_argmin(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // todo: not implemented
}




} // namespace wp
