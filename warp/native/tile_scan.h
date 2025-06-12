/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#if defined(__clang__)
// disable warnings related to C++17 extensions on CPU JIT builds
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

namespace wp
{    

#if defined(__CUDA_ARCH__)


template<typename T>
inline CUDA_CALLABLE T scan_warp_inclusive(int lane, T value)
{
//Computes an inclusive cumulative sum
#pragma unroll
    for (int i = 1; i <= 32; i *= 2)
    {
        auto n = __shfl_up_sync(0xffffffffu, value, i, 32);

        if (lane >= i)
            value = value + n;
    }
    return value;
}


template<typename T>
inline CUDA_CALLABLE T thread_block_scan_inclusive(int lane, int warp_index, int num_warps, T value)
{
    WP_TILE_SHARED T sums[1024 / WP_TILE_WARP_SIZE]; // 1024 is the maximum number of threads per block

    value = scan_warp_inclusive(lane, value);

    if (lane == 31)
    {
        sums[warp_index] = value;
    }

    WP_TILE_SYNC();

    if (warp_index == 0)
    {
        T v = lane < num_warps ? sums[lane] : T(0);
        v = scan_warp_inclusive(lane, v);
        if (lane < num_warps)
            sums[lane] = v;
    }

    WP_TILE_SYNC();

    if (warp_index > 0)
    {
        value += sums[warp_index - 1];
    }

    return value;
}

template<typename T, bool exclusive>
inline CUDA_CALLABLE void thread_block_scan(T* values, int num_elements)
{
    const int num_threads_in_block = blockDim.x;
    const int num_iterations = (num_elements + num_threads_in_block - 1) / num_threads_in_block;

    WP_TILE_SHARED T offset;
    if (threadIdx.x == 0)
        offset = T(0);

    WP_TILE_SYNC();

    const int lane = WP_TILE_THREAD_IDX % WP_TILE_WARP_SIZE;
    const int warp_index = WP_TILE_THREAD_IDX / WP_TILE_WARP_SIZE;
    const int num_warps = num_threads_in_block / WP_TILE_WARP_SIZE;

    for (int i = 0; i < num_iterations; ++i)
    {
        int element_index = WP_TILE_THREAD_IDX + i * num_threads_in_block;
        T orig_value = element_index < num_elements ? values[element_index] : T(0);
        T value = thread_block_scan_inclusive(lane, warp_index, num_warps, orig_value);
        if (element_index < num_elements)
        {
            T new_value = value + offset;
            if constexpr (exclusive)
                new_value -= orig_value;
            values[element_index] = new_value;
        }

        WP_TILE_SYNC();

        if (threadIdx.x == num_threads_in_block - 1)        
            offset += value;        

        WP_TILE_SYNC();
    }
}

template<typename Tile>
inline CUDA_CALLABLE auto tile_scan_inclusive_impl(Tile& t)
{
    using T = typename Tile::Type;
    constexpr int num_elements_to_scan = Tile::Layout::Shape::size(); 

    // create a temporary shared tile to hold the input values
    WP_TILE_SHARED T smem[num_elements_to_scan];
    tile_shared_t<T, tile_layout_strided_t<typename Tile::Layout::Shape>, false> scratch(smem, nullptr);

    // copy input values to scratch space
    scratch.assign(t);

    T* values = &scratch.data(0);
    thread_block_scan<T, false>(values, num_elements_to_scan);

    auto result =  scratch.copy_to_register();
    
    WP_TILE_SYNC();

    return result;
}

template<typename Tile>
inline CUDA_CALLABLE auto tile_scan_exclusive_impl(Tile& t)
{
    using T = typename Tile::Type;
    constexpr int num_elements_to_scan = Tile::Layout::Shape::size(); 

    // create a temporary shared tile to hold the input values
    WP_TILE_SHARED T smem[num_elements_to_scan];
    tile_shared_t<T, tile_layout_strided_t<typename Tile::Layout::Shape>, false> scratch(smem, nullptr);

    // copy input values to scratch space
    scratch.assign(t);

    T* values = &scratch.data(0);
    thread_block_scan<T, true>(values, num_elements_to_scan);

    auto result = scratch.copy_to_register();

    WP_TILE_SYNC();

    return result;
}

#else

template<typename Tile>
inline auto tile_scan_inclusive_impl(Tile& t)
{
    using T = typename Tile::Type;
    constexpr int num_elements_to_scan = Tile::Layout::Shape::size(); 
    
    auto input = t.copy_to_register();
    auto output = tile_register_like<Tile>();

    using Layout = typename decltype(input)::Layout;

    T sum = T(0);
    for (int i = 0; i < num_elements_to_scan; ++i)
    {
        sum += input.data[i];
        output.data[i] = sum;
    }

    return output;
}

template<typename Tile>
inline auto tile_scan_exclusive_impl(Tile& t)
{
    using T = typename Tile::Type;
    constexpr int num_elements_to_scan = Tile::Layout::Shape::size(); 
    
    auto input = t.copy_to_register();
    auto output = tile_register_like<Tile>();

    using Layout = typename decltype(input)::Layout;

    T sum = T(0);
    for (int i = 0; i < num_elements_to_scan; ++i)
    {
        output.data[i] = sum;
        sum += input.data[i];
    }

    return output;
}

#endif // !defined(__CUDA_ARCH__)

template <typename Tile>
auto tile_scan_inclusive(Tile& t)
{
    return tile_scan_inclusive_impl(t);
}

template <typename Tile, typename AdjTile>
void adj_tile_scan_inclusive(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // todo: not implemented
}

template <typename Tile>
auto tile_scan_exclusive(Tile& t)
{
    return tile_scan_exclusive_impl(t);
}

template <typename Tile, typename AdjTile>
void adj_tile_scan_exclusive(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // todo: not implemented
}

} // namespace wp

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
