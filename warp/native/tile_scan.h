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

namespace wp {

// Operation structs for different scan types (shared between CPU and GPU)
template <typename T> struct OpAdd {
    inline CUDA_CALLABLE T operator()(const T& a, const T& b) const { return a + b; }

    inline CUDA_CALLABLE T identity() const { return T(0); }
};

template <typename T> struct OpMax {
    inline CUDA_CALLABLE T operator()(const T& a, const T& b) const { return max(a, b); }

    inline CUDA_CALLABLE T identity() const;
};

template <> inline CUDA_CALLABLE int OpMax<int>::identity() const
{
    return -2147483648;  // INT_MIN
}

template <> inline CUDA_CALLABLE float OpMax<float>::identity() const { return -1e38f; }

template <> inline CUDA_CALLABLE double OpMax<double>::identity() const { return -1e308; }

template <typename T> struct OpMin {
    inline CUDA_CALLABLE T operator()(const T& a, const T& b) const { return min(a, b); }

    inline CUDA_CALLABLE T identity() const;
};

template <> inline CUDA_CALLABLE int OpMin<int>::identity() const
{
    return 2147483647;  // INT_MAX
}

template <> inline CUDA_CALLABLE float OpMin<float>::identity() const { return 1e38f; }

template <> inline CUDA_CALLABLE double OpMin<double>::identity() const { return 1e308; }

#if defined(__CUDA_ARCH__)

template <typename T, typename Op = OpAdd<T>> inline CUDA_CALLABLE T scan_warp_inclusive(int lane, T value)
{
    // Computes an inclusive cumulative sum/max/etc
    Op op;
#pragma unroll
    for (int i = 1; i < 32; i *= 2) {
        auto n = __shfl_up_sync(0xffffffffu, value, i, 32);

        if (lane >= i)
            value = op(value, n);
    }
    return value;
}

template <typename T, typename Op = OpAdd<T>>
inline CUDA_CALLABLE T scan_warp_exclusive(int lane, T value, T* inclusive_value)
{
    // Computes an exclusive scan by doing inclusive scan then shifting
    Op op;
    T inclusive = scan_warp_inclusive<T, Op>(lane, value);

    if (inclusive_value)
        *inclusive_value = inclusive;

    // Shift right by 1 to convert inclusive to exclusive
    T exclusive = __shfl_up_sync(0xffffffffu, inclusive, 1, 32);

    // Lane 0 gets the identity value
    if (lane == 0)
        exclusive = op.identity();

    return exclusive;
}


template <typename T, bool exclusive, typename Op = OpAdd<T>>
inline CUDA_CALLABLE T thread_block_scan(int lane, int warp_index, int num_warps, T value)
{
    __shared__ T sums[1024 / WP_TILE_WARP_SIZE];  // 1024 is the maximum number of threads per block
    Op op;

    T orig_value = value;

    if constexpr (exclusive) {
        value = scan_warp_exclusive<T, Op>(lane, value, lane == 31 ? &sums[warp_index] : nullptr);
    } else {
        value = scan_warp_inclusive<T, Op>(lane, value);
        if (lane == 31)
            sums[warp_index] = value;
    }

    WP_TILE_SYNC();

    if (warp_index == 0) {
        T v = lane < num_warps ? sums[lane] : op.identity();
        v = scan_warp_inclusive<T, Op>(lane, v);
        if (lane < num_warps)
            sums[lane] = v;
    }

    WP_TILE_SYNC();

    if (warp_index > 0) {
        value = op(value, sums[warp_index - 1]);
    }

    return value;
}

template <typename T, bool exclusive, typename Op = OpAdd<T>>
inline CUDA_CALLABLE void thread_block_scan(T* values, int num_elements)
{
    const int num_threads_in_block = blockDim.x;
    const int num_iterations = (num_elements + num_threads_in_block - 1) / num_threads_in_block;
    Op op;

    __shared__ T offset;
    if (threadIdx.x == 0)
        offset = op.identity();

    WP_TILE_SYNC();

    const int lane = WP_TILE_THREAD_IDX % WP_TILE_WARP_SIZE;
    const int warp_index = WP_TILE_THREAD_IDX / WP_TILE_WARP_SIZE;
    const int num_warps = num_threads_in_block / WP_TILE_WARP_SIZE;

    for (int i = 0; i < num_iterations; ++i) {
        int element_index = WP_TILE_THREAD_IDX + i * num_threads_in_block;
        T orig_value = element_index < num_elements ? values[element_index] : op.identity();
        T value = thread_block_scan<T, exclusive, Op>(lane, warp_index, num_warps, orig_value);
        if (element_index < num_elements) {
            values[element_index] = op(value, offset);
        }

        WP_TILE_SYNC();

        // Update offset with the inclusive total of this block
        if (threadIdx.x == num_threads_in_block - 1) {
            if constexpr (exclusive)
                // For exclusive scan, value is exclusive so add orig_value to get inclusive total
                offset = op(offset, op(value, orig_value));
            else
                // For inclusive scan, value already contains everything
                offset = op(offset, value);
        }

        WP_TILE_SYNC();
    }
}

template <typename Tile, typename Op = OpAdd<typename Tile::Type>>
inline CUDA_CALLABLE auto tile_scan_inclusive_impl(Tile& t)
{
    using T = typename Tile::Type;
    constexpr int num_elements_to_scan = Tile::Layout::Shape::size();

    // create a temporary shared tile to hold the input values
    __shared__ T smem[num_elements_to_scan];
    tile_shared_t<T, tile_layout_strided_t<typename Tile::Layout::Shape>, false> scratch(smem, nullptr);

    // copy input values to scratch space
    scratch.assign(t);

    T* values = &scratch.data(0);
    thread_block_scan<T, false, Op>(values, num_elements_to_scan);

    auto result = scratch.copy_to_register();

    WP_TILE_SYNC();

    return result;
}

template <typename Tile, typename Op = OpAdd<typename Tile::Type>>
inline CUDA_CALLABLE auto tile_scan_exclusive_impl(Tile& t)
{
    using T = typename Tile::Type;
    constexpr int num_elements_to_scan = Tile::Layout::Shape::size();

    // create a temporary shared tile to hold the input values
    __shared__ T smem[num_elements_to_scan];
    tile_shared_t<T, tile_layout_strided_t<typename Tile::Layout::Shape>, false> scratch(smem, nullptr);

    // copy input values to scratch space
    scratch.assign(t);

    T* values = &scratch.data(0);
    thread_block_scan<T, true, Op>(values, num_elements_to_scan);

    auto result = scratch.copy_to_register();

    WP_TILE_SYNC();

    return result;
}

#else

// CPU implementations
template <typename Tile, typename Op = OpAdd<typename Tile::Type>> inline auto tile_scan_inclusive_impl(Tile& t)
{
    using T = typename Tile::Type;
    constexpr int num_elements_to_scan = Tile::Layout::Shape::size();

    auto input = t.copy_to_register();
    auto output = tile_register_like<Tile>();

    using Layout = typename decltype(input)::Layout;
    Op op;

    T acc = op.identity();
    for (int i = 0; i < num_elements_to_scan; ++i) {
        acc = op(acc, input.data[i]);
        output.data[i] = acc;
    }

    return output;
}

template <typename Tile, typename Op = OpAdd<typename Tile::Type>> inline auto tile_scan_exclusive_impl(Tile& t)
{
    using T = typename Tile::Type;
    constexpr int num_elements_to_scan = Tile::Layout::Shape::size();

    auto input = t.copy_to_register();
    auto output = tile_register_like<Tile>();

    using Layout = typename decltype(input)::Layout;
    Op op;

    T acc = op.identity();
    for (int i = 0; i < num_elements_to_scan; ++i) {
        output.data[i] = acc;
        acc = op(acc, input.data[i]);
    }

    return output;
}

#endif  // !defined(__CUDA_ARCH__)

template <typename Tile> auto tile_scan_inclusive(Tile& t) { return tile_scan_inclusive_impl(t); }

template <typename Tile, typename AdjTile> void adj_tile_scan_inclusive(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // todo: not implemented
}

template <typename Tile> auto tile_scan_exclusive(Tile& t) { return tile_scan_exclusive_impl(t); }

template <typename Tile, typename AdjTile> void adj_tile_scan_exclusive(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // todo: not implemented
}

// Max scan operations
template <typename Tile> auto tile_scan_max_inclusive(Tile& t)
{
    return tile_scan_inclusive_impl<Tile, OpMax<typename Tile::Type>>(t);
}

template <typename Tile, typename AdjTile> void adj_tile_scan_max_inclusive(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // todo: not implemented
}

// Min scan operations
template <typename Tile> auto tile_scan_min_inclusive(Tile& t)
{
    return tile_scan_inclusive_impl<Tile, OpMin<typename Tile::Type>>(t);
}

template <typename Tile, typename AdjTile> void adj_tile_scan_min_inclusive(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // todo: not implemented
}

}  // namespace wp

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
