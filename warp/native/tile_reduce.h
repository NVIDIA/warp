// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tile.h"

#ifdef __clang__
// disable warnings related to C++17 extensions on CPU JIT builds
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif  // __clang__

namespace wp {


template <typename T> int argmax_tracker(T champion_value, T current_value, int champion_index, int current_index)
{
    return current_value > champion_value ? current_index : champion_index;
}

template <typename T> int argmin_tracker(T champion_value, T current_value, int champion_index, int current_index)
{
    return current_value < champion_value ? current_index : champion_index;
}


#if defined(__CUDA_ARCH__)

// half / float16 uses a dedicated overload to shuffle its 16-bit payload directly.
inline CUDA_CALLABLE half warp_shuffle_down(half val, int offset, wp_tile_lane_mask_bits_t mask)
{
    unsigned int bits = static_cast<unsigned int>(val.u);
    bits = __shfl_down_sync(mask, bits, offset, WP_TILE_WARP_SIZE);

    half result;
    result.u = static_cast<unsigned short>(bits);
    return result;
}

#ifndef WP_NO_BFLOAT16
inline CUDA_CALLABLE bfloat16 warp_shuffle_down(bfloat16 val, int offset, wp_tile_lane_mask_bits_t mask)
{
    unsigned int bits = static_cast<unsigned int>(val.u);
    bits = __shfl_down_sync(mask, bits, offset, WP_TILE_WARP_SIZE);

    bfloat16 result;
    result.u = static_cast<unsigned short>(bits);
    return result;
}
#endif  // WP_NO_BFLOAT16

template <typename T> inline CUDA_CALLABLE T warp_shuffle_down(T val, int offset, wp_tile_lane_mask_bits_t mask)
{
    // Shuffle word-by-word over the raw bytes so any trivially-copyable value type is
    // supported. A plain word buffer (rather than a union over T) avoids the deleted
    // default constructor that a union acquires when T has a non-trivial default
    // constructor (e.g. quaternions and transforms), and is padded up to a word
    // multiple so partial-word types stay in bounds. The buffers are over-aligned to
    // max(alignof(T), alignof(Word)) so reinterpreting them as a T* is well-defined for
    // over-aligned types such as wp::float64 (alignas cannot request less than the
    // array's natural Word alignment, so the Word floor is required).
    typedef unsigned int Word;

    constexpr int word_count = (sizeof(T) + sizeof(Word) - 1) / sizeof(Word);

    constexpr size_t buffer_align = alignof(T) > alignof(Word) ? alignof(T) : alignof(Word);

    alignas(buffer_align) Word input[word_count] = {};
    alignas(buffer_align) Word output[word_count] = {};

    *reinterpret_cast<T*>(input) = val;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < word_count; ++i) {
        output[i] = __shfl_down_sync(mask, input[i], offset, WP_TILE_WARP_SIZE);
    }

    return *reinterpret_cast<T*>(output);
}

// vector overload
template <unsigned Length, typename T>
inline CUDA_CALLABLE wp::vec_t<Length, T>
warp_shuffle_down(wp::vec_t<Length, T> val, int offset, wp_tile_lane_mask_bits_t mask)
{
    wp::vec_t<Length, T> result;

    for (unsigned i = 0; i < Length; ++i)
        result[i] = __shfl_down_sync(mask, val[i], offset, WP_TILE_WARP_SIZE);

    return result;
}

template <unsigned Length>
inline CUDA_CALLABLE wp::vec_t<Length, half>
warp_shuffle_down(wp::vec_t<Length, half> val, int offset, wp_tile_lane_mask_bits_t mask)
{
    wp::vec_t<Length, half> result;

    for (unsigned i = 0; i < Length; ++i)
        result[i] = warp_shuffle_down(val[i], offset, mask);

    return result;
}

#ifndef WP_NO_BFLOAT16
template <unsigned Length>
inline CUDA_CALLABLE wp::vec_t<Length, bfloat16>
warp_shuffle_down(wp::vec_t<Length, bfloat16> val, int offset, wp_tile_lane_mask_bits_t mask)
{
    wp::vec_t<Length, bfloat16> result;

    for (unsigned i = 0; i < Length; ++i)
        result[i] = warp_shuffle_down(val[i], offset, mask);

    return result;
}
#endif  // WP_NO_BFLOAT16

// matrix overload
template <unsigned Rows, unsigned Cols, typename T>
inline CUDA_CALLABLE wp::mat_t<Rows, Cols, T>
warp_shuffle_down(wp::mat_t<Rows, Cols, T> val, int offset, wp_tile_lane_mask_bits_t mask)
{
    wp::mat_t<Rows, Cols, T> result;

    for (unsigned i = 0; i < Rows; ++i)
        for (unsigned j = 0; j < Cols; ++j)
            result.data[i][j] = __shfl_down_sync(mask, val.data[i][j], offset, WP_TILE_WARP_SIZE);

    return result;
}

template <unsigned Rows, unsigned Cols>
inline CUDA_CALLABLE wp::mat_t<Rows, Cols, half>
warp_shuffle_down(wp::mat_t<Rows, Cols, half> val, int offset, wp_tile_lane_mask_bits_t mask)
{
    wp::mat_t<Rows, Cols, half> result;

    for (unsigned i = 0; i < Rows; ++i)
        for (unsigned j = 0; j < Cols; ++j)
            result.data[i][j] = warp_shuffle_down(val.data[i][j], offset, mask);

    return result;
}

#ifndef WP_NO_BFLOAT16
template <unsigned Rows, unsigned Cols>
inline CUDA_CALLABLE wp::mat_t<Rows, Cols, bfloat16>
warp_shuffle_down(wp::mat_t<Rows, Cols, bfloat16> val, int offset, wp_tile_lane_mask_bits_t mask)
{
    wp::mat_t<Rows, Cols, bfloat16> result;

    for (unsigned i = 0; i < Rows; ++i)
        for (unsigned j = 0; j < Cols; ++j)
            result.data[i][j] = warp_shuffle_down(val.data[i][j], offset, mask);

    return result;
}
#endif  // WP_NO_BFLOAT16


template <typename T> inline CUDA_CALLABLE T* warp_shuffle_down(T* val, int offset, wp_tile_lane_mask_bits_t mask)
{
    unsigned long long ptr = reinterpret_cast<unsigned long long>(val);
    unsigned int ptr_lo = static_cast<unsigned int>(ptr);
    unsigned int ptr_hi = static_cast<unsigned int>(ptr >> 32);
    ptr_lo = __shfl_down_sync(mask, ptr_lo, offset, WP_TILE_WARP_SIZE);
    ptr_hi = __shfl_down_sync(mask, ptr_hi, offset, WP_TILE_WARP_SIZE);
    ptr = (static_cast<unsigned long long>(ptr_hi) << 32) | static_cast<unsigned long long>(ptr_lo);
    return reinterpret_cast<T*>(ptr);
}

inline CUDA_CALLABLE wp::shape_t warp_shuffle_down(wp::shape_t val, int offset, wp_tile_lane_mask_bits_t mask)
{
    wp::shape_t result;

    for (int i = 0; i < wp::ARRAY_MAX_DIMS; ++i)
        result.dims[i] = __shfl_down_sync(mask, val.dims[i], offset, WP_TILE_WARP_SIZE);

    return result;
}

template <typename T>
inline CUDA_CALLABLE wp::array_t<T> warp_shuffle_down(wp::array_t<T> val, int offset, wp_tile_lane_mask_bits_t mask)
{
    wp::array_t<T> result;

    result.data = wp::warp_shuffle_down(val.data, offset, mask);
    result.grad = wp::warp_shuffle_down(val.grad, offset, mask);
    result.shape = wp::warp_shuffle_down(val.shape, offset, mask);
    for (int i = 0; i < wp::ARRAY_MAX_DIMS; ++i)
        result.strides[i] = __shfl_down_sync(mask, val.strides[i], offset, WP_TILE_WARP_SIZE);
    result.ndim
        = static_cast<uint16_t>(__shfl_down_sync(mask, static_cast<unsigned int>(val.ndim), offset, WP_TILE_WARP_SIZE));
    result.flags = static_cast<uint16_t>(
        __shfl_down_sync(mask, static_cast<unsigned int>(val.flags), offset, WP_TILE_WARP_SIZE)
    );

    return result;
}

template <typename T>
inline CUDA_CALLABLE wp::indexedarray_t<T>
warp_shuffle_down(wp::indexedarray_t<T> val, int offset, wp_tile_lane_mask_bits_t mask)
{
    wp::indexedarray_t<T> result;

    result.arr = wp::warp_shuffle_down(val.arr, offset, mask);
    for (int i = 0; i < wp::ARRAY_MAX_DIMS; ++i)
        result.indices[i] = wp::warp_shuffle_down(val.indices[i], offset, mask);
    result.shape = wp::warp_shuffle_down(val.shape, offset, mask);

    return result;
}


template <typename T, typename Op> inline CUDA_CALLABLE T warp_reduce(T val, Op f, wp_tile_lane_mask_bits_t mask)
{
    T sum = val;

    if (mask == WP_TILE_LANE_MASK_ALL) {
        // handle case where entire warp is active
        for (int offset = WP_TILE_WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum = f(sum, warp_shuffle_down(sum, offset, mask));
        }
    } else {
        // handle partial warp case - works for contiguous masks
        for (int offset = WP_TILE_WARP_SIZE / 2; offset > 0; offset /= 2) {
            T shfl_val = warp_shuffle_down(sum, offset, mask);
            if ((mask & (((wp_tile_lane_mask_bits_t)1) << ((threadIdx.x + offset) % WP_TILE_WARP_SIZE))) != 0)
                sum = f(sum, shfl_val);
        }
    }

    return sum;
}

template <typename T> struct ValueAndIndex {
    T value;
    int index;
};

template <typename T, typename Op, typename OpTrack>
inline CUDA_CALLABLE ValueAndIndex<T>
warp_reduce_tracked(T val, int idx, Op f, OpTrack track, wp_tile_lane_mask_bits_t mask)
{
    T sum = val;
    int index = idx;

    if (mask == WP_TILE_LANE_MASK_ALL) {
        // handle case where entire warp is active
        for (int offset = WP_TILE_WARP_SIZE / 2; offset > 0; offset /= 2) {
            auto shfl_val = warp_shuffle_down(sum, offset, mask);
            int shfl_idx = warp_shuffle_down(index, offset, mask);
            index = track(sum, shfl_val, index, shfl_idx);
            sum = f(sum, shfl_val);
        }
    } else {
        // handle partial warp case
        for (int offset = WP_TILE_WARP_SIZE / 2; offset > 0; offset /= 2) {
            T shfl_val = warp_shuffle_down(sum, offset, mask);
            int shfl_index = warp_shuffle_down(index, offset, mask);
            if ((mask & (((wp_tile_lane_mask_bits_t)1) << ((threadIdx.x + offset) % WP_TILE_WARP_SIZE))) != 0) {
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

// combines per-thread reduction results across warps and the entire block
// assumes each thread has already reduced its local data to thread_sum
// returns the block-wide reduced value (only valid in thread 0)
template <typename T, typename Op>
inline CUDA_CALLABLE T
block_combine_thread_results(T thread_sum, bool thread_has_data, Op f, T* partials, int& active_warps)
{
    const int warp_index = threadIdx.x / WP_TILE_WARP_SIZE;
    const int lane_index = threadIdx.x % WP_TILE_WARP_SIZE;

    // determine which threads have data
    wp_tile_lane_mask_bits_t mask = __ballot_sync(WP_TILE_LANE_MASK_ALL, thread_has_data);
    bool warp_is_active = mask != 0;

    // warp reduction
    T warp_sum;
    if (thread_has_data)
        warp_sum = warp_reduce(thread_sum, f, mask);

    // lane 0 of each active warp writes to shared memory and increments counter
    if (lane_index == 0 && warp_is_active) {
        partials[warp_index] = warp_sum;
        atomicAdd(&active_warps, 1);
    }

    // sync to ensure all warps have written their partials
    WP_TILE_SYNC();

    // thread 0 performs final reduction across active warps
    T block_sum;
    if (threadIdx.x == 0) {
        block_sum = partials[0];

        for (int w = 1; w < active_warps; ++w) {
            block_sum = f(block_sum, partials[w]);
        }
    }

    return block_sum;
}

// non-axis version which computes sum
// across the entire tile using the whole block
template <typename Tile, typename Op> CUDA_CALLABLE_DEVICE auto tile_reduce_impl(Op f, Tile& t)
{
    using T = typename Tile::Type;

    auto input = t.copy_to_register();
    auto output = tile_register_t<T, tile_layout_register_t<tile_shape_t<1>>>();

    constexpr int warp_count = (WP_TILE_BLOCK_DIM + WP_TILE_WARP_SIZE - 1) / WP_TILE_WARP_SIZE;

    using Layout = typename decltype(input)::Layout;

    // step 1: each thread reduces its own registers locally
    T thread_sum = input.data[0];
    bool thread_has_data = Layout::valid(Layout::linear_from_register(0));

    WP_PRAGMA_UNROLL
    for (int i = 1; i < Layout::NumRegs; ++i) {
        int linear = Layout::linear_from_register(i);
        if (!Layout::valid(linear))
            break;

        thread_sum = f(thread_sum, input.data[i]);
    }

    // step 2: combine thread results across block
    T block_sum;
    if constexpr (warp_count == 1) {
        // fast path: single warp, just do warp reduction
        wp_tile_lane_mask_bits_t mask = __ballot_sync(WP_TILE_LANE_MASK_ALL, thread_has_data);
        if (thread_has_data)
            block_sum = warp_reduce(thread_sum, f, mask);

        // write from first active lane (warp_reduce result is only valid there)
        int first_active = WP_TILE_LANE_MASK_FFS(mask) - 1;
        if (threadIdx.x == first_active)
            output.data[0] = block_sum;
    } else {
        // multi-warp path: cross-warp reduction via shared memory
        __shared__ T partials[warp_count];
        __shared__ int active_warps;

        if (threadIdx.x == 0)
            active_warps = 0;

        WP_TILE_SYNC();

        block_sum = block_combine_thread_results(thread_sum, thread_has_data, f, partials, active_warps);

        if (threadIdx.x == 0)
            output.data[0] = block_sum;
    }

    return output;
}

template <int Axis, typename Op, typename Tile>
CUDA_CALLABLE_DEVICE auto
tile_reduce_axis_impl(Op f, Tile& t, typename Tile::Type empty_identity, bool has_empty_identity)
{
    using T = typename Tile::Type;
    using InputShape = typename Tile::Layout::Shape;
    using OutputShape = typename tile_shape_remove_dim<Axis, InputShape>::type;

    constexpr int reduce_dim_size = InputShape::dim(Axis);
    constexpr int output_size = OutputShape::size();

    // Partial CUDA blocks cannot execute cooperative tile operations because
    // all hardware threads must reach each barrier. The identity metadata is
    // used only by the CPU cooperative-fiber path below.
    (void)empty_identity;
    (void)has_empty_identity;

    // special case: 1D input delegates to block-wide tile_reduce_impl for optimal performance
    if constexpr (InputShape::N == 1) {
        return tile_reduce_impl(f, t);
    }

    // shared memory buffer for the output (used by all tiers)
    __shared__ T output_buffer[output_size];

    // create output layout for coordinate conversion (used by all tiers)
    using OutputLayout = tile_layout_strided_t<OutputShape>;

    if constexpr (reduce_dim_size <= 32) {
        // Tier 1: Single thread per output element (optimal for small reductions)

        // each thread processes output elements, performing reduction along the axis
        for (int out_idx = WP_TILE_THREAD_IDX; out_idx < output_size; out_idx += WP_TILE_BLOCK_DIM) {
            // convert output linear index to output coordinates
            auto out_coord = OutputLayout::coord_from_linear(out_idx);

            // initialize accumulator with first element along the reduction axis
            T accumulator = t.data(tile_coord_insert_axis<Axis>(out_coord, 0));

            // reduce across the axis
            for (int i = 1; i < reduce_dim_size; ++i) {
                accumulator = f(accumulator, t.data(tile_coord_insert_axis<Axis>(out_coord, i)));
            }

            // store to output buffer
            output_buffer[out_idx] = accumulator;
        }

        // sync before reading output
        WP_TILE_SYNC();
    } else if constexpr (reduce_dim_size <= 256) {
        // Tier 2: Warp-based reduction (one warp per output element)
        constexpr int warp_count = (WP_TILE_BLOCK_DIM + WP_TILE_WARP_SIZE - 1) / WP_TILE_WARP_SIZE;
        const int warp_index = threadIdx.x / WP_TILE_WARP_SIZE;
        const int lane_index = threadIdx.x % WP_TILE_WARP_SIZE;

        constexpr int chunks_per_slice = (reduce_dim_size + WP_TILE_WARP_SIZE - 1) / WP_TILE_WARP_SIZE;

        // shared memory: one accumulator per warp
        __shared__ T warp_partials[warp_count];

        // each warp processes output slices
        for (int out_idx = warp_index; out_idx < output_size; out_idx += warp_count) {
            auto out_coord = OutputLayout::coord_from_linear(out_idx);

            // process the reduction axis in chunks of 32
            for (int chunk = 0; chunk < chunks_per_slice; ++chunk) {
                int axis_idx = chunk * WP_TILE_WARP_SIZE + lane_index;
                bool valid = axis_idx < reduce_dim_size;

                T val;
                if (valid) {
                    auto in_coord = tile_coord_insert_axis<Axis>(out_coord, axis_idx);
                    val = t.data(in_coord);
                }

                // warp reduce this chunk (only valid lanes may call warp_reduce,
                // because __shfl_down_sync requires all executing threads to be in the mask)
                wp_tile_lane_mask_bits_t mask = __ballot_sync(WP_TILE_LANE_MASK_ALL, valid);
                T chunk_result;
                if (valid)
                    chunk_result = warp_reduce(val, f, mask);

                // lane 0 accumulates the chunk result
                if (lane_index == 0) {
                    if (chunk == 0)
                        warp_partials[warp_index] = chunk_result;
                    else
                        warp_partials[warp_index] = f(warp_partials[warp_index], chunk_result);
                }
            }

            // lane 0 writes final result for this output element
            if (lane_index == 0)
                output_buffer[out_idx] = warp_partials[warp_index];
        }

        // sync before reading output
        WP_TILE_SYNC();
    } else {
        // Tier 3: Block-level reduction (entire block collaborates on each output element)
        constexpr int warp_count = (WP_TILE_BLOCK_DIM + WP_TILE_WARP_SIZE - 1) / WP_TILE_WARP_SIZE;

        // shared memory for cross-warp reduction (only needed for multi-warp)
        __shared__ T partials[warp_count];
        __shared__ int active_warps;

        // process each output element sequentially with full block cooperation
        for (int out_idx = 0; out_idx < output_size; ++out_idx) {
            auto out_coord = OutputLayout::coord_from_linear(out_idx);

            // step 1: each thread reduces its strided subset of the slice locally
            bool thread_has_data = threadIdx.x < reduce_dim_size;
            T thread_sum;

            if (thread_has_data) {
                // initialize with first element
                auto in_coord = tile_coord_insert_axis<Axis>(out_coord, threadIdx.x);
                thread_sum = t.data(in_coord);

                // reduce remaining elements with stride
                for (int i = threadIdx.x + WP_TILE_BLOCK_DIM; i < reduce_dim_size; i += WP_TILE_BLOCK_DIM) {
                    auto in_coord = tile_coord_insert_axis<Axis>(out_coord, i);
                    T val = t.data(in_coord);
                    thread_sum = f(thread_sum, val);
                }
            }

            // step 2: combine thread results across block
            T block_sum;
            if constexpr (warp_count == 1) {
                // fast path: single warp, just do warp reduction
                wp_tile_lane_mask_bits_t mask = __ballot_sync(WP_TILE_LANE_MASK_ALL, thread_has_data);
                if (thread_has_data)
                    block_sum = warp_reduce(thread_sum, f, mask);

                // write from first active lane (warp_reduce result is only valid there)
                int first_active = WP_TILE_LANE_MASK_FFS(mask) - 1;
                if (threadIdx.x == first_active)
                    output_buffer[out_idx] = block_sum;
            } else {
                // multi-warp path: cross-warp reduction via shared memory
                if (threadIdx.x == 0)
                    active_warps = 0;

                WP_TILE_SYNC();

                block_sum = block_combine_thread_results(thread_sum, thread_has_data, f, partials, active_warps);

                if (threadIdx.x == 0)
                    output_buffer[out_idx] = block_sum;
            }

            // sync before next output element
            WP_TILE_SYNC();
        }
    }

    // copy from shared memory buffer to register tile (common to all tiers)
    auto output = tile_register_t<T, tile_layout_register_t<OutputShape>>();
    using OutputRegLayout = typename decltype(output)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < OutputRegLayout::NumRegs; ++i) {
        int linear = OutputRegLayout::linear_from_register(i);
        output.data[i] = OutputRegLayout::valid(linear) ? output_buffer[linear] : T {};
    }

    return output;
}

// non-axis version which computes sum
// across the entire tile using the whole block
template <typename Tile, typename Op, typename OpTrack>
CUDA_CALLABLE_DEVICE auto tile_arg_reduce_impl(Op f, OpTrack track, Tile& t)
{
    using T = typename Tile::Type;

    auto input = t.copy_to_register();
    auto output = tile_register_t<int, tile_layout_register_t<tile_shape_t<1>>>();

    const int warp_count = (WP_TILE_BLOCK_DIM + WP_TILE_WARP_SIZE - 1) / WP_TILE_WARP_SIZE;
    const int warp_index = threadIdx.x / WP_TILE_WARP_SIZE;
    const int lane_index = threadIdx.x % WP_TILE_WARP_SIZE;

    using Layout = typename decltype(input)::Layout;

    int champion_index = Layout::NumRegs > 0 ? Layout::linear_from_register(0) : -1;
    T thread_sum = input.data[0];
    bool thread_has_data = Layout::valid(Layout::linear_from_register(0));

    // thread reduction
    WP_PRAGMA_UNROLL
    for (int i = 1; i < Layout::NumRegs; ++i) {
        int linear = Layout::linear_from_register(i);
        if (!Layout::valid(linear))
            break;

        champion_index = track(thread_sum, input.data[i], champion_index, linear);
        thread_sum = f(thread_sum, input.data[i]);
    }

    // determine which threads have valid data
    wp_tile_lane_mask_bits_t mask = __ballot_sync(WP_TILE_LANE_MASK_ALL, thread_has_data);
    bool warp_is_active = mask != 0;

    // warp reduction (only threads with valid data may participate,
    // because __shfl_down_sync requires all executing threads to be in the mask)
    ValueAndIndex<T> warp_sum;
    if (thread_has_data)
        warp_sum = warp_reduce_tracked(thread_sum, champion_index, f, track, mask);

    // fixed size scratch pad for partial results in shared memory
    __shared__ T partials[warp_count];
    __shared__ int partials_idx[warp_count];

    // count of active warps
    __shared__ int active_warps;
    if (threadIdx.x == 0)
        active_warps = 0;

    // ensure active_warps is initialized
    WP_TILE_SYNC();

    if (lane_index == 0 && warp_is_active) {
        partials[warp_index] = warp_sum.value;
        partials_idx[warp_index] = warp_sum.index;
        atomicAdd(&active_warps, 1);
    }

    // ensure partials are ready
    WP_TILE_SYNC();

    // reduce across block, todo: use warp_reduce() here
    if (threadIdx.x == 0) {
        T block_sum = partials[0];
        int block_champion_index = partials_idx[0];

        WP_PRAGMA_UNROLL
        for (int i = 1; i < active_warps; ++i) {
            block_champion_index = track(block_sum, partials[i], block_champion_index, partials_idx[i]);
            block_sum = f(block_sum, partials[i]);
        }

        output.data[0] = block_champion_index;
    }

    return output;
}

#else

// CPU implementation

template <typename Tile, typename Op> auto tile_reduce_impl(Op f, Tile& t)
{
    using T = typename Tile::Type;

    auto input = t.copy_to_register();
    auto output = tile_register_t<T, tile_layout_register_t<tile_shape_t<1>>>();

    using Layout = typename decltype(input)::Layout;

    if constexpr (WP_TILE_BLOCK_DIM == 1) {
        // Block-dim 1 fast path: full tile in one thread's registers.
        T sum = input.data[0];
        WP_PRAGMA_UNROLL
        for (int i = 1; i < Layout::NumRegs; ++i) {
            int linear = Layout::linear_from_register(i);
            if (!Layout::valid(linear))
                break;
            sum = f(sum, input.data[i]);
        }
        output.data[0] = sum;
        return output;
    } else {
        // Cross-fiber reduction: each fiber reduces its register slice into
        // a partial, drops it into shared scratch[tid], syncs, then every
        // fiber re-reduces the partials so they all return the same total.
        // O(block_dim) per fiber after the sync; comparable to the GPU
        // warp-shuffle path for moderate block_dim.
        T* scratch = (T*)tile_shared_storage_t::alloc(int(sizeof(T) * WP_TILE_BLOCK_DIM));
        bool* has_data = (bool*)tile_shared_storage_t::alloc(int(sizeof(bool) * WP_TILE_BLOCK_DIM));
        const int tid = WP_TILE_THREAD_IDX;

        // Zero `has_data` for every slot — including slots whose fibers
        // returned early in the partial-block thunk (`task_index >= dim->size`)
        // and so never reach here. The bump allocator hands back arenas that
        // are dirty across calls, so without this the stale flags from a
        // previous reduction make the combine loop read uninitialized
        // `scratch[i]`. Every participating fiber clears every flag before
        // the barrier.
        // The redundant work is intentional: lane 0, or any sparse subset
        // of lanes, may already have returned from the logical block.
        for (int i = 0; i < WP_TILE_BLOCK_DIM; ++i)
            has_data[i] = false;
        WP_TILE_SYNC();

        // Local partial. If a fiber has zero register slots (only possible at
        // block_dim > Size), it contributes the identity, but we approximate
        // by skipping it via the first-valid check.
        bool have_first = false;
        T partial {};
        WP_PRAGMA_UNROLL
        for (int i = 0; i < Layout::NumRegs; ++i) {
            int linear = Layout::linear_from_register(i);
            if (!Layout::valid(linear))
                break;
            if (!have_first) {
                partial = input.data[i];
                have_first = true;
            } else {
                partial = f(partial, input.data[i]);
            }
        }
        scratch[tid] = partial;
        has_data[tid] = have_first;
        WP_TILE_SYNC();

        // Combine across fibers in deterministic tid order.
        bool got = false;
        T total {};
        for (int i = 0; i < WP_TILE_BLOCK_DIM; ++i) {
            if (!has_data[i])
                continue;
            if (!got) {
                total = scratch[i];
                got = true;
            } else {
                total = f(total, scratch[i]);
            }
        }
        WP_TILE_SYNC();
        tile_shared_storage_t::alloc(-(int)(sizeof(bool) * WP_TILE_BLOCK_DIM));
        tile_shared_storage_t::alloc(-(int)(sizeof(T) * WP_TILE_BLOCK_DIM));

        output.data[0] = total;
        return output;
    }
}

template <int Axis, typename Op, typename Tile>
auto tile_reduce_axis_impl(Op f, Tile& t, typename Tile::Type empty_identity, bool has_empty_identity)
{
    using T = typename Tile::Type;
    using InputShape = typename Tile::Layout::Shape;
    using OutputShape = typename tile_shape_remove_dim<Axis, InputShape>::type;

    constexpr int reduce_dim_size = InputShape::dim(Axis);
    constexpr int input_size = InputShape::size();

    auto input = t.copy_to_register();
    auto output = tile_register_t<T, tile_layout_register_t<OutputShape>>();
    using InputLayout = tile_layout_register_t<InputShape>;
    using OutputLayout = typename decltype(output)::Layout;

    constexpr int output_size = OutputShape::size();

    if constexpr (WP_TILE_BLOCK_DIM == 1) {
        // Fast path: full input held in this thread's registers.
        for (int out_idx = 0; out_idx < output_size; ++out_idx) {
            T accumulator;
            if constexpr (InputShape::N == 1) {
                accumulator = input.data[0];
                for (int i = 1; i < reduce_dim_size; ++i) {
                    accumulator = f(accumulator, input.data[i]);
                }
            } else {
                auto out_coord = OutputLayout::coord_from_linear(out_idx);
                auto coord_0 = tile_coord_insert_axis<Axis>(out_coord, 0);
                int input_reg_0 = InputLayout::register_from_linear(InputLayout::linear_from_coord(coord_0));
                accumulator = input.data[input_reg_0];
                for (int i = 1; i < reduce_dim_size; ++i) {
                    auto coord_i = tile_coord_insert_axis<Axis>(out_coord, i);
                    int input_reg_i = InputLayout::register_from_linear(InputLayout::linear_from_coord(coord_i));
                    accumulator = f(accumulator, input.data[input_reg_i]);
                }
            }
            int output_reg = OutputLayout::register_from_linear(out_idx);
            output.data[output_reg] = accumulator;
        }
        return output;
    } else {
        // Block-dim>1: input registers are split across fibers. Gather the
        // whole tile into block-shared scratch via the linear<->register
        // mapping (same shape as `tile_scan_inclusive_impl`'s gather), sync,
        // then each fiber computes its output register slots by reading
        // along the reduced axis from scratch.
        T* scratch = (T*)tile_shared_storage_t::alloc(int(sizeof(T) * input_size));
        bool* active = (bool*)tile_shared_storage_t::alloc(int(sizeof(bool) * WP_TILE_BLOCK_DIM));
        const int tid = WP_TILE_THREAD_IDX;

        // In a partial CPU block, tail fibers return before entering the
        // kernel and therefore never populate their register-owned scratch
        // slots. Track the fibers that did enter this operation so reductions
        // never consume stale values left in those slots by an earlier tile
        // operation or block.
        // Do not elect lane 0 as the initializer because it may have
        // returned while the remaining fibers continue this block.
        for (int i = 0; i < WP_TILE_BLOCK_DIM; ++i)
            active[i] = false;
        WP_TILE_SYNC();
        active[tid] = true;

        WP_PRAGMA_UNROLL
        for (int r = 0; r < InputLayout::NumRegs; ++r) {
            int linear = InputLayout::linear_from_register(r);
            if (linear < input_size && InputLayout::valid(linear))
                scratch[linear] = input.data[r];
        }
        WP_TILE_SYNC();

        WP_PRAGMA_UNROLL
        for (int r = 0; r < OutputLayout::NumRegs; ++r) {
            int out_idx = OutputLayout::linear_from_register(r);
            if (out_idx >= output_size || !OutputLayout::valid(out_idx))
                continue;

            bool got = false;
            T accumulator {};
            if constexpr (InputShape::N == 1) {
                for (int i = 0; i < reduce_dim_size; ++i) {
                    if (!active[InputLayout::thread_from_linear(i)])
                        continue;
                    if (!got) {
                        accumulator = scratch[i];
                        got = true;
                    } else {
                        accumulator = f(accumulator, scratch[i]);
                    }
                }
            } else {
                auto out_coord = OutputLayout::coord_from_linear(out_idx);
                for (int i = 0; i < reduce_dim_size; ++i) {
                    auto coord_i = tile_coord_insert_axis<Axis>(out_coord, i);
                    int linear_i = InputLayout::linear_from_coord(coord_i);
                    if (!active[InputLayout::thread_from_linear(linear_i)])
                        continue;
                    if (!got) {
                        accumulator = scratch[linear_i];
                        got = true;
                    } else {
                        accumulator = f(accumulator, scratch[linear_i]);
                    }
                }
            }
            if (got) {
                output.data[r] = accumulator;
            } else if (has_empty_identity) {
                output.data[r] = empty_identity;
            } else {
                _wp_assert(
                    "Warp tile_reduce() axis slice has no active values and the reduction operator has no declared "
                    "identity",
                    __FILE__, (unsigned int)__LINE__
                );
                output.data[r] = T {};
            }
        }
        WP_TILE_SYNC();
        tile_shared_storage_t::alloc(-(int)(sizeof(bool) * WP_TILE_BLOCK_DIM));
        tile_shared_storage_t::alloc(-(int)(sizeof(T) * input_size));
        return output;
    }
}

template <typename Tile, typename Op, typename OpTrack> auto tile_arg_reduce_impl(Op f, OpTrack track, Tile& t)
{
    using T = typename Tile::Type;

    auto input = t.copy_to_register();
    auto output = tile_register_t<int, tile_layout_register_t<tile_shape_t<1>>>();

    using Layout = typename decltype(input)::Layout;

    if constexpr (WP_TILE_BLOCK_DIM == 1) {
        // Fast path.
        int champion_index = Layout::NumRegs > 0 ? Layout::linear_from_register(0) : -1;
        T sum = input.data[0];
        WP_PRAGMA_UNROLL
        for (int i = 1; i < Layout::NumRegs; ++i) {
            int linear = Layout::linear_from_register(i);
            if (!Layout::valid(linear))
                break;
            champion_index = track(sum, input.data[i], champion_index, linear);
            sum = f(sum, input.data[i]);
        }
        output.data[0] = champion_index;
        return output;
    } else {
        // Cross-fiber arg-reduction: each fiber finds its local champion
        // (value + linear index in the global tile), drops both into shared
        // scratch[tid], syncs, then every fiber re-reduces using the same
        // `track` op so they all return the same global champion index.
        T* val_scratch = (T*)tile_shared_storage_t::alloc(int(sizeof(T) * WP_TILE_BLOCK_DIM));
        int* idx_scratch = (int*)tile_shared_storage_t::alloc(int(sizeof(int) * WP_TILE_BLOCK_DIM));
        bool* has_data = (bool*)tile_shared_storage_t::alloc(int(sizeof(bool) * WP_TILE_BLOCK_DIM));
        const int tid = WP_TILE_THREAD_IDX;

        // Zero `has_data` for every slot — same partial-block reasoning as
        // `tile_reduce_impl`. Without this, fibers that returned early in
        // the bounds-check thunk leave stale `true` flags and the combine
        // loop reads uninitialized scratch.
        for (int i = 0; i < WP_TILE_BLOCK_DIM; ++i)
            has_data[i] = false;
        WP_TILE_SYNC();

        // Local champion across this fiber's registers.
        bool got = false;
        T local_val {};
        int local_idx = -1;
        WP_PRAGMA_UNROLL
        for (int i = 0; i < Layout::NumRegs; ++i) {
            int linear = Layout::linear_from_register(i);
            if (!Layout::valid(linear))
                break;
            if (!got) {
                local_val = input.data[i];
                local_idx = linear;
                got = true;
            } else {
                local_idx = track(local_val, input.data[i], local_idx, linear);
                local_val = f(local_val, input.data[i]);
            }
        }
        val_scratch[tid] = local_val;
        idx_scratch[tid] = local_idx;
        has_data[tid] = got;
        WP_TILE_SYNC();

        // Combine across fibers in deterministic tid order.
        bool combined = false;
        T total_val {};
        int total_idx = -1;
        for (int i = 0; i < WP_TILE_BLOCK_DIM; ++i) {
            if (!has_data[i])
                continue;
            if (!combined) {
                total_val = val_scratch[i];
                total_idx = idx_scratch[i];
                combined = true;
            } else {
                total_idx = track(total_val, val_scratch[i], total_idx, idx_scratch[i]);
                total_val = f(total_val, val_scratch[i]);
            }
        }
        WP_TILE_SYNC();
        tile_shared_storage_t::alloc(-(int)(sizeof(bool) * WP_TILE_BLOCK_DIM));
        tile_shared_storage_t::alloc(-(int)(sizeof(int) * WP_TILE_BLOCK_DIM));
        tile_shared_storage_t::alloc(-(int)(sizeof(T) * WP_TILE_BLOCK_DIM));

        output.data[0] = total_idx;
        return output;
    }
}

#endif  // !defined(__CUDA_ARCH__)

// entry point for Python code-gen, wraps op in a lambda to perform overload resolution
#define tile_reduce(op, t) tile_reduce_impl([](auto x, auto y) { return op(x, y);}, t)

template <typename Op, typename Tile, typename AdjOp, typename AdjTile, typename AdjRet>
void adj_tile_reduce(Op op, Tile& t, AdjOp& adj_op, AdjTile& adj_t, AdjRet& adj_ret)
{
    // MISSINGADJOINT: for differentiable ops, distribute adj_ret to all input elements via
    // op's adjoint
}

#define tile_arg_reduce(op, opTrack, t) tile_arg_reduce_impl([](auto x, auto y) { return op(x, y);}, [](auto a, auto b, auto c, auto d) { return opTrack(a, b, c, d); }, t)

// axis-specific reduction entry points
#define tile_reduce_axis(op, t, axis, identity, has_identity) \
    tile_reduce_axis_impl<axis>([](auto x, auto y) { return op(x, y);}, t, identity, has_identity)

template <typename Op, typename Tile, typename AdjOp, typename AdjTile, typename AdjRet>
void adj_tile_reduce_axis(Op op, Tile& t, int axis, AdjOp& adj_op, AdjTile& adj_t, int& adj_axis, AdjRet& adj_ret)
{
    // MISSINGADJOINT: for differentiable ops, distribute adj_ret along the reduction axis
    // via op's adjoint
}

// convenience methods for specific reductions

// whole-tile sum
template <typename Tile> auto tile_sum(Tile& t) { return tile_reduce(add, t); }

// special case adjoint for summation
template <typename Tile, typename AdjTile> CUDA_CALLABLE void adj_tile_sum(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    using T = typename Tile::Type;

    auto adj_reg = adj_ret.grad_to_register();

#if defined(__CUDA_ARCH__)
    // broadcast incoming adjoint to block
    __shared__ T scratch;
    if (WP_TILE_THREAD_IDX == 0)
        scratch = adj_reg.data[0];

    WP_TILE_SYNC();
#else
    // CPU. At block_dim==1 this thread holds the single 1-element adjoint
    // tile in `adj_reg.data[0]`. At block_dim>1 only thread 0's register
    // slot holds the meaningful value (the others' linear index is invalid),
    // so broadcast through a block-shared scratch slot, mirroring the GPU
    // path. Without the broadcast each fiber would seed `scratch` with its
    // own undefined adj_reg.data[0] and the gradient would be garbage.
    T scratch_local {};
    if constexpr (WP_TILE_BLOCK_DIM == 1) {
        scratch_local = adj_reg.data[0];
    } else {
        wp_block_shared<T> scratch_holder;
        if (WP_TILE_THREAD_IDX == 0)
            *scratch_holder = adj_reg.data[0];
        WP_TILE_SYNC();
        scratch_local = *scratch_holder;
        WP_TILE_SYNC();
    }
    T& scratch = scratch_local;
#endif

    auto adj_ret_reg = tile_register_like<Tile>();
    using Layout = typename decltype(adj_ret_reg)::Layout;
    for (int i = 0; i < Layout::NumRegs; ++i) {
        adj_ret_reg.data[i] += scratch;
    }
    adj_t.grad_add(adj_ret_reg);
}

// Fused element-wise multiply and cross-thread reduce (dot product).
// Returns a single-element tile (same convention as tile_sum / tile_reduce).
// Accesses each tile in its native storage without copying to registers.
template <typename TileA, typename TileB> CUDA_CALLABLE auto tile_dot(TileA& a, TileB& b)
{
    using T = typename TileA::Type;
    using ScalarT = decltype(tensordot(T {}, T {}));
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;

    static_assert(ShapeA::N == ShapeB::N, "Tile shapes must match for tile_dot");
    static_assert(ShapeA::size() == ShapeB::size(), "Tile sizes must match for tile_dot");
    static_assert(ShapeA::size() > 0, "tile_dot requires non-empty tiles");

    auto output = tile_register_t<ScalarT, tile_layout_register_t<tile_shape_t<1>>>();

    // Use the register layout to drive the per-thread iteration.
    using RegLayout = tile_layout_register_t<ShapeA>;

    // Phase 1: per-thread partial dot product — read each tile in native storage
    ScalarT thread_sum = ScalarT(0);
    bool has_data = false;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < RegLayout::NumRegs; ++i) {
        const int linear = RegLayout::linear_from_register(i);
        if (!RegLayout::valid(linear))
            break;

        thread_sum += tensordot(tile_read(a, i, linear), tile_read(b, i, linear));
        has_data = true;
    }

    // Phase 2: cross-thread reduction (same pattern as tile_reduce_impl)
#if defined(__CUDA_ARCH__)
    constexpr int warp_count = (WP_TILE_BLOCK_DIM + WP_TILE_WARP_SIZE - 1) / WP_TILE_WARP_SIZE;
    auto add_op = [](ScalarT x, ScalarT y) { return x + y; };

    ScalarT result {};
    if constexpr (warp_count == 1) {
        wp_tile_lane_mask_bits_t mask = __ballot_sync(WP_TILE_LANE_MASK_ALL, has_data);
        if (has_data)
            result = warp_reduce(thread_sum, add_op, mask);

        int first_active = WP_TILE_LANE_MASK_FFS(mask) - 1;
        if (threadIdx.x == first_active)
            output.data[0] = result;
    } else {
        __shared__ ScalarT partials[warp_count];
        __shared__ int active_warps;

        if (threadIdx.x == 0)
            active_warps = 0;
        WP_TILE_SYNC();

        result = block_combine_thread_results(thread_sum, has_data, add_op, partials, active_warps);

        if (threadIdx.x == 0)
            output.data[0] = result;
    }
#else
    if constexpr (WP_TILE_BLOCK_DIM == 1) {
        output.data[0] = thread_sum;
    } else {
        // Cross-fiber sum of partials. Same pattern as `tile_reduce_impl`'s
        // block_dim>1 path: each fiber drops its partial into shared scratch,
        // syncs, and re-reduces so they all see the same total.
        ScalarT* scratch = (ScalarT*)tile_shared_storage_t::alloc(int(sizeof(ScalarT) * WP_TILE_BLOCK_DIM));
        bool* has_data_arr = (bool*)tile_shared_storage_t::alloc(int(sizeof(bool) * WP_TILE_BLOCK_DIM));
        const int tid = WP_TILE_THREAD_IDX;
        // Zero `has_data_arr` for every slot — partial-block fibers that
        // returned early in the bounds-check thunk would otherwise leave
        // stale `true` flags. See the matching block in `tile_reduce_impl`.
        for (int i = 0; i < WP_TILE_BLOCK_DIM; ++i)
            has_data_arr[i] = false;
        WP_TILE_SYNC();
        scratch[tid] = thread_sum;
        has_data_arr[tid] = has_data;
        WP_TILE_SYNC();

        ScalarT total = ScalarT(0);
        for (int i = 0; i < WP_TILE_BLOCK_DIM; ++i) {
            if (has_data_arr[i])
                total += scratch[i];
        }
        WP_TILE_SYNC();
        tile_shared_storage_t::alloc(-(int)(sizeof(bool) * WP_TILE_BLOCK_DIM));
        tile_shared_storage_t::alloc(-(int)(sizeof(ScalarT) * WP_TILE_BLOCK_DIM));

        output.data[0] = total;
    }
#endif

    return output;
}

// Adjoint for tile_dot: result = sum_i(tensordot(a[i], b[i]))
// adj_a[i] += adj_ret * b[i]
// adj_b[i] += adj_ret * a[i]
// adj_ret is a single-element tile; broadcast its value to all threads
// (same pattern as adj_tile_sum).
template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB, typename AdjRet>
CUDA_CALLABLE void adj_tile_dot(TileA& a, TileB& b, AdjTileA& adj_a, AdjTileB& adj_b, AdjRet& adj_ret)
{
    using ScalarT = decltype(tensordot(typename TileA::Type {}, typename TileA::Type {}));

    auto adj_reg = adj_ret.grad_to_register();

#if defined(__CUDA_ARCH__)
    // broadcast incoming adjoint to block
    __shared__ ScalarT scratch;
    if (WP_TILE_THREAD_IDX == 0)
        scratch = adj_reg.data[0];
    WP_TILE_SYNC();
#else
    // CPU. Same shape as adj_tile_sum: at block_dim==1 the local register
    // slot holds the value; at block_dim>1 only thread 0's register is
    // valid for the size-1 adjoint tile, so broadcast through block-shared.
    ScalarT scratch_local {};
    if constexpr (WP_TILE_BLOCK_DIM == 1) {
        scratch_local = adj_reg.data[0];
    } else {
        wp_block_shared<ScalarT> scratch_holder;
        if (WP_TILE_THREAD_IDX == 0)
            *scratch_holder = adj_reg.data[0];
        WP_TILE_SYNC();
        scratch_local = *scratch_holder;
        WP_TILE_SYNC();
    }
    ScalarT& scratch = scratch_local;
#endif

    auto a_reg = a.copy_to_register();
    auto b_reg = b.copy_to_register();
    auto adj_a_reg = tile_register_like<TileA>();
    auto adj_b_reg = tile_register_like<TileB>();

    using Layout = typename decltype(a_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);
        if (!Layout::valid(linear))
            break;

        adj_a_reg.data[i] += scratch * b_reg.data[i];
        adj_b_reg.data[i] += scratch * a_reg.data[i];
    }

    adj_a.grad_add(adj_a_reg);
    adj_b.grad_add(adj_b_reg);
}

// Adjoint for tile_axpy: dest += alpha * src
// adj_src   += adj_dest * alpha       (per-register, no reduction)
// adj_alpha += tile_dot(adj_dest, src) (cross-thread reduction via tile_dot)
template <typename TileDest, typename TileSrc, typename AdjTileDest, typename AdjTileSrc>
CUDA_CALLABLE void adj_tile_axpy(
    decltype(tensordot(typename TileDest::Type {}, typename TileDest::Type {})) alpha,
    TileSrc& src,
    TileDest& dest,
    decltype(tensordot(typename TileDest::Type {}, typename TileDest::Type {}))& adj_alpha,
    AdjTileSrc& adj_src,
    AdjTileDest& adj_dest
)
{
    auto adj_dest_reg = adj_dest.grad_to_register();
    auto src_reg = src.copy_to_register();
    auto adj_src_reg = tile_register_like<TileSrc>();

    using Layout = typename decltype(adj_dest_reg)::Layout;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Layout::NumRegs; ++i) {
        const int linear = Layout::linear_from_register(i);
        if (!Layout::valid(linear))
            break;

        adj_src_reg.data[i] += adj_dest_reg.data[i] * alpha;
    }

    adj_src.grad_add(adj_src_reg);

    // adj_alpha needs a cross-thread reduction: dot(adj_dest, src).
    // tile_dot returns a 1-element tile; only thread 0 holds the valid
    // value, matching the convention that adj_alpha is a per-thread scalar
    // flowing into adj_tile_extract (which uses atomic_add for shared tiles).
    auto dot_result = tile_dot(adj_dest_reg, src_reg);
    if (WP_TILE_THREAD_IDX == 0)
        adj_alpha += dot_result.data[0];
}

// axis-specific sum
template <int Axis, typename Tile> auto tile_sum(Tile& t)
{
    return tile_reduce_axis_impl<Axis>([](auto x, auto y) { return add(x, y); }, t, typename Tile::Type(0), true);
}

// special case adjoint for axis-specific summation
template <int Axis, typename Tile, typename AdjTile> void adj_tile_sum(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    using InputShape = typename Tile::Layout::Shape;

    if constexpr (InputShape::N == 1) {
        // 1D -> scalar case: broadcast scalar to 1D
        auto broadcasted = tile_broadcast<InputShape::dim(0), 0>(adj_ret);
        tile_add_inplace(adj_t, broadcasted);
    } else if constexpr (InputShape::N == 2) {
        if constexpr (Axis == 0) {
            // broadcast from (D1,) to (D0, D1) with strides (0, 1)
            auto broadcasted = tile_broadcast<InputShape::dim(0), InputShape::dim(1), 0, 1>(adj_ret);
            tile_add_inplace(adj_t, broadcasted);
        } else  // Axis == 1
        {
            // broadcast from (D0,) to (D0, D1) with strides (1, 0)
            auto broadcasted = tile_broadcast<InputShape::dim(0), InputShape::dim(1), 1, 0>(adj_ret);
            tile_add_inplace(adj_t, broadcasted);
        }
    } else if constexpr (InputShape::N == 3) {
        if constexpr (Axis == 0) {
            // broadcast from (D1, D2) to (D0, D1, D2) with strides (0, D2, 1)
            auto broadcasted
                = tile_broadcast<InputShape::dim(0), InputShape::dim(1), InputShape::dim(2), 0, InputShape::dim(2), 1>(
                    adj_ret
                );
            tile_add_inplace(adj_t, broadcasted);
        } else if constexpr (Axis == 1) {
            // broadcast from (D0, D2) to (D0, D1, D2) with strides (D2, 0, 1)
            auto broadcasted
                = tile_broadcast<InputShape::dim(0), InputShape::dim(1), InputShape::dim(2), InputShape::dim(2), 0, 1>(
                    adj_ret
                );
            tile_add_inplace(adj_t, broadcasted);
        } else  // Axis == 2
        {
            // broadcast from (D0, D1) to (D0, D1, D2) with strides (D1, 1, 0)
            auto broadcasted
                = tile_broadcast<InputShape::dim(0), InputShape::dim(1), InputShape::dim(2), InputShape::dim(1), 1, 0>(
                    adj_ret
                );
            tile_add_inplace(adj_t, broadcasted);
        }
    } else if constexpr (InputShape::N == 4) {
        if constexpr (Axis == 0) {
            // broadcast from (D1, D2, D3) to (D0, D1, D2, D3) with strides (0, D2*D3, D3, 1)
            auto broadcasted = tile_broadcast<
                InputShape::dim(0), InputShape::dim(1), InputShape::dim(2), InputShape::dim(3), 0,
                InputShape::dim(2) * InputShape::dim(3), InputShape::dim(3), 1>(adj_ret);
            tile_add_inplace(adj_t, broadcasted);
        } else if constexpr (Axis == 1) {
            // broadcast from (D0, D2, D3) to (D0, D1, D2, D3) with strides (D2*D3, 0, D3, 1)
            auto broadcasted = tile_broadcast<
                InputShape::dim(0), InputShape::dim(1), InputShape::dim(2), InputShape::dim(3),
                InputShape::dim(2) * InputShape::dim(3), 0, InputShape::dim(3), 1>(adj_ret);
            tile_add_inplace(adj_t, broadcasted);
        } else if constexpr (Axis == 2) {
            // broadcast from (D0, D1, D3) to (D0, D1, D2, D3) with strides (D1*D3, D3, 0, 1)
            auto broadcasted = tile_broadcast<
                InputShape::dim(0), InputShape::dim(1), InputShape::dim(2), InputShape::dim(3),
                InputShape::dim(1) * InputShape::dim(3), InputShape::dim(3), 0, 1>(adj_ret);
            tile_add_inplace(adj_t, broadcasted);
        } else  // Axis == 3
        {
            // broadcast from (D0, D1, D2) to (D0, D1, D2, D3) with strides (D1*D2, D2, 1, 0)
            auto broadcasted = tile_broadcast<
                InputShape::dim(0), InputShape::dim(1), InputShape::dim(2), InputShape::dim(3),
                InputShape::dim(1) * InputShape::dim(2), InputShape::dim(2), 1, 0>(adj_ret);
            tile_add_inplace(adj_t, broadcasted);
        }
    }
}

template <typename Tile> auto tile_max(Tile& t) { return tile_reduce(max, t); }

template <typename Tile, typename AdjTile> void adj_tile_max(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // MISSINGADJOINT: subgradient: route adj_ret to the index of the maximum element
}

template <typename Tile> auto tile_min(Tile& t) { return tile_reduce(min, t); }

template <typename Tile, typename AdjTile> void adj_tile_min(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // MISSINGADJOINT: subgradient: route adj_ret to the index of the minimum element
}


template <typename Tile> auto tile_argmax(Tile& t) { return tile_arg_reduce(max, argmax_tracker, t); }

template <typename Tile> auto tile_argmin(Tile& t) { return tile_arg_reduce(min, argmin_tracker, t); }


}  // namespace wp


#ifdef __clang__
#pragma clang diagnostic pop
#endif
