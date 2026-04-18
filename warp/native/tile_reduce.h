// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tile.h"

#ifdef __clang__
// disable warnings related to C++17 extensions on CPU JIT builds
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif  // __clang__

#define WP_TILE_WARP_SIZE 32

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

template <typename T> inline CUDA_CALLABLE T warp_shuffle_down(T val, int offset, int mask)
{
    typedef unsigned int Word;

    union {
        T output;
        Word output_storage;
    };

    union {
        T input;
        Word input_storage;
    };

    input = val;

    Word* dest = reinterpret_cast<Word*>(&output);
    Word* src = reinterpret_cast<Word*>(&input);

    unsigned int shuffle_word;

    constexpr int word_count = (sizeof(T) + sizeof(Word) - 1) / sizeof(Word);

    WP_PRAGMA_UNROLL
    for (int i = 0; i < word_count; ++i) {
        shuffle_word = __shfl_down_sync(mask, src[i], offset, WP_TILE_WARP_SIZE);
        dest[i] = shuffle_word;
    }

    return output;
}

// vector overload
template <unsigned Length, typename T>
inline CUDA_CALLABLE wp::vec_t<Length, T> warp_shuffle_down(wp::vec_t<Length, T> val, int offset, int mask)
{
    wp::vec_t<Length, T> result;

    for (unsigned i = 0; i < Length; ++i)
        result[i] = __shfl_down_sync(mask, val[i], offset, WP_TILE_WARP_SIZE);

    return result;
}

// matrix overload
template <unsigned Rows, unsigned Cols, typename T>
inline CUDA_CALLABLE wp::mat_t<Rows, Cols, T> warp_shuffle_down(wp::mat_t<Rows, Cols, T> val, int offset, int mask)
{
    wp::mat_t<Rows, Cols, T> result;

    for (unsigned i = 0; i < Rows; ++i)
        for (unsigned j = 0; j < Cols; ++j)
            result.data[i][j] = __shfl_down_sync(mask, val.data[i][j], offset, WP_TILE_WARP_SIZE);

    return result;
}


template <typename T, typename Op> inline CUDA_CALLABLE T warp_reduce(T val, Op f, unsigned int mask)
{
    T sum = val;

    if (mask == 0xFFFFFFFF) {
        // handle case where entire warp is active
        for (int offset = WP_TILE_WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum = f(sum, warp_shuffle_down(sum, offset, mask));
        }
    } else {
        // handle partial warp case - works for contiguous masks
        for (int offset = WP_TILE_WARP_SIZE / 2; offset > 0; offset /= 2) {
            T shfl_val = warp_shuffle_down(sum, offset, mask);
            if ((mask & (1 << ((threadIdx.x + offset) % WP_TILE_WARP_SIZE))) != 0)
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
inline CUDA_CALLABLE ValueAndIndex<T> warp_reduce_tracked(T val, int idx, Op f, OpTrack track, unsigned int mask)
{
    T sum = val;
    int index = idx;

    if (mask == 0xFFFFFFFF) {
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
            if ((mask & (1 << ((threadIdx.x + offset) % WP_TILE_WARP_SIZE))) != 0) {
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
    unsigned int mask = __ballot_sync(0xFFFFFFFF, thread_has_data);
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
        unsigned int mask = __ballot_sync(0xFFFFFFFF, thread_has_data);
        if (thread_has_data)
            block_sum = warp_reduce(thread_sum, f, mask);

        // write from first active lane (warp_reduce result is only valid there)
        int first_active = __ffs(mask) - 1;
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

template <int Axis, typename Op, typename Tile> CUDA_CALLABLE_DEVICE auto tile_reduce_axis_impl(Op f, Tile& t)
{
    using T = typename Tile::Type;
    using InputShape = typename Tile::Layout::Shape;
    using OutputShape = typename tile_shape_remove_dim<Axis, InputShape>::type;

    constexpr int reduce_dim_size = InputShape::dim(Axis);
    constexpr int output_size = OutputShape::size();

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
                unsigned int mask = __ballot_sync(0xFFFFFFFF, valid);
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
                unsigned int mask = __ballot_sync(0xFFFFFFFF, thread_has_data);
                if (thread_has_data)
                    block_sum = warp_reduce(thread_sum, f, mask);

                // write from first active lane (warp_reduce result is only valid there)
                int first_active = __ffs(mask) - 1;
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
        if (!OutputRegLayout::valid(linear))
            break;

        output.data[i] = output_buffer[linear];
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
    unsigned int mask = __ballot_sync(0xFFFFFFFF, thread_has_data);
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
}

template <int Axis, typename Op, typename Tile> auto tile_reduce_axis_impl(Op f, Tile& t)
{
    using T = typename Tile::Type;
    using InputShape = typename Tile::Layout::Shape;
    using OutputShape = typename tile_shape_remove_dim<Axis, InputShape>::type;

    constexpr int reduce_dim_size = InputShape::dim(Axis);

    // CPU version - work directly with register tiles, no thread coordination needed
    auto input = t.copy_to_register();
    auto output = tile_register_t<T, tile_layout_register_t<OutputShape>>();
    using OutputLayout = typename decltype(output)::Layout;

    // iterate through each output element and reduce along the axis
    constexpr int output_size = OutputShape::size();
    for (int out_idx = 0; out_idx < output_size; ++out_idx) {
        T accumulator;

        // special case for 1D input (reduces to single value)
        if constexpr (InputShape::N == 1) {
            accumulator = input.data[0];
            for (int i = 1; i < reduce_dim_size; ++i) {
                // input is in registers, linear access
                accumulator = f(accumulator, input.data[i]);
            }
        } else {
            // multi-dimensional case
            auto out_coord = OutputLayout::coord_from_linear(out_idx);

            // get input coordinates by inserting axis values
            auto coord_0 = tile_coord_insert_axis<Axis>(out_coord, 0);
            int input_linear_0 = tile_layout_register_t<InputShape>::linear_from_coord(coord_0);
            int input_reg_0 = tile_layout_register_t<InputShape>::register_from_linear(input_linear_0);
            accumulator = input.data[input_reg_0];

            // reduce across the axis
            for (int i = 1; i < reduce_dim_size; ++i) {
                auto coord_i = tile_coord_insert_axis<Axis>(out_coord, i);
                int input_linear_i = tile_layout_register_t<InputShape>::linear_from_coord(coord_i);
                int input_reg_i = tile_layout_register_t<InputShape>::register_from_linear(input_linear_i);
                accumulator = f(accumulator, input.data[input_reg_i]);
            }
        }

        // store to output register
        int output_reg = OutputLayout::register_from_linear(out_idx);
        output.data[output_reg] = accumulator;
    }

    return output;
}

template <typename Tile, typename Op, typename OpTrack> auto tile_arg_reduce_impl(Op f, OpTrack track, Tile& t)
{
    using T = typename Tile::Type;

    auto input = t.copy_to_register();
    auto output = tile_register_t<int, tile_layout_register_t<tile_shape_t<1>>>();

    using Layout = typename decltype(input)::Layout;

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
}

#endif  // !defined(__CUDA_ARCH__)

inline void adj_tile_reduce_impl()
{
    // todo: general purpose reduction gradients not implemented
}

inline void adj_tile_reduce_axis_impl()
{
    // todo: axis-specific reduction gradients not implemented
}

// entry point for Python code-gen, wraps op in a lambda to perform overload resolution
#define tile_reduce(op, t) tile_reduce_impl([](auto x, auto y) { return op(x, y);}, t)
#define adj_tile_reduce(op, t, adj_op, adj_t, adj_ret) adj_tile_reduce_impl()

#define tile_arg_reduce(op, opTrack, t) tile_arg_reduce_impl([](auto x, auto y) { return op(x, y);}, [](auto a, auto b, auto c, auto d) { return opTrack(a, b, c, d); }, t)
#define adj_tile_arg_reduce(op, t, adj_op, adj_t, adj_ret) adj_tile_arg_reduce_impl()

// axis-specific reduction entry points
#define tile_reduce_axis(op, t, axis) tile_reduce_axis_impl<axis>([](auto x, auto y) { return op(x, y);}, t)
#define adj_tile_reduce_axis(op, t, axis, adj_op, adj_t, adj_axis, adj_ret) adj_tile_reduce_axis_impl()

// convenience methods for specific reductions

// whole-tile sum
template <typename Tile> auto tile_sum(Tile& t) { return tile_reduce(add, t); }

// special case adjoint for summation
template <typename Tile, typename AdjTile> CUDA_CALLABLE void adj_tile_sum(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    using T = typename Tile::Type;

    auto adj_reg = adj_ret.grad_to_register();

#if !defined(__CUDA_ARCH__)
    T scratch = adj_reg.data[0];
#else
    // broadcast incoming adjoint to block
    __shared__ T scratch;
    if (WP_TILE_THREAD_IDX == 0)
        scratch = adj_reg.data[0];

    WP_TILE_SYNC();
#endif

    auto adj_ret_reg = tile_register_like<Tile>();
    using Layout = typename decltype(adj_ret_reg)::Layout;
    for (int i = 0; i < Layout::NumRegs; ++i) {
        adj_ret_reg.data[i] += scratch;
    }
    adj_t.grad_add(adj_ret_reg);
}

// Fused element-wise multiply and cross-thread reduce (dot product).
// Returns a scalar broadcast to all threads — no intermediate tile created.
// Accesses each tile in its native storage without copying to registers.
template <typename TileA, typename TileB>
CUDA_CALLABLE auto tile_dot(TileA& a, TileB& b) -> decltype(tensordot(typename TileA::Type {}, typename TileA::Type {}))
{
    using T = typename TileA::Type;
    using ScalarT = decltype(tensordot(T {}, T {}));
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;

    static_assert(ShapeA::N == ShapeB::N, "Tile shapes must match for tile_dot");
    static_assert(ShapeA::size() == ShapeB::size(), "Tile sizes must match for tile_dot");
    static_assert(ShapeA::size() > 0, "tile_dot requires non-empty tiles");

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

    // Phase 2: cross-thread reduction + broadcast
#if defined(__CUDA_ARCH__)
    constexpr int warp_count = (WP_TILE_BLOCK_DIM + WP_TILE_WARP_SIZE - 1) / WP_TILE_WARP_SIZE;
    auto add_op = [](ScalarT x, ScalarT y) { return x + y; };

    ScalarT result {};
    if constexpr (warp_count == 1) {
        unsigned int mask = __ballot_sync(0xFFFFFFFF, has_data);
        if (has_data)
            result = warp_reduce(thread_sum, add_op, mask);

        // broadcast via shared memory so all threads get the result
        __shared__ ScalarT scratch;
        int first_active = __ffs(mask) - 1;
        if (threadIdx.x == first_active)
            scratch = result;
        WP_TILE_SYNC();
        return scratch;
    } else {
        __shared__ ScalarT partials[warp_count];
        __shared__ int active_warps;

        if (threadIdx.x == 0)
            active_warps = 0;
        WP_TILE_SYNC();

        result = block_combine_thread_results(thread_sum, has_data, add_op, partials, active_warps);

        // broadcast via shared memory so all threads get the result
        __shared__ ScalarT scratch;
        if (threadIdx.x == 0)
            scratch = result;
        WP_TILE_SYNC();
        return scratch;
    }
#else
    return thread_sum;
#endif
}

// Adjoint for tile_dot: result = sum_i(tensordot(a[i], b[i]))
// adj_a[i] += adj_ret * b[i]
// adj_b[i] += adj_ret * a[i]
// adj_ret is the scalar type (e.g., float for tile<vec3f>), already broadcast to all threads.
template <typename TileA, typename TileB, typename AdjTileA, typename AdjTileB>
CUDA_CALLABLE void adj_tile_dot(
    TileA& a,
    TileB& b,
    AdjTileA& adj_a,
    AdjTileB& adj_b,
    decltype(tensordot(typename TileA::Type {}, typename TileA::Type {})) adj_ret
)
{
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

        adj_a_reg.data[i] += adj_ret * b_reg.data[i];
        adj_b_reg.data[i] += adj_ret * a_reg.data[i];
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
    // Only one thread must accumulate the result because adj_alpha is a
    // per-thread scalar that may flow into adj_tile_extract, which uses
    // atomic_add for shared tiles — if all threads added the full dot
    // product, it would be counted BLOCK_DIM times.
    auto dot_result = tile_dot(adj_dest_reg, src_reg);
    if (WP_TILE_THREAD_IDX == 0)
        adj_alpha += dot_result;
}

// axis-specific sum
template <int Axis, typename Tile> auto tile_sum(Tile& t)
{
    return tile_reduce_axis_impl<Axis>([](auto x, auto y) { return add(x, y); }, t);
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
    // todo: not implemented
}

template <typename Tile> auto tile_min(Tile& t) { return tile_reduce(min, t); }

template <typename Tile, typename AdjTile> void adj_tile_min(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // todo: not implemented
}


template <typename Tile> auto tile_argmax(Tile& t) { return tile_arg_reduce(max, argmax_tracker, t); }

template <typename Tile, typename AdjTile> void adj_tile_argmax(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // todo: not implemented
}

template <typename Tile> auto tile_argmin(Tile& t) { return tile_arg_reduce(min, argmin_tracker, t); }

template <typename Tile, typename AdjTile> void adj_tile_argmin(Tile& t, Tile& adj_t, AdjTile& adj_ret)
{
    // todo: not implemented
}


}  // namespace wp


#ifdef __clang__
#pragma clang diagnostic pop
#endif
