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

#include "mesh.h"
#include "tile.h"
#include "tile_bvh.h"

namespace wp {

#if defined(__CUDA_ARCH__)

struct mesh_query_aabb_thread_block_t {
    CUDA_CALLABLE mesh_query_aabb_thread_block_t()
        : mesh()
        , stack_shared_mem(nullptr)
        , count_shared_mem(nullptr)
        , result_counter_shared_mem(nullptr)
        , result_buffer_shared_mem(nullptr)
        , input_lower()
        , input_upper()
    {
    }

    // Required for adjoint computations.
    CUDA_CALLABLE inline mesh_query_aabb_thread_block_t& operator+=(const mesh_query_aabb_thread_block_t& other)
    {
        return *this;
    }

    Mesh mesh;

    // BVH traversal stack (shared memory pointers):
    int* stack_shared_mem;  // [block_size] - buffer to store node indices
    int* count_shared_mem;  // [1] - counter for number of nodes on the stack
    int* result_counter_shared_mem;  // [1] - counter for number of results found
    int* result_buffer_shared_mem;  // [block_size] - buffer to store result indices
    static const int result_buffer_capacity = WP_TILE_BLOCK_DIM * 5;
    static const int stack_capacity = 64 * BVH_QUERY_STACK_SIZE;

    // inputs
    wp::vec3 input_lower;
    wp::vec3 input_upper;
};

#else

// CPU version: alias mesh_query_aabb_thread_block_t to mesh_query_aabb_t since we don't need thread block functionality
using mesh_query_aabb_thread_block_t = mesh_query_aabb_t;

#endif


#if defined(__CUDA_ARCH__)

CUDA_CALLABLE inline mesh_query_aabb_thread_block_t
mesh_query_aabb_thread_block(uint64_t id, const vec3& lower, const vec3& upper)
{
    // This routine traverses the BVH tree until it finds
    // the first overlapping bound.

    // initialize empty
    mesh_query_aabb_thread_block_t query;

    Mesh mesh = mesh_get(id);

    query.mesh = mesh;

    // Shared memory should remain available even if this method terminates - it stays available until the end of the
    // kernel
    __shared__ int stack_shared_mem[mesh_query_aabb_thread_block_t::stack_capacity];
    __shared__ int count_shared_mem[1];
    __shared__ int result_counter[1];
    __shared__ int result_buffer[mesh_query_aabb_thread_block_t::
                                     result_buffer_capacity];  // Conservative bounds for up to WP_TILE_BLOCK_DIM
                                                               // threads: Max capacity should be (BVH_LEAF_SIZE+1)*512

    query.stack_shared_mem = stack_shared_mem;
    query.count_shared_mem = count_shared_mem;
    query.result_counter_shared_mem = result_counter;
    query.result_buffer_shared_mem = result_buffer;

    // optimization: make the latest
    if (threadIdx.x == 0) {
        query.stack_shared_mem[0] = *mesh.bvh.root;
        query.count_shared_mem[0] = 1;
        query.result_counter_shared_mem[0] = 0;
    }
    __syncthreads();

    query.input_lower = lower;
    query.input_upper = upper;

    return query;
}

CUDA_CALLABLE inline mesh_query_aabb_thread_block_t
mesh_query_aabb_thread_block_impl(uint64_t id, const vec3& lower, const vec3& upper)
{
    return mesh_query_aabb_thread_block(id, lower, upper);
}

CUDA_CALLABLE inline int mesh_get_node_index_at_depth(
    mesh_query_aabb_thread_block_t& query, int node_index, int lane_id, int num_expansion_steps
)
{
    if (num_expansion_steps == 0) {
        return node_index;
    }

    BVH bvh = query.mesh.bvh;

    BVHPackedNodeHalf* node_lowers_uppers[2];
    node_lowers_uppers[0] = bvh.node_lowers;
    node_lowers_uppers[1] = bvh.node_uppers;

    int max_lanes = 1 << num_expansion_steps;
    if (lane_id >= max_lanes) {
        return -1;
    }

    bool is_leaf = bvh_load_node(bvh.node_lowers, node_index).b;
    if (is_leaf) {
        if (lane_id == 0)
            return node_index;
        else
            return -1;
    }

    for (int i = 0; i < num_expansion_steps; ++i) {
        int bit_position = num_expansion_steps - 1 - i;
        int lower_upper_select = (lane_id >> bit_position) & 1;

        node_index = bvh_load_node(node_lowers_uppers[lower_upper_select], node_index).i;

        is_leaf = bvh_load_node(bvh.node_lowers, node_index).b;
        if (is_leaf) {
            // Check if this thread is the canonical one for this leaf.
            // Multiple threads with the same prefix but different suffixes
            // reached this leaf via the same path. Only the thread with
            // all remaining lower bits zero should return this leaf.
            int mask = (1 << bit_position) - 1;
            if ((lane_id & mask) == 0) {
                return node_index;
            } else {
                return -1;
            }
        }
    }
    return node_index;
}

// block_size should be a power of 2. 2^num_expansion_steps should be less than or equal to block_size.
CUDA_CALLABLE inline bool mesh_query_aabb_next_thread_block_impl(mesh_query_aabb_thread_block_t& query, int& index)
{
    int block_size = blockDim.x;
    int lane_id = threadIdx.x;

    __syncthreads();  // Required when this method is called inside a loop because of the shared memory read below

    // Because of BVH_LEAF_SIZE>1, it is possible that already enough results are cached to immediately return
    if (query.result_counter_shared_mem[0] >= block_size) {
        index = query.result_buffer_shared_mem[query.result_counter_shared_mem[0] - block_size + lane_id];
        __syncthreads();
        if (lane_id == 0)
            query.result_counter_shared_mem[0] = max(0, query.result_counter_shared_mem[0] - block_size);
        return true;
    }

    int num_expansion_steps = 0;
    int pow_2 = 1;
    while (pow_2 < block_size) {
        pow_2 *= 2;
        num_expansion_steps += 1;
    }

    Mesh mesh = query.mesh;
    BVH bvh = mesh.bvh;
    int* stack = query.stack_shared_mem;
    int* count = query.count_shared_mem;
    __shared__ bool direct_mode_shared[1];

    index = -1;

    // Navigate through the bvh, find the first overlapping leaf node.
    while (count[0] > 0 && query.result_counter_shared_mem[0] < block_size) {
        __syncthreads();
        if (lane_id == 0) {
            if (count[0] < block_size) {
                count[0] -= 1;
                direct_mode_shared[0] = false;
            } else {
                count[0] -= block_size;
                direct_mode_shared[0] = true;
            }
        }

        __syncthreads();
        int stack_count = count[0];

        int node_index;
        if (direct_mode_shared[0])
            node_index = stack[stack_count + lane_id];
        else {
            __syncthreads();
            node_index = mesh_get_node_index_at_depth(query, stack[stack_count], lane_id, num_expansion_steps);
            if (node_index < 0) {
                // Check if there is something on the stack that could be taken instead
                int id = atomicAdd(&count[0], -1) - 1;
                if (id >= 0) {
                    node_index = stack[id];
                } else {
                    // Restore the stack underflow
                    atomicAdd(&count[0], 1);
                }
            }
        }

        __syncthreads();
        if (node_index >= 0) {
            BVHPackedNodeHalf node_lower = bvh_load_node(bvh.node_lowers, node_index);
            BVHPackedNodeHalf node_upper = bvh_load_node(bvh.node_uppers, node_index);

            if (intersect_aabb_aabb(
                    query.input_lower, query.input_upper, reinterpret_cast<vec3&>(node_lower),
                    reinterpret_cast<vec3&>(node_upper)
                )) {
                const int left_index = node_lower.i;
                const int right_index = node_upper.i;

                if (node_lower.b) {
                    // found leaf, process its primitives one at a time
                    const int start = left_index;
                    const int end = right_index;

                    if (end - start == 1) {
                        // Optimization: when a leaf contains exactly one primitive, the node bounds
                        // are identical to the primitive bounds, so we can skip the per-primitive
                        // intersection test and directly add the result
                        int primitive_index = bvh.primitive_indices[start];
                        int pos = atomicAdd(&query.result_counter_shared_mem[0], 1);
                        if (pos < mesh_query_aabb_thread_block_t::result_buffer_capacity)
                            query.result_buffer_shared_mem[pos] = primitive_index;
                    } else {
                        for (int prim_offset = 0; prim_offset < (end - start); ++prim_offset) {
                            int primitive_index = bvh.primitive_indices[start + prim_offset];

                            if (intersect_aabb_aabb(
                                    query.input_lower, query.input_upper, mesh.lowers[primitive_index],
                                    mesh.uppers[primitive_index]
                                )) {
                                int pos = atomicAdd(&query.result_counter_shared_mem[0], 1);
                                if (pos < mesh_query_aabb_thread_block_t::result_buffer_capacity)
                                    query.result_buffer_shared_mem[pos] = primitive_index;
                            }
                        }
                    }
                } else {
                    // Internal node: push children onto stack
                    int pos = atomicAdd(&count[0], 2);
                    if (pos + 1 < mesh_query_aabb_thread_block_t::stack_capacity) {
                        stack[pos] = left_index;
                        stack[pos + 1] = right_index;
                    }
                }
            }
        }
        __syncthreads();
    }

    __syncthreads();
    if (query.result_counter_shared_mem[0] >= block_size)
        index = query.result_buffer_shared_mem[query.result_counter_shared_mem[0] - block_size + lane_id];
    else
        index = lane_id < query.result_counter_shared_mem[0] ? query.result_buffer_shared_mem[lane_id] : -1;
    bool result = query.result_counter_shared_mem[0] > 0;
    __syncthreads();

    if (lane_id == 0)
        query.result_counter_shared_mem[0] = max(0, query.result_counter_shared_mem[0] - block_size);

    __syncthreads();

    return result;
}

#else

// CPU version: mesh_query_aabb_thread_block_t is aliased to mesh_query_aabb_t, so just use regular mesh query
CUDA_CALLABLE inline mesh_query_aabb_thread_block_t
mesh_query_aabb_thread_block_impl(uint64_t id, const vec3& lower, const vec3& upper)
{
    // On CPU, mesh_query_aabb_thread_block_t is just mesh_query_aabb_t
    return mesh_query_aabb(id, lower, upper);
}

// CPU version: single-threaded, just calls regular mesh_query_aabb_next
CUDA_CALLABLE inline bool mesh_query_aabb_next_thread_block_impl(mesh_query_aabb_thread_block_t& query, int& index)
{
    // On CPU, mesh_query_aabb_thread_block_t is just mesh_query_aabb_t, so call regular mesh_query_aabb_next
    return mesh_query_aabb_next(query, index);
}

#endif


// Tile-based interface
#if defined(__CUDA_ARCH__)

// CUDA implementation: uses thread-block parallel traversal
template <int Length> CUDA_CALLABLE inline auto tile_mesh_query_aabb_next_impl(mesh_query_aabb_thread_block_t& query)
{
    int index = -1;
    mesh_query_aabb_next_thread_block_impl(query, index);
    // Create a register tile from the per-thread index
    return tile<int>(index);
}

// Wrapper that works for any block size
CUDA_CALLABLE inline auto tile_mesh_query_aabb_next(mesh_query_aabb_thread_block_t& query)
{
    return tile_mesh_query_aabb_next_impl<WP_TILE_BLOCK_DIM>(query);
}

// New tile-based alias for the query function
CUDA_CALLABLE inline mesh_query_aabb_thread_block_t
tile_mesh_query_aabb(uint64_t id, const vec3& lower, const vec3& upper)
{
    return mesh_query_aabb_thread_block_impl(id, lower, upper);
}

// Stub
CUDA_CALLABLE inline void adj_tile_mesh_query_aabb(
    uint64_t id, const vec3& lower, const vec3& upper, uint64_t, vec3&, vec3&, mesh_query_aabb_thread_block_t&
)
{
}

// stub
template <int Length>
CUDA_CALLABLE inline void adj_tile_mesh_query_aabb_next_impl(
    mesh_query_aabb_thread_block_t& query, mesh_query_aabb_thread_block_t&, decltype(tile<int>(0))&
)
{
}

// stub for the wrapper
CUDA_CALLABLE inline void adj_tile_mesh_query_aabb_next(
    mesh_query_aabb_thread_block_t& query, mesh_query_aabb_thread_block_t&, decltype(tile<int>(0))&
)
{
}

#else

// CPU implementation: falls back to single-threaded query, returns index only in first element
template <int Length> inline auto tile_mesh_query_aabb_next_impl(mesh_query_aabb_thread_block_t& query)
{
    // On CPU, mesh_query_aabb_thread_block_t is aliased to mesh_query_aabb_t
    // We just call the regular query and put the result in the first element of a tile
    int index = -1;
    mesh_query_aabb_next(query, index);

    // Create a tile with the index in the first element, -1 in all others
    // This simulates a single-threaded execution where only thread 0 has work
    auto result = tile_register<int, Length>();
    using ResultLayout = typename decltype(result)::Layout;
    for (int i = 0; i < ResultLayout::NumRegs; ++i) {
        result.data[i] = (i == 0) ? index : -1;
    }
    return result;
}

// Wrapper - on CPU this needs an explicit block_dim parameter since WP_TILE_BLOCK_DIM is not defined
// However, for consistency we'll use a default value
inline auto tile_mesh_query_aabb_next(mesh_query_aabb_thread_block_t& query)
{
    // On CPU, just return a single element tile with the query result
    // Using Length=1 since we don't have block_dim available
    return tile_mesh_query_aabb_next_impl<1>(query);
}

// CPU version: tile_mesh_query_aabb just creates a regular query
inline mesh_query_aabb_thread_block_t tile_mesh_query_aabb(uint64_t id, const vec3& lower, const vec3& upper)
{
    // On CPU, this is just mesh_query_aabb since mesh_query_aabb_thread_block_t = mesh_query_aabb_t
    return mesh_query_aabb(id, lower, upper);
}

// Stub
inline void adj_tile_mesh_query_aabb(
    uint64_t id, const vec3& lower, const vec3& upper, uint64_t, vec3&, vec3&, mesh_query_aabb_thread_block_t&
)
{
}

// stub
template <int Length>
inline void adj_tile_mesh_query_aabb_next_impl(
    mesh_query_aabb_thread_block_t& query, mesh_query_aabb_thread_block_t&, decltype(tile_register<int, Length>())&
)
{
}

// stub for the wrapper
inline void adj_tile_mesh_query_aabb_next(
    mesh_query_aabb_thread_block_t& query, mesh_query_aabb_thread_block_t&, decltype(tile_register<int, 1>())&
)
{
}

#endif  // __CUDA_ARCH__

}  // namespace wp
