// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Host-side cuBQL integration: build a cuBQL tree on the CPU, then convert it
// into Warp's native BVH layout. The matching device-side code lives in
// bvh_cubql.cu.

#include "warp.h"

#include "bvh.h"
#include "error.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <vector>

#ifndef WP_DISABLE_CUBQL
#include "cuBQL/builder/cpu.h"
#endif

using namespace wp;

namespace wp {

#ifndef WP_DISABLE_CUBQL

// Node layout compatible with cuBQL::BinaryBVH<float, 3>::Node. cuBQL nodes are
// only used as temporary builder output before copying into Warp's native BVH
// layout.
struct alignas(16) CubqlNode {
    vec3 lower;
    vec3 upper;
    uint64_t admin = 0;
};

static inline cuBQL::box3f make_cubql_box(const vec3& lower, const vec3& upper)
{
    return cuBQL::box3f(cuBQL::vec3f(lower[0], lower[1], lower[2]), cuBQL::vec3f(upper[0], upper[1], upper[2]));
}

static void cubql_update_host_boxes(const BVH& bvh, cuBQL::box3f* boxes)
{
    for (int i = 0; i < bvh.num_items; ++i) {
        boxes[i] = make_cubql_box(bvh.item_lowers[i], bvh.item_uppers[i]);
    }
}

static uint16_t cubql_host_leaf_count(uint64_t admin) { return uint16_t(admin >> 48); }

static uint64_t cubql_host_admin_offset(uint64_t admin) { return admin & 0x0000FFFFFFFFFFFFull; }

struct CubqlNativeBuild {
    std::vector<BVHPackedNodeHalf> node_lowers;
    std::vector<BVHPackedNodeHalf> node_uppers;
    std::vector<int> node_parents;
    std::vector<int> primitive_indices;
    int num_leaf_nodes = 0;
    int max_depth = 0;
};

static bool cubql_decode_node(
    const CubqlNode* nodes,
    uint32_t num_nodes,
    uint32_t num_prims,
    uint32_t node_index,
    bool& is_leaf,
    uint64_t& offset,
    uint64_t& upper_offset
)
{
    if (node_index >= num_nodes) {
        wp::set_error_string("Warp error: cuBQL BVH contains an invalid node index");
        return false;
    }

    const CubqlNode& node = nodes[node_index];
    const uint16_t leaf_count = cubql_host_leaf_count(node.admin);
    offset = cubql_host_admin_offset(node.admin);
    is_leaf = leaf_count != 0;
    upper_offset = is_leaf ? offset + leaf_count : offset + 1;

    if (offset > uint64_t(INT_MAX) || upper_offset > uint64_t(INT_MAX)) {
        wp::set_error_string("Warp error: cuBQL BVH node offset is too large to convert to the native BVH layout");
        return false;
    }

    if (is_leaf && upper_offset > num_prims) {
        wp::set_error_string("Warp error: cuBQL BVH contains an invalid primitive range");
        return false;
    }

    if (!is_leaf && offset + 1 >= num_nodes) {
        wp::set_error_string("Warp error: cuBQL BVH contains an invalid child node offset");
        return false;
    }

    return true;
}

static bool cubql_append_subtree_primitives(
    const CubqlNode* nodes,
    uint32_t num_nodes,
    const uint32_t* prim_ids,
    uint32_t num_prims,
    uint32_t node_index,
    std::vector<int>& primitive_indices
)
{
    std::vector<uint32_t> stack;
    stack.push_back(node_index);

    while (!stack.empty()) {
        const uint32_t current_node = stack.back();
        stack.pop_back();

        bool is_leaf;
        uint64_t offset;
        uint64_t upper_offset;
        if (!cubql_decode_node(nodes, num_nodes, num_prims, current_node, is_leaf, offset, upper_offset)) {
            return false;
        }

        if (is_leaf) {
            for (uint64_t i = offset; i < upper_offset; ++i) {
                if (prim_ids[i] > uint32_t(INT_MAX)) {
                    wp::set_error_string("Warp error: cuBQL BVH primitive index is too large to convert");
                    return false;
                }
                primitive_indices.push_back(int(prim_ids[i]));
            }
        } else {
            stack.push_back(uint32_t(offset + 1));
            stack.push_back(uint32_t(offset));
        }
    }

    return true;
}

// Walk the cuBQL tree once to determine whether its depth exceeds what the
// native traversal stack can handle. If it fits, we can copy the tree verbatim
// and preserve cuBQL's natural memory layout (root at 0, child pairs contiguous
// at offset/offset+1) which is much friendlier to cache than the reordered
// layout used by the warp builders.
static bool
cubql_tree_exceeds_native_stack(const CubqlNode* nodes, uint32_t num_nodes, uint32_t num_prims, bool& exceeds_stack)
{
    struct StackEntry {
        uint32_t node_index;
        int depth;
    };

    exceeds_stack = false;

    std::vector<StackEntry> stack;
    stack.push_back({ 0, 0 });

    while (!stack.empty()) {
        const StackEntry entry = stack.back();
        stack.pop_back();

        bool is_leaf;
        uint64_t offset;
        uint64_t upper_offset;
        if (!cubql_decode_node(nodes, num_nodes, num_prims, entry.node_index, is_leaf, offset, upper_offset)) {
            return false;
        }

        if (!is_leaf) {
            if (entry.depth >= BVH_QUERY_STACK_SIZE - 1) {
                exceeds_stack = true;
                return true;
            }
            stack.push_back({ uint32_t(offset + 1), entry.depth + 1 });
            stack.push_back({ uint32_t(offset), entry.depth + 1 });
        }
    }

    return true;
}

// Fast-path copy that preserves cuBQL's native node layout. Used when the tree
// fits within BVH_QUERY_STACK_SIZE so we don't need to flatten deep subtrees.
static bool cubql_copy_native_order_bvh(
    BVH& bvh, const CubqlNode* nodes, const uint32_t* prim_ids, uint32_t num_nodes, uint32_t num_prims
)
{
    bvh.num_nodes = int(num_nodes);
    bvh.num_leaf_nodes = 0;
    bvh.max_depth = 0;
    bvh.max_nodes = int(num_nodes);

    bvh.node_lowers
        = static_cast<BVHPackedNodeHalf*>(wp_alloc_host(sizeof(BVHPackedNodeHalf) * num_nodes, "(native:bvh)"));
    bvh.node_uppers
        = static_cast<BVHPackedNodeHalf*>(wp_alloc_host(sizeof(BVHPackedNodeHalf) * num_nodes, "(native:bvh)"));
    bvh.node_parents = static_cast<int*>(wp_alloc_host(sizeof(int) * num_nodes, "(native:bvh)"));
    bvh.node_counts = nullptr;
    bvh.primitive_indices = static_cast<int*>(wp_alloc_host(sizeof(int) * num_prims, "(native:bvh)"));
    bvh.root = static_cast<int*>(wp_alloc_host(sizeof(int), "(native:bvh)"));

    if (!bvh.node_lowers || !bvh.node_uppers || !bvh.node_parents || !bvh.primitive_indices || !bvh.root) {
        wp::set_error_string("Warp error: failed to allocate native BVH storage for cuBQL conversion");
        bvh_destroy_host(bvh);
        return false;
    }

    std::fill(bvh.node_parents, bvh.node_parents + num_nodes, -1);
    bvh.root[0] = 0;

    for (uint32_t i = 0; i < num_prims; ++i) {
        if (prim_ids[i] > uint32_t(INT_MAX)) {
            wp::set_error_string("Warp error: cuBQL BVH primitive index is too large to convert");
            bvh_destroy_host(bvh);
            return false;
        }
        bvh.primitive_indices[i] = int(prim_ids[i]);
    }

    for (uint32_t i = 0; i < num_nodes; ++i) {
        bool is_leaf;
        uint64_t offset;
        uint64_t upper_offset;
        if (!cubql_decode_node(nodes, num_nodes, num_prims, i, is_leaf, offset, upper_offset)) {
            bvh_destroy_host(bvh);
            return false;
        }

        bvh.node_lowers[i] = make_node(nodes[i].lower, int(offset), is_leaf);
        bvh.node_uppers[i] = make_node(nodes[i].upper, int(upper_offset), false);

        if (is_leaf) {
            bvh.num_leaf_nodes++;
        } else {
            bvh.node_parents[offset] = int(i);
            bvh.node_parents[offset + 1] = int(i);
        }
    }

    // record max_depth for diagnostics; cheap since the tree fits the native stack
    std::vector<uint32_t> depth_stack;
    std::vector<int> depths;
    depth_stack.push_back(0);
    depths.push_back(0);

    while (!depth_stack.empty()) {
        const uint32_t node_index = depth_stack.back();
        const int depth = depths.back();
        depth_stack.pop_back();
        depths.pop_back();

        bvh.max_depth = std_max(bvh.max_depth, depth);

        if (!bvh.node_lowers[node_index].b) {
            depth_stack.push_back(uint32_t(bvh.node_lowers[node_index].i));
            depths.push_back(depth + 1);
            depth_stack.push_back(uint32_t(bvh.node_uppers[node_index].i));
            depths.push_back(depth + 1);
        }
    }

    return true;
}

static bool cubql_build_native_node(
    const CubqlNode* nodes,
    uint32_t num_nodes,
    const uint32_t* prim_ids,
    uint32_t num_prims,
    uint32_t cubql_node_index,
    int depth,
    int parent,
    CubqlNativeBuild& build,
    int& native_node_index
)
{
    bool is_leaf;
    uint64_t offset;
    uint64_t upper_offset;
    if (!cubql_decode_node(nodes, num_nodes, num_prims, cubql_node_index, is_leaf, offset, upper_offset)) {
        return false;
    }

    const CubqlNode& node = nodes[cubql_node_index];
    native_node_index = int(build.node_lowers.size());
    build.node_lowers.push_back({});
    build.node_uppers.push_back({});
    build.node_parents.push_back(parent);
    build.max_depth = std_max(build.max_depth, depth);

    // Native traversal uses a fixed-size stack.  If cuBQL creates a deeper
    // subtree, pack all primitives below this point into one native leaf.
    // The generic two-child-push traversal needs depth+1 stack entries, so
    // anything at depth >= BVH_QUERY_STACK_SIZE - 1 would overflow.
    if (is_leaf || depth >= BVH_QUERY_STACK_SIZE - 1) {
        const int primitive_start = int(build.primitive_indices.size());
        if (!cubql_append_subtree_primitives(
                nodes, num_nodes, prim_ids, num_prims, cubql_node_index, build.primitive_indices
            )) {
            return false;
        }
        const int primitive_end = int(build.primitive_indices.size());

        build.node_lowers[native_node_index] = make_node(node.lower, primitive_start, true);
        build.node_uppers[native_node_index] = make_node(node.upper, primitive_end, false);
        build.num_leaf_nodes++;
        return true;
    }

    int left_child;
    int right_child;
    if (!cubql_build_native_node(
            nodes, num_nodes, prim_ids, num_prims, uint32_t(offset), depth + 1, native_node_index, build, left_child
        )
        || !cubql_build_native_node(
            nodes, num_nodes, prim_ids, num_prims, uint32_t(offset + 1), depth + 1, native_node_index, build,
            right_child
        )) {
        return false;
    }

    build.node_lowers[native_node_index] = make_node(node.lower, left_child, false);
    build.node_uppers[native_node_index] = make_node(node.upper, right_child, false);
    return true;
}

static bool cubql_copy_to_native_host_bvh(BVH& bvh, const cuBQL::bvh3f& native)
{
    static_assert(
        sizeof(CubqlNode) == sizeof(cuBQL::bvh3f::node_t), "CubqlNode must match cuBQL::BinaryBVH<float, 3>::Node"
    );

    if (native.numNodes > uint32_t(INT_MAX) || native.numPrims > uint32_t(INT_MAX)) {
        wp::set_error_string("Warp error: cuBQL BVH is too large to convert to the native BVH layout");
        return false;
    }

    bvh.num_nodes = 0;
    bvh.num_leaf_nodes = 0;
    bvh.max_depth = 0;
    bvh.max_nodes = 0;

    if (native.numNodes == 0 || native.numPrims == 0) {
        return true;
    }

    const CubqlNode* nodes = reinterpret_cast<const CubqlNode*>(native.nodes);

    // If the cuBQL tree fits within the native traversal stack, copy it
    // verbatim — preserves the contiguous child-pair layout (root at 0,
    // children at offset/offset+1) that gives the fastest cache behaviour
    // during ray traversal.
    bool exceeds_stack = false;
    if (!cubql_tree_exceeds_native_stack(nodes, native.numNodes, native.numPrims, exceeds_stack)) {
        return false;
    }
    if (!exceeds_stack) {
        return cubql_copy_native_order_bvh(bvh, nodes, native.primIDs, native.numNodes, native.numPrims);
    }

    CubqlNativeBuild build;
    build.node_lowers.reserve(native.numNodes);
    build.node_uppers.reserve(native.numNodes);
    build.node_parents.reserve(native.numNodes);
    build.primitive_indices.reserve(native.numPrims);

    int root = -1;
    if (!cubql_build_native_node(nodes, native.numNodes, native.primIDs, native.numPrims, 0, 0, -1, build, root)) {
        return false;
    }

    if (root != 0 || build.primitive_indices.size() != native.numPrims) {
        wp::set_error_string("Warp error: cuBQL BVH conversion produced an invalid native BVH");
        return false;
    }

    bvh.num_nodes = int(build.node_lowers.size());
    bvh.num_leaf_nodes = build.num_leaf_nodes;
    bvh.max_depth = build.max_depth;
    bvh.max_nodes = bvh.num_nodes;

    bvh.node_lowers = static_cast<BVHPackedNodeHalf*>(
        wp_alloc_host(sizeof(BVHPackedNodeHalf) * build.node_lowers.size(), "(native:bvh)")
    );
    bvh.node_uppers = static_cast<BVHPackedNodeHalf*>(
        wp_alloc_host(sizeof(BVHPackedNodeHalf) * build.node_uppers.size(), "(native:bvh)")
    );
    bvh.node_parents = static_cast<int*>(wp_alloc_host(sizeof(int) * build.node_parents.size(), "(native:bvh)"));
    bvh.node_counts = nullptr;
    bvh.primitive_indices
        = static_cast<int*>(wp_alloc_host(sizeof(int) * build.primitive_indices.size(), "(native:bvh)"));
    bvh.root = static_cast<int*>(wp_alloc_host(sizeof(int), "(native:bvh)"));

    if (!bvh.node_lowers || !bvh.node_uppers || !bvh.node_parents || !bvh.primitive_indices || !bvh.root) {
        wp::set_error_string("Warp error: failed to allocate native BVH storage for cuBQL conversion");
        bvh_destroy_host(bvh);
        return false;
    }

    bvh.root[0] = root;

    std::copy(build.node_lowers.begin(), build.node_lowers.end(), bvh.node_lowers);
    std::copy(build.node_uppers.begin(), build.node_uppers.end(), bvh.node_uppers);
    std::copy(build.node_parents.begin(), build.node_parents.end(), bvh.node_parents);
    std::copy(build.primitive_indices.begin(), build.primitive_indices.end(), bvh.primitive_indices);

    return true;
}

void cubql_bvh_create_host(vec3* lowers, vec3* uppers, int num_items, int leaf_size, BVH& bvh)
{
    memset(&bvh, 0, sizeof(BVH));

    bvh.item_lowers = lowers;
    bvh.item_uppers = uppers;
    bvh.num_items = num_items;
    bvh.leaf_size = leaf_size;
    bvh.constructor_type = BVH_CONSTRUCTOR_CUBQL;
    bvh.context = nullptr;

    if (num_items <= 0)
        return;

    cuBQL::box3f* boxes = new cuBQL::box3f[num_items];
    cubql_update_host_boxes(bvh, boxes);

    auto free_partial_bvh = [&]() { bvh_destroy_host(bvh); };

    cuBQL::bvh3f native {};
    try {
        cuBQL::BuildConfig build_config;
        build_config.enableSAH();
        build_config.makeLeafThreshold = leaf_size;
        cuBQL::cpuBuilder(native, boxes, uint32_t(num_items), build_config);
        if (!cubql_copy_to_native_host_bvh(bvh, native))
            free_partial_bvh();
    } catch (const std::exception& e) {
        wp::set_error_string("Warp error: cuBQL BVH build failed: %s", e.what());
        free_partial_bvh();
    } catch (...) {
        wp::set_error_string("Warp error: cuBQL BVH build failed: unknown exception");
        free_partial_bvh();
    }

    bvh.constructor_type = BVH_CONSTRUCTOR_CUBQL;

    if (native.nodes || native.primIDs)
        cuBQL::cpu::freeBVH(native);

    delete[] boxes;
}

void cubql_bvh_destroy_host(BVH& bvh) { bvh_destroy_host(bvh); }

void cubql_bvh_refit_host(BVH& bvh)
{
    if (!bvh.root || bvh.num_nodes == 0)
        return;

    bvh_refit_host(bvh);
}

void cubql_bvh_rebuild_host(BVH& bvh)
{
    vec3* lowers = bvh.item_lowers;
    vec3* uppers = bvh.item_uppers;
    const int num_items = bvh.num_items;
    const int leaf_size = bvh.leaf_size;

    bvh_destroy_host(bvh);
    cubql_bvh_create_host(lowers, uppers, num_items, leaf_size, bvh);
}

#else  // WP_DISABLE_CUBQL

void cubql_bvh_create_host(vec3* lowers, vec3* uppers, int num_items, int leaf_size, BVH& bvh)
{
    wp::set_error_string("Warp error: cuBQL support disabled (WP_DISABLE_CUBQL)");
    memset(&bvh, 0, sizeof(BVH));
    bvh.constructor_type = BVH_CONSTRUCTOR_CUBQL;
}

void cubql_bvh_destroy_host(BVH&) { }
void cubql_bvh_refit_host(BVH&) { }
void cubql_bvh_rebuild_host(BVH&) { }

#endif  // WP_DISABLE_CUBQL

}  // namespace wp
