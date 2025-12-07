/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "warp.h"

#include "bvh.h"
#include "cuda_util.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <functional>
#include <map>
#include <vector>

using namespace wp;

namespace wp {


/////////////////////////////////////////////////////////////////////////////////////////////

class TopDownBVHBuilder {
public:
    void build(BVH& bvh, const vec3* lowers, const vec3* uppers, int n, int in_constructor_type, int* groups);
    void rebuild(BVH& bvh, int in_constructor_type);

private:
    void initialize_empty(BVH& bvh);

    bounds3 calc_bounds(const vec3* lowers, const vec3* uppers, const int* indices, int start, int end);
    int build_recursive(
        BVH& bvh,
        const vec3* lowers,
        const vec3* uppers,
        int start,
        int end,
        int depth,
        int parent,
        int assigned_node = -1
    );
    int
    partition_median(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds);
    int
    partition_midpoint(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds);
    float partition_sah_indices(
        const vec3* lowers,
        const vec3* uppers,
        const int* indices,
        int start,
        int end,
        bounds3 range_bounds,
        int& split_axis
    );
    void build_with_groups(BVH& bvh, const vec3* lowers, const vec3* uppers, const int* groups, int n);
    int constructor_type = -1;
};

//////////////////////////////////////////////////////////////////////

void TopDownBVHBuilder::initialize_empty(BVH& bvh)
{
    bvh.max_depth = 0;
    bvh.max_nodes = 0;
    bvh.node_lowers = nullptr;
    bvh.node_uppers = nullptr;
    bvh.node_parents = nullptr;
    bvh.node_counts = nullptr;
    bvh.root = nullptr;
    bvh.primitive_indices = nullptr;
    bvh.num_leaf_nodes = 0;
    bvh.num_items = 0;
}

static inline void compute_group_ranges(
    const int* groups,
    int n,
    std::vector<int>& unique_groups,
    std::vector<int>& group_starts,
    std::vector<int>& group_ends,
    std::vector<int>& sorted_indices
)
{
    sorted_indices.resize(n);
    for (int i = 0; i < n; ++i)
        sorted_indices[i] = i;
    std::stable_sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) { return groups[a] < groups[b]; });

    unique_groups.clear();
    group_starts.clear();
    group_ends.clear();
    if (n == 0)
        return;

    int current_group = groups[sorted_indices[0]];
    unique_groups.push_back(current_group);
    group_starts.push_back(0);
    for (int i = 1; i < n; ++i) {
        int g = groups[sorted_indices[i]];
        if (g != current_group) {
            group_ends.push_back(i);
            current_group = g;
            unique_groups.push_back(current_group);
            group_starts.push_back(i);
        }
    }
    group_ends.push_back(n);
}

void TopDownBVHBuilder::build_with_groups(BVH& bvh, const vec3* lowers, const vec3* uppers, const int* groups, int n)
{
    // Breakdown of building with groups:
    // 1. Sort and compute group ranges, finding the start and end indices of each group. Since the
    // primitives are not guaranteed to be sorted by group, we sort them by group id first.
    // 2. Find the bounds of each group
    // 3. Build group-level BVH, splitting based on group ids
    // 4. Expand each group leaf into a per-primitive subtree in ascending group order
    // 5. Reorder the tree so that all the leaf nodes are stored in the front

    // 1. Sort and compute group ranges
    std::vector<int> unique_groups, group_starts, group_ends, sorted_indices;
    compute_group_ranges(groups, n, unique_groups, group_starts, group_ends, sorted_indices);

    for (int i = 0; i < n; ++i) {
        int prim = sorted_indices[i];
        bvh.primitive_indices[i] = prim;
    }

    const int num_groups = int(unique_groups.size());

    std::vector<int> group_indices(num_groups);
    std::vector<int> group_leaf_node_index(num_groups, -1);
    std::vector<vec3> group_lowers(num_groups), group_uppers(num_groups);

    // 2. Find the bounds of each group
    for (int g = 0; g < num_groups; ++g) {
        group_indices[g] = g;
        bounds3 gb;
        for (int i = group_starts[g]; i < group_ends[g]; ++i) {
            int prim = bvh.primitive_indices[i];
            gb.add_bounds(lowers[prim], uppers[prim]);
        }
        group_lowers[g] = gb.lower;
        group_uppers[g] = gb.upper;
    }

    bvh.num_nodes = 0;
    bvh.num_leaf_nodes = 0;

    // 3. Build the group-level BVH, storing group ranges in leaf nodes
    std::function<int(int, int, int, int)> build_groups = [&](int start, int end, int depth, int parent) -> int {
        const int count = end - start;
        const int node_index = bvh.num_nodes++;
        if (depth > bvh.max_depth)
            bvh.max_depth = depth;

        bounds3 b;
        for (int i = start; i < end; ++i) {
            int g = group_indices[i];
            b.add_bounds(group_lowers[g], group_uppers[g]);
        }

        if (count == 1) {
            // leaf holds the contiguous primitive range for this group
            int g = group_indices[start];
            int prim_start = group_starts[g];
            int prim_end = group_ends[g];
            bvh.node_lowers[node_index] = make_node(b.lower, prim_start, true);
            bvh.node_uppers[node_index] = make_node(b.upper, prim_end, false);
            bvh.node_parents[node_index] = parent;
            group_leaf_node_index[g] = node_index;
            return node_index;
        }

        // then median split on based on group ids
        int mid = (start + end) / 2;
        std::nth_element(
            group_indices.begin() + start, group_indices.begin() + mid, group_indices.begin() + end,
            [&](int a, int b) { return unique_groups[a] < unique_groups[b]; }
        );

        int left = build_groups(start, mid, depth + 1, node_index);
        int right = build_groups(mid, end, depth + 1, node_index);

        bvh.node_lowers[node_index] = make_node(b.lower, left, false);
        bvh.node_uppers[node_index] = make_node(b.upper, right, false);
        bvh.node_parents[node_index] = parent;
        return node_index;
    };

    for (int i = 0; i < num_groups; ++i)
        group_indices[i] = i;
    *bvh.root = build_groups(0, num_groups, 0, -1);

    // 4. Expand each group leaf into a per-primitive subtree in ascending group order
    for (int g = 0; g < num_groups; ++g) {
        int node = group_leaf_node_index[g];
        if (node < 0)
            continue;
        int prim_start = bvh.node_lowers[node].i;
        int prim_end = bvh.node_uppers[node].i;
        // Replace this packed leaf with a full subtree over [prim_start, prim_end)
        bvh.node_lowers[node].b = 0;
        build_recursive(bvh, lowers, uppers, prim_start, prim_end, 0, bvh.node_parents[node], node);
    }

    // 5. Reorder the tree so that all the leaf nodes are stored in the front
    reorder_top_down_bvh(bvh);
}


void TopDownBVHBuilder::build(
    BVH& bvh, const vec3* lowers, const vec3* uppers, int n, int in_constructor_type, int* groups
)
{
    assert(n >= 0);
    if (n > 0) {
        assert(lowers != nullptr && uppers != nullptr && "Pointers must be valid for n > 0");
    }

    constructor_type = in_constructor_type;
    if (constructor_type != BVH_CONSTRUCTOR_SAH && constructor_type != BVH_CONSTRUCTOR_MEDIAN) {
        fprintf(
            stderr,
            "Unrecognized Constructor type: %d! For CPU constructor it should be either SAH (%d) or Median (%d)!\n",
            constructor_type, BVH_CONSTRUCTOR_SAH, BVH_CONSTRUCTOR_MEDIAN
        );
        return;
    }

    if (n < 0) {
        fprintf(stderr, "Error: Cannot build BVH with a negative primitive count: %d\n", n);
        initialize_empty(bvh);
        return;
    } else if (n == 0) {
        initialize_empty(bvh);
        return;
    } else if (n > INT_MAX / 2) {
        fprintf(stderr, "Error: Primitive count %d is too large and would cause an integer overflow.\n", n);
        initialize_empty(bvh);
        return;
    }

    bvh.max_depth = 0;
    bvh.max_nodes = 2 * n - 1;

    bvh.node_lowers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.node_uppers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.node_parents = new int[bvh.max_nodes];
    bvh.node_counts = nullptr;
    bvh.num_items = n;

    // root is always in first slot for top down builders
    bvh.root = new int[1];
    bvh.root[0] = 0;

    bvh.primitive_indices = new int[n];
    for (int i = 0; i < n; ++i)
        bvh.primitive_indices[i] = i;

    if (groups) {
        build_with_groups(bvh, lowers, uppers, groups, n);
    } else {
        build_recursive(bvh, lowers, uppers, 0, n, 0, -1);
    }
}

void TopDownBVHBuilder::rebuild(BVH& bvh, int in_constructor_type)
{
    if (in_constructor_type != BVH_CONSTRUCTOR_SAH && in_constructor_type != BVH_CONSTRUCTOR_MEDIAN) {
        fprintf(
            stderr,
            "Unrecognized Constructor type: %d! For CPU constructor it should be either SAH (%d) or Median (%d)!\n",
            in_constructor_type, BVH_CONSTRUCTOR_SAH, BVH_CONSTRUCTOR_MEDIAN
        );
        return;
    }
    if (bvh.num_items == 0)
        return;

    constructor_type = in_constructor_type;
    for (int i = 0; i < bvh.num_items; ++i)
        bvh.primitive_indices[i] = i;

    bvh.max_depth = 0;
    bvh.num_nodes = 0;
    bvh.num_leaf_nodes = 0;

    if (bvh.item_groups) {
        build_with_groups(bvh, bvh.item_lowers, bvh.item_uppers, bvh.item_groups, bvh.num_items);
    } else {
        build_recursive(bvh, bvh.item_lowers, bvh.item_uppers, 0, bvh.num_items, 0, -1);
    }
}


bounds3 TopDownBVHBuilder::calc_bounds(const vec3* lowers, const vec3* uppers, const int* indices, int start, int end)
{
    bounds3 u;

    for (int i = start; i < end; ++i) {
        u.add_bounds(lowers[indices[i]], uppers[indices[i]]);
    }

    return u;
}

struct PartitionPredicateMedian {
    PartitionPredicateMedian(const vec3* lowers, const vec3* uppers, int a)
        : lowers(lowers)
        , uppers(uppers)
        , axis(a)
    {
    }

    bool operator()(int a, int b) const
    {
        vec3 a_center = 0.5f * (lowers[a] + uppers[a]);
        vec3 b_center = 0.5f * (lowers[b] + uppers[b]);

        return a_center[axis] < b_center[axis];
    }

    const vec3* lowers;
    const vec3* uppers;
    int axis;
};


int TopDownBVHBuilder::partition_median(
    const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds
)
{
    assert(end - start >= 2);

    vec3 edges = range_bounds.edges();

    int axis = longest_axis(edges);

    const int k = (start + end) / 2;

    std::nth_element(&indices[start], &indices[k], &indices[end], PartitionPredicateMedian(lowers, uppers, axis));

    return k;
}

struct PartitionPredicateMidPoint {
    PartitionPredicateMidPoint(const vec3* lowers, const vec3* uppers, int a, float m)
        : lowers(lowers)
        , uppers(uppers)
        , axis(a)
        , mid(m)
    {
    }

    bool operator()(int index) const
    {
        vec3 center = 0.5f * (lowers[index] + uppers[index]);

        return center[axis] <= mid;
    }

    const vec3* lowers;
    const vec3* uppers;

    int axis;
    float mid;
};


int TopDownBVHBuilder::partition_midpoint(
    const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds
)
{
    assert(end - start >= 2);

    vec3 edges = range_bounds.edges();
    vec3 center = range_bounds.center();

    int axis = longest_axis(edges);
    float mid = center[axis];

    int* upper = std::partition(indices + start, indices + end, PartitionPredicateMidPoint(lowers, uppers, axis, mid));

    int k = upper - indices;

    // if we failed to split items then just split in the middle
    if (k == start || k == end)
        k = (start + end) / 2;

    return k;
}

float TopDownBVHBuilder::partition_sah_indices(
    const vec3* lowers,
    const vec3* uppers,
    const int* indices,
    int start,
    int end,
    bounds3 range_bounds,
    int& split_axis
)
{
    int buckets_counts[SAH_NUM_BUCKETS];
    bounds3 buckets[SAH_NUM_BUCKETS];
    float left_areas[SAH_NUM_BUCKETS - 1];
    float right_areas[SAH_NUM_BUCKETS - 1];

    assert(end - start >= 2);

    int n = end - start;

    bounds3 centroid_bounds;
    for (int i = start; i < end; ++i) {
        vec3 item_center = 0.5f * (lowers[indices[i]] + uppers[indices[i]]);
        centroid_bounds.add_point(item_center);
    }
    vec3 edges = centroid_bounds.edges();

    split_axis = longest_axis(edges);

    // compute each bucket
    float range_start = centroid_bounds.lower[split_axis];
    float range_end = centroid_bounds.upper[split_axis];

    // Guard against zero extent along the split axis to avoid division-by-zero
    if (range_end <= range_start) {
        // Returning range_start will cause the caller's partition to produce an empty left side;
        // the recursive builder already falls back to a midpoint split in that case.
        return range_start;
    }

    std::fill(buckets_counts, buckets_counts + SAH_NUM_BUCKETS, 0);
    for (int item_idx = start; item_idx < end; item_idx++) {
        vec3 item_center = 0.5f * (lowers[indices[item_idx]] + uppers[indices[item_idx]]);
        int bucket_idx = SAH_NUM_BUCKETS * (item_center[split_axis] - range_start) / (range_end - range_start);
        // clamp into valid range [0, SAH_NUM_BUCKETS-1]
        if (bucket_idx < 0)
            bucket_idx = 0;
        if (bucket_idx >= SAH_NUM_BUCKETS)
            bucket_idx = SAH_NUM_BUCKETS - 1;

        bounds3 item_bound(lowers[indices[item_idx]], uppers[indices[item_idx]]);

        if (buckets_counts[bucket_idx]) {
            buckets[bucket_idx] = bounds_union(item_bound, buckets[bucket_idx]);
        } else {
            buckets[bucket_idx] = item_bound;
        }

        buckets_counts[bucket_idx]++;
    }

    bounds3 left;
    bounds3 right;

    // n - 1 division points for n buckets
    int counts_l[SAH_NUM_BUCKETS - 1];
    int counts_r[SAH_NUM_BUCKETS - 1];

    int count_l = 0;
    int count_r = 0;
    // build cumulative bounds and area from left and right
    for (int i = 0; i < SAH_NUM_BUCKETS - 1; ++i) {
        bounds3 bound_start = buckets[i];
        bounds3 bound_end = buckets[SAH_NUM_BUCKETS - i - 1];

        left = bounds_union(left, bound_start);
        right = bounds_union(right, bound_end);

        left_areas[i] = left.area();
        right_areas[SAH_NUM_BUCKETS - i - 2] = right.area();

        count_l += buckets_counts[i];
        count_r += buckets_counts[SAH_NUM_BUCKETS - i - 1];

        counts_l[i] = count_l;
        counts_r[SAH_NUM_BUCKETS - i - 2] = count_r;
    }

    float invTotalArea = 1.0f / range_bounds.area();

    // find split point i that minimizes area(left[i]) * count[left[i]] + area(right[i]) * count[right[i]]
    int minSplit = 0;
    float minCost = FLT_MAX;
    for (int i = 0; i < SAH_NUM_BUCKETS - 1; ++i) {
        float pBelow = left_areas[i] * invTotalArea;
        float pAbove = right_areas[i] * invTotalArea;

        float cost = pBelow * counts_l[i] + pAbove * counts_r[i];

        if (cost < minCost) {
            minCost = cost;
            minSplit = i;
        }
    }

    // return the dividing
    assert(minSplit >= 0 && minSplit < SAH_NUM_BUCKETS - 1);
    float split_point = range_start + (minSplit + 1) * (range_end - range_start) / SAH_NUM_BUCKETS;

    return split_point;
}


int TopDownBVHBuilder::build_recursive(
    BVH& bvh, const vec3* lowers, const vec3* uppers, int start, int end, int depth, int parent, int assigned_node
)
{
    // Build subtree over [start,end). If assigned_node >= 0,
    // reuse that node index instead of allocating a new one.
    assert(start < end);

    const int n = end - start;
    const int node_index = (assigned_node >= 0) ? assigned_node : bvh.num_nodes++;

    if (assigned_node < 0)
        assert(node_index < bvh.max_nodes);

    if (depth > bvh.max_depth)
        bvh.max_depth = depth;

    bounds3 b = calc_bounds(lowers, uppers, bvh.primitive_indices, start, end);

    // If the depth exceeds BVH_QUERY_STACK_SIZE, an out-of-bounds access bug may occur during querying.
    // In that case, we merge the following nodes into a single large leaf node.
    if (n <= bvh.leaf_size || depth >= BVH_QUERY_STACK_SIZE) {
        bvh.node_lowers[node_index] = make_node(b.lower, start, true);
        bvh.node_uppers[node_index] = make_node(b.upper, end, false);
        bvh.node_parents[node_index] = parent;
        bvh.num_leaf_nodes++;
        return node_index;
    }

    int split = -1;
    if (constructor_type == BVH_CONSTRUCTOR_SAH)
    // SAH constructor
    {
        int split_axis = -1;
        float split_point = partition_sah_indices(lowers, uppers, bvh.primitive_indices, start, end, b, split_axis);
        auto boundary = std::partition(bvh.primitive_indices + start, bvh.primitive_indices + end, [&](int i) {
            return 0.5f * (lowers[i] + uppers[i])[split_axis] < split_point;
        });

        split = std::distance(bvh.primitive_indices + start, boundary) + start;
    } else if (constructor_type == BVH_CONSTRUCTOR_MEDIAN)
    // Median constructor
    {
        split = partition_median(lowers, uppers, bvh.primitive_indices, start, end, b);
    } else {
        printf("Unknown type of BVH constructor: %d!\n", constructor_type);
        return -1;
    }

    if (split == start || split == end) {
        // partitioning failed, split down the middle
        split = (start + end) / 2;
    }

    int left_child = build_recursive(bvh, lowers, uppers, start, split, depth + 1, node_index);
    int right_child = build_recursive(bvh, lowers, uppers, split, end, depth + 1, node_index);

    bvh.node_lowers[node_index] = make_node(b.lower, left_child, false);
    bvh.node_uppers[node_index] = make_node(b.upper, right_child, false);
    bvh.node_parents[node_index] = parent;
    return node_index;
}


void reorder_top_down_bvh(BVH& bvh_host)
{
    // reorder bvh_host such that its nodes are in the front
    // this is essential for the device refit
    BVHPackedNodeHalf* node_lowers_reordered = new BVHPackedNodeHalf[bvh_host.max_nodes];
    BVHPackedNodeHalf* node_uppers_reordered = new BVHPackedNodeHalf[bvh_host.max_nodes];

    int* node_parents_reordered = new int[bvh_host.max_nodes];

    std::vector<int> old_to_new(bvh_host.max_nodes, -1);

    // We will place nodes in this order:
    //   Pass 1: leaf nodes (except if it's the root index)
    //   Pass 2: non-leaf, non-root
    //   Pass 3: root node
    int next_pos = 0;

    const int root_index = *bvh_host.root;
    // Pass 1: place leaf nodes at the front
    // During grouped build, it is possible that some leaf nodes do
    // not satisfy the ascending group id assumption. This would happen if
    // a specific group has <= leaf size primitives, and so the leaf node is
    // built in-place rather than on the node frontier like the other leaves.
    // During this pass we can detect if this has occurred and call a sort to fix it.
    bool needs_sort = false;
    int prev_group = INT_MIN;
    for (int i = 0; i < bvh_host.num_nodes; ++i) {
        if (bvh_host.node_lowers[i].b) {
            if (bvh_host.item_groups) {
                int group = bvh_host.item_groups[bvh_host.primitive_indices[bvh_host.node_lowers[i].i]];
                if (group < prev_group)
                    needs_sort = true;
                prev_group = group;
            }
            node_lowers_reordered[next_pos] = bvh_host.node_lowers[i];
            node_uppers_reordered[next_pos] = bvh_host.node_uppers[i];
            old_to_new[i] = next_pos;
            next_pos++;
        }
    }

    if (needs_sort) {
        const int* groups = bvh_host.item_groups;

        std::vector<int> leaf_indices;
        leaf_indices.reserve(next_pos);
        for (int i = 0; i < bvh_host.num_nodes; ++i) {
            if (bvh_host.node_lowers[i].b) {
                leaf_indices.push_back(i);
            }
        }

        std::stable_sort(leaf_indices.begin(), leaf_indices.end(), [&](int a, int b) {
            int ga = groups[bvh_host.primitive_indices[bvh_host.node_lowers[a].i]];
            int gb = groups[bvh_host.primitive_indices[bvh_host.node_lowers[b].i]];
            if (ga == gb)
                return a < b;
            return ga < gb;
        });

        next_pos = 0;
        for (size_t i = 0; i < leaf_indices.size(); ++i) {
            int old_index = leaf_indices[i];
            node_lowers_reordered[next_pos] = bvh_host.node_lowers[old_index];
            node_uppers_reordered[next_pos] = bvh_host.node_uppers[old_index];
            old_to_new[old_index] = next_pos;
            next_pos++;
        }
    }
    bvh_host.num_leaf_nodes = next_pos;

    // Pass 2: place non-leaf, non-root nodes
    for (int i = 0; i < bvh_host.num_nodes; ++i) {
        if (i == root_index) {
            if (bvh_host.node_lowers[i].b)  // if root node is a leaf node, there must be only be one node
            {
                *bvh_host.root = 0;
            } else {
                *bvh_host.root = next_pos;
            }
        }
        if (!bvh_host.node_lowers[i].b) {
            node_lowers_reordered[next_pos] = bvh_host.node_lowers[i];
            node_uppers_reordered[next_pos] = bvh_host.node_uppers[i];
            old_to_new[i] = next_pos;
            next_pos++;
        }
    }

    // We can do that by enumerating all old->new pairs:
    for (int old_index = 0; old_index < bvh_host.num_nodes; ++old_index) {
        int new_index = old_to_new[old_index];  // new index

        int old_parent = bvh_host.node_parents[old_index];
        if (old_parent != -1) {
            node_parents_reordered[new_index] = old_to_new[old_parent];
        } else {
            node_parents_reordered[new_index] = -1;
        }

        // only need to modify the child index of non-leaf nodes
        if (!bvh_host.node_lowers[old_index].b) {
            node_lowers_reordered[new_index].i = old_to_new[bvh_host.node_lowers[old_index].i];
            node_uppers_reordered[new_index].i = old_to_new[bvh_host.node_uppers[old_index].i];
        }
    }

    delete[] bvh_host.node_lowers;
    delete[] bvh_host.node_uppers;
    delete[] bvh_host.node_parents;

    bvh_host.node_lowers = node_lowers_reordered;
    bvh_host.node_uppers = node_uppers_reordered;
    bvh_host.node_parents = node_parents_reordered;
}


void bvh_refit_recursive(BVH& bvh, int index)
{
    BVHPackedNodeHalf& lower = bvh.node_lowers[index];
    BVHPackedNodeHalf& upper = bvh.node_uppers[index];

    if (lower.b) {
        // update leaf from items
        bounds3 bound;
        for (int item_counter = lower.i; item_counter < upper.i; item_counter++) {
            const int item = bvh.primitive_indices[item_counter];
            bound.add_bounds(bvh.item_lowers[item], bvh.item_uppers[item]);
        }

        reinterpret_cast<vec3&>(lower) = bound.lower;
        reinterpret_cast<vec3&>(upper) = bound.upper;
    } else {
        int left_index = lower.i;
        int right_index = upper.i;

        bvh_refit_recursive(bvh, left_index);
        bvh_refit_recursive(bvh, right_index);

        // compute union of children
        const vec3& left_lower = reinterpret_cast<const vec3&>(bvh.node_lowers[left_index]);
        const vec3& left_upper = reinterpret_cast<const vec3&>(bvh.node_uppers[left_index]);

        const vec3& right_lower = reinterpret_cast<const vec3&>(bvh.node_lowers[right_index]);
        const vec3& right_upper = reinterpret_cast<const vec3&>(bvh.node_uppers[right_index]);

        // union of child bounds
        vec3 new_lower = min(left_lower, right_lower);
        vec3 new_upper = max(left_upper, right_upper);

        // write new BVH nodes
        reinterpret_cast<vec3&>(lower) = new_lower;
        reinterpret_cast<vec3&>(upper) = new_upper;
    }
}

void bvh_refit_host(BVH& bvh) { bvh_refit_recursive(bvh, *bvh.root); }
void bvh_rebuild_host(BVH& bvh, int constructor_type)
{
    TopDownBVHBuilder builder;
    builder.rebuild(bvh, constructor_type);
}

}  // namespace wp


// making the class accessible from python


namespace {
// host-side copy of bvh descriptors, maps GPU bvh address (id) to a CPU desc
std::map<uint64_t, BVH> g_bvh_descriptors;
}  // anonymous namespace


namespace wp {

bool bvh_get_descriptor(uint64_t id, BVH& bvh)
{
    const auto& iter = g_bvh_descriptors.find(id);
    if (iter == g_bvh_descriptors.end())
        return false;
    else
        bvh = iter->second;
    return true;
}

void bvh_add_descriptor(uint64_t id, const BVH& bvh) { g_bvh_descriptors[id] = bvh; }

void bvh_rem_descriptor(uint64_t id) { g_bvh_descriptors.erase(id); }


// create in-place given existing descriptor
void bvh_create_host(
    vec3* lowers, vec3* uppers, int num_items, int constructor_type, int* groups, int leaf_size, BVH& bvh
)
{
    memset(&bvh, 0, sizeof(BVH));

    bvh.item_lowers = lowers;
    bvh.item_uppers = uppers;
    bvh.item_groups = groups;
    bvh.num_items = num_items;
    bvh.leaf_size = leaf_size;

    TopDownBVHBuilder builder;
    builder.build(bvh, lowers, uppers, num_items, constructor_type, groups);
}


void bvh_destroy_host(BVH& bvh)
{
    delete[] bvh.node_lowers;
    delete[] bvh.node_uppers;
    delete[] bvh.node_parents;
    delete[] bvh.primitive_indices;
    delete[] bvh.root;

    bvh.node_lowers = nullptr;
    bvh.node_uppers = nullptr;
    bvh.node_parents = nullptr;
    bvh.primitive_indices = nullptr;
    bvh.root = nullptr;

    bvh.max_nodes = 0;
    bvh.num_items = 0;
}

}  // namespace wp

uint64_t wp_bvh_create_host(vec3* lowers, vec3* uppers, int num_items, int constructor_type, int* groups, int leaf_size)
{
    BVH* bvh = new BVH();
    wp::bvh_create_host(lowers, uppers, num_items, constructor_type, groups, leaf_size, *bvh);

    return (uint64_t)bvh;
}

void wp_bvh_refit_host(uint64_t id)
{
    BVH* bvh = (BVH*)(id);
    wp::bvh_refit_host(*bvh);
}

void wp_bvh_rebuild_host(uint64_t id, int constructor_type)
{
    BVH* bvh = (BVH*)(id);
    wp::bvh_rebuild_host(*bvh, constructor_type);
}

void wp_bvh_destroy_host(uint64_t id)
{
    BVH* bvh = (BVH*)(id);
    wp::bvh_destroy_host(*bvh);
    delete bvh;
}


// stubs for non-CUDA platforms
#if !WP_ENABLE_CUDA

uint64_t wp_bvh_create_device(
    void* context, wp::vec3* lowers, wp::vec3* uppers, int num_items, int constructor_type, int* groups, int leaf_size
)
{
    return 0;
}
void wp_bvh_refit_device(uint64_t id) { }
void wp_bvh_destroy_device(uint64_t id) { }
void wp_bvh_rebuild_device(uint64_t id) { }

#endif  // !WP_ENABLE_CUDA
