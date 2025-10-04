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

#include <vector>
#include <algorithm>
#include <functional>

#include "bvh.h"
#include "warp.h"
#include "cuda_util.h"

#include <cassert>
#include <map>
#include <climits>

using namespace wp;

namespace wp
{


/////////////////////////////////////////////////////////////////////////////////////////////

class TopDownBVHBuilder
{	
public:

    void build(BVH& bvh, const vec3* lowers, const vec3* uppers, int n, int in_constructor_type, int* groups);
    void rebuild(BVH& bvh, int in_constructor_type);

private:

    void initialize_empty(BVH& bvh);

    bounds3 calc_bounds(const vec3* lowers, const vec3* uppers, const int* indices, int start, int end);
    int build_recursive(BVH& bvh, const vec3* lowers, const vec3* uppers, int start, int end, int depth, int parent);
    int partition_median(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds);
    int partition_midpoint(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds);
    float partition_sah_indices(const vec3* lowers, const vec3* uppers,const int* indices, int start, int end, bounds3 range_bounds, int& split_axis);
    
    // Group-aware builds
    void build_with_groups(BVH& bvh, const vec3* lowers, const vec3* uppers, const int* groups, int n);
    int build_recursive_range(BVH& bvh, const vec3* lowers, const vec3* uppers, int start, int end, int depth, int parent, int assigned_node);

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
    bvh.keys = nullptr;
    bvh.primitive_indices = nullptr;
    bvh.num_leaf_nodes = 0;
    bvh.num_items = 0;
}

static inline void compute_group_ranges(const int* groups, int n, std::vector<int>& unique_groups,
                                        std::vector<int>& group_starts, std::vector<int>& group_ends,
                                        std::vector<int>& sorted_indices)
{
    // stable sort primitives by group id; build ranges
    sorted_indices.resize(n);
    for (int i = 0; i < n; ++i) sorted_indices[i] = i;
    std::stable_sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b){ return groups[a] < groups[b]; });

    unique_groups.clear();
    group_starts.clear();
    group_ends.clear();
    if (n == 0) return;

    int current_group = groups[sorted_indices[0]];
    unique_groups.push_back(current_group);
    group_starts.push_back(0);
    for (int i = 1; i < n; ++i)
    {
        int g = groups[sorted_indices[i]];
        if (g != current_group)
        {
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
    std::vector<int> unique_groups, group_starts, group_ends, sorted_indices;
    compute_group_ranges(groups, n, unique_groups, group_starts, group_ends, sorted_indices);

    // Reorder primitive_indices to be grouped, and set keys with group in upper 32 bits
    for (int i = 0; i < n; ++i)
    {
        int prim = sorted_indices[i];
        bvh.primitive_indices[i] = prim;
        bvh.keys[i] = (uint64_t(uint32_t(groups[prim])) << 32);
    }

    const int num_groups = int(unique_groups.size());

    std::vector<int> group_indices(num_groups);
    std::vector<vec3> group_lowers(num_groups), group_uppers(num_groups);
    for (int g = 0; g < num_groups; ++g)
    {
        group_indices[g] = g;
        bounds3 gb;
        for (int i = group_starts[g]; i < group_ends[g]; ++i)
        {
            int prim = bvh.primitive_indices[i];
            gb.add_bounds(lowers[prim], uppers[prim]);
        }
        group_lowers[g] = gb.lower;
        group_uppers[g] = gb.upper;
    }

    bvh.num_nodes = 0;
    bvh.num_leaf_nodes = 0;

    // Build the group-level BVH first, storing group ranges in leaf nodes
    // We reuse bvh.primitive_indices to refer to group_indices during group build via a small alias
    // Save original primitive_indices for later; but we already wrote per-primitive order grouped above
    // We'll build group BVH by temporarily pointing to a local index array
    std::function<int(int,int,int,int)> build_groups = [&](int start, int end, int depth, int parent)->int
    {
        const int count = end - start;
        const int node_index = bvh.num_nodes++;
        if (depth > bvh.max_depth) bvh.max_depth = depth;

        bounds3 b;
        for (int i = start; i < end; ++i)
        {
            int g = group_indices[i];
            b.add_bounds(group_lowers[g], group_uppers[g]);
        }

        if (count == 1)
        {
            // leaf holds the contiguous primitive range for this group
            int g = group_indices[start];
            int prim_start = group_starts[g];
            int prim_end = group_ends[g];
            bvh.node_lowers[node_index] = make_node(b.lower, prim_start, true);
            bvh.node_uppers[node_index] = make_node(b.upper, prim_end, false);
            bvh.node_parents[node_index] = parent;
            bvh.num_leaf_nodes++;
            return node_index;
        }

        // choose split axis using bounds, then median split on group centers
        int axis = longest_axis(b.edges());
        int mid = (start + end) / 2;
        std::nth_element(group_indices.begin() + start, group_indices.begin() + mid, group_indices.begin() + end,
            [&](int a, int b){
                vec3 ca = 0.5f*(group_lowers[a] + group_uppers[a]);
                vec3 cb = 0.5f*(group_lowers[b] + group_uppers[b]);
                return ca[axis] < cb[axis];
            });

        int left = build_groups(start, mid, depth + 1, node_index);
        int right = build_groups(mid, end, depth + 1, node_index);

        bvh.node_lowers[node_index] = make_node(b.lower, left, false);
        bvh.node_uppers[node_index] = make_node(b.upper, right, false);
        bvh.node_parents[node_index] = parent;
        return node_index;
    };

    for (int i = 0; i < num_groups; ++i) group_indices[i] = i;
    *bvh.root = build_groups(0, num_groups, 0, -1);

    // Expand each group leaf into a per-primitive subtree
    for (int node = 0; node < bvh.num_nodes; ++node)
    {
        if (bvh.node_lowers[node].b)
        {
            int prim_start = bvh.node_lowers[node].i;
            int prim_end = bvh.node_uppers[node].i;
            // Replace this packed leaf with a full subtree over [prim_start, prim_end)
            // Temporarily clear the leaf flag so our build writes internal nodes
            bvh.node_lowers[node].b = 0;
            build_recursive_range(bvh, lowers, uppers, prim_start, prim_end, 0, bvh.node_parents[node], node);
        }
    }
}

void TopDownBVHBuilder::build(BVH& bvh, const vec3* lowers, const vec3* uppers, int n, int in_constructor_type, int* groups)
{
    assert(n >= 0);
    if (n > 0)
    {
        assert(lowers != nullptr && uppers != nullptr && "Pointers must be valid for n > 0");
    }

    constructor_type = in_constructor_type;
    if (constructor_type != BVH_CONSTRUCTOR_SAH && constructor_type != BVH_CONSTRUCTOR_MEDIAN)
    {
        fprintf(stderr, "Unrecognized Constructor type: %d! For CPU constructor it should be either SAH (%d) or Median (%d)!\n",
            constructor_type, BVH_CONSTRUCTOR_SAH, BVH_CONSTRUCTOR_MEDIAN);
        return;
    }

    if (n < 0)
    {
        fprintf(stderr, "Error: Cannot build BVH with a negative primitive count: %d\n", n);
        initialize_empty(bvh);
        return;
    }
    else if (n == 0)
    {
        initialize_empty(bvh);
        return;
    }
    else if (n > INT_MAX / 2)
    {
        fprintf(stderr, "Error: Primitive count %d is too large and would cause an integer overflow.\n", n);
        initialize_empty(bvh);
        return;
    }

    bvh.max_depth = 0;
    bvh.max_nodes = 2*n-1;

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
    
    bvh.keys = new uint64_t[n];

    if (groups)
    {
        // Build using two-level grouping to ensure each group maps to a single subtree
        build_with_groups(bvh, lowers, uppers, groups, n);
    }
    else
    {
        for (int i = 0; i < n; ++i)
            bvh.keys[i] = 0;

        build_recursive(bvh, lowers, uppers,  0, n, 0, -1);
    }
}

void TopDownBVHBuilder::rebuild(BVH& bvh, int in_constructor_type)
{
    if (in_constructor_type != BVH_CONSTRUCTOR_SAH && in_constructor_type != BVH_CONSTRUCTOR_MEDIAN)
    {
        fprintf(stderr, "Unrecognized Constructor type: %d! For CPU constructor it should be either SAH (%d) or Median (%d)!\n",
            in_constructor_type, BVH_CONSTRUCTOR_SAH, BVH_CONSTRUCTOR_MEDIAN);
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

    if (bvh.item_groups)
    {
        build_with_groups(bvh, bvh.item_lowers, bvh.item_uppers, bvh.item_groups, bvh.num_items);
    }
    else
    {
        // ensure keys initialized
        for (int i = 0; i < bvh.num_items; ++i) bvh.keys[i] = 0;
        build_recursive(bvh, bvh.item_lowers, bvh.item_uppers, 0, bvh.num_items, 0, -1);
    }
}


bounds3 TopDownBVHBuilder::calc_bounds(const vec3* lowers, const vec3* uppers, const int* indices, int start, int end)
{
    bounds3 u;

    for (int i=start; i < end; ++i)
    {
        u.add_bounds(lowers[indices[i]], uppers[indices[i]]);
    }

    return u;
}

struct PartitionPredicateMedian
{
    PartitionPredicateMedian(const vec3* lowers, const vec3* uppers, int a) : lowers(lowers), uppers(uppers), axis(a) {}

    bool operator()(int a, int b) const
    {
        vec3 a_center = 0.5f*(lowers[a] + uppers[a]);
        vec3 b_center = 0.5f*(lowers[b] + uppers[b]);
        
        return a_center[axis] < b_center[axis];
    }

    const vec3* lowers;
    const vec3* uppers;
    int axis;
};


int TopDownBVHBuilder::partition_median(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds)
{
    assert(end-start >= 2);

    vec3 edges = range_bounds.edges();

    int axis = longest_axis(edges);

    const int k = (start+end)/2;

    std::nth_element(&indices[start], &indices[k], &indices[end], PartitionPredicateMedian(lowers, uppers, axis));

    return k;
}	
    
struct PartitionPredicateMidPoint
{
    PartitionPredicateMidPoint(const vec3* lowers, const vec3* uppers, int a, float m) : lowers(lowers), uppers(uppers), axis(a), mid(m) {}

    bool operator()(int index) const 
    {
        vec3 center = 0.5f*(lowers[index] + uppers[index]);

        return center[axis] <= mid;
    }

    const vec3* lowers;
    const vec3* uppers;
    
    int axis;
    float mid;
};


int TopDownBVHBuilder::partition_midpoint(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds)
{
    assert(end-start >= 2);

    vec3 edges = range_bounds.edges();
    vec3 center = range_bounds.center();

    int axis = longest_axis(edges);
    float mid = center[axis];

    int* upper = std::partition(indices+start, indices+end, PartitionPredicateMidPoint(lowers, uppers, axis, mid));

    int k = upper-indices;

    // if we failed to split items then just split in the middle
    if (k == start || k == end)
        k = (start+end)/2;

    return k;
}

float TopDownBVHBuilder::partition_sah_indices(const vec3* lowers, const vec3* uppers, const int* indices, int start, int end, bounds3 range_bounds, int& split_axis)
{
    int buckets_counts[SAH_NUM_BUCKETS];
    bounds3 buckets[SAH_NUM_BUCKETS];
    float left_areas[SAH_NUM_BUCKETS - 1];
    float right_areas[SAH_NUM_BUCKETS - 1];

    assert(end - start >= 2);

    int n = end - start;
    vec3 edges = range_bounds.edges();

    bounds3 b = calc_bounds(lowers, uppers, indices, start, end);

    split_axis = longest_axis(edges);

    // compute each bucket
    float range_start = b.lower[split_axis];
    float range_end = b.upper[split_axis];

    // Guard against zero extent along the split axis to avoid division-by-zero
    if (range_end <= range_start)
    {
        // Returning range_start will cause the caller's partition to produce an empty left side;
        // the recursive builder already falls back to a midpoint split in that case.
        return range_start;
    }

    std::fill(buckets_counts, buckets_counts + SAH_NUM_BUCKETS, 0);
    for (int item_idx = start; item_idx < end; item_idx++)
    {
        vec3 item_center = 0.5f * (lowers[indices[item_idx]] + uppers[indices[item_idx]]);
        int bucket_idx = SAH_NUM_BUCKETS * (item_center[split_axis] - range_start) / (range_end - range_start);
        // clamp into valid range [0, SAH_NUM_BUCKETS-1]
        if (bucket_idx < 0) bucket_idx = 0;
        if (bucket_idx >= SAH_NUM_BUCKETS) bucket_idx = SAH_NUM_BUCKETS - 1;

        bounds3 item_bound(lowers[indices[item_idx]], uppers[indices[item_idx]]);

        if (buckets_counts[bucket_idx])
        {
            buckets[bucket_idx] = bounds_union(item_bound, buckets[bucket_idx]);
        }
        else
        {
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
    for (int i = 0; i < SAH_NUM_BUCKETS - 1; ++i)
    {
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
    for (int i = 0; i < SAH_NUM_BUCKETS - 1; ++i)
    {
        float pBelow = left_areas[i] * invTotalArea;
        float pAbove = right_areas[i] * invTotalArea;

        float cost = pBelow * counts_l[i] + pAbove * counts_r[i];

        if (cost < minCost)
        {
            minCost = cost;
            minSplit = i;
        }
    }

    // return the dividing 
    assert(minSplit >= 0 && minSplit < SAH_NUM_BUCKETS - 1);
    float split_point = range_start + (minSplit + 1) * (range_end - range_start) / SAH_NUM_BUCKETS;

    return split_point;
}

int TopDownBVHBuilder::build_recursive(BVH& bvh, const vec3* lowers, const vec3* uppers, int start, int end, int depth, int parent)
{
    assert(start < end);

    const int n = end - start;
    const int node_index = bvh.num_nodes++;

    assert(node_index < bvh.max_nodes);

    if (depth > bvh.max_depth)
        bvh.max_depth = depth;

    bounds3 b = calc_bounds(lowers, uppers, bvh.primitive_indices, start, end);

    // If the depth exceeds BVH_QUERY_STACK_SIZE, an out-of-bounds access bug may occur during querying.
    // In that case, we merge the following nodes into a single large leaf node.
    if (n <= BVH_LEAF_SIZE || depth >= BVH_QUERY_STACK_SIZE - 1)
    {
        bvh.node_lowers[node_index] = make_node(b.lower, start, true);
        bvh.node_uppers[node_index] = make_node(b.upper, end, false);
        bvh.node_parents[node_index] = parent;
        bvh.num_leaf_nodes++;
    }
    else
    {
        int split = -1;
        if (constructor_type == BVH_CONSTRUCTOR_SAH)
            // SAH constructor
        {
            int split_axis = -1;
            float split_point = partition_sah_indices(lowers, uppers, bvh.primitive_indices, start, end, b, split_axis);
            auto boundary = std::partition(bvh.primitive_indices + start, bvh.primitive_indices + end,
                [&](int i) {
                    return 0.5f * (lowers[i] + uppers[i])[split_axis] < split_point;
                });

            split = std::distance(bvh.primitive_indices + start, boundary) + start;
        }
        else if (constructor_type == BVH_CONSTRUCTOR_MEDIAN)
            // Median constructor
        {
            split = partition_median(lowers, uppers, bvh.primitive_indices, start, end, b);
        }
        else
        {
            printf("Unknown type of BVH constructor: %d!\n", constructor_type);
            return -1;
        }

        if (split == start || split == end)
        {
            // partitioning failed, split down the middle
            split = (start + end) / 2;
        }

        int left_child = build_recursive(bvh, lowers, uppers, start, split, depth + 1, node_index);
        int right_child = build_recursive(bvh, lowers, uppers, split, end, depth + 1, node_index);

        bvh.node_lowers[node_index] = make_node(b.lower, left_child, false);
        bvh.node_uppers[node_index] = make_node(b.upper, right_child, false);
        bvh.node_parents[node_index] = parent;
    }

    return node_index;
}

// Build subtree over [start,end) into an already-reserved node index if assigned_node >= 0.
// Returns the index of the subtree root.
int TopDownBVHBuilder::build_recursive_range(BVH& bvh, const vec3* lowers, const vec3* uppers, int start, int end, int depth, int parent, int assigned_node)
{
    assert(start < end);

    const int n = end - start;
    const int node_index = (assigned_node >= 0) ? assigned_node : bvh.num_nodes++;

    if (assigned_node < 0)
        assert(node_index < bvh.max_nodes);

    if (depth > bvh.max_depth)
        bvh.max_depth = depth;

    bounds3 b = calc_bounds(lowers, uppers, bvh.primitive_indices, start, end);

    if (n <= BVH_LEAF_SIZE || depth >= BVH_QUERY_STACK_SIZE - 1)
    {
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
        auto boundary = std::partition(bvh.primitive_indices + start, bvh.primitive_indices + end,
            [&](int i) {
                return 0.5f * (lowers[i] + uppers[i])[split_axis] < split_point;
            });

        split = std::distance(bvh.primitive_indices + start, boundary) + start;
    }
    else if (constructor_type == BVH_CONSTRUCTOR_MEDIAN)
        // Median constructor
    {
        split = partition_median(lowers, uppers, bvh.primitive_indices, start, end, b);
    }
    else
    {
        printf("Unknown type of BVH constructor: %d!\n", constructor_type);
        return -1;
    }

    if (split == start || split == end)
    {
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


void bvh_refit_recursive(BVH& bvh, int index)
{
    BVHPackedNodeHalf& lower = bvh.node_lowers[index];
    BVHPackedNodeHalf& upper = bvh.node_uppers[index];

    if (lower.b)
    {
        // update leaf from items
        bounds3 bound;
        for (int item_counter = lower.i; item_counter < upper.i; item_counter++)
        {
            const int item = bvh.primitive_indices[item_counter];
            bound.add_bounds(bvh.item_lowers[item], bvh.item_uppers[item]);
        }

        reinterpret_cast<vec3&>(lower) = bound.lower;
        reinterpret_cast<vec3&>(upper) = bound.upper;
    }
    else
    {
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

void bvh_refit_host(BVH& bvh)
{
    bvh_refit_recursive(bvh, 0);
}
void bvh_rebuild_host(BVH& bvh, int constructor_type)
{
    TopDownBVHBuilder builder;
    builder.rebuild(bvh, constructor_type);
}

} // namespace wp


// making the class accessible from python


namespace 
{
    // host-side copy of bvh descriptors, maps GPU bvh address (id) to a CPU desc
    std::map<uint64_t, BVH> g_bvh_descriptors;

} // anonymous namespace


namespace wp
{

bool bvh_get_descriptor(uint64_t id, BVH& bvh)
{
    const auto& iter = g_bvh_descriptors.find(id);
    if (iter == g_bvh_descriptors.end())
        return false;
    else
        bvh = iter->second;
        return true;
}

void bvh_add_descriptor(uint64_t id, const BVH& bvh)
{
    g_bvh_descriptors[id] = bvh;
    
}

void bvh_rem_descriptor(uint64_t id)
{
    g_bvh_descriptors.erase(id);

}


// create in-place given existing descriptor
void bvh_create_host(vec3* lowers, vec3* uppers, int num_items, int constructor_type, int* groups, BVH& bvh)
{
    memset(&bvh, 0, sizeof(BVH));

    bvh.item_lowers = lowers;
    bvh.item_uppers = uppers;
    bvh.num_items = num_items;

    TopDownBVHBuilder builder;
    builder.build(bvh, lowers, uppers, num_items, constructor_type, groups);
}

void bvh_destroy_host(BVH& bvh)
{
    delete[] bvh.node_lowers;
    delete[] bvh.node_uppers;
    delete[] bvh.node_parents;
    delete[] bvh.primitive_indices;
    delete[] bvh.keys;
    delete[] bvh.root;

    bvh.node_lowers = nullptr;
    bvh.node_uppers = nullptr;
    bvh.node_parents = nullptr;
    bvh.primitive_indices = nullptr;
    bvh.keys = nullptr;
    bvh.root = nullptr;

    bvh.max_nodes = 0;
    bvh.num_items = 0;
}

} // namespace wp

uint64_t wp_bvh_create_host(vec3* lowers, vec3* uppers, int num_items, int constructor_type, int* groups)
{
    BVH* bvh = new BVH();
    wp::bvh_create_host(lowers, uppers, num_items, constructor_type, groups, *bvh);

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

uint64_t wp_bvh_create_device(void* context, wp::vec3* lowers, wp::vec3* uppers, int num_items, int constructor_type, int* groups) { return 0; }
void wp_bvh_refit_device(uint64_t id) {}
void wp_bvh_destroy_device(uint64_t id) {}
void wp_bvh_rebuild_device(uint64_t id) {}

#endif // !WP_ENABLE_CUDA
