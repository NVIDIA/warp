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

#include "bvh.h"
#include "warp.h"
#include "cuda_util.h"

#include <map>

using namespace wp;

namespace wp
{


/////////////////////////////////////////////////////////////////////////////////////////////

class TopDownBVHBuilder
{	
public:

    void build(BVH& bvh, const vec3* lowers, const vec3* uppers, int n, int in_constructor_type);

private:

    bounds3 calc_bounds(const vec3* lowers, const vec3* uppers, const int* indices, int start, int end);

    int partition_median(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds);
    int partition_midpoint(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds);
    float partition_sah(BVH& bvh, const vec3* lowers, const vec3* uppers,
       int start, int end, bounds3 range_bounds, int& split_axis);

    int build_recursive(BVH& bvh, const vec3* lowers, const vec3* uppers, int start, int end, int depth, int parent);

    int constructor_type = -1;
};

//////////////////////////////////////////////////////////////////////

void TopDownBVHBuilder::build(BVH& bvh, const vec3* lowers, const vec3* uppers, int n, int in_constructor_type)
{
    constructor_type = in_constructor_type;
    if (constructor_type != BVH_CONSTRUCTOR_SAH && constructor_type != BVH_CONSTRUCTOR_MEDIAN)
    {
        printf("Unrecognized Constructor type: %d! For CPU constructor it should be either SAH (%d) or Median (%d)!\n",
            constructor_type, BVH_CONSTRUCTOR_SAH, BVH_CONSTRUCTOR_MEDIAN);
        return;
    }

    bvh.max_depth = 0;
    bvh.max_nodes = 2*n-1;

    bvh.node_lowers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.node_uppers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.node_parents = new int[bvh.max_nodes];
    bvh.node_counts = NULL;

    // root is always in first slot for top down builders
    bvh.root = new int[1];
    bvh.root[0] = 0;

    if (n == 0)
        return;
    
    bvh.primitive_indices = new int[n];
    for (int i = 0; i < n; ++i)
        bvh.primitive_indices[i] = i;

    build_recursive(bvh, lowers, uppers,  0, n, 0, -1);
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

float TopDownBVHBuilder::partition_sah(BVH& bvh, const vec3* lowers, const vec3* uppers, int start, int end, bounds3 range_bounds, int& split_axis)
{
    int buckets_counts[SAH_NUM_BUCKETS];
    bounds3 buckets[SAH_NUM_BUCKETS];
    float left_areas[SAH_NUM_BUCKETS - 1];
    float right_areas[SAH_NUM_BUCKETS - 1];

    assert(end - start >= 2);

    int n = end - start;
    vec3 edges = range_bounds.edges();

    bounds3 b = calc_bounds(lowers, uppers, bvh.primitive_indices, start, end);

    split_axis = longest_axis(edges);

    // compute each bucket
    float range_start = b.lower[split_axis];
    float range_end = b.upper[split_axis];

    std::fill(buckets_counts, buckets_counts + SAH_NUM_BUCKETS, 0);
    for (int item_idx = start; item_idx < end; item_idx++)
    {
        vec3 item_center = 0.5f * (lowers[bvh.primitive_indices[item_idx]] + uppers[bvh.primitive_indices[item_idx]]);
        int bucket_idx = SAH_NUM_BUCKETS * (item_center[split_axis] - range_start) / (range_end - range_start);
        assert(bucket_idx >= 0 && bucket_idx <= SAH_NUM_BUCKETS);
        // one of them will have the range_end, we put it into the last bucket
        bucket_idx = bucket_idx < SAH_NUM_BUCKETS ? bucket_idx : SAH_NUM_BUCKETS - 1;

        bounds3 item_bound(lowers[bvh.primitive_indices[item_idx]], uppers[bvh.primitive_indices[item_idx]]);

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

    // printf("start %d end %d\n", start, end);

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
            float split_point = partition_sah(bvh, lowers, uppers, start, end, b, split_axis);
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

        (vec3&)lower = bound.lower;
        (vec3&)upper = bound.upper;
    }
    else
    {
        int left_index = lower.i;
        int right_index = upper.i;

        bvh_refit_recursive(bvh, left_index);
        bvh_refit_recursive(bvh, right_index);

        // compute union of children
        const vec3& left_lower = (vec3&)bvh.node_lowers[left_index];
        const vec3& left_upper = (vec3&)bvh.node_uppers[left_index];

        const vec3& right_lower = (vec3&)bvh.node_lowers[right_index];
        const vec3& right_upper = (vec3&)bvh.node_uppers[right_index];

        // union of child bounds
        vec3 new_lower = min(left_lower, right_lower);
        vec3 new_upper = max(left_upper, right_upper);
        
        // write new BVH nodes
        (vec3&)lower = new_lower;
        (vec3&)upper = new_upper;
    }
}

void bvh_refit_host(BVH& bvh)
{
    bvh_refit_recursive(bvh, 0);
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
void bvh_create_host(vec3* lowers, vec3* uppers, int num_items, int constructor_type, BVH& bvh)
{
    memset(&bvh, 0, sizeof(BVH));

    bvh.item_lowers = lowers;
    bvh.item_uppers = uppers;
    bvh.num_items = num_items;

    TopDownBVHBuilder builder;
    builder.build(bvh, lowers, uppers, num_items, constructor_type);
}

void bvh_destroy_host(BVH& bvh)
{
    delete[] bvh.node_lowers;
    delete[] bvh.node_uppers;
    delete[] bvh.node_parents;
    delete[] bvh.primitive_indices;
    delete[] bvh.root;

    bvh.node_lowers = NULL;
    bvh.node_uppers = NULL;
    bvh.node_parents = NULL;
    bvh.primitive_indices = NULL;
    bvh.root = NULL;

    bvh.max_nodes = 0;
    bvh.num_items = 0;
}

} // namespace wp

uint64_t wp_bvh_create_host(vec3* lowers, vec3* uppers, int num_items, int constructor_type)
{
    BVH* bvh = new BVH();
    wp::bvh_create_host(lowers, uppers, num_items, constructor_type, *bvh);

    return (uint64_t)bvh;
}

void wp_bvh_refit_host(uint64_t id)
{
    BVH* bvh = (BVH*)(id);
    wp::bvh_refit_host(*bvh);
}

void wp_bvh_destroy_host(uint64_t id)
{
    BVH* bvh = (BVH*)(id);
    wp::bvh_destroy_host(*bvh);
    delete bvh;
}


// stubs for non-CUDA platforms
#if !WP_ENABLE_CUDA

uint64_t wp_bvh_create_device(void* context, wp::vec3* lowers, wp::vec3* uppers, int num_items, int constructor_type) { return 0; }
void wp_bvh_refit_device(uint64_t id) {}
void wp_bvh_destroy_device(uint64_t id) {}

#endif // !WP_ENABLE_CUDA
