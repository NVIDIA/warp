/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
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

class MedianBVHBuilder
{	
public:

    void build(BVH& bvh, const vec3* lowers, const vec3* uppers, int n);

private:

    bounds3 calc_bounds(const vec3* lowers, const vec3* uppers, const int* indices, int start, int end);

    int partition_median(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds);
    int partition_midpoint(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds);
    int partition_sah(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds);

    int build_recursive(BVH& bvh, const vec3* lowers, const vec3* uppers, int* indices, int start, int end, int depth, int parent);
};

//////////////////////////////////////////////////////////////////////

void MedianBVHBuilder::build(BVH& bvh, const vec3* lowers, const vec3* uppers, int n)
{
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
    
    std::vector<int> indices(n);
    for (int i=0; i < n; ++i)
        indices[i] = i;

    build_recursive(bvh, lowers, uppers, &indices[0], 0, n, 0, -1);
}


bounds3 MedianBVHBuilder::calc_bounds(const vec3* lowers, const vec3* uppers, const int* indices, int start, int end)
{
    bounds3 u;

    for (int i=start; i < end; ++i)
    {
        u.add_point(lowers[indices[i]]);
        u.add_point(uppers[indices[i]]);
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


int MedianBVHBuilder::partition_median(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds)
{
    assert(end-start >= 2);

    vec3 edges = range_bounds.edges();

    int axis = longest_axis(edges);

    const int k = (start+end)/2;

    std::nth_element(&indices[start], &indices[k], &indices[end], PartitionPredicateMedian(lowers, uppers, axis));

    return k;
}	
    
struct PartitionPredictateMidPoint
{
    PartitionPredictateMidPoint(const vec3* lowers, const vec3* uppers, int a, float m) : lowers(lowers), uppers(uppers), axis(a), mid(m) {}

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


int MedianBVHBuilder::partition_midpoint(const vec3* lowers, const vec3* uppers, int* indices, int start, int end, bounds3 range_bounds)
{
    assert(end-start >= 2);

    vec3 edges = range_bounds.edges();
    vec3 center = range_bounds.center();

    int axis = longest_axis(edges);
    float mid = center[axis];

    int* upper = std::partition(indices+start, indices+end, PartitionPredictateMidPoint(lowers, uppers, axis, mid));

    int k = upper-indices;

    // if we failed to split items then just split in the middle
    if (k == start || k == end)
        k = (start+end)/2;

    return k;
}

// disable std::sort workaround for macOS error
#if 0
int MedianBVHBuilder::partition_sah(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds)
{
    assert(end-start >= 2);

    int n = end-start;
    vec3 edges = range_bounds.edges();

    int longestAxis = longest_axis(edges);

    // sort along longest axis
    std::sort(&indices[0]+start, &indices[0]+end, PartitionPredicateMedian(&bounds[0], longestAxis));

    // total area for range from [0, split]
    std::vector<float> left_areas(n);
    // total area for range from (split, end]
    std::vector<float> right_areas(n);

    bounds3 left;
    bounds3 right;

    // build cumulative bounds and area from left and right
    for (int i=0; i < n; ++i)
    {
        left = bounds_union(left, bounds[indices[start+i]]);
        right = bounds_union(right, bounds[indices[end-i-1]]);

        left_areas[i] = left.area();
        right_areas[n-i-1] = right.area();
    }

    float invTotalArea = 1.0f/range_bounds.area();

    // find split point i that minimizes area(left[i]) + area(right[i])
    int minSplit = 0;
    float minCost = FLT_MAX;

    for (int i=0; i < n; ++i)
    {
        float pBelow = left_areas[i]*invTotalArea;
        float pAbove = right_areas[i]*invTotalArea;

        float cost = pBelow*i + pAbove*(n-i);

        if (cost < minCost)
        {
            minCost = cost;
            minSplit = i;
        }
    }

    return start + minSplit + 1;
}
#endif

int MedianBVHBuilder::build_recursive(BVH& bvh, const vec3* lowers, const vec3* uppers, int* indices, int start, int end, int depth, int parent)
{
    assert(start < end);

    const int n = end-start;
    const int node_index = bvh.num_nodes++;

    assert(node_index < bvh.max_nodes);

    if (depth > bvh.max_depth)
        bvh.max_depth = depth;

    bounds3 b = calc_bounds(lowers, uppers, indices, start, end);
    
    const int kMaxItemsPerLeaf = 1;

    if (n <= kMaxItemsPerLeaf)
    {
        bvh.node_lowers[node_index] = make_node(b.lower, indices[start], true);
        bvh.node_uppers[node_index] = make_node(b.upper, indices[start], false);
        bvh.node_parents[node_index] = parent;
    }
    else    
    {
        //int split = partition_midpoint(bounds, indices, start, end, b);
        int split = partition_median(lowers, uppers, indices, start, end, b);
        //int split = partition_sah(bounds, indices, start, end, b);

        if (split == start || split == end)
        {
            // partitioning failed, split down the middle
            split = (start+end)/2;
        }
    
        int left_child = build_recursive(bvh, lowers, uppers, indices, start, split, depth+1, node_index);
        int right_child = build_recursive(bvh, lowers, uppers, indices, split, end, depth+1, node_index);
        
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
        const int leaf_index = lower.i;

        // update leaf from items
        (vec3&)lower = bvh.item_lowers[leaf_index];
        (vec3&)upper = bvh.item_uppers[leaf_index];
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


void bvh_destroy_host(BVH& bvh)
{
    delete[] bvh.node_lowers;
    delete[] bvh.node_uppers;
    delete[] bvh.node_parents;
    delete[] bvh.root;

    bvh.node_lowers = NULL;
    bvh.node_uppers = NULL;
    bvh.node_parents = NULL;
    bvh.root = NULL;

    bvh.max_nodes = 0;
    bvh.num_items = 0;
}

} // namespace wp

uint64_t bvh_create_host(vec3* lowers, vec3* uppers, int num_items)
{
    BVH* bvh = new BVH();
    memset(bvh, 0, sizeof(BVH));

    bvh->context = NULL;

    bvh->item_lowers = lowers;
    bvh->item_uppers = uppers;
    bvh->num_items = num_items;

    MedianBVHBuilder builder;
    builder.build(*bvh, lowers, uppers, num_items);

    return (uint64_t)bvh;
}

void bvh_refit_host(uint64_t id)
{
    BVH* bvh = (BVH*)(id);
    bvh_refit_host(*bvh);
}

void bvh_destroy_host(uint64_t id)
{
    BVH* bvh = (BVH*)(id);
    bvh_destroy_host(*bvh);
    delete bvh;
}


// stubs for non-CUDA platforms
#if !WP_ENABLE_CUDA

uint64_t bvh_create_device(void* context, wp::vec3* lowers, wp::vec3* uppers, int num_items) { return 0; }
void bvh_refit_device(uint64_t id) {}
void bvh_destroy_device(uint64_t id) {}

#endif // !WP_ENABLE_CUDA
