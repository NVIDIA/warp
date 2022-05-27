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


namespace wp
{


/////////////////////////////////////////////////////////////////////////////////////////////

class MedianBVHBuilder
{	
public:

    void build(BVH& bvh, const bounds3* items, int n);

private:

    bounds3 calc_bounds(const bounds3* bounds, const int* indices, int start, int end);

    int partition_median(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds);
    int partition_midpoint(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds);
    int partition_sah(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds);

    int build_recursive(BVH& bvh, const bounds3* bounds, int* indices, int start, int end, int depth, int parent);
};

//////////////////////////////////////////////////////////////////////

void MedianBVHBuilder::build(BVH& bvh, const bounds3* items, int n)
{
    memset(&bvh, 0, sizeof(BVH));

    bvh.max_nodes = 2*n-1;

    bvh.node_lowers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.node_uppers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.node_parents = new int[bvh.max_nodes];

    bvh.num_nodes = 0;

    // root is always in first slot for top down builders
    bvh.root = 0;

    if (n == 0)
        return;
    
    std::vector<int> indices(n);
    for (int i=0; i < n; ++i)
        indices[i] = i;

    build_recursive(bvh, items, &indices[0], 0, n, 0, -1);
}


bounds3 MedianBVHBuilder::calc_bounds(const bounds3* bounds, const int* indices, int start, int end)
{
    bounds3 u;

    for (int i=start; i < end; ++i)
        u = bounds_union(u, bounds[indices[i]]);

    return u;
}

struct PartitionPredicateMedian
{
    PartitionPredicateMedian(const bounds3* bounds, int a) : bounds(bounds), axis(a) {}

    bool operator()(int a, int b) const
    {
        return bounds[a].center()[axis] < bounds[b].center()[axis];
    }

    const bounds3* bounds;
    int axis;
};


int MedianBVHBuilder::partition_median(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds)
{
    assert(end-start >= 2);

    vec3 edges = range_bounds.edges();

    int axis = longest_axis(edges);

    const int k = (start+end)/2;

    std::nth_element(&indices[start], &indices[k], &indices[end], PartitionPredicateMedian(&bounds[0], axis));

    return k;
}	
    
struct PartitionPredictateMidPoint
{
    PartitionPredictateMidPoint(const bounds3* bounds, int a, float m) : bounds(bounds), axis(a), mid(m) {}

    bool operator()(int index) const 
    {
        return bounds[index].center()[axis] <= mid;
    }

    const bounds3* bounds;
    int axis;
    float mid;
};


int MedianBVHBuilder::partition_midpoint(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds)
{
    assert(end-start >= 2);

    vec3 edges = range_bounds.edges();
    vec3 center = range_bounds.center();

    int axis = longest_axis(edges);
    float mid = center[axis];

    int* upper = std::partition(indices+start, indices+end, PartitionPredictateMidPoint(&bounds[0], axis, mid));

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

int MedianBVHBuilder::build_recursive(BVH& bvh, const bounds3* bounds, int* indices, int start, int end, int depth, int parent)
{
    assert(start < end);

    const int n = end-start;
    const int node_index = bvh.num_nodes++;

    assert(node_index < bvh.max_nodes);

    if (depth > bvh.max_depth)
        bvh.max_depth = depth;

    bounds3 b = calc_bounds(bounds, indices, start, end);
    
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
        int split = partition_median(bounds, indices, start, end, b);
        //int split = partition_sah(bounds, indices, start, end, b);

        if (split == start || split == end)
        {
            // partitioning failed, split down the middle
            split = (start+end)/2;
        }
    
        int left_child = build_recursive(bvh, bounds, indices, start, split, depth+1, node_index);
        int right_child = build_recursive(bvh, bounds, indices, split, end, depth+1, node_index);
        
        bvh.node_lowers[node_index] = make_node(b.lower, left_child, false);
        bvh.node_uppers[node_index] = make_node(b.upper, right_child, false);
        bvh.node_parents[node_index] = parent;
    }

    return node_index;
}

class LinearBVHBuilderCPU
{
public:
    
    void build(BVH& bvh, const bounds3* items, int n);

private:

    // calculate Morton codes
    struct KeyIndexPair
    {
        uint32_t key;
        int index;

        inline bool operator < (const KeyIndexPair& rhs) const { return key < rhs.key; }
    };	

    bounds3 calc_bounds(const bounds3* bounds, const KeyIndexPair* keys, int start, int end);
    int find_split(const KeyIndexPair* pairs, int start, int end);
    int build_recursive(BVH& bvh, const KeyIndexPair* keys, const bounds3* bounds, int start, int end, int depth);

};


// disable std::sort workaround for macOS error
#if 0
void LinearBVHBuilderCPU::build(BVH& bvh, const bounds3* items, int n)
{
	memset(&bvh, 0, sizeof(BVH));

	bvh.max_nodes = 2*n-1;

	bvh.node_lowers = new BVHPackedNodeHalf[bvh.max_nodes];
	bvh.node_uppers = new BVHPackedNodeHalf[bvh.max_nodes];
	bvh.num_nodes = 0;

	// root is always in first slot for top down builders
	bvh.root = 0;

	std::vector<KeyIndexPair> keys;
	keys.reserve(n);

	bounds3 totalbounds3;
	for (int i=0; i < n; ++i)
		totalbounds3 = bounds_union(totalbounds3, items[i]);

	// ensure non-zero edge length in all dimensions
	totalbounds3.expand(0.001f);

	vec3 edges = totalbounds3.edges();
	vec3 invEdges = cw_div(vec3(1.0f), edges);

	for (int i=0; i < n; ++i)
	{
		vec3 center = items[i].center();
		vec3 local = cw_mul(center-totalbounds3.lower, invEdges);

		KeyIndexPair l;
		l.key = morton3<1024>(local.x, local.y, local.z);
		l.index = i;

		keys.push_back(l);
	}

	// sort by key
	std::sort(keys.begin(), keys.end());

	build_recursive(bvh, &keys[0], items,  0, n, 0);

	printf("Created BVH for %d items with %d nodes, max depth of %d\n", n, bvh.num_nodes, bvh.max_depth);
}
#endif

inline bounds3 LinearBVHBuilderCPU::calc_bounds(const bounds3* bounds, const KeyIndexPair* keys, int start, int end)
{
	bounds3 u;

	for (int i=start; i < end; ++i)
		u = bounds_union(u, bounds[keys[i].index]);

	return u;
}

inline int LinearBVHBuilderCPU::find_split(const KeyIndexPair* pairs, int start, int end)
{
	if (pairs[start].key == pairs[end-1].key)
		return (start+end)/2;

	// find split point between keys, xor here means all bits 
	// of the result are zero up until the first differing bit
	int common_prefix = clz(pairs[start].key ^ pairs[end-1].key);

	// use binary search to find the point at which this bit changes
	// from zero to a 1		
	const int mask = 1 << (31-common_prefix);

	while (end-start > 0)
	{
		int index = (start+end)/2;

		if (pairs[index].key&mask)
		{
			end = index;
		}
		else
			start = index+1;
	}

	assert(start == end);

	return start;
}

int LinearBVHBuilderCPU::build_recursive(BVH& bvh, const KeyIndexPair* keys, const bounds3* bounds, int start, int end, int depth)
{
	assert(start < end);

	const int n = end-start;
	const int nodeIndex = bvh.num_nodes++;

	assert(nodeIndex < bvh.max_nodes);

	if (depth > bvh.max_depth)
		bvh.max_depth = depth;

	bounds3 b = calc_bounds(bounds, keys, start, end);
		
	const int kMaxItemsPerLeaf = 1;

	if (n <= kMaxItemsPerLeaf)
	{
		bvh.node_lowers[nodeIndex] = make_node(b.lower, keys[start].index, true);
		bvh.node_uppers[nodeIndex] = make_node(b.upper, keys[start].index, false);
	}
	else
	{
		int split = find_split(keys, start, end);
		
		int leftChild = build_recursive(bvh, keys, bounds, start, split, depth+1);
		int rightChild = build_recursive(bvh, keys, bounds, split, end, depth+1);
			
		bvh.node_lowers[nodeIndex] = make_node(b.lower, leftChild, false);
		bvh.node_uppers[nodeIndex] = make_node(b.upper, rightChild, false);		
	}

	return nodeIndex;
}



// create only happens on host currently, use bvh_clone() to transfer BVH To device
BVH bvh_create(const bounds3* bounds, int num_bounds)
{
    BVH bvh;
    memset(&bvh, 0, sizeof(bvh));

    MedianBVHBuilder builder;
    //LinearBVHBuilderCPU builder;
    builder.build(bvh, bounds, num_bounds);

    return bvh;
}

void bvh_destroy_host(BVH& bvh)
{
    delete[] bvh.node_lowers;
    delete[] bvh.node_uppers;
    delete[] bvh.node_parents;

    bvh.node_lowers = 0;
    bvh.node_uppers = 0;
    bvh.max_nodes = 0;
    bvh.num_nodes = 0;
}

void bvh_destroy_device(BVH& bvh)
{
    free_device(bvh.node_lowers); bvh.node_lowers = NULL;
    free_device(bvh.node_uppers); bvh.node_uppers = NULL;
    free_device(bvh.node_parents); bvh.node_parents = NULL;
    free_device(bvh.node_counts); bvh.node_counts = NULL;
}


BVH bvh_clone(const BVH& bvh_host)
{
    BVH bvh_device = bvh_host;

    bvh_device.node_lowers = (BVHPackedNodeHalf*)alloc_device(sizeof(BVHPackedNodeHalf)*bvh_host.max_nodes);
    bvh_device.node_uppers = (BVHPackedNodeHalf*)alloc_device(sizeof(BVHPackedNodeHalf)*bvh_host.max_nodes);
    bvh_device.node_parents = (int*)alloc_device(sizeof(int)*bvh_host.max_nodes);
    bvh_device.node_counts = (int*)alloc_device(sizeof(int)*bvh_host.max_nodes);

    // copy host data to device
    memcpy_h2d(bvh_device.node_lowers, bvh_host.node_lowers, sizeof(BVHPackedNodeHalf)*bvh_host.max_nodes);
    memcpy_h2d(bvh_device.node_uppers, bvh_host.node_uppers, sizeof(BVHPackedNodeHalf)*bvh_host.max_nodes);
    memcpy_h2d(bvh_device.node_parents, bvh_host.node_parents, sizeof(int)*bvh_host.max_nodes);

    return bvh_device;
}

void bvh_refit_recursive(BVH& bvh, int index, const bounds3* bounds)
{
    BVHPackedNodeHalf& lower = bvh.node_lowers[index];
    BVHPackedNodeHalf& upper = bvh.node_uppers[index];

    if (lower.b)
    {
        const int leaf_index = lower.i;

        (vec3&)lower = bounds[leaf_index].lower;
        (vec3&)upper = bounds[leaf_index].upper;
    }
    else
    {
        int left_index = lower.i;
        int right_index = upper.i;

        bvh_refit_recursive(bvh, left_index, bounds);
        bvh_refit_recursive(bvh, right_index, bounds);

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

void bvh_refit_host(BVH& bvh, const bounds3* b)
{
    bvh_refit_recursive(bvh, 0, b);
}


} // namespace wp



