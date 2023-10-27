/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "builtin.h"
#include "intersect.h"

namespace wp
{

struct bounds3
{
	CUDA_CALLABLE inline bounds3() : lower( FLT_MAX)
						           , upper(-FLT_MAX) {}

	CUDA_CALLABLE inline bounds3(const vec3& lower, const vec3& upper) : lower(lower), upper(upper) {}

	CUDA_CALLABLE inline vec3 center() const { return 0.5f*(lower+upper); }
	CUDA_CALLABLE inline vec3 edges() const { return upper-lower; }

	CUDA_CALLABLE inline void expand(float r)
	{
		lower -= vec3(r);
		upper += vec3(r);
	}

	CUDA_CALLABLE inline void expand(const vec3& r)
	{
		lower -= r;
		upper += r;
	}

	CUDA_CALLABLE inline bool empty() const { return lower[0] >= upper[0] || lower[1] >= upper[1] || lower[2] >= upper[2]; }

	CUDA_CALLABLE inline bool overlaps(const vec3& p) const
	{
		if (p[0] < lower[0] ||
			p[1] < lower[1] ||
			p[2] < lower[2] ||
			p[0] > upper[0] ||
			p[1] > upper[1] ||
			p[2] > upper[2])
		{
			return false;
		}
		else
		{
			return true;
		}
	}

	CUDA_CALLABLE inline bool overlaps(const bounds3& b) const
	{
		if (lower[0] > b.upper[0] ||
			lower[1] > b.upper[1] ||
			lower[2] > b.upper[2] ||
			upper[0] < b.lower[0] ||
			upper[1] < b.lower[1] ||
			upper[2] < b.lower[2])
		{
			return false;
		}
		else
		{
			return true;
		}
	}

	CUDA_CALLABLE inline void add_point(const vec3& p)
	{
		lower = min(lower, p);
		upper = max(upper, p);
	}

	CUDA_CALLABLE inline float area() const
	{
		vec3 e = upper-lower;
		return 2.0f*(e[0]*e[1] + e[0]*e[2] + e[1]*e[2]);
	}

	vec3 lower;
	vec3 upper;
};

CUDA_CALLABLE inline bounds3 bounds_union(const bounds3& a, const vec3& b) 
{
	return bounds3(min(a.lower, b), max(a.upper, b));
}

CUDA_CALLABLE inline bounds3 bounds_union(const bounds3& a, const bounds3& b) 
{
	return bounds3(min(a.lower, b.lower), max(a.upper, b.upper));
}

CUDA_CALLABLE inline bounds3 bounds_intersection(const bounds3& a, const bounds3& b)
{
	return bounds3(max(a.lower, b.lower), min(a.upper, b.upper));
}

struct BVHPackedNodeHalf
{
	float x;
	float y;
	float z;
	unsigned int i : 31;
	unsigned int b : 1;
};

struct BVH
{		
    BVHPackedNodeHalf* node_lowers;
    BVHPackedNodeHalf* node_uppers;

	// used for fast refits
	int* node_parents;
	int* node_counts;
	
	int max_depth;
	int max_nodes;
	int num_nodes;
	
	// pointer (CPU or GPU) to a single integer index in node_lowers, node_uppers
	// representing the root of the tree, this is not always the first node
	// for bottom-up builders
	int* root;

	// item bounds are not owned by the BVH but by the caller
    vec3* item_lowers;
	vec3* item_uppers;
	int num_items;

	// cuda context
	void* context;
};

CUDA_CALLABLE inline BVHPackedNodeHalf make_node(const vec3& bound, int child, bool leaf)
{
    BVHPackedNodeHalf n;
    n.x = bound[0];
    n.y = bound[1];
    n.z = bound[2];
    n.i = (unsigned int)child;
    n.b = (unsigned int)(leaf?1:0);

    return n;
}

// variation of make_node through volatile pointers used in build_hierarchy
CUDA_CALLABLE inline void make_node(volatile BVHPackedNodeHalf* n, const vec3& bound, int child, bool leaf)
{
    n->x = bound[0];
    n->y = bound[1];
    n->z = bound[2];
    n->i = (unsigned int)child;
    n->b = (unsigned int)(leaf?1:0);
}

CUDA_CALLABLE inline int clz(int x)
{
    int n;
    if (x == 0) return 32;
    for (n = 0; ((x & 0x80000000) == 0); n++, x <<= 1);
    return n;
}

CUDA_CALLABLE inline uint32_t part1by2(uint32_t n)
{
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n <<  8)) & 0x0300f00f;
    n = (n ^ (n <<  4)) & 0x030c30c3;
    n = (n ^ (n <<  2)) & 0x09249249;

    return n;
}

// Takes values in the range [0, 1] and assigns an index based Morton codes of length 3*lwp2(dim) bits 
template <int dim>
CUDA_CALLABLE inline uint32_t morton3(float x, float y, float z)
{
    uint32_t ux = clamp(int(x*dim), 0, dim-1);
    uint32_t uy = clamp(int(y*dim), 0, dim-1);
    uint32_t uz = clamp(int(z*dim), 0, dim-1);

    return (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux);
}

// making the class accessible from python

CUDA_CALLABLE inline BVH bvh_get(uint64_t id)
{
    return *(BVH*)(id);
}

CUDA_CALLABLE inline int bvh_get_num_bounds(uint64_t id)
{
	BVH bvh = bvh_get(id);
	return bvh.num_items;
}


// stores state required to traverse the BVH nodes that 
// overlap with a query AABB.
struct bvh_query_t
{
    CUDA_CALLABLE bvh_query_t()
    {
    }
    CUDA_CALLABLE bvh_query_t(int)
    {
    } // for backward pass

    BVH bvh;

	// BVH traversal stack:
	int stack[32];
	int count;

    // inputs
	bool is_ray;
    wp::vec3 input_lower;	// start for ray
    wp::vec3 input_upper;	// dir for ray

	int bounds_nr;
};


CUDA_CALLABLE inline bvh_query_t bvh_query(
    uint64_t id, bool is_ray, const vec3& lower, const vec3& upper)
{
    // This routine traverses the BVH tree until it finds
	// the first overlapping bound. 

    // initialize empty
	bvh_query_t query;

	query.bounds_nr = -1;

	BVH bvh = bvh_get(id);

	query.bvh = bvh;
	query.is_ray = is_ray;

    // optimization: make the latest	
	query.stack[0] = *bvh.root;
	query.count = 1;
    query.input_lower = lower;
    query.input_upper = upper;

    wp::bounds3 input_bounds(query.input_lower, query.input_upper);

    // Navigate through the bvh, find the first overlapping leaf node.
    while (query.count)
    {
		const int node_index = query.stack[--query.count];

		BVHPackedNodeHalf node_lower = bvh.node_lowers[node_index];
		BVHPackedNodeHalf node_upper = bvh.node_uppers[node_index];

		wp::vec3 lower_pos(node_lower.x, node_lower.y, node_lower.z);
		wp::vec3 upper_pos(node_upper.x, node_upper.y, node_upper.z);
        wp::bounds3 current_bounds(lower_pos, upper_pos);

		if (query.is_ray) 
		{
			float t = 0.0f;
			if (!intersect_ray_aabb(query.input_lower, query.input_upper, current_bounds.lower, current_bounds.upper, t))
			// Skip this box, it doesn't overlap with our ray.
				continue;
		}
		else 
		{
	        if (!input_bounds.overlaps(current_bounds))
				// Skip this box, it doesn't overlap with our target box.
				continue;
		}

		const int left_index = node_lower.i;
		const int right_index = node_upper.i;

        // Make bounds from this AABB
        if (node_lower.b)
        {
			// found very first leaf index.
			// Back up one level and return 
			query.stack[query.count++] = node_index;
			return query;
        }
        else
        {	
		  query.stack[query.count++] = left_index;
		  query.stack[query.count++] = right_index;
		}
	}	

	return query;
}

CUDA_CALLABLE inline bvh_query_t bvh_query_aabb(
    uint64_t id, const vec3& lower, const vec3& upper)
{
	return bvh_query(id, false, lower, upper);
}


CUDA_CALLABLE inline bvh_query_t bvh_query_ray(
    uint64_t id, const vec3& start, const vec3& dir)
{
	return bvh_query(id, true, start, dir);
}

//Stub
CUDA_CALLABLE inline void adj_bvh_query_aabb(uint64_t id, const vec3& lower, const vec3& upper,
											   uint64_t, vec3&, vec3&, bvh_query_t&)
{
}


CUDA_CALLABLE inline void adj_bvh_query_ray(uint64_t id, const vec3& start, const vec3& dir,
											   uint64_t, vec3&, vec3&, bvh_query_t&)
{
}


CUDA_CALLABLE inline bool bvh_query_next(bvh_query_t& query, int& index)
{
    BVH bvh = query.bvh;
	
	wp::bounds3 input_bounds(query.input_lower, query.input_upper);

    // Navigate through the bvh, find the first overlapping leaf node.
    while (query.count)
    {
        const int node_index = query.stack[--query.count];
        BVHPackedNodeHalf node_lower = bvh.node_lowers[node_index];
        BVHPackedNodeHalf node_upper = bvh.node_uppers[node_index];

        wp::vec3 lower_pos(node_lower.x, node_lower.y, node_lower.z);
        wp::vec3 upper_pos(node_upper.x, node_upper.y, node_upper.z);
        wp::bounds3 current_bounds(lower_pos, upper_pos);

		if (query.is_ray) 
		{
			float t = 0.0f;
			if (!intersect_ray_aabb(query.input_lower, query.input_upper, current_bounds.lower, current_bounds.upper, t))
			// Skip this box, it doesn't overlap with our ray.
				continue;
		}
		else {
	        if (!input_bounds.overlaps(current_bounds))
				// Skip this box, it doesn't overlap with our target box.
				continue;
		}

        const int left_index = node_lower.i;
        const int right_index = node_upper.i;

        if (node_lower.b)
        {
            // found leaf
            query.bounds_nr = left_index;
			index = left_index;
            return true;
        }
        else
        {

            query.stack[query.count++] = left_index;
            query.stack[query.count++] = right_index;
        }
    }
    return false;
}


CUDA_CALLABLE inline int iter_next(bvh_query_t& query)
{
    return query.bounds_nr;
}

CUDA_CALLABLE inline bool iter_cmp(bvh_query_t& query)
{
    bool finished = bvh_query_next(query, query.bounds_nr);
    return finished;
}

CUDA_CALLABLE inline bvh_query_t iter_reverse(const bvh_query_t& query)
{
    // can't reverse BVH queries, users should not rely on traversal ordering
    return query;
}


// stub
CUDA_CALLABLE inline void adj_bvh_query_next(bvh_query_t& query, int& index, bvh_query_t&, int&, bool&) 
{

}

CUDA_CALLABLE bool bvh_get_descriptor(uint64_t id, BVH& bvh);
CUDA_CALLABLE void bvh_add_descriptor(uint64_t id, const BVH& bvh);
CUDA_CALLABLE void bvh_rem_descriptor(uint64_t id);

#if !__CUDA_ARCH__

void bvh_destroy_host(wp::BVH& bvh);
void bvh_refit_host(wp::BVH& bvh);

void bvh_destroy_device(wp::BVH& bvh);
void bvh_refit_device(uint64_t id);

#endif

} // namespace wp

