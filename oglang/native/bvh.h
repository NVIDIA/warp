#pragma once

#include "core.h"

namespace og
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

	CUDA_CALLABLE inline bool empty() const { return lower.x >= upper.x || lower.y >= upper.y || lower.z >= upper.z; }

	CUDA_CALLABLE inline bool overlaps(const vec3& p) const
	{
		if (p.x < lower.x ||
			p.y < lower.y ||
			p.z < lower.z ||
			p.x > upper.x ||
			p.y > upper.y ||
			p.z > upper.z)
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
		if (lower.x > b.upper.x ||
			lower.y > b.upper.y ||
			lower.z > b.upper.z ||
			upper.x < b.lower.x ||
			upper.y < b.lower.y ||
			upper.z < b.lower.z)
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
		lower = og::min(lower, p);
		upper = og::max(upper, p);
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

CUDA_CALLABLE inline float bounds_surface_area(const bounds3& b)
{
	vec3 e = b.upper-b.lower;
	return 2.0f*(e.x*e.y + e.x*e.z + e.y*e.z);
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

	int root;
};

BVH bvh_create(const bounds3* bounds, int num_bounds);

void bvh_destroy_host(BVH& bvh);
void bvh_destroy_device(BVH& bvh);

void bvh_refit_host(BVH& bvh, const bounds3* bounds);
void bvh_refit_device(BVH& bvh, const bounds3* bounds);

// copy host BVH to device
BVH bvh_clone(const BVH& bvh_host);



CUDA_CALLABLE inline BVHPackedNodeHalf make_node(const vec3& bound, int child, bool leaf)
{
    BVHPackedNodeHalf n;
    n.x = bound.x;
    n.y = bound.y;
    n.z = bound.z;
    n.i = (unsigned int)child;
    n.b = (unsigned int)(leaf?1:0);

    return n;
}

// variation of make_node through volatile pointers used in BuildHierarchy
CUDA_CALLABLE inline void make_node(volatile BVHPackedNodeHalf* n, const vec3& bound, int child, bool leaf)
{
    n->x = bound.x;
    n->y = bound.y;
    n->z = bound.z;
    n->i = (unsigned int)child;
    n->b = (unsigned int)(leaf?1:0);
}



} // namespace og

