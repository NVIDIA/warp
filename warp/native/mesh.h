/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "builtin.h"
#include "bvh.h"
#include "intersect.h"

#define BVH_DEBUG 0

namespace wp
{

struct Mesh
{
    vec3* points;
	vec3* velocities;

    int* indices;

	bounds3* bounds;

    int num_points;
    int num_tris;

    BVH bvh;
};

CUDA_CALLABLE inline Mesh mesh_get(uint64_t id)
{
    return *(Mesh*)(id);
}


CUDA_CALLABLE inline float distance_to_aabb_sq(const vec3& p, const vec3& lower, const vec3& upper)
{
	vec3 cp = closest_point_to_aabb(p, lower, upper);

	return length_sq(p-cp);
}



// these can be called inside kernels so need to be inline
CUDA_CALLABLE inline vec3 mesh_query_point_old(uint64_t id, const vec3& point, float max_dist, float& inside)
{
    Mesh mesh = mesh_get(id);

	if (mesh.bvh.num_nodes == 0)
		return vec3();

	int stack[32];
    stack[0] = mesh.bvh.root;

	int count = 1;

	float min_dist_sq = max_dist*max_dist;
	vec3 min_point;
	inside = 1.0f;

#if BVH_DEBUG
	int tests = 0;
#endif

	while (count)
	{
		const int nodeIndex = stack[--count];

		BVHPackedNodeHalf lower = mesh.bvh.node_lowers[nodeIndex];
		BVHPackedNodeHalf upper = mesh.bvh.node_uppers[nodeIndex];

		const float dist_sq = distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));

		if (dist_sq < min_dist_sq)
		{
			const int left_index = lower.i;
			const int right_index = upper.i;

			if (lower.b)
			{	
                // compute closest point on tri
				int i = mesh.indices[left_index*3+0];
				int j = mesh.indices[left_index*3+1];
				int k = mesh.indices[left_index*3+2];

				vec3 p = mesh.points[i];
				vec3 q = mesh.points[j];
				vec3 r = mesh.points[k];

				vec2 barycentric = closest_point_to_triangle(p, q, r, point);
				float u = barycentric[0];
				float v = barycentric[1];
				float w = 1.f - u - v;
				vec3 c = u*p + v*q + w*r;

				float dist_sq = length_sq(c-point);

				if (dist_sq < min_dist_sq)
				{
					min_dist_sq = dist_sq;
					min_point = c;

					// if this is a 'new point', i.e.: strictly closer than previous best then update sign
					vec3 normal = cross(q-p, r-p);
					inside = sign(dot(normal, point-c));
				}
				else if (dist_sq == min_dist_sq)
				{
					// if the test point is equal, then test if inside should be updated
					// point considered inside if *any* of the incident faces enclose the point
					vec3 normal = cross(q-p, r-p);
					if (dot(normal, point-c) < 0.0f)
						inside = -1.0f;
				}
#if BVH_DEBUG
				tests++;
#endif 				
			}
			else
			{
				stack[count++] = left_index;
				stack[count++] = right_index;
			}
		}
	}


#if BVH_DEBUG
    static int max_tests = 0;
	static vec3 max_point;
	static float max_point_dist = 0.0f;

    if (tests > max_tests)
	{
        max_tests = tests;
		max_point = point;
		max_point_dist = sqrtf(min_dist_sq);

		printf("max_tests: %d max_point: %f %f %f max_point_dist: %f\n", max_tests, max_point.x, max_point.y, max_point.z, max_point_dist);
	}
#endif

	return min_point;
}

// returns true if there is a point (strictly) < distance max_dist
CUDA_CALLABLE inline bool mesh_query_point(uint64_t id, const vec3& point, float max_dist, float& inside, int& face, float& u, float& v)
{
    Mesh mesh = mesh_get(id);

	if (mesh.bvh.num_nodes == 0)
		return false;

	int stack[32];
    stack[0] = mesh.bvh.root;

	int count = 1;

	float min_dist_sq = max_dist*max_dist;
	int min_face;
	float min_v;
	float min_w;
	float min_inside = 1.0f;

	int tests = 0;

#if BVH_DEBUG
	
	int secondary_culls = 0;

	std::vector<int> test_history;
	std::vector<vec3> test_centers;
	std::vector<vec3> test_extents;
#endif

	while (count)
	{
		const int nodeIndex = stack[--count];

		BVHPackedNodeHalf lower = mesh.bvh.node_lowers[nodeIndex];
		BVHPackedNodeHalf upper = mesh.bvh.node_uppers[nodeIndex];
	
		// re-test distance
		float node_dist_sq = distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));
		if (node_dist_sq > min_dist_sq)
		{
#if BVH_DEBUG			
			secondary_culls++;
#endif			
			continue;
		}

		const int left_index = lower.i;
		const int right_index = upper.i;

		if (lower.b)
		{	
			// compute closest point on tri
			int i = mesh.indices[left_index*3+0];
			int j = mesh.indices[left_index*3+1];
			int k = mesh.indices[left_index*3+2];

			vec3 p = mesh.points[i];
			vec3 q = mesh.points[j];
			vec3 r = mesh.points[k];
			
			vec3 e0 = q-p;
			vec3 e1 = r-p;
			vec3 e2 = r-q;
			vec3 normal = cross(e0, e1);
			
			// sliver detection
			if (length(normal)/(dot(e0,e0) + dot(e1,e1) + dot(e2,e2)) < 1.e-6f)
				continue;

			vec2 barycentric = closest_point_to_triangle(p, q, r, point);
			float u = barycentric[0];
			float v = barycentric[1];
			float w = 1.f - u - v;
			vec3 c = u*p + v*q + w*r;

			float angle = dot(normal, point-c);
			float dist_sq = length_sq(c-point);

			if (dist_sq < min_dist_sq)
			{
				min_dist_sq = dist_sq;
				min_v = v;
				min_w = w;
				min_face = left_index;
				
				// this is a 'new point', i.e.: strictly closer than previous so update sign
				min_inside = sign(angle);
			}
			else if (dist_sq == min_dist_sq)	// todo: should probably use fuzzy equality here
			{
				// if the test point is equally close, then the point is point considered 
				// inside if *any* of the incident faces enclose the point
				if (angle < 0.0f)
					min_inside = -1.0f;
			}

			tests++;

#if BVH_DEBUG

			bounds3 b;
			b = bounds_union(b, p);
			b = bounds_union(b, q);
			b = bounds_union(b, r);

			if (distance_to_aabb_sq(point, b.lower, b.upper) < max_dist*max_dist)
			{
				//if (dist_sq < max_dist*max_dist)
				test_history.push_back(left_index);
				test_centers.push_back(b.center());
				test_extents.push_back(b.edges());
			}
#endif

		}
		else
		{
			BVHPackedNodeHalf left_lower = mesh.bvh.node_lowers[left_index];
			BVHPackedNodeHalf left_upper = mesh.bvh.node_uppers[left_index];

			BVHPackedNodeHalf right_lower = mesh.bvh.node_lowers[right_index];
			BVHPackedNodeHalf right_upper = mesh.bvh.node_uppers[right_index];

			float left_dist_sq = distance_to_aabb_sq(point, vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z));
			float right_dist_sq = distance_to_aabb_sq(point, vec3(right_lower.x, right_lower.y, right_lower.z), vec3(right_upper.x, right_upper.y, right_upper.z));

			//float left_score = bounds3(vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)).area();
			//float right_score = bounds3(vec3(right_lower.x, right_lower.y, right_lower.z), vec3(right_upper.x, right_upper.y, right_upper.z)).area();

			float left_score = left_dist_sq;
			float right_score = right_dist_sq;

			// if (left_score == right_score)
			// {
			// 	// use distance from box centroid to order children
			// 	//left_score = -length_sq(point-bounds3(vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)).center());
			// 	//right_score = -length_sq(point-bounds3(vec3(right_lower.x, right_lower.y, right_lower.z), vec3(right_upper.x, right_upper.y, right_upper.z)).center());
			// 	left_score = bounds3(vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)).area();
			// 	right_score = bounds3(vec3(right_lower.x, right_lower.y, right_lower.z), vec3(right_upper.x, right_upper.y, right_upper.z)).area();

			// }

			//if (left_dist_sq < right_dist_sq)
			if (left_score < right_score)
			{
				// put left on top of the stack
				if (right_dist_sq < min_dist_sq)
					stack[count++] = right_index;

				if (left_dist_sq < min_dist_sq)
					stack[count++] = left_index;
			}
			else
			{
				// put right on top of the stack
				if (left_dist_sq < min_dist_sq)
					stack[count++] = left_index;

				if (right_dist_sq < min_dist_sq)
					stack[count++] = right_index;
			}
		}
	}


#if BVH_DEBUG
printf("%d\n", tests);

    static int max_tests = 0;
	static vec3 max_point;
	static float max_point_dist = 0.0f;
	static int max_secondary_culls = 0;

	if (secondary_culls > max_secondary_culls)
		max_secondary_culls = secondary_culls;

    if (tests > max_tests)
	{
        max_tests = tests;
		max_point = point;
		max_point_dist = sqrtf(min_dist_sq);

		printf("max_tests: %d max_point: %f %f %f max_point_dist: %f max_second_culls: %d\n", max_tests, max_point.x, max_point.y, max_point.z, max_point_dist, max_secondary_culls);

		FILE* f = fopen("test_history.txt", "w");
		for (int i=0; i < test_history.size(); ++i)
		{
			fprintf(f, "%d, %f, %f, %f, %f, %f, %f\n", 
			test_history[i], 
			test_centers[i].x, test_centers[i].y, test_centers[i].z,
			test_extents[i].x, test_extents[i].y, test_extents[i].z);
		}

		fclose(f);
	}
#endif

	// check if we found a point, and write outputs
	if (min_dist_sq < max_dist*max_dist)
	{
		u = 1.0f - min_v - min_w;
		v = min_v;
		face = min_face;
		inside = min_inside;
		
		return true;
	}
	else
	{
		return false;
	}
}

CUDA_CALLABLE inline void adj_mesh_query_point(uint64_t id, const vec3& point, float max_dist, float& inside, int& face, float& u, float& v,
											   uint64_t adj_id, vec3& adj_point, float& adj_max_dist, float& adj_inside, int& adj_face, float& adj_u, float& adj_v, bool& adj_ret)
{
	Mesh mesh = mesh_get(id);
	
	// face is determined by BVH in forward pass
	int i = mesh.indices[face*3+0];
	int j = mesh.indices[face*3+1];
	int k = mesh.indices[face*3+2];

	vec3 p = mesh.points[i];
	vec3 q = mesh.points[j];
	vec3 r = mesh.points[k];

	vec3 adj_p, adj_q, adj_r;

	vec2 adj_uv(adj_u, adj_v);

	adj_closest_point_to_triangle(p, q, r, point, adj_p, adj_q, adj_r, adj_point, adj_uv);
}


CUDA_CALLABLE inline bool mesh_query_ray(uint64_t id, const vec3& start, const vec3& dir, float max_t, float& t, float& u, float& v, float& sign, vec3& normal, int& face)
{
    Mesh mesh = mesh_get(id);

	if (mesh.bvh.num_nodes == 0)
		return false;

    int stack[32];
	stack[0] = mesh.bvh.root;
	int count = 1;

	vec3 rcp_dir = vec3(1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z);

	float min_t = max_t;
	int min_face;
	float min_u;
	float min_v;
	float min_sign = 1.0f;
	vec3 min_normal;

	while (count)
	{
		const int nodeIndex = stack[--count];

		BVHPackedNodeHalf lower = mesh.bvh.node_lowers[nodeIndex];
		BVHPackedNodeHalf upper = mesh.bvh.node_uppers[nodeIndex];

		// todo: switch to robust ray-aabb, or expand bounds in build stage
		float eps = 1.e-3f;
		float t = 0.0f;
		bool hit = intersect_ray_aabb(start, rcp_dir, vec3(lower.x-eps, lower.y-eps, lower.z-eps), vec3(upper.x+eps, upper.y+eps, upper.z+eps), t);

		if (hit && t < min_t)
		{
			const int left_index = lower.i;
			const int right_index = upper.i;

			if (lower.b)
			{	
				// compute closest point on tri
				int i = mesh.indices[left_index*3+0];
				int j = mesh.indices[left_index*3+1];
				int k = mesh.indices[left_index*3+2];

				vec3 p = mesh.points[i];
				vec3 q = mesh.points[j];
				vec3 r = mesh.points[k];

				float t, u, v, sign;
				vec3 n;
				
				if (intersect_ray_tri_woop(start, dir, p, q, r, t, u, v, sign, &n))
				{
					if (t < min_t && t >= 0.0f)
					{
						min_t = t;
						min_face = left_index;
						min_u = u;
						min_v = v;
						min_sign = sign;
						min_normal = n;
					}
				}
			}
			else
			{
				stack[count++] = left_index;
				stack[count++] = right_index;
			}
		}
	}

	if (min_t < max_t)
	{
		// write outputs
		u = min_u;
		v = min_v;
		sign = min_sign;
		t = min_t;
		normal = normalize(min_normal);
		face = min_face;

		return true;
	}
	else
	{
		return false;
	}
	
}


CUDA_CALLABLE inline void adj_mesh_query_ray(
	uint64_t id, const vec3& start, const vec3& dir, float max_t, float& t, float& u, float& v, float& sign, vec3& n, int& face,
	uint64_t adj_id, vec3& adj_start, vec3& adj_dir, float& adj_max_t, float& adj_t, float& adj_u, float& adj_v, float& adj_sign, vec3& adj_n, int& adj_face, bool adj_ret)
{

	Mesh mesh = mesh_get(id);
	
	// face is determined by BVH in forward pass
	int i = mesh.indices[face*3+0];
	int j = mesh.indices[face*3+1];
	int k = mesh.indices[face*3+2];

	vec3 a = mesh.points[i];
	vec3 b = mesh.points[j];
	vec3 c = mesh.points[k];

	vec3 adj_a, adj_b, adj_c;

	adj_intersect_ray_tri_woop(start, dir, a, b, c, t, u, v, sign, &n, adj_start, adj_dir, adj_a, adj_b, adj_c, adj_t, adj_u, adj_v, adj_sign, &adj_n, adj_ret);

}

// stores state required to traverse the BVH nodes that 
// overlap with a query AABB.
struct mesh_query_aabb_t
{
    CUDA_CALLABLE mesh_query_aabb_t()
    {
    }
    CUDA_CALLABLE mesh_query_aabb_t(int)
    {
    } // for backward pass

    // Mesh Id
    Mesh mesh;
	// BVH traversal stack:
	int stack[32];
	int count;

    // inputs
    wp::vec3 input_lower;
    wp::vec3 input_upper;

	// Face
	int face;
};



CUDA_CALLABLE inline mesh_query_aabb_t mesh_query_aabb(
    uint64_t id, const vec3& lower, const vec3& upper)
{
    // This routine traverses the BVH tree until it finds
	// the first triangle with an overlapping bvh. 

    // initialize empty
	mesh_query_aabb_t query;
	query.face = -1;

	Mesh mesh = mesh_get(id);

	query.mesh = mesh;
	
    // if no bvh nodes, return empty query.
    if (mesh.bvh.num_nodes == 0)
    {
		query.count = 0;
		return query;
	}

    // optimization: make the latest
	
	query.stack[0] = mesh.bvh.root;
	query.count = 1;
    query.input_lower = lower;
    query.input_upper = upper;

    wp::bounds3 input_bounds(query.input_lower, query.input_upper);
	
    // Navigate through the bvh, find the first overlapping leaf node.
    while (query.count)
    {
		const int nodeIndex = query.stack[--query.count];
		BVHPackedNodeHalf node_lower = mesh.bvh.node_lowers[nodeIndex];
		BVHPackedNodeHalf node_upper = mesh.bvh.node_uppers[nodeIndex];

		wp::vec3 lower_pos(node_lower.x, node_lower.y, node_lower.z);
		wp::vec3 upper_pos(node_upper.x, node_upper.y, node_upper.z);
        wp::bounds3 current_bounds(lower_pos, upper_pos);
        if (!input_bounds.overlaps(current_bounds))
        {
            // Skip this box, it doesn't overlap with our target box.
			continue;
		}

		const int left_index = node_lower.i;
		const int right_index = node_upper.i;

        // Make bounds from this AABB
        if (node_lower.b)
        {
			// found very first triangle index.
			// Back up one level and return 
			query.stack[query.count++] = nodeIndex;
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

//Stub
CUDA_CALLABLE inline void adj_mesh_query_aabb(uint64_t id, const vec3& lower, const vec3& upper,
											   uint64_t, vec3&, vec3&, mesh_query_aabb_t&)
{

}

CUDA_CALLABLE inline bool mesh_query_aabb_next(mesh_query_aabb_t& query, int& index)
{
    Mesh mesh = query.mesh;
	
	wp::bounds3 input_bounds(query.input_lower, query.input_upper);
    // Navigate through the bvh, find the first overlapping leaf node.
    while (query.count)
    {
        const int nodeIndex = query.stack[--query.count];
        BVHPackedNodeHalf node_lower = mesh.bvh.node_lowers[nodeIndex];
        BVHPackedNodeHalf node_upper = mesh.bvh.node_uppers[nodeIndex];

        wp::vec3 lower_pos(node_lower.x, node_lower.y, node_lower.z);
        wp::vec3 upper_pos(node_upper.x, node_upper.y, node_upper.z);
        wp::bounds3 current_bounds(lower_pos, upper_pos);
        if (!input_bounds.overlaps(current_bounds))
        {
            // Skip this box, it doesn't overlap with our target box.
            continue;
        }

        const int left_index = node_lower.i;
        const int right_index = node_upper.i;

        // Make bounds from this AABB
        if (node_lower.b)
        {
            // found very first triangle index
            query.face = left_index;
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

CUDA_CALLABLE inline int iter_next(mesh_query_aabb_t& query)
{
    return query.face;
}

CUDA_CALLABLE inline bool iter_cmp(mesh_query_aabb_t& query)
{
    bool finished = mesh_query_aabb_next(query, query.face);
    return finished;
}

CUDA_CALLABLE inline mesh_query_aabb_t iter_reverse(const mesh_query_aabb_t& query)
{
    // can't reverse BVH queries, users should not rely on neighbor ordering
    return query;
}


// stub
CUDA_CALLABLE inline void adj_mesh_query_aabb_next(mesh_query_aabb_t& query, int& index, mesh_query_aabb_t&, int&, bool&) 
{

}

// // determine if a point is inside (ret >0 ) or outside the mesh (ret < 0)
// CUDA_CALLABLE inline float mesh_query_inside(uint64_t id, const vec3& p)
// {
//     float t, u, v, sign;
//     int parity = 0;

//     // x-axis
//     if (mesh_query_ray(id, p, vec3(1.0f, 0.0f, 0.0f), FLT_MAX, t, u, v, sign))
//         parity++;
//     // y-axis
//     if (mesh_query_ray(id, p, vec3(0.0f, 1.0f, 0.0f), FLT_MAX, t, u, v, sign))
//         parity++;
//     // z-axis
//     if (mesh_query_ray(id, p, vec3(0.0f, 0.0f, 1.0f), FLT_MAX, t, u, v, sign))
//         parity++;

//     // if all 3 rays inside then return -1
//     if (parity == 3)
//         return -1.0f;
//     else
//         return 1.0f;
// }


CUDA_CALLABLE inline vec3 mesh_eval_position(uint64_t id, int tri, float u, float v)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.points)
		return vec3();

	assert(tri < mesh.num_tris);

	int i = mesh.indices[tri*3+0];
	int j = mesh.indices[tri*3+1];
	int k = mesh.indices[tri*3+2];

	vec3 p = mesh.points[i];
	vec3 q = mesh.points[j];
	vec3 r = mesh.points[k];

	return p*u + q*v + r*(1.0f-u-v);
}

CUDA_CALLABLE inline vec3 mesh_eval_velocity(uint64_t id, int tri, float u, float v)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.velocities)
		return vec3();

	assert(tri < mesh.num_tris);

	int i = mesh.indices[tri*3+0];
	int j = mesh.indices[tri*3+1];
	int k = mesh.indices[tri*3+2];

	vec3 vp = mesh.velocities[i];
	vec3 vq = mesh.velocities[j];
	vec3 vr = mesh.velocities[k];

	return vp*u + vq*v + vr*(1.0f-u-v);
}


CUDA_CALLABLE inline void adj_mesh_eval_position(uint64_t id, int tri, float u, float v,
												 uint64_t& adj_id, int& adj_tri, float& adj_u, float& adj_v, const vec3& adj_ret)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.points)
		return;

	assert(tri < mesh.num_tris);

	int i = mesh.indices[tri*3+0];
	int j = mesh.indices[tri*3+1];
	int k = mesh.indices[tri*3+2];

	vec3 p = mesh.points[i];
	vec3 q = mesh.points[j];
	vec3 r = mesh.points[k];

	adj_u += (p.x - r.x) * adj_ret.x + (p.y - r.y) * adj_ret.y + (p.z - r.z) * adj_ret.z;
	adj_v += (q.x - r.x) * adj_ret.x + (q.y - r.y) * adj_ret.y + (q.z - r.z) * adj_ret.z;
}

CUDA_CALLABLE inline void adj_mesh_eval_velocity(uint64_t id, int tri, float u, float v,
												 uint64_t& adj_id, int& adj_tri, float& adj_u, float& adj_v, const vec3& adj_ret)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.velocities)
		return;

	assert(tri < mesh.num_tris);

	int i = mesh.indices[tri*3+0];
	int j = mesh.indices[tri*3+1];
	int k = mesh.indices[tri*3+2];

	vec3 vp = mesh.velocities[i];
	vec3 vq = mesh.velocities[j];
	vec3 vr = mesh.velocities[k];

	adj_u += (vp.x - vr.x) * adj_ret.x + (vp.y - vr.y) * adj_ret.y + (vp.z - vr.z) * adj_ret.z;
	adj_v += (vq.x - vr.x) * adj_ret.x + (vq.y - vr.y) * adj_ret.y + (vq.z - vr.z) * adj_ret.z;
}

CUDA_CALLABLE inline vec3 mesh_eval_face_normal(uint64_t id, int tri)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.points)
		return vec3();

	assert(tri < mesh.num_tris);

	int i = mesh.indices[tri*3+0];
	int j = mesh.indices[tri*3+1];
	int k = mesh.indices[tri*3+2];

	vec3 p = mesh.points[i];
	vec3 q = mesh.points[j];
	vec3 r = mesh.points[k];

	return normalize(cross(q - p, r - p));
}

CUDA_CALLABLE inline void adj_mesh_eval_face_normal(uint64_t id, int tri,
												    uint64_t& adj_id, int& adj_tri, const vec3& adj_ret)
{
	// no-op
}

CUDA_CALLABLE inline vec3 mesh_get_point(uint64_t id, int index)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.points)
		return vec3();

	assert(index < mesh.num_tris * 3);

	int i = mesh.indices[index];

	return mesh.points[i];
}

CUDA_CALLABLE inline void adj_mesh_get_point(uint64_t id, int index,
										     uint64_t& adj_id, int& adj_index, const vec3& adj_ret)
{
	// no-op
}

CUDA_CALLABLE inline vec3 mesh_get_velocity(uint64_t id, int index)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.velocities)
		return vec3();

	assert(index < mesh.num_tris * 3);

	int i = mesh.indices[index];

	return mesh.velocities[i];
}

CUDA_CALLABLE inline void adj_mesh_get_velocity(uint64_t id, int index,
										     uint64_t& adj_id, int& adj_index, const vec3& adj_ret)
{
	// no-op
}

CUDA_CALLABLE inline int mesh_get_index(uint64_t id, int face_vertex_index)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.indices)
		return -1;

	assert(face_vertex_index < mesh.num_tris * 3);

	return mesh.indices[face_vertex_index];
}

CUDA_CALLABLE inline void adj_mesh_get_index(uint64_t id, int index,
										     uint64_t& adj_id, int& adj_index, const vec3& adj_ret)
{
	// no-op
}

bool mesh_get_descriptor(uint64_t id, Mesh& mesh);
void mesh_add_descriptor(uint64_t id, const Mesh& mesh);
void mesh_rem_descriptor(uint64_t id);


} // namespace wp
