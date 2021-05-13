#pragma once

#include "core.h"
#include "bvh.h"
#include "intersect.h"

#define BVH_DEBUG 0

namespace og
{

struct Mesh
{
    vec3* points;
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

				float v, w;
				vec3 c = closest_point_to_triangle(p, q, r, point, v, w);

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


CUDA_CALLABLE inline vec3 mesh_query_point(uint64_t id, const vec3& point, float max_dist, float& inside)
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

			float v, w;
			vec3 c = closest_point_to_triangle(p, q, r, point, v, w);

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

	return min_point;
}

CUDA_CALLABLE inline void adj_mesh_query_point(uint64_t id, const vec3& point, float max_dist, float sign, uint64_t& adj_id, vec3& adj_point, float& adj_max_dist, float& adj_sign, const vec3& adj_ret)
{

}


CUDA_CALLABLE inline float mesh_query_ray(uint64_t id, const vec3& start, const vec3& dir, float max_t, float& u, float& v, float& sign)
{
	/*
    Mesh mesh = mesh_get(id);

	if (mesh.bvh.mNumNodes == 0)
		return 0.0f;

    int stack[64]
	stack[0] = mesh.bvh.root;
	int count = 1;

	vec3 rcp_dir = vec3(1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z);

	while (count)
	{
		const int nodeIndex = stack[--count];

		// union to allow 128-bit loads
		union { BVHPackedNodeHalf lower; vec4 lowerf; };
		union {	BVHPackedNodeHalf upper; vec4 upperf; };
		
		lowerf = tex1Dfetch<vec4>(bvh.node_lowers, nodeIndex);
		upperf = tex1Dfetch<vec4>(bvh.node_uppers, nodeIndex);

		Bounds nodeBounds(vec3(&lower.x)*bvhScale, vec3(&upper.x)*bvhScale);
		nodeBounds.Expand(thickness);

		float t;
		bool hit = intersect_ray_aabb(start, rcp_dir, nodeBounds.lower, nodeBounds.upper, t);

		if (hit && t < maxT)
		{
			const int leftIndex = lower.i;
			const int rightIndex = upper.i;

			if (lower.b)
			{	
				//f(leftIndex, maxT);
			}
			else
			{
				stack[count++] = leftIndex;
				stack[count++] = rightIndex;
			}
		}
	}

	// todo:
	return 0.0f;
	*/
	return 0.0f;
}

// determine if a point is inside (ret >0 ) or outside the mesh (ret < 0)
CUDA_CALLABLE inline float mesh_query_inside(uint64_t id, const vec3& p)
{
    float u, v, sign;
    int parity = 0;

    // x-axis
    if (mesh_query_ray(id, p, vec3(1.0f, 0.0f, 0.0f), FLT_MAX, u, v, sign) < 0.0f)
        parity++;
    // y-axis
    if (mesh_query_ray(id, p, vec3(0.0f, 1.0f, 0.0f), FLT_MAX, u, v, sign) < 0.0f)
        parity++;
    // z-axis
    if (mesh_query_ray(id, p, vec3(0.0f, 0.0f, 1.0f), FLT_MAX, u, v, sign) < 0.0f)
        parity++;

    // if all 3 rays inside then return -1
    if (parity == 3)
        return -1.0f;
    else
        return 1.0f;
}

bool mesh_get_descriptor(uint64_t id, Mesh& mesh);
void mesh_add_descriptor(uint64_t id, const Mesh& mesh);
void mesh_rem_descriptor(uint64_t id);


} // namespace og
