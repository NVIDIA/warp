// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "builtin.h"

#include "array.h"
#include "bvh.h"
#include "intersect.h"
#include "rand.h"
#include "solid_angle.h"

namespace wp {

struct Mesh {
    array_t<vec3> points;
    array_t<vec3> velocities;

    array_t<int> indices;

    vec3* lowers;
    vec3* uppers;

    SolidAngleProps* solid_angle_props;

    int num_points;
    int num_tris;

    BVH bvh;

    void* context;
    float average_edge_length;

    inline CUDA_CALLABLE Mesh(int id = 0)
    {
        // for backward a = 0 initialization syntax
        lowers = nullptr;
        uppers = nullptr;
        num_points = 0;
        num_tris = 0;
        context = nullptr;
        solid_angle_props = nullptr;
        average_edge_length = 0.0f;
        bvh = BVH {};
    }

    inline CUDA_CALLABLE Mesh(
        array_t<vec3> points,
        array_t<vec3> velocities,
        array_t<int> indices,
        int num_points,
        int num_tris,
        void* context = nullptr
    )
        : points(points)
        , velocities(velocities)
        , indices(indices)
        , num_points(num_points)
        , num_tris(num_tris)
        , context(context)
    {
        lowers = nullptr;
        uppers = nullptr;
        solid_angle_props = nullptr;
        average_edge_length = 0.0f;
        bvh = BVH {};
    }
};

CUDA_CALLABLE inline Mesh mesh_get(uint64_t id) { return *(Mesh*)(id); }

// Return the id of the mesh's internal BVH so it can be queried directly with the bvh_query_* builtins.
// The id is simply the address of the embedded BVH (bvh_get() casts it straight back to a BVH*).
CUDA_CALLABLE inline uint64_t mesh_get_bvh(uint64_t id) { return (uint64_t)&(((Mesh*)id)->bvh); }

CUDA_CALLABLE inline int mesh_get_group_root(uint64_t id, int group_id)
{
    Mesh* mesh = (Mesh*)(id);
    return bvh_get_group_root((uint64_t)&mesh->bvh, group_id);
}


CUDA_CALLABLE inline Mesh& operator+=(Mesh& a, const Mesh& b)
{
    // dummy operator needed for adj_select involving meshes
    return a;
}

CUDA_CALLABLE inline float distance_to_aabb_sq(const vec3& p, const vec3& lower, const vec3& upper)
{
    const float dx = min(upper[0], max(lower[0], p[0])) - p[0];
    const float dy = min(upper[1], max(lower[1], p[1])) - p[1];
    const float dz = min(upper[2], max(lower[2], p[2])) - p[2];
    return dx * dx + dy * dy + dz * dz;
}

CUDA_CALLABLE inline float furthest_distance_to_aabb_sq(const vec3& p, const vec3& lower, const vec3& upper)
{
    // X-axis
    float dist_lower_x = fabs(p[0] - lower[0]);
    float dist_upper_x = fabs(p[0] - upper[0]);
    float corner_diff_x = (dist_lower_x > dist_upper_x) ? dist_lower_x : dist_upper_x;

    // Y-axis
    float dist_lower_y = fabs(p[1] - lower[1]);
    float dist_upper_y = fabs(p[1] - upper[1]);
    float corner_diff_y = (dist_lower_y > dist_upper_y) ? dist_lower_y : dist_upper_y;

    // Z-axis
    float dist_lower_z = fabs(p[2] - lower[2]);
    float dist_upper_z = fabs(p[2] - upper[2]);
    float corner_diff_z = (dist_lower_z > dist_upper_z) ? dist_lower_z : dist_upper_z;

    // Calculate and return the distance
    return corner_diff_x * corner_diff_x + corner_diff_y * corner_diff_y + corner_diff_z * corner_diff_z;
}

CUDA_CALLABLE inline int
mesh_query_ray_count_intersections(uint64_t id, const vec3& start, const vec3& dir, int root = -1);
CUDA_CALLABLE inline float mesh_query_inside_ray_tracing(uint64_t id, const vec3& p);
CUDA_CALLABLE inline float
mesh_query_inside_parity(uint64_t id, const vec3& p, const vec3 base_dir, int n_sample, float perturbation_scale);

// returns true if there is a point (strictly) < distance max_dist
CUDA_CALLABLE inline bool
mesh_query_point(uint64_t id, const vec3& point, float max_dist, float& inside, int& face, float& u, float& v)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    stack[0] = *mesh.bvh.root;

    int count = 1;

    float min_dist_sq = max_dist * max_dist;
    int min_face;
    float min_v;
    float min_w;

    while (count) {
        const int nodeIndex = stack[--count];

        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, nodeIndex);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, nodeIndex);

        // re-test distance
        float node_dist_sq
            = distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));
        if (node_dist_sq > min_dist_sq) {
            continue;
        }

        const int left_index = lower.i;
        const int right_index = upper.i;

        if (lower.b) {
            const int start = left_index;
            const int end = right_index;
            // loops through primitives in the leaf
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, primitive_counter);
                int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
                int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
                int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                vec3 e0 = q - p;
                vec3 e1 = r - p;
                vec3 e2 = r - q;
                vec3 normal = cross(e0, e1);

                // sliver detection
                if (length(normal) / (dot(e0, e0) + dot(e1, e1) + dot(e2, e2)) < 1.e-6f)
                    continue;

                vec2 barycentric = closest_point_to_triangle(p, q, r, point);
                float u = barycentric[0];
                float v = barycentric[1];
                float w = 1.f - u - v;
                vec3 c = u * p + v * q + w * r;

                float dist_sq = length_sq(c - point);

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    min_v = v;
                    min_w = w;
                    min_face = primitive_index;
                }
            }
        } else {
            BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
            BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);

            BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
            BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

            float left_dist_sq = distance_to_aabb_sq(
                point, vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)
            );
            float right_dist_sq = distance_to_aabb_sq(
                point, vec3(right_lower.x, right_lower.y, right_lower.z),
                vec3(right_upper.x, right_upper.y, right_upper.z)
            );

            wp::vec2i child_indices;
            wp::vec2 child_dist;
            if (left_dist_sq < right_dist_sq) {
                child_indices = wp::vec2i(right_index, left_index);
                child_dist = wp::vec2(right_dist_sq, left_dist_sq);
            } else {
                child_indices = wp::vec2i(left_index, right_index);
                child_dist = wp::vec2(left_dist_sq, right_dist_sq);
            }

            if (child_dist[0] < min_dist_sq)
                stack[count++] = child_indices[0];

            if (child_dist[1] < min_dist_sq)
                stack[count++] = child_indices[1];
        }
    }

    // check if we found a point, and write outputs
    if (min_dist_sq < max_dist * max_dist) {
        u = 1.0f - min_v - min_w;
        v = min_v;
        face = min_face;

        // determine inside outside using ray-cast parity check
        inside = mesh_query_inside_ray_tracing(id, point);

        return true;
    } else {
        return false;
    }
}

// returns true if there is a point (strictly) < distance max_dist
CUDA_CALLABLE inline bool mesh_query_point_sign_parity(
    uint64_t id,
    const vec3& point,
    float max_dist,
    float& inside,
    int& face,
    float& u,
    float& v,
    int n_sample = 1,
    float perturbation_scale = 0.1f
)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    stack[0] = *mesh.bvh.root;

    int count = 1;

    float min_dist_sq = max_dist * max_dist;
    int min_face;
    float min_v;
    float min_w;

    while (count) {
        const int nodeIndex = stack[--count];

        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, nodeIndex);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, nodeIndex);

        // re-test distance
        float node_dist_sq
            = distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));
        if (node_dist_sq > min_dist_sq) {
            continue;
        }

        const int left_index = lower.i;
        const int right_index = upper.i;

        if (lower.b) {
            const int start = left_index;
            const int end = right_index;
            // loops through primitives in the leaf
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, primitive_counter);
                int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
                int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
                int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                vec3 e0 = q - p;
                vec3 e1 = r - p;
                vec3 e2 = r - q;
                vec3 normal = cross(e0, e1);

                // sliver detection
                if (length(normal) / (dot(e0, e0) + dot(e1, e1) + dot(e2, e2)) < 1.e-6f)
                    continue;

                vec2 barycentric = closest_point_to_triangle(p, q, r, point);
                float u = barycentric[0];
                float v = barycentric[1];
                float w = 1.f - u - v;
                vec3 c = u * p + v * q + w * r;

                float dist_sq = length_sq(c - point);

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    min_v = v;
                    min_w = w;
                    min_face = primitive_index;
                }
            }
        } else {
            BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
            BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);

            BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
            BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

            float left_dist_sq = distance_to_aabb_sq(
                point, vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)
            );
            float right_dist_sq = distance_to_aabb_sq(
                point, vec3(right_lower.x, right_lower.y, right_lower.z),
                vec3(right_upper.x, right_upper.y, right_upper.z)
            );

            wp::vec2i child_indices;
            wp::vec2 child_dist;
            if (left_dist_sq < right_dist_sq) {
                child_indices = wp::vec2i(right_index, left_index);
                child_dist = wp::vec2(right_dist_sq, left_dist_sq);
            } else {
                child_indices = wp::vec2i(left_index, right_index);
                child_dist = wp::vec2(left_dist_sq, right_dist_sq);
            }

            if (child_dist[0] < min_dist_sq)
                stack[count++] = child_indices[0];

            if (child_dist[1] < min_dist_sq)
                stack[count++] = child_indices[1];
        }
    }

    // check if we found a point, and write outputs
    if (min_dist_sq < max_dist * max_dist) {
        u = 1.0f - min_v - min_w;
        v = min_v;
        face = min_face;

        // determine inside outside using ray-cast parity check
        inside = mesh_query_inside_parity(id, point, vec3(1.f, 1.f, 1.f), n_sample, perturbation_scale);

        return true;
    } else {
        return false;
    }
}

// returns true if there is a point (strictly) < distance max_dist
CUDA_CALLABLE inline bool
mesh_query_point_no_sign(uint64_t id, const vec3& point, float max_dist, int& face, float& u, float& v)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    stack[0] = *mesh.bvh.root;

    int count = 1;

    float min_dist_sq = max_dist * max_dist;
    int min_face;
    float min_v;
    float min_w;

    while (count) {
        const int nodeIndex = stack[--count];

        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, nodeIndex);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, nodeIndex);

        // re-test distance
        float node_dist_sq
            = distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));
        if (node_dist_sq > min_dist_sq) {
            continue;
        }

        const int left_index = lower.i;
        const int right_index = upper.i;

        if (lower.b) {
            const int start = left_index;
            const int end = right_index;
            // loops through primitives in the leaf
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, primitive_counter);
                int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
                int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
                int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];
                vec3 e0 = q - p;
                vec3 e1 = r - p;
                vec3 e2 = r - q;
                vec3 normal = cross(e0, e1);

                // sliver detection
                if (length(normal) / (dot(e0, e0) + dot(e1, e1) + dot(e2, e2)) < 1.e-6f)
                    continue;

                vec2 barycentric = closest_point_to_triangle(p, q, r, point);
                float u = barycentric[0];
                float v = barycentric[1];
                float w = 1.f - u - v;
                vec3 c = u * p + v * q + w * r;

                float dist_sq = length_sq(c - point);

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    min_v = v;
                    min_w = w;
                    min_face = primitive_index;
                }
            }
        } else {
            BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
            BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);

            BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
            BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

            float left_dist_sq = distance_to_aabb_sq(
                point, vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)
            );
            float right_dist_sq = distance_to_aabb_sq(
                point, vec3(right_lower.x, right_lower.y, right_lower.z),
                vec3(right_upper.x, right_upper.y, right_upper.z)
            );

            wp::vec2i child_indices;
            wp::vec2 child_dist;
            if (left_dist_sq < right_dist_sq) {
                child_indices = wp::vec2i(right_index, left_index);
                child_dist = wp::vec2(right_dist_sq, left_dist_sq);
            } else {
                child_indices = wp::vec2i(left_index, right_index);
                child_dist = wp::vec2(left_dist_sq, right_dist_sq);
            }

            if (child_dist[0] < min_dist_sq)
                stack[count++] = child_indices[0];

            if (child_dist[1] < min_dist_sq)
                stack[count++] = child_indices[1];
        }
    }

    // check if we found a point, and write outputs
    if (min_dist_sq < max_dist * max_dist) {
        u = 1.0f - min_v - min_w;
        v = min_v;
        face = min_face;

        return true;
    } else {
        return false;
    }
}

// returns true if there is a point (strictly) > distance min_dist
CUDA_CALLABLE inline bool
mesh_query_furthest_point_no_sign(uint64_t id, const vec3& point, float min_dist, int& face, float& u, float& v)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    stack[0] = *mesh.bvh.root;

    int count = 1;

    float min_dist_sq = min_dist * min_dist;
    int max_face;
    float max_v;
    float max_w;

    while (count) {
        const int nodeIndex = stack[--count];

        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, nodeIndex);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, nodeIndex);

        // re-test distance
        float node_dist_sq
            = furthest_distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));

        // if maximum distance to this node is less than our existing furthest max then skip
        if (node_dist_sq < min_dist_sq) {
            continue;
        }

        const int left_index = lower.i;
        const int right_index = upper.i;

        if (lower.b) {
            const int start = left_index;
            const int end = right_index;
            // loops through primitives in the leaf
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, primitive_counter);
                int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
                int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
                int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                vec3 e0 = q - p;
                vec3 e1 = r - p;
                vec3 e2 = r - q;
                vec3 normal = cross(e0, e1);

                // sliver detection
                if (length(normal) / (dot(e0, e0) + dot(e1, e1) + dot(e2, e2)) < 1.e-6f)
                    continue;

                vec2 barycentric = furthest_point_to_triangle(p, q, r, point);
                float u = barycentric[0];
                float v = barycentric[1];
                float w = 1.f - u - v;
                vec3 c = u * p + v * q + w * r;

                float dist_sq = length_sq(c - point);

                if (dist_sq > min_dist_sq) {
                    min_dist_sq = dist_sq;
                    max_v = v;
                    max_w = w;
                    max_face = primitive_index;
                }
            }
        } else {
            BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
            BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);

            BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
            BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

            float left_dist_sq = furthest_distance_to_aabb_sq(
                point, vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)
            );
            float right_dist_sq = furthest_distance_to_aabb_sq(
                point, vec3(right_lower.x, right_lower.y, right_lower.z),
                vec3(right_upper.x, right_upper.y, right_upper.z)
            );

            wp::vec2i child_indices;
            wp::vec2 child_dist;
            if (left_dist_sq > right_dist_sq) {
                child_indices = wp::vec2i(right_index, left_index);
                child_dist = wp::vec2(right_dist_sq, left_dist_sq);
            } else {
                child_indices = wp::vec2i(left_index, right_index);
                child_dist = wp::vec2(left_dist_sq, right_dist_sq);
            }

            if (child_dist[0] > min_dist_sq)
                stack[count++] = child_indices[0];

            if (child_dist[1] > min_dist_sq)
                stack[count++] = child_indices[1];
        }
    }

    // check if we found a point, and write outputs
    if (min_dist_sq > min_dist * min_dist) {
        u = 1.0f - max_v - max_w;
        v = max_v;
        face = max_face;

        return true;
    } else {
        return false;
    }
}

// returns true if there is a point (strictly) < distance max_dist
CUDA_CALLABLE inline bool mesh_query_point_sign_normal(
    uint64_t id,
    const vec3& point,
    float max_dist,
    float& inside,
    int& face,
    float& u,
    float& v,
    const float epsilon = 1e-3f
)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    stack[0] = *mesh.bvh.root;
    int count = 1;
    float min_dist = max_dist;
    int min_face;
    float min_v;
    float min_w;
    vec3 accumulated_angle_weighted_normal;

    float epsilon_min_dist = mesh.average_edge_length * epsilon;
    float epsilon_min_dist_sq = epsilon_min_dist * epsilon_min_dist;

    while (count) {
        const int nodeIndex = stack[--count];
        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, nodeIndex);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, nodeIndex);

        // re-test distance
        float node_dist_sq
            = distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));
        if (node_dist_sq > (min_dist + epsilon_min_dist) * (min_dist + epsilon_min_dist)) {
            continue;
        }

        const int left_index = lower.i;
        const int right_index = upper.i;

        if (lower.b) {
            const int start = left_index;
            const int end = right_index;
            // loops through primitives in the leaf
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, primitive_counter);
                int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
                int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
                int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                vec3 e0 = q - p;
                vec3 e1 = r - p;
                vec3 e2 = r - q;
                vec3 normal = cross(e0, e1);

                // sliver detection
                float e0_norm_sq = dot(e0, e0);
                float e1_norm_sq = dot(e1, e1);
                float e2_norm_sq = dot(e2, e2);
                if (length(normal) / (e0_norm_sq + e1_norm_sq + e2_norm_sq) < 1.e-6f)
                    continue;

                vec2 barycentric = closest_point_to_triangle(p, q, r, point);
                float u = barycentric[0];
                float v = barycentric[1];
                float w = 1.f - u - v;
                vec3 c = u * p + v * q + w * r;
                float dist = sqrtf(length_sq(c - point));
                if (dist < min_dist + epsilon_min_dist) {
                    float weight = 0.0f;
                    vec3 cp = c - p;
                    vec3 cq = c - q;
                    vec3 cr = c - r;
                    float len_cp_sq = length_sq(cp);
                    float len_cq_sq = length_sq(cq);
                    float len_cr_sq = length_sq(cr);

                    // Check if near vertex
                    if (len_cp_sq < epsilon_min_dist_sq) {
                        // Vertex 0 is the closest feature
                        weight = acosf(dot(normalize(e0), normalize(e1)));
                    } else if (len_cq_sq < epsilon_min_dist_sq) {
                        // Vertex 1 is the closest feature
                        weight = acosf(dot(normalize(e2), normalize(-e0)));
                    } else if (len_cr_sq < epsilon_min_dist_sq) {
                        // Vertex 2 is the closest feature
                        weight = acosf(dot(normalize(-e1), normalize(-e2)));
                    } else {
                        float e0cp = dot(e0, cp);
                        float e2cq = dot(e2, cq);
                        float e1cp = dot(e1, cp);

                        if ((len_cp_sq * e0_norm_sq - e0cp * e0cp < epsilon_min_dist_sq * e0_norm_sq)
                            || (len_cq_sq * e2_norm_sq - e2cq * e2cq < epsilon_min_dist_sq * e2_norm_sq)
                            || (len_cp_sq * e1_norm_sq - e1cp * e1cp < epsilon_min_dist_sq * e1_norm_sq)) {
                            // One of the edge
                            weight = 3.14159265359f;  // PI
                        } else {
                            weight = 2.0f * 3.14159265359f;  // 2*PI
                        }
                    }

                    if (dist > min_dist - epsilon_min_dist) {
                        // Treat as equal
                        accumulated_angle_weighted_normal += weight * normalize(normal);
                        if (dist < min_dist) {
                            min_dist = dist;
                            min_v = v;
                            min_w = w;
                            min_face = primitive_index;
                        }
                    } else {
                        // Less
                        min_dist = dist;
                        min_v = v;
                        min_w = w;
                        min_face = primitive_index;
                        accumulated_angle_weighted_normal = weight * normalize(normal);
                    }
                }
            }
        } else {
            BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
            BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);

            BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
            BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

            float left_dist_sq = distance_to_aabb_sq(
                point, vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)
            );
            float right_dist_sq = distance_to_aabb_sq(
                point, vec3(right_lower.x, right_lower.y, right_lower.z),
                vec3(right_upper.x, right_upper.y, right_upper.z)
            );

            wp::vec2i child_indices;
            wp::vec2 child_dist;
            if (left_dist_sq < right_dist_sq) {
                child_indices = wp::vec2i(right_index, left_index);
                child_dist = wp::vec2(right_dist_sq, left_dist_sq);
            } else {
                child_indices = wp::vec2i(left_index, right_index);
                child_dist = wp::vec2(left_dist_sq, right_dist_sq);
            }

            if (child_dist[0] < (min_dist + epsilon_min_dist) * (min_dist + epsilon_min_dist))
                stack[count++] = child_indices[0];

            if (child_dist[1] < (min_dist + epsilon_min_dist) * (min_dist + epsilon_min_dist))
                stack[count++] = child_indices[1];
        }
    }
    // check if we found a point, and write outputs
    if (min_dist < max_dist) {
        u = 1.0f - min_v - min_w;
        v = min_v;
        face = min_face;
        // determine inside outside using ray-cast parity check
        // inside = mesh_query_inside(id, point);
        int i = mesh.indices[min_face * 3 + 0];
        int j = mesh.indices[min_face * 3 + 1];
        int k = mesh.indices[min_face * 3 + 2];
        vec3 p = mesh.points[i];
        vec3 q = mesh.points[j];
        vec3 r = mesh.points[k];
        vec3 closest_point = p * u + q * v + r * min_w;
        if (dot(accumulated_angle_weighted_normal, point - closest_point) > 0.0) {
            inside = 1.0f;
        } else {
            inside = -1.0f;
        }
        return true;
    } else {
        return false;
    }
}

CUDA_CALLABLE inline float solid_angle_iterative(uint64_t id, const vec3& p, const float accuracy_sq)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    int at_child[BVH_QUERY_STACK_SIZE];  // 0 for left, 1 for right, 2 for done
    float angle[BVH_QUERY_STACK_SIZE];
    stack[0] = *mesh.bvh.root;
    at_child[0] = 0;

    int count = 1;
    angle[0] = 0.0f;

    while (count) {
        const int nodeIndex = stack[count - 1];
        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, nodeIndex);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, nodeIndex);

        const int left_index = lower.i;
        const int right_index = upper.i;
        if (lower.b) {
            // compute closest point on tri
            const int start = left_index;
            const int end = right_index;
            angle[count - 1] = 0.f;
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, primitive_counter);
                int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
                int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
                int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);
                angle[count - 1] += robust_solid_angle(mesh.points[i], mesh.points[j], mesh.points[k], p);
                // printf("Leaf %d, got %f\n", leaf_index, my_data[count - 1]);
            }
            count--;
        } else {
            // See if I have to descend
            if (at_child[count - 1] == 0) {
                // First visit
                bool des
                    = evaluate_node_solid_angle(p, &mesh.solid_angle_props[nodeIndex], angle[count - 1], accuracy_sq);

                // printf("Non-Leaf %d, got %f\n", nodeIndex, angle[count - 1]);
                if (des) {
                    // Go left
                    stack[count] = left_index;
                    at_child[count - 1] = 1;
                    angle[count] = 0.0f;
                    at_child[count] = 0;
                    count++;
                } else {
                    // Does not descend done
                    count--;
                }
            } else if (at_child[count - 1] == 1) {
                // Add data to parent
                angle[count - 1] += angle[count];
                // Go right
                stack[count] = right_index;
                at_child[count - 1] = 2;
                angle[count] = 0.0f;
                at_child[count] = 0;
                count++;
            } else {
                // Descend both sides already
                angle[count - 1] += angle[count];
                count--;
            }
        }
    }
    return angle[0];
}

CUDA_CALLABLE inline float mesh_query_winding_number(uint64_t id, const vec3& p, const float accuracy)
{
    float angle = solid_angle_iterative(id, p, accuracy * accuracy);
    return angle * 0.07957747154;  // divided by 4 PI
}

// returns true if there is a point (strictly) < distance max_dist
CUDA_CALLABLE inline bool mesh_query_point_sign_winding_number(
    uint64_t id,
    const vec3& point,
    float max_dist,
    float& inside,
    int& face,
    float& u,
    float& v,
    const float accuracy,
    const float winding_number_threshold
)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    stack[0] = *mesh.bvh.root;

    int count = 1;

    float min_dist_sq = max_dist * max_dist;
    int min_face;
    float min_v;
    float min_w;

    while (count) {
        const int nodeIndex = stack[--count];

        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, nodeIndex);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, nodeIndex);

        // re-test distance
        float node_dist_sq
            = distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));
        if (node_dist_sq > min_dist_sq) {
            continue;
        }

        const int left_index = lower.i;
        const int right_index = upper.i;

        if (lower.b) {
            const int start = left_index;
            const int end = right_index;
            // loops through primitives in the leaf
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, primitive_counter);
                int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
                int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
                int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                vec3 e0 = q - p;
                vec3 e1 = r - p;
                vec3 e2 = r - q;
                vec3 normal = cross(e0, e1);

                // sliver detection
                if (length(normal) / (dot(e0, e0) + dot(e1, e1) + dot(e2, e2)) < 1.e-6f)
                    continue;

                vec2 barycentric = closest_point_to_triangle(p, q, r, point);
                float u = barycentric[0];
                float v = barycentric[1];
                float w = 1.f - u - v;
                vec3 c = u * p + v * q + w * r;

                float dist_sq = length_sq(c - point);

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    min_v = v;
                    min_w = w;
                    min_face = primitive_index;
                }
            }
        } else {
            BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
            BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);

            BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
            BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

            float left_dist_sq = distance_to_aabb_sq(
                point, vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)
            );
            float right_dist_sq = distance_to_aabb_sq(
                point, vec3(right_lower.x, right_lower.y, right_lower.z),
                vec3(right_upper.x, right_upper.y, right_upper.z)
            );

            wp::vec2i child_indices;
            wp::vec2 child_dist;
            if (left_dist_sq < right_dist_sq) {
                child_indices = wp::vec2i(right_index, left_index);
                child_dist = wp::vec2(right_dist_sq, left_dist_sq);
            } else {
                child_indices = wp::vec2i(left_index, right_index);
                child_dist = wp::vec2(left_dist_sq, right_dist_sq);
            }

            if (child_dist[0] < min_dist_sq)
                stack[count++] = child_indices[0];

            if (child_dist[1] < min_dist_sq)
                stack[count++] = child_indices[1];
        }
    }

    // check if we found a point, and write outputs
    if (min_dist_sq < max_dist * max_dist) {
        u = 1.0f - min_v - min_w;
        v = min_v;
        face = min_face;

        // determine inside outside using ray-cast parity check
        if (!mesh.solid_angle_props) {
            inside = mesh_query_inside_ray_tracing(id, point);
        } else {
            float winding_number = mesh_query_winding_number(id, point, accuracy);
            inside = (winding_number > winding_number_threshold) ? -1.0f : 1.0f;
        }

        return true;
    } else {
        return false;
    }
}

CUDA_CALLABLE inline void adj_mesh_query_point_no_sign(
    uint64_t id,
    const vec3& point,
    float max_dist,
    const int& face,
    const float& u,
    const float& v,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    bool& adj_ret
)
{
    Mesh mesh = mesh_get(id);

    // face is determined by BVH in forward pass
    int i = mesh.indices[face * 3 + 0];
    int j = mesh.indices[face * 3 + 1];
    int k = mesh.indices[face * 3 + 2];

    vec3 p = mesh.points[i];
    vec3 q = mesh.points[j];
    vec3 r = mesh.points[k];

    vec3 adj_p, adj_q, adj_r;

    vec2 adj_uv(adj_u, adj_v);

    adj_closest_point_to_triangle(p, q, r, point, adj_p, adj_q, adj_r, adj_point, adj_uv);
}

CUDA_CALLABLE inline void adj_mesh_query_furthest_point_no_sign(
    uint64_t id,
    const vec3& point,
    float min_dist,
    const int& face,
    const float& u,
    const float& v,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_min_dist,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    bool& adj_ret
)
{
    Mesh mesh = mesh_get(id);

    // face is determined by BVH in forward pass
    int i = mesh.indices[face * 3 + 0];
    int j = mesh.indices[face * 3 + 1];
    int k = mesh.indices[face * 3 + 2];

    vec3 p = mesh.points[i];
    vec3 q = mesh.points[j];
    vec3 r = mesh.points[k];

    vec3 adj_p, adj_q, adj_r;

    vec2 adj_uv(adj_u, adj_v);

    adj_closest_point_to_triangle(p, q, r, point, adj_p, adj_q, adj_r, adj_point, adj_uv);  // Todo for Miles :>
}

CUDA_CALLABLE inline void adj_mesh_query_point_sign_parity(
    uint64_t id,
    const vec3& point,
    float max_dist,
    const float& inside,
    const int& face,
    const float& u,
    const float& v,
    int n_sample,
    float perturbation_scale,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    float& adj_inside,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    int& adj_n_sample,
    float& adj_perturbation_scale,
    bool& adj_ret
)
{
    adj_mesh_query_point_no_sign(
        id, point, max_dist, face, u, v, adj_id, adj_point, adj_max_dist, adj_face, adj_u, adj_v, adj_ret
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_sign_normal(
    uint64_t id,
    const vec3& point,
    float max_dist,
    const float& inside,
    const int& face,
    const float& u,
    const float& v,
    const float epsilon,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    float& adj_inside,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    float& adj_epsilon,
    bool& adj_ret
)
{
    adj_mesh_query_point_no_sign(
        id, point, max_dist, face, u, v, adj_id, adj_point, adj_max_dist, adj_face, adj_u, adj_v, adj_ret
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_sign_winding_number(
    uint64_t id,
    const vec3& point,
    float max_dist,
    const float& inside,
    const int& face,
    const float& u,
    const float& v,
    const float accuracy,
    const float winding_number_threshold,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    float& adj_inside,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    float& adj_accuracy,
    float& adj_winding_number_threshold,
    bool& adj_ret
)
{
    adj_mesh_query_point_no_sign(
        id, point, max_dist, face, u, v, adj_id, adj_point, adj_max_dist, adj_face, adj_u, adj_v, adj_ret
    );
}


// Stores the result of querying the closest point on a mesh.
struct mesh_query_point_t {
    CUDA_CALLABLE mesh_query_point_t()
        : result(false)
        , sign(0.0f)
        , face(0)
        , u(0.0f)
        , v(0.0f)
    {
    }

    // Required for adjoint computations.
    CUDA_CALLABLE inline mesh_query_point_t& operator+=(const mesh_query_point_t& other)
    {
        result |= other.result;  // Use OR for bool accumulation
        sign += other.sign;
        face += other.face;
        u += other.u;
        v += other.v;
        return *this;
    }

    bool result;
    float sign;
    int face;
    float u;
    float v;
};


CUDA_CALLABLE inline void adj_mesh_query_point(
    uint64_t id,
    const vec3& point,
    float max_dist,
    const float& inside,
    const int& face,
    const float& u,
    const float& v,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    float& adj_inside,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    bool& adj_ret
)
{
    adj_mesh_query_point_no_sign(
        id, point, max_dist, face, u, v, adj_id, adj_point, adj_max_dist, adj_face, adj_u, adj_v, adj_ret
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point(
    uint64_t id,
    const vec3& point,
    float max_dist,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_point(
        id, point, max_dist, ret.sign, ret.face, ret.u, ret.v, adj_id, adj_point, adj_max_dist, adj_ret.sign,
        adj_ret.face, adj_ret.u, adj_ret.v, adj_ret.result
    );
}

CUDA_CALLABLE inline mesh_query_point_t mesh_query_point(uint64_t id, const vec3& point, float max_dist)
{
    mesh_query_point_t query;
    query.result = mesh_query_point(id, point, max_dist, query.sign, query.face, query.u, query.v);
    return query;
}


CUDA_CALLABLE inline mesh_query_point_t mesh_query_point_sign_parity(
    uint64_t id, const vec3& point, float max_dist, int n_sample = 1, float perturbation_scale = 0.1f
)
{
    mesh_query_point_t query;
    query.result = mesh_query_point_sign_parity(
        id, point, max_dist, query.sign, query.face, query.u, query.v, n_sample, perturbation_scale
    );
    return query;
}

CUDA_CALLABLE inline mesh_query_point_t mesh_query_point_no_sign(uint64_t id, const vec3& point, float max_dist)
{
    mesh_query_point_t query;
    query.sign = 0.0;
    query.result = mesh_query_point_no_sign(id, point, max_dist, query.face, query.u, query.v);
    return query;
}

CUDA_CALLABLE inline mesh_query_point_t
mesh_query_furthest_point_no_sign(uint64_t id, const vec3& point, float min_dist)
{
    mesh_query_point_t query;
    query.sign = 0.0;
    query.result = mesh_query_furthest_point_no_sign(id, point, min_dist, query.face, query.u, query.v);
    return query;
}

CUDA_CALLABLE inline mesh_query_point_t
mesh_query_point_sign_normal(uint64_t id, const vec3& point, float max_dist, const float epsilon = 1e-3f)
{
    mesh_query_point_t query;
    query.result = mesh_query_point_sign_normal(id, point, max_dist, query.sign, query.face, query.u, query.v, epsilon);
    return query;
}

CUDA_CALLABLE inline mesh_query_point_t mesh_query_point_sign_winding_number(
    uint64_t id, const vec3& point, float max_dist, float accuracy, float winding_number_threshold
)
{
    mesh_query_point_t query;
    query.result = mesh_query_point_sign_winding_number(
        id, point, max_dist, query.sign, query.face, query.u, query.v, accuracy, winding_number_threshold
    );
    return query;
}

CUDA_CALLABLE inline void adj_mesh_query_point_sign_parity(
    uint64_t id,
    const vec3& point,
    float max_dist,
    int n_sample,
    float perturbation_scale,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    int& adj_n_sample,
    float& adj_perturbation_scale,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_point_sign_parity(
        id, point, max_dist, ret.sign, ret.face, ret.u, ret.v, n_sample, perturbation_scale, adj_id, adj_point,
        adj_max_dist, adj_ret.sign, adj_ret.face, adj_ret.u, adj_ret.v, adj_n_sample, adj_perturbation_scale,
        adj_ret.result
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_no_sign(
    uint64_t id,
    const vec3& point,
    float max_dist,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_point_no_sign(
        id, point, max_dist, ret.face, ret.u, ret.v, adj_id, adj_point, adj_max_dist, adj_ret.face, adj_ret.u,
        adj_ret.v, adj_ret.result
    );
}

CUDA_CALLABLE inline void adj_mesh_query_furthest_point_no_sign(
    uint64_t id,
    const vec3& point,
    float min_dist,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_min_dist,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_furthest_point_no_sign(
        id, point, min_dist, ret.face, ret.u, ret.v, adj_id, adj_point, adj_min_dist, adj_ret.face, adj_ret.u,
        adj_ret.v, adj_ret.result
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_sign_normal(
    uint64_t id,
    const vec3& point,
    float max_dist,
    float epsilon,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    float& adj_epsilon,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_point_sign_normal(
        id, point, max_dist, ret.sign, ret.face, ret.u, ret.v, epsilon, adj_id, adj_point, adj_max_dist, adj_ret.sign,
        adj_ret.face, adj_ret.u, adj_ret.v, adj_epsilon, adj_ret.result
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_sign_winding_number(
    uint64_t id,
    const vec3& point,
    float max_dist,
    float accuracy,
    float winding_number_threshold,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    float& adj_accuracy,
    float& adj_winding_number_threshold,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_point_sign_winding_number(
        id, point, max_dist, ret.sign, ret.face, ret.u, ret.v, accuracy, winding_number_threshold, adj_id, adj_point,
        adj_max_dist, adj_ret.sign, adj_ret.face, adj_ret.u, adj_ret.v, adj_accuracy, adj_winding_number_threshold,
        adj_ret.result
    );
}

CUDA_CALLABLE inline vec3 mesh_query_ray_safe_dir(const vec3& dir)
{
    vec3 ray_dir = dir;
    if (ray_dir[0] == 0.0f)
        ray_dir[0] = 1.0e-20f;
    if (ray_dir[1] == 0.0f)
        ray_dir[1] = 1.0e-20f;
    if (ray_dir[2] == 0.0f)
        ray_dir[2] = 1.0e-20f;
    return ray_dir;
}

CUDA_CALLABLE inline bool mesh_query_ray_use_fast_aabb(const vec3& dir)
{
    return dir[0] != 0.0f && dir[1] != 0.0f && dir[2] != 0.0f;
}

CUDA_CALLABLE inline bool mesh_query_ray_intersect_aabb(
    const vec3& start,
    const vec3& dir,
    const vec3& rcp_dir,
    bool fast_aabb,
    const vec3& lower,
    const vec3& upper,
    float& t
)
{
    if (fast_aabb)
        return intersect_ray_aabb(start, rcp_dir, lower, upper, t);
    else
        return intersect_ray_aabb_robust(start, dir, rcp_dir, lower, upper, t);
}

CUDA_CALLABLE inline bool mesh_query_ray(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    float& t,
    float& u,
    float& v,
    float& sign,
    vec3& normal,
    int& face,
    int root = -1
)
{
    Mesh mesh = mesh_get(id);

    uint64_t stack[BVH_QUERY_STACK_SIZE];
    int stack_size = 0;
    uint64_t cur_node = bvh_query_node_load(mesh.bvh, (root == -1) ? *mesh.bvh.root : root);

    vec3 ray_dir = mesh_query_ray_safe_dir(dir);
    vec3 rcp_dir(1.0f / ray_dir[0], 1.0f / ray_dir[1], 1.0f / ray_dir[2]);
    const bool fast_aabb = mesh_query_ray_use_fast_aabb(dir);

    float min_t = max_t;
    int min_face;
    float min_u;
    float min_v;
    float min_sign = 1.0f;
    vec3 min_normal;
    bool hit = false;

    while (true) {
        if (bvh_query_node_is_leaf(cur_node)) {
            const int primitive_begin = bvh_query_node_lower_payload(cur_node);
            const int primitive_end = bvh_query_node_upper_payload(cur_node);
            // Leaf: test all primitives in the leaf.
            for (int pc = primitive_begin; pc < primitive_end; ++pc) {
                int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, pc);
                int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
                int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
                int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                float tri_t, tri_u, tri_v, tri_sign;
                vec3 n;

                if (intersect_ray_tri_woop(start, dir, p, q, r, tri_t, tri_u, tri_v, tri_sign, &n)) {
                    if (tri_t < min_t && tri_t >= 0.0f) {
                        min_t = tri_t;
                        min_face = primitive_index;
                        min_u = tri_u;
                        min_v = tri_v;
                        min_sign = tri_sign;
                        min_normal = n;
                        hit = true;
                    }
                }
            }
            if (stack_size == 0)
                break;
            cur_node = stack[--stack_size];
            continue;
        }

        // Inner node: load both children so we can sort by entry distance.
        const int left_index = bvh_query_node_lower_payload(cur_node);
        const int right_index = bvh_query_node_upper_payload(cur_node);

        BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
        BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);
        BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
        BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

        float t0 = FLT_MAX;
        float t1 = FLT_MAX;
        const bool h0 = mesh_query_ray_intersect_aabb(
                            start, dir, rcp_dir, fast_aabb, vec3(left_lower.x, left_lower.y, left_lower.z),
                            vec3(left_upper.x, left_upper.y, left_upper.z), t0
                        )
            && t0 < min_t;
        const bool h1 = mesh_query_ray_intersect_aabb(
                            start, dir, rcp_dir, fast_aabb, vec3(right_lower.x, right_lower.y, right_lower.z),
                            vec3(right_upper.x, right_upper.y, right_upper.z), t1
                        )
            && t1 < min_t;

        if (h0 && h1) {
            const bool near_left = (t0 < t1);
            if (stack_size >= BVH_QUERY_STACK_SIZE)
                break;
            const uint64_t left_node = bvh_query_node_pack(left_lower, left_upper);
            const uint64_t right_node = bvh_query_node_pack(right_lower, right_upper);
            stack[stack_size++] = near_left ? right_node : left_node;
            cur_node = near_left ? left_node : right_node;
        } else if (h0) {
            cur_node = bvh_query_node_pack(left_lower, left_upper);
        } else if (h1) {
            cur_node = bvh_query_node_pack(right_lower, right_upper);
        } else {
            // Neither child reachable; pop.
            if (stack_size == 0)
                break;
            cur_node = stack[--stack_size];
        }
    }

    if (hit) {
        // write outputs
        u = min_u;
        v = min_v;
        sign = min_sign;
        t = min_t;
        normal = normalize(min_normal);
        face = min_face;

        return true;
    } else {
        return false;
    }
}

CUDA_CALLABLE inline bool
mesh_query_ray_anyhit(uint64_t id, const vec3& start, const vec3& dir, float max_t, int root = -1)
{
    Mesh mesh = mesh_get(id);

    uint64_t stack[BVH_QUERY_STACK_SIZE];
    int stack_size = 0;
    uint64_t cur_node = bvh_query_node_load(mesh.bvh, (root == -1) ? *mesh.bvh.root : root);

    vec3 ray_dir = mesh_query_ray_safe_dir(dir);
    vec3 rcp_dir(1.0f / ray_dir[0], 1.0f / ray_dir[1], 1.0f / ray_dir[2]);
    const bool fast_aabb = mesh_query_ray_use_fast_aabb(dir);

    while (true) {
        if (bvh_query_node_is_leaf(cur_node)) {
            const int primitive_begin = bvh_query_node_lower_payload(cur_node);
            const int primitive_end = bvh_query_node_upper_payload(cur_node);
            for (int pc = primitive_begin; pc < primitive_end; ++pc) {
                int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, pc);
                int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
                int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
                int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                float tri_t, tri_u, tri_v, tri_sign;
                vec3 n;

                if (intersect_ray_tri_woop(start, dir, p, q, r, tri_t, tri_u, tri_v, tri_sign, &n)) {
                    if (tri_t < max_t && tri_t >= 0.0f) {
                        return true;
                    }
                }
            }
            if (stack_size == 0)
                return false;
            cur_node = stack[--stack_size];
            continue;
        }

        const int left_index = bvh_query_node_lower_payload(cur_node);
        const int right_index = bvh_query_node_upper_payload(cur_node);

        BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
        BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);
        BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
        BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

        float t0 = FLT_MAX;
        float t1 = FLT_MAX;
        const bool h0 = mesh_query_ray_intersect_aabb(
                            start, dir, rcp_dir, fast_aabb, vec3(left_lower.x, left_lower.y, left_lower.z),
                            vec3(left_upper.x, left_upper.y, left_upper.z), t0
                        )
            && t0 < max_t;
        const bool h1 = mesh_query_ray_intersect_aabb(
                            start, dir, rcp_dir, fast_aabb, vec3(right_lower.x, right_lower.y, right_lower.z),
                            vec3(right_upper.x, right_upper.y, right_upper.z), t1
                        )
            && t1 < max_t;

        if (h0 && h1) {
            const bool near_left = (t0 < t1);
            if (stack_size >= BVH_QUERY_STACK_SIZE)
                return false;
            const uint64_t left_node = bvh_query_node_pack(left_lower, left_upper);
            const uint64_t right_node = bvh_query_node_pack(right_lower, right_upper);
            stack[stack_size++] = near_left ? right_node : left_node;
            cur_node = near_left ? left_node : right_node;
        } else if (h0) {
            cur_node = bvh_query_node_pack(left_lower, left_upper);
        } else if (h1) {
            cur_node = bvh_query_node_pack(right_lower, right_upper);
        } else {
            if (stack_size == 0)
                return false;
            cur_node = stack[--stack_size];
        }
    }
}

CUDA_CALLABLE inline int mesh_query_ray_count_intersections(uint64_t id, const vec3& start, const vec3& dir, int root)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];

    stack[0] = root == -1 ? *mesh.bvh.root : root;
    int count = 1;

    vec3 rcp_dir(1.0f / dir[0], 1.0f / dir[1], 1.0f / dir[2]);

    int num_hit = 0;
    float temp_t;

    while (count) {
        const int node_index = stack[--count];

        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, node_index);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, node_index);

        bool hit = intersect_ray_aabb_robust(
            start, dir, rcp_dir, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z), temp_t
        );

        if (hit) {
            if (lower.b) {
                const int start_index = lower.i;
                const int end_index = upper.i;
                // loops through primitives in the leaf
                for (int primitive_counter = start_index; primitive_counter < end_index; primitive_counter++) {
                    int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, primitive_counter);
                    int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
                    int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
                    int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);

                    vec3 p = mesh.points[i];
                    vec3 q = mesh.points[j];
                    vec3 r = mesh.points[k];

                    float temp_t, temp_u, temp_v, temp_sign;
                    vec3 n;

                    if (intersect_ray_tri_woop(start, dir, p, q, r, temp_t, temp_u, temp_v, temp_sign, &n)) {
                        if (temp_t >= 0.0f) {
                            num_hit++;
                        }
                    }
                }
            } else {
                stack[count++] = lower.i;
                stack[count++] = upper.i;
            }
        }
    }

    return num_hit;
}

template <typename T> CUDA_CALLABLE inline void _swap(T& a, T& b)
{
    T t = a;
    a = b;
    b = t;
}

CUDA_CALLABLE inline bool mesh_query_ray_ordered(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    float& t,
    float& u,
    float& v,
    float& sign,
    vec3& normal,
    int& face,
    int root = -1
)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    float stack_dist[BVH_QUERY_STACK_SIZE];

    stack[0] = root == -1 ? *mesh.bvh.root : root;
    stack_dist[0] = -FLT_MAX;

    int count = 1;

    vec3 rcp_dir(1.0f / dir[0], 1.0f / dir[1], 1.0f / dir[2]);

    float min_t = max_t;
    int min_face;
    float min_u;
    float min_v;
    float min_sign = 1.0f;
    vec3 min_normal;

    while (count) {
        count -= 1;

        const int node_index = stack[count];
        const float node_dist = stack_dist[count];

        if (node_dist < min_t) {
            int left_index = mesh.bvh.node_lowers[node_index].i;
            int right_index = mesh.bvh.node_uppers[node_index].i;
            bool leaf = mesh.bvh.node_lowers[node_index].b;

            if (leaf) {
                const int start_index = left_index;
                const int end_index = right_index;
                // loops through primitives in the leaf
                for (int primitive_counter = start_index; primitive_counter < end_index; primitive_counter++) {
                    int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, primitive_counter);
                    int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
                    int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
                    int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);

                    vec3 p = mesh.points[i];
                    vec3 q = mesh.points[j];
                    vec3 r = mesh.points[k];

                    float temp_t, temp_u, temp_v, temp_sign;
                    vec3 n;

                    if (intersect_ray_tri_woop(start, dir, p, q, r, temp_t, temp_u, temp_v, temp_sign, &n)) {
                        if (temp_t < min_t && temp_t >= 0.0f) {
                            min_t = temp_t;
                            min_face = primitive_index;
                            min_u = temp_u;
                            min_v = temp_v;
                            min_sign = temp_sign;
                            min_normal = n;
                        }
                    }
                }
            } else {
                BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
                BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);

                BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
                BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

                float left_dist = FLT_MAX;
                bool left_hit = intersect_ray_aabb_robust(
                    start, dir, rcp_dir, vec3(left_lower.x, left_lower.y, left_lower.z),
                    vec3(left_upper.x, left_upper.y, left_upper.z), left_dist
                );

                float right_dist = FLT_MAX;
                bool right_hit = intersect_ray_aabb_robust(
                    start, dir, rcp_dir, vec3(right_lower.x, right_lower.y, right_lower.z),
                    vec3(right_upper.x, right_upper.y, right_upper.z), right_dist
                );


                if (left_dist < right_dist) {
                    _swap(left_index, right_index);
                    _swap(left_dist, right_dist);
                    _swap(left_hit, right_hit);
                }

                if (left_hit && left_dist < min_t) {
                    stack[count] = left_index;
                    stack_dist[count] = left_dist;
                    count += 1;
                }

                if (right_hit && right_dist < min_t) {
                    stack[count] = right_index;
                    stack_dist[count] = right_dist;
                    count += 1;
                }
            }
        }
    }

    if (min_t < max_t) {
        // write outputs
        u = min_u;
        v = min_v;
        sign = min_sign;
        t = min_t;
        normal = normalize(min_normal);
        face = min_face;

        return true;
    } else {
        return false;
    }
}

CUDA_CALLABLE inline void adj_mesh_query_ray(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    float t,
    float u,
    float v,
    float sign,
    const vec3& n,
    int face,
    int root,
    uint64_t adj_id,
    vec3& adj_start,
    vec3& adj_dir,
    float& adj_max_t,
    float& adj_t,
    float& adj_u,
    float& adj_v,
    float& adj_sign,
    vec3& adj_n,
    int& adj_face,
    int& adj_root,
    bool& adj_ret
)
{

    Mesh mesh = mesh_get(id);

    // face is determined by BVH in forward pass
    int i = mesh.indices[face * 3 + 0];
    int j = mesh.indices[face * 3 + 1];
    int k = mesh.indices[face * 3 + 2];

    vec3 a = mesh.points[i];
    vec3 b = mesh.points[j];
    vec3 c = mesh.points[k];

    vec3 adj_a, adj_b, adj_c;

    adj_intersect_ray_tri_woop(
        start, dir, a, b, c, t, u, v, sign, n, adj_start, adj_dir, adj_a, adj_b, adj_c, adj_t, adj_u, adj_v, adj_sign,
        adj_n, adj_ret
    );
}

// Stores the result of querying the closest point on a mesh.
struct mesh_query_ray_t {
    CUDA_CALLABLE mesh_query_ray_t()
        : result(false)
        , sign(0.0f)
        , face(0)
        , t(0.0f)
        , u(0.0f)
        , v(0.0f)
        , normal()
    {
    }

    // Required for adjoint computations.
    CUDA_CALLABLE inline mesh_query_ray_t& operator+=(const mesh_query_ray_t& other)
    {
        result |= other.result;  // Use OR for bool accumulation
        sign += other.sign;
        face += other.face;
        t += other.t;
        u += other.u;
        v += other.v;
        normal += other.normal;
        return *this;
    }

    float sign;
    int face;
    float t;
    float u;
    float v;
    vec3 normal;
    bool result;
};

CUDA_CALLABLE inline mesh_query_ray_t
mesh_query_ray(uint64_t id, const vec3& start, const vec3& dir, float max_t, int root)
{
    mesh_query_ray_t query;
    query.result
        = mesh_query_ray(id, start, dir, max_t, query.t, query.u, query.v, query.sign, query.normal, query.face, root);
    return query;
}

CUDA_CALLABLE inline void adj_mesh_query_ray(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    int root,
    const mesh_query_ray_t& ret,
    uint64_t adj_id,
    vec3& adj_start,
    vec3& adj_dir,
    float& adj_max_t,
    int& adj_root,
    mesh_query_ray_t& adj_ret
)
{
    adj_mesh_query_ray(
        id, start, dir, max_t, ret.t, ret.u, ret.v, ret.sign, ret.normal, ret.face, root, adj_id, adj_start, adj_dir,
        adj_max_t, adj_ret.t, adj_ret.u, adj_ret.v, adj_ret.sign, adj_ret.normal, adj_ret.face, adj_root, adj_ret.result
    );
}

// Flat-stack closest-hit traversal that returns only the sign of the closest
// hit. Used by mesh_query_inside_ray_tracing for the three axis-aligned probe
// rays: those rays penetrate the whole mesh and rarely allow pruning, so the
// eager-child-loading overhead of the near-far mesh_query_ray traversal is not
// worth paying. This function uses the classic push-both-children approach,
// which has half the BVH node loads per inner step.
CUDA_CALLABLE inline bool
mesh_query_ray_closest_sign(const Mesh& mesh, const vec3& start, const vec3& dir, float& out_sign)
{
    int stack[BVH_QUERY_STACK_SIZE];
    int stack_size = 0;
    int node_index = *mesh.bvh.root;

    vec3 rcp_dir(1.0f / dir[0], 1.0f / dir[1], 1.0f / dir[2]);
    float min_t = FLT_MAX;
    float temp_t;
    bool hit = false;

    while (true) {
        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, node_index);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, node_index);

        if (intersect_ray_aabb_robust(
                start, dir, rcp_dir, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z), temp_t
            )
            && temp_t < min_t) {
            if (lower.b) {
                for (int pc = lower.i; pc < upper.i; ++pc) {
                    int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, pc);
                    int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
                    int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
                    int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);

                    vec3 p = mesh.points[i];
                    vec3 q = mesh.points[j];
                    vec3 r = mesh.points[k];

                    float tri_t, tri_u, tri_v, tri_sign;
                    vec3 n;

                    if (intersect_ray_tri_woop(start, dir, p, q, r, tri_t, tri_u, tri_v, tri_sign, &n)) {
                        if (tri_t >= 0.0f && tri_t < min_t) {
                            min_t = tri_t;
                            out_sign = tri_sign;
                            hit = true;
                        }
                    }
                }
            } else {
                stack[stack_size++] = lower.i;
                stack[stack_size++] = upper.i;
            }
        }

        if (stack_size == 0)
            break;
        node_index = stack[--stack_size];
    }
    return hit;
}

// determine if a point is inside (ret < 0 ) or outside the mesh (ret > 0) using ray tracing
CUDA_CALLABLE inline float mesh_query_inside_ray_tracing(uint64_t id, const vec3& p)
{
    Mesh mesh = mesh_get(id);

    int vote = 0;
    float sign;

    for (int i = 0; i < 3; ++i) {
        if (mesh_query_ray_closest_sign(mesh, p, vec3(float(i == 0), float(i == 1), float(i == 2)), sign) && sign < 0) {
            vote++;
        }
    }

    if (vote >= 2)
        return -1.0f;
    else
        return 1.0f;
}


// determine if a point is inside (ret < 0 ) or outside the mesh (ret > 0)
CUDA_CALLABLE inline float
mesh_query_inside_parity(uint64_t id, const vec3& p, const vec3 base_dir, int n_sample, float perturbation_scale)
{
    int vote = 0;

    // deterministic
    uint32_t rand_state = rand_init(42);

    for (int i = 0; i < n_sample; ++i) {

        vec3 dir;
        do {
            dir = base_dir
                + vec3(
                      randf(rand_state, -perturbation_scale, perturbation_scale),
                      randf(rand_state, -perturbation_scale, perturbation_scale),
                      randf(rand_state, -perturbation_scale, perturbation_scale)
                );
        } while (length_sq(dir) < 1e-8f);

        if (mesh_query_ray_count_intersections(id, p, dir) % 2) {
            vote++;
        }
    }

    if (vote * 2 >= n_sample)
        return -1.0f;
    else
        return 1.0f;
}

// Mesh query kind, stored in mesh_query_aabb_t::kind for the code paths shared
// across kinds at runtime (mesh_query_next_dynamic for kind-erased queries, and
// iter_cmp, the `for face in query:` protocol). The per-kind while-loop iterators
// are selected at Warp codegen time and never read it.
enum class MeshQueryKind : uint8_t { AABB = 0, SPHERE = 1 };

// stores state required to traverse the BVH nodes that
// overlap with a query AABB.
struct mesh_query_aabb_t {
    CUDA_CALLABLE mesh_query_aabb_t()
        : mesh()
        , stack()
        , count(0)
        , input_lower()
        , input_upper()
        , face(0)
        , prim_cur(0)
        , prim_end(0)
        , last_query_valid(true)
        , kind(MeshQueryKind::AABB)
        , radius_sq(0.0f)
    {
    }

    // Required for adjoint computations.
    CUDA_CALLABLE inline mesh_query_aabb_t& operator+=(const mesh_query_aabb_t& other) { return *this; }

    // Mesh Id
    Mesh mesh;
    // BVH traversal stack:
#if BVH_SHARED_STACK
    bvh_stack_t stack;
#else
    int stack[BVH_QUERY_STACK_SIZE];
#endif

    int count;

    // inputs
    wp::vec3 input_lower;
    wp::vec3 input_upper;

    // primitive range of the packed leaf currently being enumerated;
    // when prim_cur < prim_end the query resumes mid-leaf on the next
    // mesh_query_next() call, without re-visiting the leaf node
    int prim_cur;
    int prim_end;

    // Face
    int face;

    // Tracks whether the most recent mesh_query_aabb_next() / tile_mesh_query_aabb_next()
    // call produced a valid face index. Seeded to true so an initial tile_query_valid()
    // check (before any next() call) reports valid.
    bool last_query_valid;
    // Read only by the paths shared by all query kinds, which therefore need a runtime
    // discriminant: mesh_query_next_dynamic (kind-erased queries) and iter_cmp (the
    // `for face in query:` protocol). The per-kind while-loop iterators are selected at
    // Warp codegen time from the Python type and never read this.
    MeshQueryKind kind;
    // Squared sphere radius for sphere queries (0 for plain AABB queries). Only the square
    // is stored: every traversal test compares squared distances, and the mesh side has no
    // capsule mode that would need the linear radius.
    float radius_sq;
};


#if BVH_SHARED_STACK
// One shared-memory traversal stack per kernel, shared by every mesh query kind.
// Allocated outside the templated factory: each template instantiation would
// otherwise declare its own slab, so a kernel constructing both AABB and sphere
// queries would exceed the shared-memory budget.
CUDA_CALLABLE inline int* mesh_query_shared_stack()
{
    __shared__ int stack[BVH_QUERY_STACK_SIZE * WP_TILE_BLOCK_DIM];
    return stack;
}
#endif

// Shared factory for all mesh query kinds. IsSphere selects the stored kind
// (read only on the kind-erased paths; statically-typed iterators are selected
// at Warp codegen time via the Python return type).
template <bool IsSphere>
CUDA_CALLABLE inline mesh_query_aabb_t mesh_query_impl(uint64_t id, const vec3& a, const vec3& b, float radius)
{
    mesh_query_aabb_t query;
    query.face = -1;
    query.kind = IsSphere ? MeshQueryKind::SPHERE : MeshQueryKind::AABB;
    const float r = max(radius, 0.0f);
    query.radius_sq = r * r;

    Mesh mesh = mesh_get(id);
    query.mesh = mesh;

#if BVH_SHARED_STACK
    query.stack.ptr = &mesh_query_shared_stack()[threadIdx.x];
#endif

    query.stack[0] = *mesh.bvh.root;
    query.count = 1;
    query.input_lower = a;
    query.input_upper = b;

    return query;
}

CUDA_CALLABLE inline mesh_query_aabb_t mesh_query_aabb(uint64_t id, const vec3& lower, const vec3& upper)
{
    return mesh_query_impl<false>(id, lower, upper, 0.0f);
}

// Sphere query: iterate triangles that intersect the sphere. The broad phase keeps triangles whose AABB is
// within `radius` of `center` (exact sphere-AABB test); the narrow phase keeps only those whose closest
// point to `center` is within `radius`.
CUDA_CALLABLE inline mesh_query_aabb_t mesh_query_sphere(uint64_t id, const vec3& center, float radius)
{
    return mesh_query_impl<true>(id, center, center, radius);
}

// Sphere per-primitive test: broad-phase sphere test against the cached triangle AABB,
// then an exact closest-point narrow phase (with a fallback for degenerate faces).
CUDA_CALLABLE inline bool mesh_query_prim_test(const mesh_query_aabb_t& query, const Mesh& mesh, int primitive_index)
{
    if (!intersect_sphere_aabb(
            query.input_lower, query.radius_sq, mesh.lowers[primitive_index], mesh.uppers[primitive_index]
        ))
        return false;

    int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
    int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
    int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);
    vec3 a = mesh.points[i];
    vec3 b = mesh.points[j];
    vec3 c = mesh.points[k];

    const vec3& center = query.input_lower;
    vec3 cp;
    // Guard against degenerate (zero-area) faces to avoid NaN from closest_point_to_triangle.
    vec3 ab = b - a, ac = c - a;
    if (dot(cross(ab, ac), cross(ab, ac)) == 0.0f) {
        // Degenerate: collapse to segment or point. Find the longest edge.
        vec3 bc = c - b;
        float lab2 = dot(ab, ab), lac2 = dot(ac, ac), lbc2 = dot(bc, bc);
        vec3 p, q;
        float len2;
        if (lab2 >= lac2 && lab2 >= lbc2) {
            p = a;
            q = b;
            len2 = lab2;
        } else if (lac2 >= lbc2) {
            p = a;
            q = c;
            len2 = lac2;
        } else {
            p = b;
            q = c;
            len2 = lbc2;
        }
        vec3 pq = q - p;
        float t = (len2 > 0.0f) ? clamp(dot(center - p, pq) / len2, 0.0f, 1.0f) : 0.0f;
        cp = p + t * pq;
    } else {
        vec2 uv = closest_point_to_triangle(a, b, c, center);
        cp = a * uv[0] + b * uv[1] + c * (1.0f - uv[0] - uv[1]);
    }
    vec3 d = cp - center;
    return dot(d, d) <= query.radius_sq;
}

// Stub
CUDA_CALLABLE inline void
adj_mesh_query_aabb(uint64_t id, const vec3& lower, const vec3& upper, uint64_t, vec3&, vec3&, mesh_query_aabb_t&)
{
}


// ---------------------------------------------------------------------------
// Mesh query traversal skeleton and per-kind iterators
//
// The skeleton is written once as a function template parameterized by a
// NodeTest and a PrimitiveTest functor.  Each public iterator instantiates
// it with the matching pair, producing a dispatch-free loop.  Because each
// public function is called only from the Python type that represents its
// query kind, NVCC compiles exactly one loop per kernel -- no dead-branch
// code bloat and no runtime/literal performance difference.
// ---------------------------------------------------------------------------

struct AabbNodeTest {
    CUDA_CALLABLE bool operator()(const mesh_query_aabb_t& q, const vec3& lo, const vec3& hi) const
    {
        return intersect_aabb_aabb(q.input_lower, q.input_upper, lo, hi);
    }
};

struct SphereNodeTest {
    CUDA_CALLABLE bool operator()(const mesh_query_aabb_t& q, const vec3& lo, const vec3& hi) const
    {
        return intersect_sphere_aabb(q.input_lower, q.radius_sq, lo, hi);
    }
};

// Broad-phase AABB primitive test: check the triangle's cached AABB only.
struct AabbPrimitiveTest {
    CUDA_CALLABLE bool operator()(const mesh_query_aabb_t& q, const Mesh& m, int pi) const
    {
        const vec3 face_lower = bvh_load_vec3(m.lowers, pi);
        const vec3 face_upper = bvh_load_vec3(m.uppers, pi);
        return intersect_aabb_aabb(q.input_lower, q.input_upper, face_lower, face_upper);
    }
};

// Sphere primitive test: delegates to mesh_query_prim_test which handles
// the degenerate-face guard and the closest-point narrow phase.
struct SpherePrimitiveTest {
    CUDA_CALLABLE bool operator()(const mesh_query_aabb_t& q, const Mesh& m, int pi) const
    {
        return mesh_query_prim_test(q, m, pi);
    }
};

// TEST_SINGLETON: whether PrimitiveTest must run on singleton leaves. For the plain
// (broad-phase) AABB iterator the singleton leaf's bounds equal the cached primitive
// bounds, so the node test already decided the answer; re-running the primitive test
// costs two extra dependent global loads per visited singleton leaf (measured +9-21%
// on singleton-leaf trees, e.g. the cuBQL mesh default). The sphere iterator keeps
// the test - it is its narrow phase.
template <typename NodeTest, typename PrimitiveTest, bool TEST_SINGLETON = true>
CUDA_CALLABLE inline bool mesh_query_next_impl(mesh_query_aabb_t& query, int& index)
{
    Mesh mesh = query.mesh;

    // A single flat loop: every iteration either emits one primitive from the
    // packed leaf currently being enumerated, or pops and processes one node.
    for (;;) {
        if (query.prim_cur < query.prim_end) {
            const int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, query.prim_cur++);

            if (PrimitiveTest {}(query, mesh, primitive_index)) {
                index = primitive_index;
                query.face = primitive_index;
                return true;
            }
            continue;
        }

        if (!query.count)
            return false;

        const int node_index = query.stack[--query.count];
        BVHPackedNodeHalf node_lower = bvh_load_node(mesh.bvh.node_lowers, node_index);
        BVHPackedNodeHalf node_upper = bvh_load_node(mesh.bvh.node_uppers, node_index);

        if (!NodeTest {}(query, reinterpret_cast<vec3&>(node_lower), reinterpret_cast<vec3&>(node_upper)))
            continue;

        const int left_index = node_lower.i;
        const int right_index = node_upper.i;

        if (node_lower.b) {
            const int start = left_index;
            const int end = right_index;

            // Fast path when the leaf contains exactly one primitive: its AABB
            // is the leaf node's AABB, which just passed the node test above
            if (end - start == 1) {
                int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, start);
                bool singleton_hit = true;
                if constexpr (TEST_SINGLETON)
                    singleton_hit = PrimitiveTest {}(query, mesh, primitive_index);
                if (singleton_hit) {
                    index = primitive_index;
                    query.face = primitive_index;
                    return true;
                }
                continue;
            }

            // packed leaf: enumerate its primitives through the scalar cursors,
            // one per loop iteration, without re-pushing the leaf node
            query.prim_cur = start;
            query.prim_end = end;
        } else {
            query.stack[query.count++] = left_index;
            query.stack[query.count++] = right_index;
        }
    }
}

// Backward-compatible broad-phase AABB iterator. Called by mesh_query_aabb_next
// (the alias) and by mesh_query_next when the query object is MeshQueryAABB.
// AabbPrimitiveTest always passes after AabbNodeTest on singleton leaves
// (leaf AABB == primitive AABB), so the compiler folds it to a single test.
CUDA_CALLABLE inline bool mesh_query_aabb_next(mesh_query_aabb_t& query, int& index)
{
    return mesh_query_next_impl<AabbNodeTest, AabbPrimitiveTest, false>(query, index);
}

// Sphere iterator -- called from mesh_query_next when the query object is _MeshQuerySphere.
CUDA_CALLABLE inline bool mesh_query_sphere_next(mesh_query_aabb_t& query, int& index)
{
    return mesh_query_next_impl<SphereNodeTest, SpherePrimitiveTest>(query, index);
}

// Kind-erased iterator (MeshQuery type): used only when the concrete kind is unknown
// at compile time (a kernel branch merging two kinds, or a function parameter
// annotated with the parent type), so it dispatches on the kind stored at
// construction (radius_sq alone cannot distinguish a zero-radius sphere query from
// an AABB query). The statically-typed iterators above never route through here.
CUDA_CALLABLE inline bool mesh_query_next_dynamic(mesh_query_aabb_t& query, int& index)
{
    if (query.kind == MeshQueryKind::SPHERE)
        return mesh_query_sphere_next(query, index);
    return mesh_query_aabb_next(query, index);
}

CUDA_CALLABLE inline int iter_next(mesh_query_aabb_t& query) { return query.face; }

CUDA_CALLABLE inline bool iter_cmp(mesh_query_aabb_t& query)
{
    // The for-loop protocol shares one iter_cmp across all query kinds, so it must
    // dispatch on the stored kind.
    return mesh_query_next_dynamic(query, query.face);
}

CUDA_CALLABLE inline mesh_query_aabb_t iter_reverse(const mesh_query_aabb_t& query)
{
    // can't reverse BVH queries, users should not rely on neighbor ordering
    return query;
}

CUDA_CALLABLE inline vec3 mesh_eval_position(uint64_t id, int tri, float u, float v)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.points)
        return vec3();

    assert(tri < mesh.num_tris);

    int i = mesh.indices[tri * 3 + 0];
    int j = mesh.indices[tri * 3 + 1];
    int k = mesh.indices[tri * 3 + 2];

    vec3 p = mesh.points[i];
    vec3 q = mesh.points[j];
    vec3 r = mesh.points[k];

    return p * u + q * v + r * (1.0f - u - v);
}

CUDA_CALLABLE inline vec3 mesh_eval_velocity(uint64_t id, int tri, float u, float v)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.velocities)
        return vec3();

    assert(tri < mesh.num_tris);

    int i = mesh.indices[tri * 3 + 0];
    int j = mesh.indices[tri * 3 + 1];
    int k = mesh.indices[tri * 3 + 2];

    vec3 vp = mesh.velocities[i];
    vec3 vq = mesh.velocities[j];
    vec3 vr = mesh.velocities[k];

    return vp * u + vq * v + vr * (1.0f - u - v);
}


CUDA_CALLABLE inline void adj_mesh_eval_position(
    uint64_t id,
    int tri,
    float u,
    float v,
    uint64_t& adj_id,
    int& adj_tri,
    float& adj_u,
    float& adj_v,
    const vec3& adj_ret
)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.points)
        return;

    assert(tri < mesh.num_tris);

    int i = mesh.indices[tri * 3 + 0];
    int j = mesh.indices[tri * 3 + 1];
    int k = mesh.indices[tri * 3 + 2];

    vec3 p = mesh.points[i];
    vec3 q = mesh.points[j];
    vec3 r = mesh.points[k];

    adj_u += (p[0] - r[0]) * adj_ret[0] + (p[1] - r[1]) * adj_ret[1] + (p[2] - r[2]) * adj_ret[2];
    adj_v += (q[0] - r[0]) * adj_ret[0] + (q[1] - r[1]) * adj_ret[1] + (q[2] - r[2]) * adj_ret[2];
}

CUDA_CALLABLE inline void adj_mesh_eval_velocity(
    uint64_t id,
    int tri,
    float u,
    float v,
    uint64_t& adj_id,
    int& adj_tri,
    float& adj_u,
    float& adj_v,
    const vec3& adj_ret
)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.velocities)
        return;

    assert(tri < mesh.num_tris);

    int i = mesh.indices[tri * 3 + 0];
    int j = mesh.indices[tri * 3 + 1];
    int k = mesh.indices[tri * 3 + 2];

    vec3 vp = mesh.velocities[i];
    vec3 vq = mesh.velocities[j];
    vec3 vr = mesh.velocities[k];

    adj_u += (vp[0] - vr[0]) * adj_ret[0] + (vp[1] - vr[1]) * adj_ret[1] + (vp[2] - vr[2]) * adj_ret[2];
    adj_v += (vq[0] - vr[0]) * adj_ret[0] + (vq[1] - vr[1]) * adj_ret[1] + (vq[2] - vr[2]) * adj_ret[2];
}

CUDA_CALLABLE inline vec3 mesh_eval_face_normal(uint64_t id, int tri)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.points)
        return vec3();

    assert(tri < mesh.num_tris);

    int i = mesh.indices[tri * 3 + 0];
    int j = mesh.indices[tri * 3 + 1];
    int k = mesh.indices[tri * 3 + 2];

    vec3 p = mesh.points[i];
    vec3 q = mesh.points[j];
    vec3 r = mesh.points[k];

    return normalize(cross(q - p, r - p));
}

CUDA_CALLABLE inline void
adj_mesh_eval_face_normal(uint64_t id, int tri, uint64_t& adj_id, int& adj_tri, const vec3& adj_ret)
{
    // MISSINGADJOINT: backprop through normalize(cross(q-p, r-p)) to
    // mesh.points.grad slots for the three face vertex indices
}

CUDA_CALLABLE inline vec3 mesh_get_point(uint64_t id, int index)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.points)
        return vec3();

#if FP_CHECK
    if (index >= mesh.num_tris * 3) {
        printf("mesh_get_point (%llu, %d) out of bounds at %s:%d\n", id, index, __FILE__, __LINE__);
        assert(0);
    }
#endif

    int i = mesh.indices[index];
    return mesh.points[i];
}

CUDA_CALLABLE inline void
adj_mesh_get_point(uint64_t id, int index, uint64_t& adj_id, int& adj_index, const vec3& adj_ret)
{
    // MISSINGADJOINT: atomic-add adj_ret into mesh.points.grad[index] when the gradient
    // buffer is allocated
}

CUDA_CALLABLE inline vec3 mesh_get_velocity(uint64_t id, int index)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.velocities)
        return vec3();

#if FP_CHECK
    if (index >= mesh.num_tris * 3) {
        printf("mesh_get_velocity (%llu, %d) out of bounds at %s:%d\n", id, index, __FILE__, __LINE__);
        assert(0);
    }
#endif

    int i = mesh.indices[index];
    return mesh.velocities[i];
}

CUDA_CALLABLE inline void
adj_mesh_get_velocity(uint64_t id, int index, uint64_t& adj_id, int& adj_index, const vec3& adj_ret)
{
    // MISSINGADJOINT: atomic-add adj_ret into mesh.velocities.grad[index] when the
    // gradient buffer is allocated
}

CUDA_CALLABLE inline int mesh_get_index(uint64_t id, int face_vertex_index)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.indices)
        return -1;

    assert(face_vertex_index < mesh.num_tris * 3);

    return mesh.indices[face_vertex_index];
}

CUDA_CALLABLE bool mesh_get_descriptor(uint64_t id, Mesh& mesh);
CUDA_CALLABLE bool mesh_set_descriptor(uint64_t id, const Mesh& mesh);
CUDA_CALLABLE void mesh_add_descriptor(uint64_t id, const Mesh& mesh);
CUDA_CALLABLE void mesh_rem_descriptor(uint64_t id);

}  // namespace wp


#include "tile_mesh.h"
