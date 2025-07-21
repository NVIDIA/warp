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

#pragma once

#include "builtin.h"

namespace wp
{

CUDA_CALLABLE inline vec3 closest_point_to_aabb(const vec3& p, const vec3& lower, const vec3& upper)
{
	vec3 c;

	{
		float v = p[0];
		if (v < lower[0]) v = lower[0];
		if (v > upper[0]) v = upper[0];
		c[0] = v;
	}

	{
		float v = p[1];
		if (v < lower[1]) v = lower[1];
		if (v > upper[1]) v = upper[1];
		c[1] = v;
	}

	{
		float v = p[2];
		if (v < lower[2]) v = lower[2];
		if (v > upper[2]) v = upper[2];
		c[2] = v;
	}

	return c;
}

CUDA_CALLABLE inline vec2 closest_point_to_triangle(const vec3& a, const vec3& b, const vec3& c, const vec3& p)
{
	vec3 ab = b-a;
	vec3 ac = c-a;
	vec3 ap = p-a;
	
	float u, v, w;
	float d1 = dot(ab, ap);
	float d2 = dot(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f)
	{
		v = 0.0f;
		w = 0.0f;
		u = 1.0f - v - w;
		return vec2(u, v);
	}

	vec3 bp = p-b;
	float d3 = dot(ab, bp);
	float d4 = dot(ac, bp);
	if (d3 >= 0.0f && d4 <= d3)
	{
		v = 1.0f;
		w = 0.0f;
		u = 1.0f - v - w;
		return vec2(u, v);
	}

	float vc = d1*d4 - d3*d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		v = d1 / (d1-d3);
		w = 0.0f;
		u = 1.0f - v - w;
		return vec2(u, v);
	}

	vec3 cp = p-c;
	float d5 = dot(ab, cp);
	float d6 = dot(ac, cp);
	if (d6 >= 0.0f && d5 <= d6)
	{
		v = 0.0f;
		w = 1.0f;
		u = 1.0f - v - w;
		return vec2(u, v);
	}

	float vb = d5*d2 - d1*d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		v = 0.0f;
		w = d2 / (d2 - d6);
		u = 1.0f - v - w;
		return vec2(u, v);
	}

	float va = d3*d6 - d5*d4;
	if (va <= 0.0f && (d4 -d3) >= 0.0f && (d5-d6) >= 0.0f)
	{
		w = (d4-d3)/((d4-d3) + (d5-d6));
		v = 1.0f - w;
		u = 1.0f - v - w;
		return vec2(u, v);
	}

	float denom = 1.0f / (va + vb + vc);
	v = vb * denom;
	w = vc * denom;
	u = 1.0f - v - w;
	return vec2(u, v);
}

CUDA_CALLABLE inline vec2 furthest_point_to_triangle(const vec3& a, const vec3& b, const vec3& c, const vec3& p)
{
    vec3 pa = p-a;
    vec3 pb = p-b;
    vec3 pc = p-c;
    float dist_a = dot(pa, pa);
    float dist_b = dot(pb, pb);
    float dist_c = dot(pc, pc);

    if (dist_a > dist_b && dist_a > dist_c) 
        return vec2(1.0f, 0.0f); // a is furthest
    if (dist_b > dist_c)
        return vec2(0.0f, 1.0f); // b is furthest
    return vec2(0.0f, 0.0f); // c is furthest
}

CUDA_CALLABLE inline bool intersect_ray_aabb(const vec3& pos, const vec3& rcp_dir, const vec3& lower, const vec3& upper, float& t)
{
	float l1, l2, lmin, lmax;

    l1 = (lower[0] - pos[0]) * rcp_dir[0];
    l2 = (upper[0] - pos[0]) * rcp_dir[0];
    lmin = min(l1,l2);
    lmax = max(l1,l2);

    l1 = (lower[1] - pos[1]) * rcp_dir[1];
    l2 = (upper[1] - pos[1]) * rcp_dir[1];
    lmin = max(min(l1,l2), lmin);
    lmax = min(max(l1,l2), lmax);

    l1 = (lower[2] - pos[2]) * rcp_dir[2];
    l2 = (upper[2] - pos[2]) * rcp_dir[2];
    lmin = max(min(l1,l2), lmin);
    lmax = min(max(l1,l2), lmax);

    bool hit = ((lmax >= 0.f) & (lmax >= lmin));
    if (hit)
        t = lmin;

    return hit;
}

CUDA_CALLABLE inline bool intersect_aabb_aabb(const vec3& a_lower, const vec3& a_upper, const vec3& b_lower, const vec3& b_upper)
{
    if (a_lower[0] > b_upper[0] ||
        a_lower[1] > b_upper[1] ||
        a_lower[2] > b_upper[2] ||
        a_upper[0] < b_lower[0] ||
        a_upper[1] < b_lower[1] ||
        a_upper[2] < b_lower[2])
    {
        return false;
    }
    else
    {
        return true;
    }
}


// Moller and Trumbore's method
CUDA_CALLABLE inline bool intersect_ray_tri_moller(const vec3& p, const vec3& dir, const vec3& a, const vec3& b, const vec3& c, float& t, float& u, float& v, float& w, float& sign, vec3* normal)
{
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 n = cross(ab, ac);

    float d = dot(-dir, n);
    float ood = 1.0f / d; // No need to check for division by zero here as infinity arithmetic will save us...
    vec3 ap = p - a;

    t = dot(ap, n) * ood;
    if (t < 0.0f)
        return false;

    vec3 e = cross(-dir, ap);
    v = dot(ac, e) * ood;
    if (v < 0.0f || v > 1.0f) // ...here...
        return false;
    w = -dot(ab, e) * ood;
    if (w < 0.0f || (v + w) > 1.0f) // ...and here
        return false;

    u = 1.0f - v - w;
    if (normal)
        *normal = n;

	sign = d;

    return true;
}


CUDA_CALLABLE inline bool intersect_ray_tri_rtcd(const vec3& p, const vec3& dir, const vec3& a, const vec3& b, const vec3& c, float& t, float& u, float& v, float& w, float& sign, vec3* normal)
{
	const vec3 ab = b-a;
	const vec3 ac = c-a;

	// calculate normal
	vec3 n = cross(ab, ac);

	// need to solve a system of three equations to give t, u, v
	float d = dot(-dir, n);

	// if dir is parallel to triangle plane or points away from triangle 
	if (d <= 0.0f)
        return false;

	vec3 ap = p-a;
	t = dot(ap, n);

	// ignores tris behind 
	if (t < 0.0f)
		return false;

	// compute barycentric coordinates
	vec3 e = cross(-dir, ap);
	v = dot(ac, e);
	if (v < 0.0f || v > d) return false;

	w = -dot(ab, e);
	if (w < 0.0f || v + w > d) return false;

	float ood = 1.0f / d;
	t *= ood;
	v *= ood;
	w *= ood;
	u = 1.0f-v-w;

	// optionally write out normal (todo: this branch is a performance concern, should probably remove)
	if (normal)
		*normal = n;

	return true;
}

#ifndef  __CUDA_ARCH__

// these are provided as built-ins by CUDA
inline float __int_as_float(int i)
{
	return *(float*)(&i);
}

inline int __float_as_int(float f)
{
	return *(int*)(&f);
}

#endif


CUDA_CALLABLE inline float xorf(float x, int y)
{
	return __int_as_float(__float_as_int(x) ^ y);
}

CUDA_CALLABLE inline int sign_mask(float x)
{
	return __float_as_int(x) & 0x80000000;
}

CUDA_CALLABLE inline int max_dim(vec3 a)
{
	float x = abs(a[0]);
	float y = abs(a[1]);
	float z = abs(a[2]);

	return longest_axis(vec3(x, y, z));
}

// computes the difference of products a*b - c*d using 
// FMA instructions for improved numerical precision
CUDA_CALLABLE inline float diff_product(float a, float b, float c, float d) 
{
    float cd = c * d;
    float diff = fmaf(a, b, -cd);
    float error = fmaf(-c, d, cd);

    return diff + error;
}

// http://jcgt.org/published/0002/01/05/
CUDA_CALLABLE inline bool intersect_ray_tri_woop(const vec3& p, const vec3& dir, const vec3& a, const vec3& b, const vec3& c, float& t, float& u, float& v, float& sign, vec3* normal)
{
	// todo: precompute for ray

	int kz = max_dim(dir);
	int kx = kz+1; if (kx == 3) kx = 0;
	int ky = kx+1; if (ky == 3) ky = 0;

	if (dir[kz] < 0.0f)
	{
		int tmp = kx;
		kx = ky;
		ky = tmp;
	}

	float Sx = dir[kx]/dir[kz];
	float Sy = dir[ky]/dir[kz];
	float Sz = 1.0f/dir[kz];

	// todo: end precompute

	const vec3 A = a-p;
	const vec3 B = b-p;
	const vec3 C = c-p;
	
	const float Ax = A[kx] - Sx*A[kz];
	const float Ay = A[ky] - Sy*A[kz];
	const float Bx = B[kx] - Sx*B[kz];
	const float By = B[ky] - Sy*B[kz];
	const float Cx = C[kx] - Sx*C[kz];
	const float Cy = C[ky] - Sy*C[kz];
		
    float U = diff_product(Cx, By, Cy, Bx);
    float V = diff_product(Ax, Cy, Ay, Cx);
    float W = diff_product(Bx, Ay, By, Ax);

	if (U == 0.0f || V == 0.0f || W == 0.0f) 
	{
		double CxBy = (double)Cx*(double)By;
		double CyBx = (double)Cy*(double)Bx;
		U = (float)(CxBy - CyBx);
		double AxCy = (double)Ax*(double)Cy;
		double AyCx = (double)Ay*(double)Cx;
		V = (float)(AxCy - AyCx);
		double BxAy = (double)Bx*(double)Ay;
		double ByAx = (double)By*(double)Ax;
		W = (float)(BxAy - ByAx);
	}

	if ((U<0.0f || V<0.0f || W<0.0f) &&	(U>0.0f || V>0.0f || W>0.0f)) 
    {
        return false;
    }

	float det = U+V+W;

	if (det == 0.0f) 
    {
		return false;
    }

	const float Az = Sz*A[kz];
	const float Bz = Sz*B[kz];
	const float Cz = Sz*C[kz];
	const float T = U*Az + V*Bz + W*Cz;

	int det_sign = sign_mask(det);
	if (xorf(T,det_sign) < 0.0f)// || xorf(T,det_sign) > hit.t * xorf(det, det_sign)) // early out if hit.t is specified
    {
		return false;
    }

	const float rcpDet = 1.0f/det;
	u = U*rcpDet;
	v = V*rcpDet;
	t = T*rcpDet;
	sign = det;
	
	// optionally write out normal (todo: this branch is a performance concern, should probably remove)
	if (normal)
	{
		const vec3 ab = b-a;
		const vec3 ac = c-a;

		// calculate normal
		*normal = cross(ab, ac); 
	}

	return true;
}

CUDA_CALLABLE inline void adj_intersect_ray_tri_woop(
    const vec3& p, const vec3& dir, const vec3& a, const vec3& b, const vec3& c, float t, float u, float v, float sign, const vec3& normal,
    vec3& adj_p, vec3& adj_dir, vec3& adj_a, vec3& adj_b, vec3& adj_c, float& adj_t, float& adj_u, float& adj_v, float& adj_sign, vec3& adj_normal, bool& adj_ret)
{

	// todo: precompute for ray

	int kz = max_dim(dir);
	int kx = kz+1; if (kx == 3) kx = 0;
	int ky = kx+1; if (ky == 3) ky = 0;

	if (dir[kz] < 0.0f)
	{
		int tmp = kx;
		kx = ky;
		ky = tmp;
	}

    const float Dx = dir[kx];
    const float Dy = dir[ky];
    const float Dz = dir[kz];

	const float Sx = dir[kx]/dir[kz];
	const float Sy = dir[ky]/dir[kz];
	const float Sz = 1.0f/dir[kz];

	// todo: end precompute

	const vec3 A = a-p;
	const vec3 B = b-p;
	const vec3 C = c-p;
	
	const float Ax = A[kx] - Sx*A[kz];
	const float Ay = A[ky] - Sy*A[kz];
	const float Bx = B[kx] - Sx*B[kz];
	const float By = B[ky] - Sy*B[kz];
	const float Cx = C[kx] - Sx*C[kz];
	const float Cy = C[ky] - Sy*C[kz];
		
	float U = Cx*By - Cy*Bx;
	float V = Ax*Cy - Ay*Cx;
	float W = Bx*Ay - By*Ax;

	if (U == 0.0f || V == 0.0f || W == 0.0f) 
	{
		double CxBy = (double)Cx*(double)By;
		double CyBx = (double)Cy*(double)Bx;
		U = (float)(CxBy - CyBx);
		double AxCy = (double)Ax*(double)Cy;
		double AyCx = (double)Ay*(double)Cx;
		V = (float)(AxCy - AyCx);
		double BxAy = (double)Bx*(double)Ay;
		double ByAx = (double)By*(double)Ax;
		W = (float)(BxAy - ByAx);
	}

	if ((U<0.0f || V<0.0f || W<0.0f) &&	(U>0.0f || V>0.0f || W>0.0f)) 
		return;

	float det = U+V+W;

	if (det == 0.0f) 
		return;

	const float Az = Sz*A[kz];
	const float Bz = Sz*B[kz];
	const float Cz = Sz*C[kz];
	const float T = U*Az + V*Bz + W*Cz;

	int det_sign = sign_mask(det);
	if (xorf(T,det_sign) < 0.0f)// || xorf(T,det_sign) > hit.t * xorf(det, det_sign)) // early out if hit.t is specified
		return;

    const float rcpDet = (1.f / det);
    const float rcpDetSq = rcpDet * rcpDet;

    // adj_p

    const float dAx_dpx = -1.f;
    const float dBx_dpx = -1.f;
    const float dCx_dpx = -1.f;
    const float dAy_dpx = 0.f;
    const float dBy_dpx = 0.f;
    const float dCy_dpx = 0.f;
    const float dAz_dpx = 0.f;
    const float dBz_dpx = 0.f;
    const float dCz_dpx = 0.f;

    const float dAx_dpy = 0.f;
    const float dBx_dpy = 0.f;
    const float dCx_dpy = 0.f;
    const float dAy_dpy = -1.f;
    const float dBy_dpy = -1.f;
    const float dCy_dpy = -1.f;
    const float dAz_dpy = 0.f;
    const float dBz_dpy = 0.f;
    const float dCz_dpy = 0.f;
    
    const float dAx_dpz = Sx;
    const float dBx_dpz = Sx;
    const float dCx_dpz = Sx;
    const float dAy_dpz = Sy;
    const float dBy_dpz = Sy;
    const float dCy_dpz = Sy;
    const float dAz_dpz = -Sz;
    const float dBz_dpz = -Sz;
    const float dCz_dpz = -Sz;

    const float dU_dpx = Cx * dBy_dpx + By * dCx_dpx - Cy * dBx_dpx - Bx * dCy_dpx;
    const float dU_dpy = Cx * dBy_dpy + By * dCx_dpy - Cy * dBx_dpy - Bx * dCy_dpy;
    const float dU_dpz = Cx * dBy_dpz + By * dCx_dpz - Cy * dBx_dpz - Bx * dCy_dpz;
    const vec3 dU_dp = vec3(dU_dpx, dU_dpy, dU_dpz);

    const float dV_dpx = Ax * dCy_dpx + Cy * dAx_dpx - Ay * dCx_dpx - Cx * dAy_dpx;
    const float dV_dpy = Ax * dCy_dpy + Cy * dAx_dpy - Ay * dCx_dpy - Cx * dAy_dpy;
    const float dV_dpz = Ax * dCy_dpz + Cy * dAx_dpz - Ay * dCx_dpz - Cx * dAy_dpz;
    const vec3 dV_dp = vec3(dV_dpx, dV_dpy, dV_dpz);

    const float dW_dpx = Bx * dAy_dpx + Ay * dBx_dpx - By * dAx_dpx - Ax * dBy_dpx;
    const float dW_dpy = Bx * dAy_dpy + Ay * dBx_dpy - By * dAx_dpy - Ax * dBy_dpy;
    const float dW_dpz = Bx * dAy_dpz + Ay * dBx_dpz - By * dAx_dpz - Ax * dBy_dpz;
    const vec3 dW_dp = vec3(dW_dpx, dW_dpy, dW_dpz);

    const float dT_dpx = dU_dpx * Az + U * dAz_dpx + dV_dpx * Bz + V * dBz_dpx + dW_dpx * Cz + W * dCz_dpx;
    const float dT_dpy = dU_dpy * Az + U * dAz_dpy + dV_dpy * Bz + V * dBz_dpy + dW_dpy * Cz + W * dCz_dpy;
    const float dT_dpz = dU_dpz * Az + U * dAz_dpz + dV_dpz * Bz + V * dBz_dpz + dW_dpz * Cz + W * dCz_dpz;
    const vec3 dT_dp = vec3(dT_dpx, dT_dpy, dT_dpz);

    const float dDet_dpx = dU_dpx + dV_dpx + dW_dpx;
    const float dDet_dpy = dU_dpy + dV_dpy + dW_dpy;
    const float dDet_dpz = dU_dpz + dV_dpz + dW_dpz;
    const vec3 dDet_dp = vec3(dDet_dpx, dDet_dpy, dDet_dpz);

    const vec3 du_dp = rcpDet * dU_dp + -U * rcpDetSq * dDet_dp;
    const vec3 dv_dp = rcpDet * dV_dp + -V * rcpDetSq * dDet_dp;
    const vec3 dt_dp = rcpDet * dT_dp + -T * rcpDetSq * dDet_dp;

    vec3 adj_p_swapped = adj_u*du_dp + adj_v*dv_dp + adj_t*dt_dp;
    adj_p[kx] += adj_p_swapped[0];
    adj_p[ky] += adj_p_swapped[1];
    adj_p[kz] += adj_p_swapped[2];

    // adj_dir

    const float dAx_dDx = -Sz * A[kz];
    const float dBx_dDx = -Sz * B[kz];
    const float dCx_dDx = -Sz * C[kz];
    const float dAy_dDx = 0.f;
    const float dBy_dDx = 0.f;
    const float dCy_dDx = 0.f;
    const float dAz_dDx = 0.f;
    const float dBz_dDx = 0.f;
    const float dCz_dDx = 0.f;
    
    const float dAx_dDy = 0.f;
    const float dBx_dDy = 0.f;
    const float dCx_dDy = 0.f;
    const float dAy_dDy = -Sz * A[kz];
    const float dBy_dDy = -Sz * B[kz];
    const float dCy_dDy = -Sz * C[kz];
    const float dAz_dDy = 0.f;
    const float dBz_dDy = 0.f;
    const float dCz_dDy = 0.f;
    
    const float dAx_dDz = Dx * Sz * Sz * A[kz];
    const float dBx_dDz = Dx * Sz * Sz * B[kz];
    const float dCx_dDz = Dx * Sz * Sz * C[kz];
    const float dAy_dDz = Dy * Sz * Sz * A[kz];
    const float dBy_dDz = Dy * Sz * Sz * B[kz];
    const float dCy_dDz = Dy * Sz * Sz * C[kz];
    const float dAz_dDz = -Sz * Sz * A[kz];
    const float dBz_dDz = -Sz * Sz * B[kz];
    const float dCz_dDz = -Sz * Sz * C[kz];

    const float dU_dDx = Cx * dBy_dDx + By * dCx_dDx - Cy * dBx_dDx - Bx * dCy_dDx;
    const float dU_dDy = Cx * dBy_dDy + By * dCx_dDy - Cy * dBx_dDy - Bx * dCy_dDy;
    const float dU_dDz = Cx * dBy_dDz + By * dCx_dDz - Cy * dBx_dDz - Bx * dCy_dDz;
    const vec3 dU_dD = vec3(dU_dDx, dU_dDy, dU_dDz);

    const float dV_dDx = Ax * dCy_dDx + Cy * dAx_dDx - Ay * dCx_dDx - Cx * dAy_dDx;
    const float dV_dDy = Ax * dCy_dDy + Cy * dAx_dDy - Ay * dCx_dDy - Cx * dAy_dDy;
    const float dV_dDz = Ax * dCy_dDz + Cy * dAx_dDz - Ay * dCx_dDz - Cx * dAy_dDz;
    const vec3 dV_dD = vec3(dV_dDx, dV_dDy, dV_dDz);

    const float dW_dDx = Bx * dAy_dDx + Ay * dBx_dDx - By * dAx_dDx - Ax * dBy_dDx;
    const float dW_dDy = Bx * dAy_dDy + Ay * dBx_dDy - By * dAx_dDy - Ax * dBy_dDy;
    const float dW_dDz = Bx * dAy_dDz + Ay * dBx_dDz - By * dAx_dDz - Ax * dBy_dDz;
    const vec3 dW_dD = vec3(dW_dDx, dW_dDy, dW_dDz);

    const float dT_dDx = dU_dDx * Az + U * dAz_dDx + dV_dDx * Bz + V * dBz_dDx + dW_dDx * Cz + W * dCz_dDx;
    const float dT_dDy = dU_dDy * Az + U * dAz_dDy + dV_dDy * Bz + V * dBz_dDy + dW_dDy * Cz + W * dCz_dDy;
    const float dT_dDz = dU_dDz * Az + U * dAz_dDz + dV_dDz * Bz + V * dBz_dDz + dW_dDz * Cz + W * dCz_dDz;
    const vec3 dT_dD = vec3(dT_dDx, dT_dDy, dT_dDz);

    const float dDet_dDx = dU_dDx + dV_dDx + dW_dDx;
    const float dDet_dDy = dU_dDy + dV_dDy + dW_dDy;
    const float dDet_dDz = dU_dDz + dV_dDz + dW_dDz;
    const vec3 dDet_dD = vec3(dDet_dDx, dDet_dDy, dDet_dDz);

    const vec3 du_dD = rcpDet * dU_dD + -U * rcpDetSq * dDet_dD;
    const vec3 dv_dD = rcpDet * dV_dD + -V * rcpDetSq * dDet_dD;
    const vec3 dt_dD = rcpDet * dT_dD + -T * rcpDetSq * dDet_dD;

    vec3 adj_dir_swapped = adj_u*du_dD + adj_v*dv_dD + adj_t*dt_dD;
    adj_dir[kx] += adj_dir_swapped[0];
    adj_dir[ky] += adj_dir_swapped[1];
    adj_dir[kz] += adj_dir_swapped[2];
}

// Möller's method
#include "intersect_tri.h"

CUDA_CALLABLE inline int intersect_tri_tri(
    vec3& v0, vec3& v1, vec3& v2,
    vec3& u0, vec3& u1, vec3& u2)
{
	return NoDivTriTriIsect(&v0[0], &v1[0], &v2[0], &u0[0], &u1[0], &u2[0]);
}

CUDA_CALLABLE inline void adj_intersect_tri_tri(const vec3& var_v0,
												const vec3& var_v1,
												const vec3& var_v2,
												const vec3& var_u0,
												const vec3& var_u1,
												const vec3& var_u2,
												vec3& adj_v0,
												vec3& adj_v1,
												vec3& adj_v2,
												vec3& adj_u0,
												vec3& adj_u1,
												vec3& adj_u2,
												int adj_ret) {}


CUDA_CALLABLE inline void adj_closest_point_to_triangle(
	const vec3& var_a, const vec3& var_b, const vec3& var_c, const vec3& var_p,
	vec3& adj_a, vec3& adj_b, vec3& adj_c, vec3& adj_p, vec2& adj_ret)
{

    // primal vars
    vec3 var_0;
    vec3 var_1;
    vec3 var_2;
    float32 var_3;
    float32 var_4;
    const float32 var_5 = 0.0;
    bool var_6;
    bool var_7;
    bool var_8;
    const float32 var_9 = 1.0;
    vec2 var_10;
    vec3 var_11;
    float32 var_12;
    float32 var_13;
    bool var_14;
    bool var_15;
    bool var_16;
    vec2 var_17;
    vec2 var_18;
    float32 var_19;
    float32 var_20;
    float32 var_21;
    float32 var_22;
    float32 var_23;
    bool var_24;
    bool var_25;
    bool var_26;
    bool var_27;
    float32 var_28 = 0.0;
    vec2 var_29;
    vec2 var_30;
    vec3 var_31;
    float32 var_32;
    float32 var_33;
    bool var_34;
    bool var_35;
    bool var_36;
    vec2 var_37;
    vec2 var_38;
    float32 var_39;
    float32 var_40;
    float32 var_41;
    float32 var_42;
    float32 var_43;
    bool var_44;
    bool var_45;
    bool var_46;
    bool var_47;
    float32 var_48 = 0.0;
    vec2 var_49;
    vec2 var_50;
    float32 var_51;
    float32 var_52;
    float32 var_53;
    float32 var_54;
    float32 var_55;
    float32 var_56;
    float32 var_57;
    float32 var_58;
    bool var_59;
    float32 var_60;
    bool var_61;
    float32 var_62;
    bool var_63;
    bool var_64;
    float32 var_65 = 0.0;
    vec2 var_66;
    // vec2 var_67;
    float32 var_68;
    float32 var_69;
    float32 var_70;
    float32 var_71;
    float32 var_72;
    float32 var_73;
    float32 var_74;
    // vec2 var_75;
    //---------
    // dual vars
    vec3 adj_0 = 0;
    vec3 adj_1 = 0;
    vec3 adj_2 = 0;
    float32 adj_3 = 0;
    float32 adj_4 = 0;
    float32 adj_5 = 0;
    //bool adj_6 = 0;
    //bool adj_7 = 0;
    //bool adj_8 = 0;
    float32 adj_9 = 0;
    vec2 adj_10 = 0;
    vec3 adj_11 = 0;
    float32 adj_12 = 0;
    float32 adj_13 = 0;
    //bool adj_14 = 0;
    //bool adj_15 = 0;
    bool adj_16 = 0;
    vec2 adj_17 = 0;
    vec2 adj_18 = 0;
    float32 adj_19 = 0;
    float32 adj_20 = 0;
    float32 adj_21 = 0;
    float32 adj_22 = 0;
    float32 adj_23 = 0;
    //bool adj_24 = 0;
    //bool adj_25 = 0;
    //bool adj_26 = 0;
    bool adj_27 = 0;
    float32 adj_28 = 0;
    vec2 adj_29 = 0;
    vec2 adj_30 = 0;
    vec3 adj_31 = 0;
    float32 adj_32 = 0;
    float32 adj_33 = 0;
    //bool adj_34 = 0;
    //bool adj_35 = 0;
    bool adj_36 = 0;
    vec2 adj_37 = 0;
    vec2 adj_38 = 0;
    float32 adj_39 = 0;
    float32 adj_40 = 0;
    float32 adj_41 = 0;
    float32 adj_42 = 0;
    float32 adj_43 = 0;
    //bool adj_44 = 0;
    //bool adj_45 = 0;
    //bool adj_46 = 0;
    bool adj_47 = 0;
    float32 adj_48 = 0;
    vec2 adj_49 = 0;
    vec2 adj_50 = 0;
    float32 adj_51 = 0;
    float32 adj_52 = 0;
    float32 adj_53 = 0;
    float32 adj_54 = 0;
    float32 adj_55 = 0;
    float32 adj_56 = 0;
    float32 adj_57 = 0;
    float32 adj_58 = 0;
    //bool adj_59 = 0;
    float32 adj_60 = 0;
    //bool adj_61 = 0;
    float32 adj_62 = 0;
    //bool adj_63 = 0;
    bool adj_64 = 0;
    float32 adj_65 = 0;
    vec2 adj_66 = 0;
    vec2 adj_67 = 0;
    float32 adj_68 = 0;
    float32 adj_69 = 0;
    float32 adj_70 = 0;
    float32 adj_71 = 0;
    float32 adj_72 = 0;
    float32 adj_73 = 0;
    float32 adj_74 = 0;
    vec2 adj_75 = 0;
    //---------
    // forward
    var_0 = wp::sub(var_b, var_a);
    var_1 = wp::sub(var_c, var_a);
    var_2 = wp::sub(var_p, var_a);
    var_3 = wp::dot(var_0, var_2);
    var_4 = wp::dot(var_1, var_2);
    var_6 = (var_3 <= var_5);
    var_7 = (var_4 <= var_5);
    var_8 = var_6 && var_7;
    if (var_8) {
    	var_10 = wp::vec2(var_9, var_5);
    	goto label0;
    }
    var_11 = wp::sub(var_p, var_b);
    var_12 = wp::dot(var_0, var_11);
    var_13 = wp::dot(var_1, var_11);
    var_14 = (var_12 >= var_5);
    var_15 = (var_13 <= var_12);
    var_16 = var_14 && var_15;
    if (var_16) {
    	var_17 = wp::vec2(var_5, var_9);
    	goto label1;
    }
    var_18 = wp::where(var_16, var_17, var_10);
    var_19 = wp::mul(var_3, var_13);
    var_20 = wp::mul(var_12, var_4);
    var_21 = wp::sub(var_19, var_20);
    var_22 = wp::sub(var_3, var_12);
    var_23 = wp::div(var_3, var_22);
    var_24 = (var_21 <= var_5);
    var_25 = (var_3 >= var_5);
    var_26 = (var_12 <= var_5);
    var_27 = var_24 && var_25 && var_26;
    if (var_27) {
    	var_28 = wp::sub(var_9, var_23);
    	var_29 = wp::vec2(var_28, var_23);
    	goto label2;
    }
    var_30 = wp::where(var_27, var_29, var_18);
    var_31 = wp::sub(var_p, var_c);
    var_32 = wp::dot(var_0, var_31);
    var_33 = wp::dot(var_1, var_31);
    var_34 = (var_33 >= var_5);
    var_35 = (var_32 <= var_33);
    var_36 = var_34 && var_35;
    if (var_36) {
    	var_37 = wp::vec2(var_5, var_5);
    	goto label3;
    }
    var_38 = wp::where(var_36, var_37, var_30);
    var_39 = wp::mul(var_32, var_4);
    var_40 = wp::mul(var_3, var_33);
    var_41 = wp::sub(var_39, var_40);
    var_42 = wp::sub(var_4, var_33);
    var_43 = wp::div(var_4, var_42);
    var_44 = (var_41 <= var_5);
    var_45 = (var_4 >= var_5);
    var_46 = (var_33 <= var_5);
    var_47 = var_44 && var_45 && var_46;
    if (var_47) {
    	var_48 = wp::sub(var_9, var_43);
    	var_49 = wp::vec2(var_48, var_5);
    	goto label4;
    }
    var_50 = wp::where(var_47, var_49, var_38);
    var_51 = wp::mul(var_12, var_33);
    var_52 = wp::mul(var_32, var_13);
    var_53 = wp::sub(var_51, var_52);
    var_54 = wp::sub(var_13, var_12);
    var_55 = wp::sub(var_13, var_12);
    var_56 = wp::sub(var_32, var_33);
    var_57 = wp::add(var_55, var_56);
    var_58 = wp::div(var_54, var_57);
    var_59 = (var_53 <= var_5);
    var_60 = wp::sub(var_13, var_12);
    var_61 = (var_60 >= var_5);
    var_62 = wp::sub(var_32, var_33);
    var_63 = (var_62 >= var_5);
    var_64 = var_59 && var_61 && var_63;
    if (var_64) {
    	var_65 = wp::sub(var_9, var_58);
    	var_66 = wp::vec2(var_5, var_65);
    	goto label5;
    }
    // var_67 = wp::where(var_64, var_66, var_50);
    var_68 = wp::add(var_53, var_41);
    var_69 = wp::add(var_68, var_21);
    var_70 = wp::div(var_9, var_69);
    var_71 = wp::mul(var_41, var_70);
    var_72 = wp::mul(var_21, var_70);
    var_73 = wp::sub(var_9, var_71);
    var_74 = wp::sub(var_73, var_72);
    // var_75 = wp::vec2(var_74, var_71);
    goto label6;
    //---------
    // reverse
    label6:;
    adj_75 += adj_ret;
    wp::adj_vec2(var_74, var_71, adj_74, adj_71, adj_75);
    wp::adj_sub(var_73, var_72, adj_73, adj_72, adj_74);
    wp::adj_sub(var_9, var_71, adj_9, adj_71, adj_73);
    wp::adj_mul(var_21, var_70, adj_21, adj_70, adj_72);
    wp::adj_mul(var_41, var_70, adj_41, adj_70, adj_71);
    wp::adj_div(var_9, var_69, var_70, adj_9, adj_69, adj_70);
    wp::adj_add(var_68, var_21, adj_68, adj_21, adj_69);
    wp::adj_add(var_53, var_41, adj_53, adj_41, adj_68);
    wp::adj_where(var_64, var_66, var_50, adj_64, adj_66, adj_50, adj_67);
    if (var_64) {
    	label5:;
    	adj_66 += adj_ret;
    	wp::adj_vec2(var_5, var_65, adj_5, adj_65, adj_66);
    	wp::adj_sub(var_9, var_58, adj_9, adj_58, adj_65);
    }
    wp::adj_sub(var_32, var_33, adj_32, adj_33, adj_62);
    wp::adj_sub(var_13, var_12, adj_13, adj_12, adj_60);
    wp::adj_div(var_54, var_57, var_58, adj_54, adj_57, adj_58);
    wp::adj_add(var_55, var_56, adj_55, adj_56, adj_57);
    wp::adj_sub(var_32, var_33, adj_32, adj_33, adj_56);
    wp::adj_sub(var_13, var_12, adj_13, adj_12, adj_55);
    wp::adj_sub(var_13, var_12, adj_13, adj_12, adj_54);
    wp::adj_sub(var_51, var_52, adj_51, adj_52, adj_53);
    wp::adj_mul(var_32, var_13, adj_32, adj_13, adj_52);
    wp::adj_mul(var_12, var_33, adj_12, adj_33, adj_51);
    wp::adj_where(var_47, var_49, var_38, adj_47, adj_49, adj_38, adj_50);
    if (var_47) {
    	label4:;
    	adj_49 += adj_ret;
    	wp::adj_vec2(var_48, var_5, adj_48, adj_5, adj_49);
    	wp::adj_sub(var_9, var_43, adj_9, adj_43, adj_48);
    }
    wp::adj_div(var_4, var_42, var_43, adj_4, adj_42, adj_43);
    wp::adj_sub(var_4, var_33, adj_4, adj_33, adj_42);
    wp::adj_sub(var_39, var_40, adj_39, adj_40, adj_41);
    wp::adj_mul(var_3, var_33, adj_3, adj_33, adj_40);
    wp::adj_mul(var_32, var_4, adj_32, adj_4, adj_39);
    wp::adj_where(var_36, var_37, var_30, adj_36, adj_37, adj_30, adj_38);
    if (var_36) {
    	label3:;
    	adj_37 += adj_ret;
    	wp::adj_vec2(var_5, var_5, adj_5, adj_5, adj_37);
    }
    wp::adj_dot(var_1, var_31, adj_1, adj_31, adj_33);
    wp::adj_dot(var_0, var_31, adj_0, adj_31, adj_32);
    wp::adj_sub(var_p, var_c, adj_p, adj_c, adj_31);
    wp::adj_where(var_27, var_29, var_18, adj_27, adj_29, adj_18, adj_30);
    if (var_27) {
    	label2:;
    	adj_29 += adj_ret;
    	wp::adj_vec2(var_28, var_23, adj_28, adj_23, adj_29);
    	wp::adj_sub(var_9, var_23, adj_9, adj_23, adj_28);
    }
    wp::adj_div(var_3, var_22, var_23, adj_3, adj_22, adj_23);
    wp::adj_sub(var_3, var_12, adj_3, adj_12, adj_22);
    wp::adj_sub(var_19, var_20, adj_19, adj_20, adj_21);
    wp::adj_mul(var_12, var_4, adj_12, adj_4, adj_20);
    wp::adj_mul(var_3, var_13, adj_3, adj_13, adj_19);
    wp::adj_where(var_16, var_17, var_10, adj_16, adj_17, adj_10, adj_18);
    if (var_16) {
    	label1:;
    	adj_17 += adj_ret;
    	wp::adj_vec2(var_5, var_9, adj_5, adj_9, adj_17);
    }
    wp::adj_dot(var_1, var_11, adj_1, adj_11, adj_13);
    wp::adj_dot(var_0, var_11, adj_0, adj_11, adj_12);
    wp::adj_sub(var_p, var_b, adj_p, adj_b, adj_11);
    if (var_8) {
    	label0:;
    	adj_10 += adj_ret;
    	wp::adj_vec2(var_9, var_5, adj_9, adj_5, adj_10);
    }
    wp::adj_dot(var_1, var_2, adj_1, adj_2, adj_4);
    wp::adj_dot(var_0, var_2, adj_0, adj_2, adj_3);
    wp::adj_sub(var_p, var_a, adj_p, adj_a, adj_2);
    wp::adj_sub(var_c, var_a, adj_c, adj_a, adj_1);
    wp::adj_sub(var_b, var_a, adj_b, adj_a, adj_0);
    return;

}


    
// ----------------------------------------------------------------
// jleaf: I needed to replace "float(" with "cast_float(" manually below because 
// "#define float(x) cast_float(x)"" in this header affects other files.
// See adjoint in "intersect_adj.h" for the generated adjoint.
/* 
  Here is the original warp implementation that was used to generate this code:

# https://books.google.ca/books?id=WGpL6Sk9qNAC&printsec=frontcover&hl=en#v=onepage&q=triangle&f=false
# From 5.1.9
# p1 and q1 are points of edge 1.
# p2 and q2 are points of edge 2.
# epsilon zero tolerance for determining if points in an edge are degenerate
# output: A single wp.vec3, containing s and t for edges 1 and 2 respectively,
# and the distance between their closest points.
@wp.func
def closest_point_edge_edge(
    p1: wp.vec3, q1: wp.vec3, p2: wp.vec3, q2: wp.vec3, epsilon: float
):
    # direction vectors of each segment/edge
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2

    a = wp.dot(d1, d1)  # squared length of segment s1, always nonnegative
    e = wp.dot(d2, d2)  # squared length of segment s2, always nonnegative
    f = wp.dot(d2, r)

    s = float(0.0)
    t = float(0.0)
    dist = wp.length(p2 - p1)

    # Check if either or both segments degenerate into points
    if a <= epsilon and e <= epsilon:
        # both segments degenerate into points
        return wp.vec3(s, t, dist)

    if a <= epsilon:
        s = float(0.0)
        t = float(f / e)  # s = 0 => t = (b*s + f) / e = f / e
    else:
        c = wp.dot(d1, r)
        if e <= epsilon:
            # second segment generates into a point
            s = wp.clamp(-c / a, 0.0, 1.0)  # t = 0 => s = (b*t-c)/a = -c/a
            t = float(0.0)
        else:
            # The general nondegenerate case starts here
            b = wp.dot(d1, d2)
            denom = a * e - b * b  # always nonnegative

            # if segments not parallel, compute closest point on L1 to L2 and
            # clamp to segment S1. Else pick arbitrary s (here 0)
            if denom != 0.0:
                s = wp.clamp((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0

            # compute point on L2 closest to S1(s) using
            # t = dot((p1+d2*s) - p2,d2)/dot(d2,d2) = (b*s+f)/e
            t = (b * s + f) / e

            # if t in [0,1] done. Else clamp t, recompute s for the new value
            # of t using s = dot((p2+d2*t-p1,d1)/dot(d1,d1) = (t*b - c)/a
            # and clamp s to [0,1]
            if t < 0.0:
                t = 0.0
                s = wp.clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = wp.clamp((b - c) / a, 0.0, 1.0)

    c1 = p1 + (q1 - p1) * s
    c2 = p2 + (q2 - p2) * t
    dist = wp.length(c2 - c1)
    return wp.vec3(s, t, dist)

*/

static CUDA_CALLABLE vec3 closest_point_edge_edge(vec3 var_p1,
	vec3 var_q1,
	vec3 var_p2,
	vec3 var_q2,
	float32 var_epsilon)
{
    //---------
    // primal vars
    vec3 var_0;
    vec3 var_1;
    vec3 var_2;
    float32 var_3;
    float32 var_4;
    float32 var_5;
    const float32 var_6 = 0.0;
    float32 var_7;
    float32 var_8;
    vec3 var_9;
    float32 var_10;
    bool var_11;
    bool var_12;
    bool var_13;
    vec3 var_14;
    bool var_15;
    float32 var_16;
    float32 var_17;
    float32 var_18;
    float32 var_19;
    float32 var_20;
    float32 var_21;
    bool var_22;
    float32 var_23;
    float32 var_24;
    const float32 var_25 = 1.0;
    float32 var_26;
    float32 var_27;
    float32 var_28;
    float32 var_29;
    float32 var_30;
    float32 var_31;
    float32 var_32;
    float32 var_33;
    bool var_34;
    float32 var_35;
    float32 var_36;
    float32 var_37;
    float32 var_38;
    float32 var_39;
    float32 var_40;
    float32 var_41;
    float32 var_42;
    float32 var_43;
    float32 var_44;
    bool var_45;
    float32 var_46;
    float32 var_47;
    float32 var_48;
    float32 var_49;
    float32 var_50;
    bool var_51;
    float32 var_52;
    float32 var_53;
    float32 var_54;
    float32 var_55;
    float32 var_56;
    float32 var_57;
    float32 var_58;
    float32 var_59;
    float32 var_60;
    float32 var_61;
    float32 var_62;
    vec3 var_63;
    vec3 var_64;
    vec3 var_65;
    vec3 var_66;
    vec3 var_67;
    vec3 var_68;
    vec3 var_69;
    float32 var_70;
    vec3 var_71;
    //---------
    // forward
    var_0 = wp::sub(var_q1, var_p1);
    var_1 = wp::sub(var_q2, var_p2);
    var_2 = wp::sub(var_p1, var_p2);
    var_3 = wp::dot(var_0, var_0);
    var_4 = wp::dot(var_1, var_1);
    var_5 = wp::dot(var_1, var_2);
    var_7 = wp::cast_float(var_6);
    var_8 = wp::cast_float(var_6);
    var_9 = wp::sub(var_p2, var_p1);
    var_10 = wp::length(var_9);
    var_11 = (var_3 <= var_epsilon);
    var_12 = (var_4 <= var_epsilon);
    var_13 = var_11 && var_12;
    if (var_13) {
    	var_14 = wp::vec3(var_7, var_8, var_10);
    	return var_14;
    }
    var_15 = (var_3 <= var_epsilon);
    if (var_15) {
    	var_16 = wp::cast_float(var_6);
    	var_17 = wp::div(var_5, var_4);
    	var_18 = wp::cast_float(var_17);
    }
    var_19 = wp::where(var_15, var_16, var_7);
    var_20 = wp::where(var_15, var_18, var_8);
    if (!var_15) {
    	var_21 = wp::dot(var_0, var_2);
    	var_22 = (var_4 <= var_epsilon);
    	if (var_22) {
    		var_23 = wp::neg(var_21);
    		var_24 = wp::div(var_23, var_3);
    		var_26 = wp::clamp(var_24, var_6, var_25);
    		var_27 = wp::cast_float(var_6);
    	}
    	var_28 = wp::where(var_22, var_26, var_19);
    	var_29 = wp::where(var_22, var_27, var_20);
    	if (!var_22) {
    		var_30 = wp::dot(var_0, var_1);
    		var_31 = wp::mul(var_3, var_4);
    		var_32 = wp::mul(var_30, var_30);
    		var_33 = wp::sub(var_31, var_32);
    		var_34 = (var_33 != var_6);
    		if (var_34) {
    			var_35 = wp::mul(var_30, var_5);
    			var_36 = wp::mul(var_21, var_4);
    			var_37 = wp::sub(var_35, var_36);
    			var_38 = wp::div(var_37, var_33);
    			var_39 = wp::clamp(var_38, var_6, var_25);
    		}
    		var_40 = wp::where(var_34, var_39, var_28);
    		if (!var_34) {
    		}
    		var_41 = wp::where(var_34, var_40, var_6);
    		var_42 = wp::mul(var_30, var_41);
    		var_43 = wp::add(var_42, var_5);
    		var_44 = wp::div(var_43, var_4);
    		var_45 = (var_44 < var_6);
    		if (var_45) {
    			var_46 = wp::neg(var_21);
    			var_47 = wp::div(var_46, var_3);
    			var_48 = wp::clamp(var_47, var_6, var_25);
    		}
    		var_49 = wp::where(var_45, var_48, var_41);
    		var_50 = wp::where(var_45, var_6, var_44);
    		if (!var_45) {
    			var_51 = (var_50 > var_25);
    			if (var_51) {
    				var_52 = wp::sub(var_30, var_21);
    				var_53 = wp::div(var_52, var_3);
    				var_54 = wp::clamp(var_53, var_6, var_25);
    			}
    			var_55 = wp::where(var_51, var_54, var_49);
    			var_56 = wp::where(var_51, var_25, var_50);
    		}
    		var_57 = wp::where(var_45, var_49, var_55);
    		var_58 = wp::where(var_45, var_50, var_56);
    	}
    	var_59 = wp::where(var_22, var_28, var_57);
    	var_60 = wp::where(var_22, var_29, var_58);
    }
    var_61 = wp::where(var_15, var_19, var_59);
    var_62 = wp::where(var_15, var_20, var_60);
    var_63 = wp::sub(var_q1, var_p1);
    var_64 = wp::mul(var_63, var_61);
    var_65 = wp::add(var_p1, var_64);
    var_66 = wp::sub(var_q2, var_p2);
    var_67 = wp::mul(var_66, var_62);
    var_68 = wp::add(var_p2, var_67);
    var_69 = wp::sub(var_68, var_65);
    var_70 = wp::length(var_69);
    var_71 = wp::vec3(var_61, var_62, var_70);
    return var_71;

}
} // namespace wp
