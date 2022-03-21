/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
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

CUDA_CALLABLE inline vec3 closest_point_to_triangle(const vec3& a, const vec3& b, const vec3& c, const vec3& p, float& v, float& w)
{
	vec3 ab = b-a;
	vec3 ac = c-a;
	vec3 ap = p-a;
	
	float d1 = dot(ab, ap);
	float d2 = dot(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f)
	{
		v = 0.0f;
		w = 0.0f;
		return a;
	}

	vec3 bp = p-b;
	float d3 = dot(ab, bp);
	float d4 = dot(ac, bp);
	if (d3 >= 0.0f && d4 <= d3)
	{
		v = 1.0f;
		w = 0.0f;
		return b;
	}

	float vc = d1*d4 - d3*d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		v = d1 / (d1-d3);
		w = 0.0f;
		return a + v*ab;
	}

	vec3 cp =p-c;
	float d5 = dot(ab, cp);
	float d6 = dot(ac, cp);
	if (d6 >= 0.0f && d5 <= d6)
	{
		v = 0.0f;
		w = 1.0f;
		return c;
	}

	float vb = d5*d2 - d1*d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		v = 0.0f;
		w = d2 / (d2 - d6);
		return a + w * ac;
	}

	float va = d3*d6 - d5*d4;
	if (va <= 0.0f && (d4 -d3) >= 0.0f && (d5-d6) >= 0.0f)
	{
		w = (d4-d3)/((d4-d3) + (d5-d6));
		v = 1.0f-w;		
		return b + w * (c-b);
	}

	float denom = 1.0f / (va + vb + vc);
	v = vb * denom;
	w = vc * denom;
	return a + ab*v + ac*w;
}


CUDA_CALLABLE inline bool intersect_ray_aabb(const vec3& pos, const vec3& rcp_dir, const vec3& lower, const vec3& upper, float& t)
{
	float l1, l2, lmin, lmax;

    l1 = (lower.x - pos.x) * rcp_dir.x;
    l2 = (upper.x - pos.x) * rcp_dir.x;
    lmin = min(l1,l2);
    lmax = max(l1,l2);

    l1 = (lower.y - pos.y) * rcp_dir.y;
    l2 = (upper.y - pos.y) * rcp_dir.y;
    lmin = max(min(l1,l2), lmin);
    lmax = min(max(l1,l2), lmax);

    l1 = (lower.z - pos.z) * rcp_dir.z;
    l2 = (upper.z - pos.z) * rcp_dir.z;
    lmin = max(min(l1,l2), lmin);
    lmax = min(max(l1,l2), lmax);

    bool hit = ((lmax >= 0.f) & (lmax >= lmin));
    if (hit)
        t = lmin;

    return hit;
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
	float x = abs(a.x);
	float y = abs(a.y);
	float z = abs(a.z);

	return longest_axis(vec3(x, y, z));
}

// http://jcgt.org/published/0002/01/05/
CUDA_CALLABLE inline bool intersect_ray_tri_woop(const vec3& p, const vec3& dir, const vec3& a, const vec3& b, const vec3& c, float& t, float& u, float& v, float& w, float& sign, vec3* normal)
{
	// todo: precompute for ray

	int kz = max_dim(dir);
	int kx = kz+1; if (kx == 3) kx = 0;
	int ky = kx+1; if (ky == 3) ky = 0;

	if (dir[kz] < 0.0f)
	{
		float tmp = kx;
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
		return false;

	float det = U+V+W;

	if (det == 0.0f) 
		return false;

	const float Az = Sz*A[kz];
	const float Bz = Sz*B[kz];
	const float Cz = Sz*C[kz];
	const float T = U*Az + V*Bz + W*Cz;

	int det_sign = sign_mask(det);
	if (xorf(T,det_sign) < 0.0f)// || xorf(T,det_sign) > hit.t * xorf(det, det_sign)) // early out if hit.t is specified
		return false;

	const float rcpDet = 1.0f/det;
	u = U*rcpDet;
	v = V*rcpDet;
	w = W*rcpDet;
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

} // namespace wp