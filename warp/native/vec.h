/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#if !defined(__CUDACC__)
#include <initializer_list>
#include "string.h" // memset
#endif

namespace wp
{

    template <unsigned Length, typename Type>
    struct vec
    {
        Type c[Length];

        inline CUDA_CALLABLE vec()
        {
            memset(c, 0, Length * sizeof(Type));
        }

        inline CUDA_CALLABLE vec(Type s)
        {
            for (unsigned i = 0; i < Length; ++i)
            {
                c[i] = s;
            }
        }

        // Implementing constructors with fixed argument lists.

        // I wonder if not having these for Length > 4 will cause
        // problems... These things are immutable in kernels aren't
        // they so you need these constructors to acutally put data
        // in them...
        inline CUDA_CALLABLE vec(Type x, Type y)
        {
            assert(Length == 2);
            c[0] = x;
            c[1] = y;
        }

        inline CUDA_CALLABLE vec(Type x, Type y, Type z)
        {
            assert(Length == 3);
            c[0] = x;
            c[1] = y;
            c[2] = z;
        }

        inline CUDA_CALLABLE vec(Type x, Type y, Type z, Type w)
        {
            assert(Length == 4);
            c[0] = x;
            c[1] = y;
            c[2] = z;
            c[3] = w;
        }

        inline CUDA_CALLABLE vec(std::initializer_list<Type> l)
        {
            assert(l.size() == Length);
            auto src = l.begin();
            auto end = l.end();
            for (auto dst = c; src != end; ++src, ++dst)
            {
                *dst = *src;
            }
        }

        inline CUDA_CALLABLE Type operator[](int index) const
        {
            assert(index < Length);
            return c[index];
        }

        inline CUDA_CALLABLE Type &operator[](int index)
        {
            assert(index < Length);
            return c[index];
        }
    };

    //--------------
    // vec<Length, Type> methods

    // Should these accept const references as arguments? It's all
    // inlined so maybe it doesn't matter? Even if it does, it
    // probably depends on the Length of the vector...

    // negation:
    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> operator-(vec<Length, Type> a)
    {
        // NB: this constructor will initialize all ret's components to 0, which is
        // unnecessary...
        vec<Length, Type> ret;
        for (unsigned i = 0; i < Length; ++i)
        {
            ret[i] = -a[i];
        }

        // Wonder if this does a load of copying when it returns... hopefully not as it's inlined?
        return ret;
    }

    template <unsigned Length, typename Type>
    CUDA_CALLABLE inline vec<Length, Type> neg(const vec<Length, Type> &x)
    {
        return -x;
    }

    template <typename Type>
    CUDA_CALLABLE inline vec<3, Type> neg(const vec<3, Type> &x)
    {
        return vec<3, Type>(-x.c[0], -x.c[1], -x.c[2]);
    }

    template <typename Type>
    CUDA_CALLABLE inline vec<2, Type> neg(const vec<2, Type> &x)
    {
        return vec<2, Type>(-x.c[0], -x.c[1]);
    }

    template <unsigned Length, typename Type>
    CUDA_CALLABLE inline void adj_neg(const vec<Length, Type> &x, vec<Length, Type> &adj_x, const vec<Length, Type> &adj_ret)
    {
        adj_x -= adj_ret;
    }

    // equality:
    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE bool operator==(const vec<Length, Type> &a, const vec<Length, Type> &b)
    {
        for (unsigned i = 0; i < Length; ++i)
        {
            if (a[i] != b[i])
            {
                return false;
            }
        }
        return true;
    }

    // scalar multiplication:
    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> mul(vec<Length, Type> a, Type s)
    {
        vec<Length, Type> ret;
        for (unsigned i = 0; i < Length; ++i)
        {
            ret[i] = a[i] * s;
        }
        return ret;
    }

    template <typename Type>
    inline CUDA_CALLABLE vec<3, Type> mul(vec<3, Type> a, Type s)
    {
        return vec<3, Type>(a.c[0] * s, a.c[1] * s, a.c[2] * s);
    }

    template <typename Type>
    inline CUDA_CALLABLE vec<2, Type> mul(vec<2, Type> a, Type s)
    {
        return vec<2, Type>(a.c[0] * s, a.c[1] * s);
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> mul(Type s, vec<Length, Type> a)
    {
        return mul(a, s);
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> operator*(Type s, vec<Length, Type> a)
    {
        return mul(a, s);
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> operator*(vec<Length, Type> a, Type s)
    {
        return mul(a, s);
    }

    // component wise multiplication:
    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> cw_mul(vec<Length, Type> a, vec<Length, Type> b)
    {
        vec<Length, Type> ret;
        for (unsigned i = 0; i < Length; ++i)
        {
            ret[i] = a[i] * b[i];
        }
        return ret;
    }

    // division
    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> div(vec<Length, Type> a, Type s)
    {
        vec<Length, Type> ret;
        for (unsigned i = 0; i < Length; ++i)
        {
            ret[i] = a[i] / s;
        }
        return ret;
    }

    template <typename Type>
    inline CUDA_CALLABLE vec<3, Type> div(vec<3, Type> a, Type s)
    {
        return vec<3, Type>(a.c[0] / s, a.c[1] / s, a.c[2] / s);
    }

    template <typename Type>
    inline CUDA_CALLABLE vec<2, Type> div(vec<2, Type> a, Type s)
    {
        return vec<2, Type>(a.c[0] / s, a.c[1] / s);
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> operator/(vec<Length, Type> a, Type s)
    {
        return div(a, s);
    }

    // component wise division
    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> cw_div(vec<Length, Type> a, vec<Length, Type> b)
    {
        vec<Length, Type> ret;
        for (unsigned i = 0; i < Length; ++i)
        {
            ret[i] = a[i] / b[i];
        }
        return ret;
    }

    // addition
    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> add(vec<Length, Type> a, vec<Length, Type> b)
    {
        vec<Length, Type> ret;
        for (unsigned i = 0; i < Length; ++i)
        {
            ret[i] = a[i] + b[i];
        }
        return ret;
    }

    template <typename Type>
    inline CUDA_CALLABLE vec<2, Type> add(vec<2, Type> a, vec<2, Type> b)
    {
        return vec<2, Type>(a.c[0] + b.c[0], a.c[1] + b.c[1]);
    }

    template <typename Type>
    inline CUDA_CALLABLE vec<3, Type> add(vec<3, Type> a, vec<3, Type> b)
    {
        return vec<3, Type>(a.c[0] + b.c[0], a.c[1] + b.c[1], a.c[2] + b.c[2]);
    }

    // subtraction
    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> sub(vec<Length, Type> a, vec<Length, Type> b)
    {
        vec<Length, Type> ret;
        for (unsigned i = 0; i < Length; ++i)
        {
            ret[i] = Type(a[i] - b[i]);
        }
        return ret;
    }

    template <typename Type>
    inline CUDA_CALLABLE vec<2, Type> sub(vec<2, Type> a, vec<2, Type> b)
    {
        return vec<2, Type>(a.c[0] - b.c[0], a.c[1] - b.c[1]);
    }

    template <typename Type>
    inline CUDA_CALLABLE vec<3, Type> sub(vec<3, Type> a, vec<3, Type> b)
    {
        return vec<3, Type>(a.c[0] - b.c[0], a.c[1] - b.c[1], a.c[2] - b.c[2]);
    }

    // dot product:
    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE Type dot(vec<Length, Type> a, vec<Length, Type> b)
    {
        Type ret(0);
        for (unsigned i = 0; i < Length; ++i)
        {
            ret += a[i] * b[i];
        }
        return ret;
    }

    template <typename Type>
    inline CUDA_CALLABLE Type dot(vec<2, Type> a, vec<2, Type> b)
    {
        return a.c[0] * b.c[0] + a.c[1] * b.c[1];
    }

    template <typename Type>
    inline CUDA_CALLABLE Type dot(vec<3, Type> a, vec<3, Type> b)
    {
        return a.c[0] * b.c[0] + a.c[1] * b.c[1] + a.c[2] * b.c[2];
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE Type tensordot(vec<Length, Type> a, vec<Length, Type> b)
    {
        // corresponds to `np.tensordot()` with all axes being contracted
        return dot(a, b);
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE Type index(const vec<Length, Type> &a, int idx)
    {
#if FP_CHECK
        if (idx < 0 || idx > Length)
        {
            printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
            assert(0);
        }
#endif

        return a[idx];
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE Type length(vec<Length, Type> a)
    {
        return sqrt(dot(a, a));
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE Type length_sq(vec<Length, Type> a)
    {
        return dot(a, a);
    }

    template <typename Type>
    inline CUDA_CALLABLE Type length(vec<2, Type> a)
    {
        return sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1]);
    }

    template <typename Type>
    inline CUDA_CALLABLE Type length(vec<3, Type> a)
    {
        return sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1] + a.c[2] * a.c[2]);
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> normalize(vec<Length, Type> a)
    {
        Type l = length(a);
        if (l > Type(kEps))
            return div(a, l);
        else
            return vec<Length, Type>();
    }

    template <typename Type>
    inline CUDA_CALLABLE vec<2, Type> normalize(vec<2, Type> a)
    {
        Type l = sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1]);
        if (l > Type(kEps))
            return vec<2, Type>(a.c[0] / l, a.c[1] / l);
        else
            return vec<2, Type>();
    }

    template <typename Type>
    inline CUDA_CALLABLE vec<3, Type> normalize(vec<3, Type> a)
    {
        Type l = sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1] + a.c[2] * a.c[2]);
        if (l > Type(kEps))
            return vec<3, Type>(a.c[0] / l, a.c[1] / l, a.c[2] / l);
        else
            return vec<3, Type>();
    }

    template <typename Type>
    inline CUDA_CALLABLE vec<3, Type> cross(vec<3, Type> a, vec<3, Type> b)
    {
        return {
            Type(a[1] * b[2] - a[2] * b[1]),
            Type(a[2] * b[0] - a[0] * b[2]),
            Type(a[0] * b[1] - a[1] * b[0])};
    }

    template <unsigned Length, typename Type>
    inline bool CUDA_CALLABLE isfinite(vec<Length, Type> x)
    {
        for (unsigned i = 0; i < Length; ++i)
        {
            if (!isfinite(x[i]))
            {
                return false;
            }
        }
        return true;
    }

    // These two functions seem to compile very slowly
    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> min(vec<Length, Type> a, vec<Length, Type> b)
    {
        vec<Length, Type> ret;
        for (unsigned i = 0; i < Length; ++i)
        {
            ret[i] = a[i] < b[i] ? a[i] : b[i];
        }
        return ret;
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> max(vec<Length, Type> a, vec<Length, Type> b)
    {
        vec<Length, Type> ret;
        for (unsigned i = 0; i < Length; ++i)
        {
            ret[i] = a[i] > b[i] ? a[i] : b[i];
        }
        return ret;
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> abs(vec<Length, Type> a)
    {
        vec<Length, Type> ret;
        for (unsigned i = 0; i < Length; ++i)
        {
            ret[i] = abs(a[i]);
        }
        return ret;
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> sign(vec<Length, Type> a)
    {
        vec<Length, Type> ret;
        for (unsigned i = 0; i < Length; ++i)
        {
            ret[i] = sign(a[i]);
        }
        return ret;
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void expect_near(const vec<Length, Type> &actual, const vec<Length, Type> &expected, const Type &tolerance)
    {
        const Type diff(0);
        for (size_t i = 0; i < Length; ++i)
        {
            diff = max(diff, abs(actual[i] - expected[i]));
        }
        if (diff > tolerance)
        {
            printf("Error, expect_near() failed with torerance ");
            print(tolerance);
            printf("\t Expected: ");
            print(expected);
            printf("\t Actual: ");
            print(actual);
        }
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_expect_near(const vec<Length, Type> &actual, const vec<Length, Type> &expected, Type tolerance, vec<Length, Type> &adj_actual, vec<Length, Type> &adj_expected, Type adj_tolerance)
    {
        // nop
    }

    // adjoint for the initializer_list constructor:
    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_vec(std::initializer_list<Type> cmps, std::initializer_list<Type *> adj_cmps, const vec<Length, Type> &adj_ret)
    {
        auto it = adj_cmps.begin();
        for (unsigned i = 0; i < Length; ++i, ++it)
        {
            *(*it) += adj_ret[i];
        }
    }

    // adjoint for the component constructors:
    template <typename Type>
    inline CUDA_CALLABLE void adj_vec(Type cmpx, Type cmpy, Type &adj_cmpx, Type &adj_cmpy, const vec<2, Type> &adj_ret)
    {
        adj_cmpx += adj_ret.c[0];
        adj_cmpy += adj_ret.c[1];
    }

    template <typename Type>
    inline CUDA_CALLABLE void adj_vec(Type cmpx, Type cmpy, Type cmpz, Type &adj_cmpx, Type &adj_cmpy, Type &adj_cmpz, const vec<3, Type> &adj_ret)
    {
        adj_cmpx += adj_ret.c[0];
        adj_cmpy += adj_ret.c[1];
        adj_cmpz += adj_ret.c[2];
    }

    template <typename Type>
    inline CUDA_CALLABLE void adj_vec(Type cmpx, Type cmpy, Type cmpz, Type cmpw, Type &adj_cmpx, Type &adj_cmpy, Type &adj_cmpz, Type &adj_cmpw, const vec<4, Type> &adj_ret)
    {
        adj_cmpx += adj_ret.c[0];
        adj_cmpy += adj_ret.c[1];
        adj_cmpz += adj_ret.c[2];
        adj_cmpw += adj_ret.c[3];
    }

    // adjoint for the constant constructor:
    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_vec(Type s, Type &adj_s, const vec<Length, Type> &adj_ret)
    {
        for (unsigned i = 0; i < Length; ++i)
        {
            adj_s += adj_ret[i];
        }
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_mul(vec<Length, Type> a, Type s, vec<Length, Type> &adj_a, Type &adj_s, const vec<Length, Type> &adj_ret)
    {
        for (unsigned i = 0; i < Length; ++i)
        {
            adj_a[i] += s * adj_ret[i];
        }

        adj_s += dot(a, adj_ret);

#if FP_CHECK
        if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) || !isfinite(adj_ret))
        {
            // \TODO: How shall we implement this error message?
            // printf("adj_mul((%f %f %f %f), %f, (%f %f %f %f), %f, (%f %f %f %f)\n", a.x, a.y, a.z, a.w, s, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_s, adj_ret.x, adj_ret.y, adj_ret.z, adj_ret.w);
            assert(0);
        }
#endif
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_mul(Type s, vec<Length, Type> a, Type &adj_s, vec<Length, Type> &adj_a, const vec<Length, Type> &adj_ret)
    {
        adj_mul(a, s, adj_a, adj_s, adj_ret);
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_cw_mul(vec<Length, Type> a, vec<Length, Type> b, vec<Length, Type> &adj_a, vec<Length, Type> &adj_b, const vec<Length, Type> &adj_ret)
    {
        adj_a += cw_mul(b, adj_ret);
        adj_b += cw_mul(a, adj_ret);
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_div(vec<Length, Type> a, Type s, vec<Length, Type> &adj_a, Type &adj_s, const vec<Length, Type> &adj_ret)
    {

        adj_s -= dot(a, adj_ret) / (s * s); // - a / s^2

        for (unsigned i = 0; i < Length; ++i)
        {
            adj_a[i] += adj_ret[i] / s;
        }

#if FP_CHECK
        if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) || !isfinite(adj_ret))
        {
            // \TODO: How shall we implement this error message?
            // printf("adj_div((%f %f %f %f), %f, (%f %f %f %f), %f, (%f %f %f %f)\n", a.x, a.y, a.z, a.w, s, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_s, adj_ret.x, adj_ret.y, adj_ret.z, adj_ret.w);
            assert(0);
        }
#endif
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_cw_div(vec<Length, Type> a, vec<Length, Type> b, vec<Length, Type> &adj_a, vec<Length, Type> &adj_b, const vec<Length, Type> &adj_ret)
    {
        adj_a += cw_div(adj_ret, b);
        adj_b -= cw_mul(adj_ret, cw_div(cw_div(a, b), b));
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_add(vec<Length, Type> a, vec<Length, Type> b, vec<Length, Type> &adj_a, vec<Length, Type> &adj_b, const vec<Length, Type> &adj_ret)
    {
        adj_a += adj_ret;
        adj_b += adj_ret;
    }

    template <typename Type>
    inline CUDA_CALLABLE void adj_add(vec<2, Type> a, vec<2, Type> b, vec<2, Type> &adj_a, vec<2, Type> &adj_b, const vec<2, Type> &adj_ret)
    {
        adj_a.c[0] += adj_ret.c[0];
        adj_a.c[1] += adj_ret.c[1];
        adj_b.c[0] += adj_ret.c[0];
        adj_b.c[1] += adj_ret.c[1];
    }

    template <typename Type>
    inline CUDA_CALLABLE void adj_add(vec<3, Type> a, vec<3, Type> b, vec<3, Type> &adj_a, vec<3, Type> &adj_b, const vec<3, Type> &adj_ret)
    {
        adj_a.c[0] += adj_ret.c[0];
        adj_a.c[1] += adj_ret.c[1];
        adj_a.c[2] += adj_ret.c[2];
        adj_b.c[0] += adj_ret.c[0];
        adj_b.c[1] += adj_ret.c[1];
        adj_b.c[2] += adj_ret.c[2];
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_sub(vec<Length, Type> a, vec<Length, Type> b, vec<Length, Type> &adj_a, vec<Length, Type> &adj_b, const vec<Length, Type> &adj_ret)
    {
        adj_a += adj_ret;
        adj_b -= adj_ret;
    }

    template <typename Type>
    inline CUDA_CALLABLE void adj_sub(vec<2, Type> a, vec<2, Type> b, vec<2, Type> &adj_a, vec<2, Type> &adj_b, const vec<2, Type> &adj_ret)
    {
        adj_a.c[0] += adj_ret.c[0];
        adj_a.c[1] += adj_ret.c[1];
        adj_b.c[0] -= adj_ret.c[0];
        adj_b.c[1] -= adj_ret.c[1];
    }

    template <typename Type>
    inline CUDA_CALLABLE void adj_sub(vec<3, Type> a, vec<3, Type> b, vec<3, Type> &adj_a, vec<3, Type> &adj_b, const vec<3, Type> &adj_ret)
    {
        adj_a.c[0] += adj_ret.c[0];
        adj_a.c[1] += adj_ret.c[1];
        adj_a.c[2] += adj_ret.c[2];
        adj_b.c[0] -= adj_ret.c[0];
        adj_b.c[1] -= adj_ret.c[1];
        adj_b.c[2] -= adj_ret.c[2];
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_dot(vec<Length, Type> a, vec<Length, Type> b, vec<Length, Type> &adj_a, vec<Length, Type> &adj_b, const Type adj_ret)
    {
        adj_a += b * adj_ret;
        adj_b += a * adj_ret;

#if FP_CHECK
        if (!isfinite(a) || !isfinite(b) || !isfinite(adj_a) || !isfinite(adj_b) || !isfinite(adj_ret))
        {
            // \TODO: How shall we implement this error message?
            // printf("adj_dot((%f %f %f %f), (%f %f %f %f), (%f %f %f %f), (%f %f %f %f), %f)\n", a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_b.x, adj_b.y, adj_b.z, adj_b.w, adj_ret);
            assert(0);
        }
#endif
    }

    template <typename Type>
    inline CUDA_CALLABLE void adj_dot(vec<2, Type> a, vec<2, Type> b, vec<2, Type> &adj_a, vec<2, Type> &adj_b, const Type adj_ret)
    {
        adj_a.c[0] += b.c[0] * adj_ret;
        adj_a.c[1] += b.c[1] * adj_ret;

        adj_b.c[0] += a.c[0] * adj_ret;
        adj_b.c[1] += a.c[1] * adj_ret;
    }

    template <typename Type>
    inline CUDA_CALLABLE void adj_dot(vec<3, Type> a, vec<3, Type> b, vec<3, Type> &adj_a, vec<3, Type> &adj_b, const Type adj_ret)
    {
        adj_a.c[0] += b.c[0] * adj_ret;
        adj_a.c[1] += b.c[1] * adj_ret;
        adj_a.c[2] += b.c[2] * adj_ret;

        adj_b.c[0] += a.c[0] * adj_ret;
        adj_b.c[1] += a.c[1] * adj_ret;
        adj_b.c[2] += a.c[2] * adj_ret;
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_index(const vec<Length, Type> &a, int idx, vec<Length, Type> &adj_a, int &adj_idx, Type &adj_ret)
    {
#if FP_CHECK
        if (idx < 0 || idx > Length)
        {
            printf("Tvec2<Scalar> index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
            assert(0);
        }
#endif

        adj_a[idx] += adj_ret;
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_length(vec<Length, Type> a, vec<Length, Type> &adj_a, const Type adj_ret)
    {
        adj_a += normalize(a) * adj_ret;

#if FP_CHECK
        if (!isfinite(adj_a))
        {
            // \TODO: How shall we implement this error message?
            // printf("%s:%d - adj_length((%f %f %f %f), (%f %f %f %f), (%f))\n", __FILE__, __LINE__, a.x, a.y, a.z, a.w, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_ret);
            assert(0);
        }
#endif
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_length_sq(vec<Length, Type> a, vec<Length, Type> &adj_a, const Type adj_ret)
    {
        adj_a += Type(2.0) * a * adj_ret;

#if FP_CHECK
        if (!isfinite(adj_a))
        {
            // \TODO: How shall we implement this error message?
            // printf("%s:%d - adj_length((%f %f %f %f), (%f %f %f %f), (%f))\n", __FILE__, __LINE__, a.x, a.y, a.z, a.w, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_ret);
            assert(0);
        }
#endif
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_normalize(vec<Length, Type> a, vec<Length, Type> &adj_a, const vec<Length, Type> &adj_ret)
    {
        Type d = length(a);

        if (d > Type(kEps))
        {
            Type invd = Type(1.0f) / d;

            vec<Length, Type> ahat = normalize(a);

            adj_a += (adj_ret * invd - ahat * (dot(ahat, adj_ret)) * invd);

#if FP_CHECK
            if (!isfinite(adj_a))
            {
                // \TODO: How shall we implement this error message?
                // printf("%s:%d - adj_normalize((%f %f %f %f), (%f %f %f %f), (%f, %f, %f, %f))\n", __FILE__, __LINE__, a.x, a.y, a.z, a.w, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_ret.x, adj_ret.y, adj_ret.z, adj_ret.w);
                assert(0);
            }
#endif
        }
    }

    template <typename Type>
    inline CUDA_CALLABLE void adj_cross(vec<3, Type> a, vec<3, Type> b, vec<3, Type> &adj_a, vec<3, Type> &adj_b, const vec<3, Type> &adj_ret)
    {
        // todo: sign check
        adj_a += cross(b, adj_ret);
        adj_b -= cross(a, adj_ret);
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_min(const vec<Length, Type> &a, const vec<Length, Type> &b, vec<Length, Type> &adj_a, vec<Length, Type> &adj_b, const vec<Length, Type> &adj_ret)
    {
        for (unsigned i = 0; i < Length; ++i)
        {
            if (a[i] < b[i])
                adj_a[i] += adj_ret[i];
            else
                adj_b[i] += adj_ret[i];
        }
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_max(const vec<Length, Type> &a, const vec<Length, Type> &b, vec<Length, Type> &adj_a, vec<Length, Type> &adj_b, const vec<Length, Type> &adj_ret)
    {
        for (unsigned i = 0; i < Length; ++i)
        {
            if (a[i] > b[i])
                adj_a[i] += adj_ret[i];
            else
                adj_b[i] += adj_ret[i];
        }
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_abs(const vec<Length, Type> &a, vec<Length, Type> &adj_a, const vec<Length, Type> &adj_ret)
    {
        for (unsigned i = 0; i < Length; ++i)
        {
            if (a[i] > Type(0))
                adj_a[i] += adj_ret[i];
            else if (a[i] < Type(0))
                adj_a[i] -= adj_ret[i];
            else
                adj_a[i] += Type(0); // derivative of abs(x) at x=0 is not defined, we take 0 here
        }
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE void adj_sign(const vec<Length, Type> &a, vec<Length, Type> &adj_a, const vec<Length, Type> &adj_ret)
    {
        for (unsigned i = 0; i < Length; ++i)
        {
            if (a[i] > Type(0))
                adj_a[i] += adj_ret[i];
            else if (a[i] < Type(0))
                adj_a[i] -= adj_ret[i];
            else
                adj_a[i] += Type(0); // derivative of sign(x) at x=0 is not defined, we take 0 here
        }
    }

    // Do I need to specialize these for different lengths?
    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> atomic_add(vec<Length, Type> *addr, vec<Length, Type> value)
    {

        vec<Length, Type> ret;
        for (unsigned i = 0; i < Length; ++i)
        {
            ret[i] = atomic_add(&(addr->c[i]), value[i]);
        }

        return ret;
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> atomic_min(vec<Length, Type> *addr, vec<Length, Type> value)
    {

        vec<Length, Type> ret;
        for (unsigned i = 0; i < Length; ++i)
        {
            ret[i] = atomic_min(&(addr->c[i]), value[i]);
        }

        return ret;
    }

    template <unsigned Length, typename Type>
    inline CUDA_CALLABLE vec<Length, Type> atomic_max(vec<Length, Type> *addr, vec<Length, Type> value)
    {

        vec<Length, Type> ret;
        for (unsigned i = 0; i < Length; ++i)
        {
            ret[i] = atomic_max(&(addr->c[i]), value[i]);
        }

        return ret;
    }

    // ok, the original implementation of this didn't take the absolute values.
    // I wouldn't consider this expected behavior. It looks like it's only
    // being used for bounding boxes at the moment, where this doesn't matter,
    // but you often use it for ray tracing where it does. Not sure if the
    // fabs() incurs a performance hit...
    template <unsigned Length, typename Type>
    CUDA_CALLABLE inline int longest_axis(const vec<Length, Type> &v)
    {
        Type lmax = fabs(v[0]);
        int ret(0);
        for (unsigned i = 1; i < Length; ++i)
        {
            Type l = fabs(v[i]);
            if (l > lmax)
            {
                ret = i;
                lmax = l;
            }
        }
        return ret;
    }

    template <unsigned Length, typename Type>
    CUDA_CALLABLE inline vec<Length, Type> lerp(const vec<Length, Type> &a, const vec<Length, Type> &b, Type t)
    {
        return a * (Type(1) - t) + b * t;
    }

    template <unsigned Length, typename Type>
    CUDA_CALLABLE inline void adj_lerp(const vec<Length, Type> &a, const vec<Length, Type> &b, Type t, vec<Length, Type> &adj_a, vec<Length, Type> &adj_b, Type &adj_t, const vec<Length, Type> &adj_ret)
    {
        adj_a += adj_ret * (Type(1) - t);
        adj_b += adj_ret * t;
        adj_t += tensordot(b, adj_ret) - tensordot(a, adj_ret);
    }

    // for integral types we do not accumulate gradients
    template <unsigned Length>
    CUDA_CALLABLE inline void adj_atomic_add(vec<Length, int8> *buf, const vec<Length, int8> &value) {}
    template <unsigned Length>
    CUDA_CALLABLE inline void adj_atomic_add(vec<Length, uint8> *buf, const vec<Length, uint8> &value) {}
    template <unsigned Length>
    CUDA_CALLABLE inline void adj_atomic_add(vec<Length, int16> *buf, const vec<Length, int16> &value) {}
    template <unsigned Length>
    CUDA_CALLABLE inline void adj_atomic_add(vec<Length, uint16> *buf, const vec<Length, uint16> &value) {}
    template <unsigned Length>
    CUDA_CALLABLE inline void adj_atomic_add(vec<Length, int32> *buf, const vec<Length, int32> &value) {}
    template <unsigned Length>
    CUDA_CALLABLE inline void adj_atomic_add(vec<Length, uint32> *buf, const vec<Length, uint32> &value) {}
    template <unsigned Length>
    CUDA_CALLABLE inline void adj_atomic_add(vec<Length, int64> *buf, const vec<Length, int64> &value) {}
    template <unsigned Length>
    CUDA_CALLABLE inline void adj_atomic_add(vec<Length, uint64> *buf, const vec<Length, uint64> &value) {}

    using vec2ub = vec<2, uint8>;
    using vec3ub = vec<3, uint8>;
    using vec4ub = vec<4, uint8>;

    using vec2h = vec<2, half>;
    using vec3h = vec<3, half>;
    using vec4h = vec<4, half>;

    using vec2 = vec<2, float>;
    using vec3 = vec<3, float>;
    using vec4 = vec<4, float>;

    using vec2f = vec<2, float>;
    using vec3f = vec<3, float>;
    using vec4f = vec<4, float>;

    using vec2d = vec<2, double>;
    using vec3d = vec<3, double>;
    using vec4d = vec<4, double>;

    // adjoints for some of the constructors, used in intersect.h
    inline CUDA_CALLABLE void adj_vec2(float x, float y, float &adj_x, float &adj_y, const vec2 &adj_ret)
    {
        adj_x += adj_ret[0];
        adj_y += adj_ret[1];
    }

    inline CUDA_CALLABLE void adj_vec3(float x, float y, float z, float &adj_x, float &adj_y, float &adj_z, const vec3 &adj_ret)
    {
        adj_x += adj_ret[0];
        adj_y += adj_ret[1];
        adj_z += adj_ret[2];
    }

    inline CUDA_CALLABLE void adj_vec4(float x, float y, float z, float w, float &adj_x, float &adj_y, float &adj_z, float &adj_w, const vec4 &adj_ret)
    {
        adj_x += adj_ret[0];
        adj_y += adj_ret[1];
        adj_z += adj_ret[2];
        adj_w += adj_ret[3];
    }

    inline CUDA_CALLABLE void adj_vec3(float s, float &adj_s, const vec3 &adj_ret)
    {
        adj_vec(s, adj_s, adj_ret);
    }

    inline CUDA_CALLABLE void adj_vec4(float s, float &adj_s, const vec4 &adj_ret)
    {
        adj_vec(s, adj_s, adj_ret);
    }

} // namespace wp