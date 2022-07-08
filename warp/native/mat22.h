/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

namespace wp
{

//----------------------------------------------------------
// mat22

struct mat22
{
    inline CUDA_CALLABLE mat22(vec2 c0, vec2 c1)
    {
        data[0][0] = c0.x;
        data[1][0] = c0.y;

        data[0][1] = c1.x;
        data[1][1] = c1.y;
    }

    inline CUDA_CALLABLE mat22(float m00=0.0f, float m01=0.0f, float m10=0.0f, float m11=0.0f) 
    {
        data[0][0] = m00;
        data[1][0] = m10;
        data[0][1] = m01;
        data[1][1] = m11;
    }

    CUDA_CALLABLE vec2 get_row(int index) const
    {
        return (vec2&)data[index]; 
    }

    CUDA_CALLABLE void set_row(int index, const vec2& v)
    {
        (vec2&)data[index] = v;
    }

    CUDA_CALLABLE vec2 get_col(int index) const
    {
        return vec2(data[0][index], data[1][index]);
    }

    CUDA_CALLABLE void set_col(int index, const vec2& v)
    {
        data[0][index] = v.x;
        data[1][index] = v.y;
    }


    // row major storage assumed to be compatible with PyTorch
    float data[2][2];
};

inline CUDA_CALLABLE bool operator==(const mat22& a, const mat22& b)
{
    for (int i=0; i < 2; ++i)
        for (int j=0; j < 2; ++j)
            if (a.data[i][j] != b.data[i][j])
                return false;

    return true;
}


inline CUDA_CALLABLE mat22 atomic_add(mat22 * addr, mat22 value) {
    // *addr += value;
    mat22 m;
    m.data[0][0] = atomic_add(&((addr -> data)[0][0]), value.data[0][0]);
    m.data[0][1] = atomic_add(&((addr -> data)[0][1]), value.data[0][1]);
    m.data[1][0] = atomic_add(&((addr -> data)[1][0]), value.data[1][0]);
    m.data[1][1] = atomic_add(&((addr -> data)[1][1]), value.data[1][1]);

    return m;
}

inline CUDA_CALLABLE void adj_mat22(float m00, float m01, float m10, float m11, float& adj_m00, float& adj_m01, float& adj_m10, float& adj_m11, const mat22& adj_ret)
{
    adj_m00 += adj_ret.data[0][0];
    adj_m01 += adj_ret.data[0][1];
    adj_m10 += adj_ret.data[1][0];
    adj_m11 += adj_ret.data[1][1];
}

inline CUDA_CALLABLE vec2 index(const mat22& m, int row)
{
    return vec2(m.data[row][0], m.data[row][1]);
}

inline CUDA_CALLABLE void adj_index(const mat22& m, int row, mat22& adj_m, int& adj_row, const vec2& adj_ret)
{
    adj_m.data[row][0] += adj_ret[0];
    adj_m.data[row][1] += adj_ret[1];
}



inline bool CUDA_CALLABLE isfinite(const mat22& m)
{
    for (int i=0; i < 2; ++i)
        for (int j=0; j < 2; ++j)
            if (!::isfinite(m.data[i][j]))
                return false;
    return true;
}

inline CUDA_CALLABLE float index(const mat22& m, int row, int col)
{
#if FP_CHECK
    if (row < 0 || row > 1)
    {
        printf("mat22 row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < 0 || col > 1)
    {
        printf("mat22 col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif
    return m.data[row][col];
}

inline CUDA_CALLABLE mat22 add(const mat22& a, const mat22& b)
{
    mat22 t;
    for (int i=0; i < 2; ++i)
    {
        for (int j=0; j < 2; ++j)
        {
            t.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }

    return t;
}

inline CUDA_CALLABLE mat22 sub(const mat22& a, const mat22& b)
{
    mat22 t;
    for (int i=0; i < 2; ++i)
    {
        for (int j=0; j < 2; ++j)
        {
            t.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }

    return t;
}


inline CUDA_CALLABLE mat22 mul(const mat22& a, float b)
{
    mat22 t;
    for (int i=0; i < 2; ++i)
    {
        for (int j=0; j < 2; ++j)
        {
            t.data[i][j] = a.data[i][j]*b;
        }
    }

    return t;
}
inline CUDA_CALLABLE vec2 mul(const mat22& a, const vec2& b)
{
    vec2 r = a.get_col(0)*b.x +
             a.get_col(1)*b.y;
    
    return r;
}


inline CUDA_CALLABLE mat22 mul(const mat22& a, const mat22& b)
{
    mat22 t;
    for (int i=0; i < 2; ++i)
    {
        for (int j=0; j < 2; ++j)
        {
            for (int k=0; k < 2; ++k)
            {
                t.data[i][j] += a.data[i][k]*b.data[k][j];
            }
        }
    }

    return t;
}

inline CUDA_CALLABLE mat22 transpose(const mat22& a)
{
    mat22 t;
    for (int i=0; i < 2; ++i)
    {
        for (int j=0; j < 2; ++j)
        {
            t.data[i][j] = a.data[j][i];
        }
    }

    return t;
}


inline CUDA_CALLABLE float determinant(const mat22& m)
{
    return m.data[0][0]*m.data[1][1] - m.data[1][0]*m.data[0][1];
}

inline CUDA_CALLABLE mat22 inverse(const mat22& m)
{
    float det = determinant(m);
    if (fabs(det) > kEps)
    {
        return mat22( m.data[1][1], -m.data[0][1],
                     -m.data[1][0],  m.data[0][0])*(1.0f/det);
    }
    else
    {
        return mat22();
    }
}

inline CUDA_CALLABLE void adj_inverse(const mat22& m, mat22& adj_m, const mat22& adj_ret)
{
    // todo: how to cache this from the forward pass?
    mat22 invt = transpose(inverse(m));

    // see https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf 2.2.3
    adj_m -= mul(mul(invt, adj_ret), invt);
}

inline CUDA_CALLABLE mat22 diag(const vec2& d) 
{
    return mat22(d.x, 0.f, 
                 0.f, d.y);
}

inline CUDA_CALLABLE mat22 outer(const vec2& a, const vec2& b)
{
    return mat22(a*b.x, a*b.y);
}

inline CUDA_CALLABLE void adj_outer(const vec2& a, const vec2& b, vec2& adj_a, vec2& adj_b, const mat22& adj_ret)
{
    adj_a += mul(adj_ret, b);
    adj_b += mul(transpose(adj_ret), a);
}


inline void CUDA_CALLABLE adj_index(const mat22& m, int row, int col, mat22& adj_m, int& adj_row, int& adj_col, float adj_ret)
{
#if FP_CHECK
    if (row < 0 || row > 1)
    {
        printf("mat22 row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < 0 || col > 1)
    {
        printf("mat22 col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif
    adj_m.data[row][col] += adj_ret;
}

inline CUDA_CALLABLE void adj_add(const mat22& a, const mat22& b, mat22& adj_a, mat22& adj_b, const mat22& adj_ret)
{
    for (int i=0; i < 2; ++i)
    {
        for (int j=0; j < 2; ++j)
        {
            adj_a.data[i][j] += adj_ret.data[i][j];
            adj_b.data[i][j] += adj_ret.data[i][j];
        }
    }
}

inline CUDA_CALLABLE void adj_sub(const mat22& a, const mat22& b, mat22& adj_a, mat22& adj_b, const mat22& adj_ret)
{
    for (int i=0; i < 2; ++i)
    {
        for (int j=0; j < 2; ++j)
        {
            adj_a.data[i][j] += adj_ret.data[i][j];
            adj_b.data[i][j] -= adj_ret.data[i][j];
        }
    }
}

inline CUDA_CALLABLE void adj_mul(const mat22& a, float b, mat22& adj_a, float& adj_b, const mat22& adj_ret)
{
    for (int i=0; i < 2; ++i)
    {
        for (int j=0; j < 2; ++j)
        {
            adj_a.data[i][j] += b*adj_ret.data[i][j];
            adj_b += a.data[i][j]*adj_ret.data[i][j];
        }
    }
}

inline CUDA_CALLABLE void adj_mul(const mat22& a, const vec2& b, mat22& adj_a, vec2& adj_b, const vec2& adj_ret)
{
    adj_a += outer(adj_ret, b);
    adj_b += mul(transpose(a), adj_ret);
}

inline CUDA_CALLABLE void adj_mul(const mat22& a, const mat22& b, mat22& adj_a, mat22& adj_b, const mat22& adj_ret)
{
    adj_a += mul(adj_ret, transpose(b));
    adj_b += mul(transpose(a), adj_ret);
}

inline CUDA_CALLABLE void adj_transpose(const mat22& a, mat22& adj_a, const mat22& adj_ret)
{
    adj_a += transpose(adj_ret);
}

inline CUDA_CALLABLE void adj_determinant(const mat22& m, mat22& adj_m, float adj_ret)
{
    adj_m.data[0][0] += m.data[1][1]*adj_ret;
    adj_m.data[1][1] += m.data[0][0]*adj_ret;
    adj_m.data[0][1] -= m.data[1][0]*adj_ret;
    adj_m.data[1][0] -= m.data[0][1]*adj_ret;
}


inline CUDA_CALLABLE void adj_diag(const vec2& d, vec2& adj_d, const mat22& adj_ret) 
{
    adj_d += vec2(adj_ret.data[0][0], adj_ret.data[1][1]);
}


} // namespace wp