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
// mat33

struct mat33
{
    inline CUDA_CALLABLE mat33(vec3 c0, vec3 c1, vec3 c2)
    {
        data[0][0] = c0.x;
        data[1][0] = c0.y;
        data[2][0] = c0.z;

        data[0][1] = c1.x;
        data[1][1] = c1.y;
        data[2][1] = c1.z;

        data[0][2] = c2.x;
        data[1][2] = c2.y;
        data[2][2] = c2.z;
    }

    inline CUDA_CALLABLE mat33(
        float m00=0.0f, float m01=0.0f, float m02=0.0f,
        float m10=0.0f, float m11=0.0f, float m12=0.0f,
        float m20=0.0f, float m21=0.0f, float m22=0.0f) 
    {
        data[0][0] = m00;
        data[1][0] = m10;
        data[2][0] = m20;

        data[0][1] = m01;
        data[1][1] = m11;
        data[2][1] = m21;

        data[0][2] = m02;
        data[1][2] = m12;
        data[2][2] = m22;
    }

    CUDA_CALLABLE vec3 get_row(int index) const
    {
        return (vec3&)data[index]; 
    }

    CUDA_CALLABLE void set_row(int index, const vec3& v)
    {
        (vec3&)data[index] = v;
    }

    CUDA_CALLABLE vec3 get_col(int index) const
    {
        return vec3(data[0][index], data[1][index], data[2][index]);
    }

    CUDA_CALLABLE void set_col(int index, const vec3& v)
    {
        data[0][index] = v.x;
        data[1][index] = v.y;
        data[2][index] = v.z;
    }

    // row major storage assumed to be compatible with PyTorch
    float data[3][3];
};

inline CUDA_CALLABLE bool operator==(const mat33& a, const mat33& b)
{
    for (int i=0; i < 3; ++i)
        for (int j=0; j < 3; ++j)
            if (a.data[i][j] != b.data[i][j])
                return false;

    return true;
}

inline CUDA_CALLABLE mat33 diag(const vec3& d) {
  return mat33(d.x, 0.f, 0.f, 0.f, d.y, 0.f, 0.f, 0.f, d.z);
}

inline CUDA_CALLABLE void adj_diag(const vec3& d, vec3& adj_d, const mat33& adj_ret) {
  adj_d += vec3(adj_ret.data[0][0], adj_ret.data[1][1], adj_ret.data[2][2]);
}

inline CUDA_CALLABLE mat33 atomic_add(mat33 * addr, mat33 value) 
{
    mat33 m;

    m.data[0][0] = atomic_add(&((addr -> data)[0][0]), value.data[0][0]);
    m.data[1][0] = atomic_add(&((addr -> data)[1][0]), value.data[1][0]);
    m.data[2][0] = atomic_add(&((addr -> data)[2][0]), value.data[2][0]);
    m.data[0][1] = atomic_add(&((addr -> data)[0][1]), value.data[0][1]);
    m.data[1][1] = atomic_add(&((addr -> data)[1][1]), value.data[1][1]);
    m.data[2][1] = atomic_add(&((addr -> data)[2][1]), value.data[2][1]);
    m.data[0][2] = atomic_add(&((addr -> data)[0][2]), value.data[0][2]);
    m.data[1][2] = atomic_add(&((addr -> data)[1][2]), value.data[1][2]);
    m.data[2][2] = atomic_add(&((addr -> data)[2][2]), value.data[2][2]);

    return m;
}

inline CUDA_CALLABLE void adj_mat33(vec3 c0, vec3 c1, vec3 c2,
                      vec3& a0, vec3& a1, vec3& a2,
                      const mat33& adj_ret)
{
    // column constructor
    a0 += adj_ret.get_col(0);
    a1 += adj_ret.get_col(1);
    a2 += adj_ret.get_col(2);

}

inline CUDA_CALLABLE void adj_mat33(float m00, float m01, float m02,
                      float m10, float m11, float m12,
                      float m20, float m21, float m22,
                      float& a00, float& a01, float& a02,
                      float& a10, float& a11, float& a12,
                      float& a20, float& a21, float& a22,
                      const mat33& adj_ret)
{
    printf("todo\n");
}

inline bool CUDA_CALLABLE isfinite(const mat33& m)
{
    for (int i=0; i < 3; ++i)
        for (int j=0; j < 3; ++j)
            if (!::isfinite(m.data[i][j]))
                return false;
    return true;
}

inline CUDA_CALLABLE float index(const mat33& m, int row, int col)
{
#if FP_CHECK
    if (row < 0 || row > 2)
    {
        printf("mat33 row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < 0 || col > 2)
    {
        printf("mat33 col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif
    return m.data[row][col];
}


inline CUDA_CALLABLE vec3 index(const mat33& m, int row)
{
    return vec3(m.data[row][0], m.data[row][1], m.data[row][2]);
}

inline CUDA_CALLABLE void adj_index(const mat33& m, int row, mat33& adj_m, int& adj_row, const vec3& adj_ret)
{
    adj_m.data[row][0] += adj_ret[0];
    adj_m.data[row][1] += adj_ret[1];
    adj_m.data[row][2] += adj_ret[2];
}


inline CUDA_CALLABLE mat33 add(const mat33& a, const mat33& b)
{
    mat33 t;
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            t.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }

    return t;
}

inline CUDA_CALLABLE mat33 sub(const mat33& a, const mat33& b)
{
  mat33 t;
  for (int i=0; i < 3; ++i)
  {
    for (int j=0; j < 3; ++j)
    {
      t.data[i][j] = a.data[i][j] - b.data[i][j];
    }
  }

  return t;
}

inline CUDA_CALLABLE mat33 mul(const mat33& a, float b)
{
    mat33 t;
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            t.data[i][j] = a.data[i][j]*b;
        }
    }

    return t;   
}


inline CUDA_CALLABLE vec3 mul(const mat33& a, const vec3& b)
{
    vec3 r = a.get_col(0)*b.x +
               a.get_col(1)*b.y +
               a.get_col(2)*b.z;
    
    return r;
}

inline CUDA_CALLABLE mat33 mul(const mat33& a, const mat33& b)
{
    mat33 t;
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            for (int k=0; k < 3; ++k)
            {
                t.data[i][j] += a.data[i][k]*b.data[k][j];
            }
        }
    }

    return t;
}

inline CUDA_CALLABLE mat33 element_mul(const mat33& a, const mat33& b)
{
  mat33 t;
  for (int i=0; i < 3; ++i)
  {
    for (int j=0; j < 3; ++j)
    {
      t.data[i][j] = a.data[i][j] * b.data[i][j];
    }
  }

  return t;
}

inline CUDA_CALLABLE mat33 transpose(const mat33& a)
{
    mat33 t;
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            t.data[i][j] = a.data[j][i];
        }
    }

    return t;
}


inline CUDA_CALLABLE float determinant(const mat33& m)
{
    return dot(vec3(m.data[0]), cross(vec3(m.data[1]), vec3(m.data[2])));
}

inline CUDA_CALLABLE mat33 inverse(const mat33& m)
{
	float det = determinant(m);

	if (fabsf(det) > kEps)
	{
		mat33 b;
		
		b.data[0][0] = m.data[1][1]*m.data[2][2] - m.data[1][2]*m.data[2][1]; 
		b.data[1][0] = m.data[1][2]*m.data[2][0] - m.data[1][0]*m.data[2][2]; 
		b.data[2][0] = m.data[1][0]*m.data[2][1] - m.data[1][1]*m.data[2][0]; 
		
        b.data[0][1] = m.data[0][2]*m.data[2][1] - m.data[0][1]*m.data[2][2]; 
        b.data[1][1] = m.data[0][0]*m.data[2][2] - m.data[0][2]*m.data[2][0]; 
        b.data[2][1] = m.data[0][1]*m.data[2][0] - m.data[0][0]*m.data[2][1]; 

        b.data[0][2] = m.data[0][1]*m.data[1][2] - m.data[0][2]*m.data[1][1];
        b.data[1][2] = m.data[0][2]*m.data[1][0] - m.data[0][0]*m.data[1][2];
        b.data[2][2] = m.data[0][0]*m.data[1][1] - m.data[0][1]*m.data[1][0];

		return b*(1.0f/det);
	}
	else
	{
		return mat33();
	}
}

inline CUDA_CALLABLE void adj_inverse(const mat33& m, mat33& adj_m, const mat33& adj_ret)
{
    // todo: how to cache this from the forward pass?
    mat33 invt = transpose(inverse(m));

    // see https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf 2.2.3
    adj_m -= mul(mul(invt, adj_ret), invt);
}

inline CUDA_CALLABLE mat33 outer(const vec3& a, const vec3& b)
{
    return mat33(a*b.x, a*b.y, a*b.z);    
}

inline CUDA_CALLABLE mat33 skew(const vec3& a)
{
    mat33 out(0.0f, -a.z,   a.y,
              a.z,   0.0f, -a.x,
             -a.y,   a.x,   0.0f);

    return out;
}

inline void CUDA_CALLABLE adj_index(const mat33& m, int row, int col, mat33& adj_m, int& adj_row, int& adj_col, float adj_ret)
{
#if FP_CHECK
    if (row < 0 || row > 2)
    {
        printf("mat33 row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < 0 || col > 2)
    {
        printf("mat33 col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif
    adj_m.data[row][col] += adj_ret;
}

inline CUDA_CALLABLE void adj_add(const mat33& a, const mat33& b, mat33& adj_a, mat33& adj_b, const mat33& adj_ret)
{
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            adj_a.data[i][j] += adj_ret.data[i][j];
            adj_b.data[i][j] += adj_ret.data[i][j];
        }
    }
}

inline CUDA_CALLABLE void adj_sub(const mat33& a, const mat33& b, mat33& adj_a, mat33& adj_b, const mat33& adj_ret)
{
  for (int i=0; i < 3; ++i)
  {
    for (int j=0; j < 3; ++j)
    {
      adj_a.data[i][j] += adj_ret.data[i][j];
      adj_b.data[i][j] -= adj_ret.data[i][j];
    }
  }
}

inline CUDA_CALLABLE void adj_mul(const mat33& a, float b, mat33& adj_a, float& adj_b, const mat33& adj_ret)
{
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            adj_a.data[i][j] += b*adj_ret.data[i][j];
            adj_b += a.data[i][j]*adj_ret.data[i][j];
        }
    }
}

inline CUDA_CALLABLE void adj_mul(const mat33& a, const vec3& b, mat33& adj_a, vec3& adj_b, const vec3& adj_ret)
{
    adj_a += outer(adj_ret, b);
    adj_b += mul(transpose(a), adj_ret);
}

inline CUDA_CALLABLE void adj_mul(const mat33& a, const mat33& b, mat33& adj_a, mat33& adj_b, const mat33& adj_ret)
{
    adj_a += mul(adj_ret, transpose(b));
    adj_b += mul(transpose(a), adj_ret);
}

inline CUDA_CALLABLE void adj_transpose(const mat33& a, mat33& adj_a, const mat33& adj_ret)
{
    adj_a += transpose(adj_ret);
}

inline CUDA_CALLABLE void adj_determinant(const mat33& m, mat33& adj_m, float adj_ret)
{
    (vec3&)adj_m.data[0] += cross(m.get_row(1), m.get_row(2))*adj_ret;
    (vec3&)adj_m.data[1] += cross(m.get_row(2), m.get_row(0))*adj_ret;
    (vec3&)adj_m.data[2] += cross(m.get_row(0), m.get_row(1))*adj_ret;
}

inline CUDA_CALLABLE void adj_outer(const vec3& a, const vec3& b, vec3& adj_a, vec3& adj_b, const mat33& adj_ret)
{
  adj_a += mul(adj_ret, b);
  adj_b += mul(transpose(adj_ret), a);
}

inline CUDA_CALLABLE void adj_skew(const vec3& a, vec3& adj_a, const mat33& adj_ret)
{
    mat33 out(0.0f, -a.z,   a.y,
              a.z,   0.0f, -a.x,
             -a.y,   a.x,   0.0f);


    adj_a.x += adj_ret.data[2][1] - adj_ret.data[1][2];
    adj_a.y += adj_ret.data[0][2] - adj_ret.data[2][0];
    adj_a.z += adj_ret.data[1][0] - adj_ret.data[0][1];
}

} // namespace wp
