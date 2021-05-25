#pragma once

//---------------------------------------------------------------------------------
// Represents a twist in se(3)

struct spatial_vector
{
    vec3 w;
    vec3 v;

    CUDA_CALLABLE inline spatial_vector(float a, float b, float c, float d, float e, float f) : w(a, b, c), v(d, e, f) {}
    CUDA_CALLABLE inline spatial_vector(vec3 w=vec3(), vec3 v=vec3()) : w(w), v(v) {}
    CUDA_CALLABLE inline spatial_vector(float a) : w(a, a, a), v(a, a, a) {}

    CUDA_CALLABLE inline float operator[](int index) const
    {
        assert(index < 6);

        return (&w.x)[index];
    }

    CUDA_CALLABLE inline float& operator[](int index)
    {
        assert(index < 6);

        return (&w.x)[index];
    }
};

CUDA_CALLABLE inline spatial_vector operator - (spatial_vector a)
{
    return spatial_vector(-a.w, -a.v);
}


CUDA_CALLABLE inline spatial_vector add(const spatial_vector& a, const spatial_vector& b)
{
    return { a.w + b.w, a.v + b.v };
}

CUDA_CALLABLE inline spatial_vector sub(const spatial_vector& a, const spatial_vector& b)
{
    return { a.w - b.w, a.v - b.v };
}

CUDA_CALLABLE inline spatial_vector mul(const spatial_vector& a, float s)
{
    return { a.w*s, a.v*s };
}

CUDA_CALLABLE inline spatial_vector mul(float s, const spatial_vector& a)
{
    return mul(a, s);
}

CUDA_CALLABLE inline float spatial_dot(const spatial_vector& a, const spatial_vector& b)
{
    return dot(a.w, b.w) + dot(a.v, b.v);
}

CUDA_CALLABLE inline spatial_vector spatial_cross(const spatial_vector& a,  const spatial_vector& b)
{
    vec3 w = cross(a.w, b.w);
    vec3 v = cross(a.v, b.w) + cross(a.w, b.v);
    
    return spatial_vector(w, v);
}

CUDA_CALLABLE inline spatial_vector spatial_cross_dual(const spatial_vector& a,  const spatial_vector& b)
{
    vec3 w = cross(a.w, b.w) + cross(a.v, b.v);
    vec3 v = cross(a.w, b.v);

    return spatial_vector(w, v);
}

CUDA_CALLABLE inline vec3 spatial_top(const spatial_vector& a)
{
    return a.w;
}

CUDA_CALLABLE inline vec3 spatial_bottom(const spatial_vector& a)
{
    return a.v;
}

// adjoint methods
CUDA_CALLABLE inline void adj_spatial_vector(
    float a, float b, float c, 
    float d, float e, float f, 
    float& adj_a, float& adj_b, float& adj_c,
    float& adj_d, float& adj_e,float& adj_f, 
    const spatial_vector& adj_ret)
{
    adj_a += adj_ret.w.x;
    adj_b += adj_ret.w.y;
    adj_c += adj_ret.w.z;
    
    adj_d += adj_ret.v.x;
    adj_e += adj_ret.v.y;
    adj_f += adj_ret.v.z;
}

CUDA_CALLABLE inline void adj_spatial_vector(const vec3& w, const vec3& v, vec3& adj_w, vec3& adj_v, const spatial_vector& adj_ret)
{
    adj_w += adj_ret.w;
    adj_v += adj_ret.v;
}

CUDA_CALLABLE inline void adj_add(const spatial_vector& a, const spatial_vector& b, spatial_vector& adj_a, spatial_vector& adj_b, const spatial_vector& adj_ret)
{
    adj_add(a.w, b.w, adj_a.w, adj_b.w, adj_ret.w);
    adj_add(a.v, b.v, adj_a.v, adj_b.v, adj_ret.v);
}

CUDA_CALLABLE inline void adj_sub(const spatial_vector& a, const spatial_vector& b, spatial_vector& adj_a, spatial_vector& adj_b, const spatial_vector& adj_ret)
{
    adj_sub(a.w, b.w, adj_a.w, adj_b.w, adj_ret.w);
    adj_sub(a.v, b.v, adj_a.v, adj_b.v, adj_ret.v);
}

CUDA_CALLABLE inline void adj_mul(const spatial_vector& a, float s, spatial_vector& adj_a, float& adj_s, const spatial_vector& adj_ret)
{
    adj_mul(a.w, s, adj_a.w, adj_s, adj_ret.w);
    adj_mul(a.v, s, adj_a.v, adj_s, adj_ret.v);
}

CUDA_CALLABLE inline void adj_spatial_dot(const spatial_vector& a, const spatial_vector& b, spatial_vector& adj_a, spatial_vector& adj_b, const float& adj_ret)
{
    adj_dot(a.w, b.w, adj_a.w, adj_b.w, adj_ret);
    adj_dot(a.v, b.v, adj_a.v, adj_b.v, adj_ret);
}

CUDA_CALLABLE inline void adj_spatial_cross(const spatial_vector& a,  const spatial_vector& b, spatial_vector& adj_a,  spatial_vector& adj_b, const spatial_vector& adj_ret)
{
    adj_cross(a.w, b.w, adj_a.w, adj_b.w, adj_ret.w);
    
    adj_cross(a.v, b.w, adj_a.v, adj_b.w, adj_ret.v);
    adj_cross(a.w, b.v, adj_a.w, adj_b.v, adj_ret.v);
}

CUDA_CALLABLE inline void adj_spatial_cross_dual(const spatial_vector& a,  const spatial_vector& b, spatial_vector& adj_a, spatial_vector& adj_b, const spatial_vector& adj_ret)
{
    adj_cross(a.w, b.w, adj_a.w, adj_b.w, adj_ret.w);
    adj_cross(a.v, b.v, adj_a.v, adj_b.v, adj_ret.w);

    adj_cross(a.w, b.v, adj_a.w, adj_b.v, adj_ret.v);
}

CUDA_CALLABLE inline void adj_spatial_top(const spatial_vector& a, spatial_vector& adj_a, const vec3& adj_ret)
{
    adj_a.w += adj_ret;
}

CUDA_CALLABLE inline void adj_spatial_bottom(const spatial_vector& a, spatial_vector& adj_a, const vec3& adj_ret)
{
    adj_a.v += adj_ret;
}

#ifdef CUDA
inline __device__ spatial_vector atomic_add(spatial_vector* addr, const spatial_vector& value) {
    
    vec3 w = atomic_add(&addr->w, value.w);
    vec3 v = atomic_add(&addr->v, value.v);

    return spatial_vector(w, v);
}
#endif

//---------------------------------------------------------------------------------
// Represents a rigid body transformation

struct spatial_transform
{
    vec3 p;
    quat q;

    CUDA_CALLABLE inline spatial_transform(vec3 p=vec3(), quat q=quat()) : p(p), q(q) {}
    CUDA_CALLABLE inline spatial_transform(float)  {}  // helps uniform initialization
};

CUDA_CALLABLE inline spatial_transform spatial_transform_identity()
{
    return spatial_transform(vec3(), quat_identity());
}

CUDA_CALLABLE inline vec3 spatial_transform_get_translation(const spatial_transform& t)
{
    return t.p;
}

CUDA_CALLABLE inline quat spatial_transform_get_rotation(const spatial_transform& t)
{
    return t.q;
}

CUDA_CALLABLE inline spatial_transform spatial_transform_multiply(const spatial_transform& a, const spatial_transform& b)
{
    return { rotate(a.q, b.p) + a.p, mul(a.q, b.q) };
}

/*
CUDA_CALLABLE inline spatial_transform spatial_transform_inverse(const spatial_transform& t)
{
    quat q_inv = inverse(t.q);
    return spatial_transform(-rotate(q_inv, t.p), q_inv);
}
*/
    
CUDA_CALLABLE inline vec3 spatial_transform_vector(const spatial_transform& t, const vec3& x)
{
    return rotate(t.q, x);
}

CUDA_CALLABLE inline vec3 spatial_transform_point(const spatial_transform& t, const vec3& x)
{
    return t.p + rotate(t.q, x);
}
/*
// Frank & Park definition 3.20, pg 100
CUDA_CALLABLE inline spatial_vector spatial_transform_twist(const spatial_transform& t, const spatial_vector& x)
{
    vec3 w = rotate(t.q, x.w);
    vec3 v = rotate(t.q, x.v) + cross(t.p, w);

    return spatial_vector(w, v);
}

CUDA_CALLABLE inline spatial_vector spatial_transform_wrench(const spatial_transform& t, const spatial_vector& x)
{
    vec3 v = rotate(t.q, x.v);
    vec3 w = rotate(t.q, x.w) + cross(t.p, v);

    return spatial_vector(w, v);
}
*/

CUDA_CALLABLE inline spatial_transform add(const spatial_transform& a, const spatial_transform& b)
{
    return { a.p + b.p, a.q + b.q };
}

CUDA_CALLABLE inline spatial_transform sub(const spatial_transform& a, const spatial_transform& b)
{
    return { a.p - b.p, a.q - b.q };
}

CUDA_CALLABLE inline spatial_transform mul(const spatial_transform& a, float s)
{
    return { a.p*s, a.q*s };
}


// adjoint methods
CUDA_CALLABLE inline void adj_add(const spatial_transform& a, const spatial_transform& b, spatial_transform& adj_a, spatial_transform& adj_b, const spatial_transform& adj_ret)
{
    adj_add(a.p, b.p, adj_a.p, adj_b.p, adj_ret.p);
    adj_add(a.q, b.q, adj_a.q, adj_b.q, adj_ret.q);
}

CUDA_CALLABLE inline void adj_sub(const spatial_transform& a, const spatial_transform& b, spatial_transform& adj_a, spatial_transform& adj_b, const spatial_transform& adj_ret)
{
    adj_sub(a.p, b.p, adj_a.p, adj_b.p, adj_ret.p);
    adj_sub(a.q, b.q, adj_a.q, adj_b.q, adj_ret.q);
}

CUDA_CALLABLE inline void adj_mul(const spatial_transform& a, float s, spatial_transform& adj_a, float& adj_s, const spatial_transform& adj_ret)
{
    adj_mul(a.p, s, adj_a.p, adj_s, adj_ret.p);
    adj_mul(a.q, s, adj_a.q, adj_s, adj_ret.q);
}

#ifdef CUDA
inline __device__ spatial_transform atomic_add(spatial_transform* addr, const spatial_transform& value) {
    
    vec3 p = atomic_add(&addr->p, value.p);
    quat q = atomic_add(&addr->q, value.q);

    return spatial_transform(p, q);
}
#endif

CUDA_CALLABLE inline void adj_spatial_transform(const vec3& p, const quat& q, vec3& adj_p, quat& adj_q, const spatial_transform& adj_ret)
{
    adj_p += adj_ret.p;
    adj_q += adj_ret.q;
}

CUDA_CALLABLE inline void adj_spatial_transform_identity(const spatial_transform& adj_ret)
{
    // nop
}


CUDA_CALLABLE inline void adj_spatial_transform_get_translation(const spatial_transform& t, spatial_transform& adj_t, const vec3& adj_ret)
{
    adj_t.p += adj_ret;
}

CUDA_CALLABLE inline void adj_spatial_transform_get_rotation(const spatial_transform& t, spatial_transform& adj_t, const quat& adj_ret)
{
    adj_t.q += adj_ret;
}

/*
CUDA_CALLABLE inline void adj_spatial_transform_inverse(const spatial_transform& t, spatial_transform& adj_t, const spatial_transform& adj_ret)
{
    //quat q_inv = inverse(t.q);
    //return spatial_transform(-rotate(q_inv, t.p), q_inv);

    quat q_inv = inverse(t.q); 
    vec3 p = rotate(q_inv, t.p);
    vec3 np = -p;

    quat adj_q_inv = 0.0f;
    quat adj_q = 0.0f;
    vec3 adj_p = 0.0f;
    vec3 adj_np = 0.0f;

    adj_spatial_transform(np, q_inv, adj_np, adj_q_inv, adj_ret);
    adj_p = -adj_np;
    adj_rotate(q_inv, t.p, adj_q_inv, adj_t.p, adj_p);
    adj_inverse(t.q, adj_t.q, adj_q_inv);
    
}
*/

CUDA_CALLABLE inline void adj_spatial_transform_multiply(const spatial_transform& a, const spatial_transform& b, spatial_transform& adj_a, spatial_transform& adj_b, const spatial_transform& adj_ret)
{
    // translational part
    adj_rotate(a.q, b.p, adj_a.q, adj_b.p, adj_ret.p);
    adj_a.p += adj_ret.p;

    // rotational part
    adj_mul(a.q, b.q, adj_a.q, adj_b.q, adj_ret.q);
}

CUDA_CALLABLE inline void adj_spatial_transform_vector(const spatial_transform& t, const vec3& x, spatial_transform& adj_t, vec3& adj_x, const vec3& adj_ret)
{
    adj_rotate(t.q, x, adj_t.q, adj_x, adj_ret);
}

CUDA_CALLABLE inline void adj_spatial_transform_point(const spatial_transform& t, const vec3& x, spatial_transform& adj_t, vec3& adj_x, const vec3& adj_ret)
{
    adj_rotate(t.q, x, adj_t.q, adj_x, adj_ret);
    adj_t.p += adj_ret;
}

/*
CUDA_CALLABLE inline void adj_spatial_transform_twist(const spatial_transform& a, const spatial_vector& s, spatial_transform& adj_a, spatial_vector& adj_s, const spatial_vector& adj_ret)
{
    printf("todo, %s, %d\n", __FILE__, __LINE__);

    // vec3 w = rotate(t.q, x.w);
    // vec3 v = rotate(t.q, x.v) + cross(t.p, w);

    // return spatial_vector(w, v);    
}

CUDA_CALLABLE inline void adj_spatial_transform_wrench(const spatial_transform& t, const spatial_vector& x, spatial_transform& adj_t, spatial_vector& adj_x, const spatial_vector& adj_ret)
{
    printf("todo, %s, %d\n", __FILE__, __LINE__);
    // vec3 v = rotate(t.q, x.v);
    // vec3 w = rotate(t.q, x.w) + cross(t.p, v);

    // return spatial_vector(w, v);
}
*/

/*
// should match model.py
#define JOINT_PRISMATIC 0
#define JOINT_REVOLUTE 1
#define JOINT_FIXED 2
#define JOINT_FREE 3


CUDA_CALLABLE inline spatial_transform spatial_jcalc(int type, float* joint_q, vec3 axis, int start)
{
    if (type == JOINT_REVOLUTE)
    {
        float q = joint_q[start];
        spatial_transform X_jc = spatial_transform(vec3(), quat_from_axis_angle(axis, q));
        return X_jc;
    }
    else if (type == JOINT_PRISMATIC)
    {
        float q = joint_q[start];
        spatial_transform X_jc = spatial_transform(axis*q, quat_identity());
        return X_jc;
    }
    else if (type == JOINT_FREE)
    {
        float px = joint_q[start+0];
        float py = joint_q[start+1];
        float pz = joint_q[start+2];
        
        float qx = joint_q[start+3];
        float qy = joint_q[start+4];
        float qz = joint_q[start+5];
        float qw = joint_q[start+6];
        
        spatial_transform X_jc = spatial_transform(vec3(px, py, pz), quat(qx, qy, qz, qw));
        return X_jc;
    }

    // JOINT_FIXED
    return spatial_transform(vec3(), quat_identity());
}

CUDA_CALLABLE inline void adj_spatial_jcalc(int type, float* q, vec3 axis, int start, int& adj_type, float* adj_q, vec3& adj_axis, int& adj_start, const spatial_transform& adj_ret)
{
    if (type == JOINT_REVOLUTE)
    {
        adj_quat_from_axis_angle(axis, q[start], adj_axis, adj_q[start], adj_ret.q);
    }
    else if (type == JOINT_PRISMATIC)
    {
        adj_mul(axis, q[start], adj_axis, adj_q[start], adj_ret.p);
    }
    else if (type == JOINT_FREE)
    {
        adj_q[start+0] += adj_ret.p.x;
        adj_q[start+1] += adj_ret.p.y;
        adj_q[start+2] += adj_ret.p.z;
        
        adj_q[start+3] += adj_ret.q.x;
        adj_q[start+4] += adj_ret.q.y;
        adj_q[start+5] += adj_ret.q.z;
        adj_q[start+6] += adj_ret.q.w;
    }
}
*/

struct spatial_matrix
{
    float data[6][6] = { { 0 } };

    CUDA_CALLABLE inline spatial_matrix(float f=0.0f)
    {
    }

    CUDA_CALLABLE inline spatial_matrix(
        float a00, float a01, float a02, float a03, float a04, float a05,
        float a10, float a11, float a12, float a13, float a14, float a15,
        float a20, float a21, float a22, float a23, float a24, float a25,
        float a30, float a31, float a32, float a33, float a34, float a35,
        float a40, float a41, float a42, float a43, float a44, float a45,
        float a50, float a51, float a52, float a53, float a54, float a55)
    {
        data[0][0] = a00;
        data[0][1] = a01;
        data[0][2] = a02;
        data[0][3] = a03;
        data[0][4] = a04;
        data[0][5] = a05;

        data[1][0] = a10;
        data[1][1] = a11;
        data[1][2] = a12;
        data[1][3] = a13;
        data[1][4] = a14;
        data[1][5] = a15;

        data[2][0] = a20;
        data[2][1] = a21;
        data[2][2] = a22;
        data[2][3] = a23;
        data[2][4] = a24;
        data[2][5] = a25;

        data[3][0] = a30;
        data[3][1] = a31;
        data[3][2] = a32;
        data[3][3] = a33;
        data[3][4] = a34;
        data[3][5] = a35;

        data[4][0] = a40;
        data[4][1] = a41;
        data[4][2] = a42;
        data[4][3] = a43;
        data[4][4] = a44;
        data[4][5] = a45;

        data[5][0] = a50;
        data[5][1] = a51;
        data[5][2] = a52;
        data[5][3] = a53;
        data[5][4] = a54;
        data[5][5] = a55;
    }

};



inline CUDA_CALLABLE float index(const spatial_matrix& m, int row, int col)
{
    return m.data[row][col];
}


inline CUDA_CALLABLE spatial_matrix add(const spatial_matrix& a, const spatial_matrix& b)
{
    spatial_matrix out;

    for (int i=0; i < 6; ++i)
        for (int j=0; j < 6; ++j)
            out.data[i][j] = a.data[i][j] + b.data[i][j];

    return out;
}


inline CUDA_CALLABLE spatial_vector mul(const spatial_matrix& a, const spatial_vector& b)
{
    spatial_vector out;

    for (int i=0; i < 6; ++i)
        for (int j=0; j < 6; ++j)
            out[i] += a.data[i][j]*b[j];

    return out;
}

inline CUDA_CALLABLE spatial_matrix mul(const spatial_matrix& a, const spatial_matrix& b)
{
    spatial_matrix out;

    for (int i=0; i < 6; ++i)
    {
        for (int j=0; j < 6; ++j)
        {
            for (int k=0; k < 6; ++k)
            {
                out.data[i][j] += a.data[i][k]*b.data[k][j];
            }
        }
    }
    return out;
}

inline CUDA_CALLABLE spatial_matrix transpose(const spatial_matrix& a)
{
    spatial_matrix out;

    for (int i=0; i < 6; i++)
        for (int j=0; j < 6; j++)
            out.data[i][j] = a.data[j][i];

    return out;
}

inline CUDA_CALLABLE spatial_matrix outer(const spatial_vector& a, const spatial_vector& b)
{
    spatial_matrix out;

    for (int i=0; i < 6; i++)
        for (int j=0; j < 6; j++)
            out.data[i][j] = a[i]*b[j];

    return out;
}

CUDA_CALLABLE void print(spatial_transform t);
CUDA_CALLABLE void print(spatial_matrix m);

inline CUDA_CALLABLE spatial_matrix spatial_adjoint(const mat33& R, const mat33& S)
{    
    spatial_matrix adT;

    // T = [R          0]
    //     [skew(p)*R  R]

    // diagonal blocks    
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            adT.data[i][j] = R.data[i][j];
            adT.data[i+3][j+3] = R.data[i][j];
        }
    }

    // lower off diagonal
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            adT.data[i+3][j] = S.data[i][j];
        }
    }

    return adT;
}

inline CUDA_CALLABLE void adj_spatial_adjoint(const mat33& R, const mat33& S, mat33& adj_R, mat33& adj_S, const spatial_matrix& adj_ret)
{
    // diagonal blocks    
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            adj_R.data[i][j] += adj_ret.data[i][j];
            adj_R.data[i][j] += adj_ret.data[i+3][j+3];
        }
    }

    // lower off diagonal
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            adj_S.data[i][j] += adj_ret.data[i+3][j];
        }
    }
}

/*
// computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates), Frank & Park, section 8.2.3, pg 290
inline CUDA_CALLABLE spatial_matrix spatial_transform_inertia(const spatial_transform& t, const spatial_matrix& I)
{
    spatial_transform t_inv = spatial_transform_inverse(t);

    vec3 r1 = rotate(t_inv.q, vec3(1.0, 0.0, 0.0));
    vec3 r2 = rotate(t_inv.q, vec3(0.0, 1.0, 0.0));
    vec3 r3 = rotate(t_inv.q, vec3(0.0, 0.0, 1.0));

    mat33 R(r1, r2, r3);    
    mat33 S = mul(skew(t_inv.p), R);

    spatial_matrix T = spatial_adjoint(R, S);

    // first quadratic form, for derivation of the adjoint see https://people.maths.ox.ac.uk/gilesm/files/AD2008.pog, section 2.3.2
    return mul(mul(transpose(T), I), T);
}
*/



inline CUDA_CALLABLE void adj_add(const spatial_matrix& a, const spatial_matrix& b, spatial_matrix& adj_a, spatial_matrix& adj_b, const spatial_matrix& adj_ret)
{
    adj_a += adj_ret;
    adj_b += adj_ret;
}

inline CUDA_CALLABLE void adj_mul(const spatial_matrix& a, const spatial_vector& b, spatial_matrix& adj_a, spatial_vector& adj_b, const spatial_vector& adj_ret)
{
    adj_a += outer(adj_ret, b);
    adj_b += mul(transpose(a), adj_ret);
}

inline CUDA_CALLABLE void adj_mul(const spatial_matrix& a, const spatial_matrix& b, spatial_matrix& adj_a, spatial_matrix& adj_b, const spatial_matrix& adj_ret)
{
    adj_a += mul(adj_ret, transpose(b));
    adj_b += mul(transpose(a), adj_ret);
}

inline CUDA_CALLABLE void adj_transpose(const spatial_matrix& a, spatial_matrix& adj_a, const spatial_matrix& adj_ret)
{
    adj_a += transpose(adj_ret);
}



inline CUDA_CALLABLE void adj_spatial_transform_inertia(
    const spatial_transform& xform, const spatial_matrix& I,
    const spatial_transform& adj_xform, const spatial_matrix& adj_I,
    spatial_matrix& adj_ret)
{
    //printf("todo, %s, %d\n", __FILE__, __LINE__);
}


inline void CUDA_CALLABLE adj_index(const spatial_matrix& m, int row, int col, spatial_matrix& adj_m, int& adj_row, int& adj_col, float adj_ret)
{
    adj_m.data[row][col] += adj_ret;
}

#ifdef CUDA
inline __device__ spatial_matrix atomic_add(spatial_matrix* addr, const spatial_matrix& value) 
{
    spatial_matrix m;

    for (int i=0; i < 6; ++i)
    {
        for (int j=0; j < 6; ++j)
        {
            m.data[i][j] = atomicAdd(&addr->data[i][j], value.data[i][j]);
        }
    }   

    return m;
}
#endif


CUDA_CALLABLE inline int row_index(int stride, int i, int j)
{
    return i*stride + j;
}

// builds spatial Jacobian J which is an (joint_count*6)x(dof_count) matrix
CUDA_CALLABLE inline void spatial_jacobian(
    const spatial_vector* S,
    const int* joint_parents, 
    const int* joint_qd_start, 
    int joint_start,    // offset of the first joint for the articulation
    int joint_count,    
    int J_start,
    float* J)
{
    const int articulation_dof_start = joint_qd_start[joint_start];
    const int articulation_dof_end = joint_qd_start[joint_start + joint_count];
    const int articulation_dof_count = articulation_dof_end-articulation_dof_start;

	// shift output pointers
	const int S_start = articulation_dof_start;

	S += S_start;
	J += J_start;
	
    for (int i=0; i < joint_count; ++i)
    {
        const int row_start = i * 6;

        int j = joint_start + i;
        while (j != -1)
        {
            const int joint_dof_start = joint_qd_start[j];
            const int joint_dof_end = joint_qd_start[j+1];
            const int joint_dof_count = joint_dof_end-joint_dof_start;

            // fill out each row of the Jacobian walking up the tree
            //for (int col=dof_start; col < dof_end; ++col)
            for (int dof=0; dof < joint_dof_count; ++dof)
            {
                const int col = (joint_dof_start-articulation_dof_start) + dof;

                J[row_index(articulation_dof_count, row_start+0, col)] = S[col].w.x;
                J[row_index(articulation_dof_count, row_start+1, col)] = S[col].w.y;
                J[row_index(articulation_dof_count, row_start+2, col)] = S[col].w.z;
                J[row_index(articulation_dof_count, row_start+3, col)] = S[col].v.x;
                J[row_index(articulation_dof_count, row_start+4, col)] = S[col].v.y;
                J[row_index(articulation_dof_count, row_start+5, col)] = S[col].v.z;
            }

            j = joint_parents[j];
        }
    }
}

CUDA_CALLABLE inline void adj_spatial_jacobian(
    const spatial_vector* S, 
    const int* joint_parents, 
    const int* joint_qd_start, 
    const int joint_start,
    const int joint_count, 
    const int J_start, 
    const float* J,
    // adjs
    spatial_vector* adj_S, 
    int* adj_joint_parents, 
    int* adj_joint_qd_start, 
    int& adj_joint_start,
    int& adj_joint_count, 
    int& adj_J_start, 
    const float* adj_J)
{   
    const int articulation_dof_start = joint_qd_start[joint_start];
    const int articulation_dof_end = joint_qd_start[joint_start + joint_count];
    const int articulation_dof_count = articulation_dof_end-articulation_dof_start;

	// shift output pointers
	const int S_start = articulation_dof_start;

	S += S_start;
	J += J_start;

    adj_S += S_start;
    adj_J += J_start;
	
    for (int i=0; i < joint_count; ++i)
    {
        const int row_start = i * 6;

        int j = joint_start + i;
        while (j != -1)
        {
            const int joint_dof_start = joint_qd_start[j];
            const int joint_dof_end = joint_qd_start[j+1];
            const int joint_dof_count = joint_dof_end-joint_dof_start;

            // fill out each row of the Jacobian walking up the tree
            //for (int col=dof_start; col < dof_end; ++col)
            for (int dof=0; dof < joint_dof_count; ++dof)
            {
                const int col = (joint_dof_start-articulation_dof_start) + dof;

                adj_S[col].w.x += adj_J[row_index(articulation_dof_count, row_start+0, col)];
                adj_S[col].w.y += adj_J[row_index(articulation_dof_count, row_start+1, col)];
                adj_S[col].w.z += adj_J[row_index(articulation_dof_count, row_start+2, col)];
                adj_S[col].v.x += adj_J[row_index(articulation_dof_count, row_start+3, col)];
                adj_S[col].v.y += adj_J[row_index(articulation_dof_count, row_start+4, col)];
                adj_S[col].v.z += adj_J[row_index(articulation_dof_count, row_start+5, col)];
            }

            j = joint_parents[j];
        }
    }
}


CUDA_CALLABLE inline void spatial_mass(const spatial_matrix* I_s, int joint_start, int joint_count, int M_start, float* M)
{
    const int stride = joint_count*6;

    for (int l=0; l < joint_count; ++l)
    {
        for (int i=0; i < 6; ++i)
        {
            for (int j=0; j < 6; ++j)
            {
                M[M_start + row_index(stride, l*6 + i, l*6 + j)] = I_s[joint_start + l].data[i][j];
            }
        }
    } 
}

CUDA_CALLABLE inline void adj_spatial_mass(
    const spatial_matrix* I_s, 
    const int joint_start,
    const int joint_count, 
    const int M_start,
    const float* M,
    spatial_matrix* adj_I_s, 
    int& adj_joint_start,
    int& adj_joint_count, 
    int& adj_M_start,
    const float* adj_M)
{
    const int stride = joint_count*6;

    for (int l=0; l < joint_count; ++l)
    {
        for (int i=0; i < 6; ++i)
        {
            for (int j=0; j < 6; ++j)
            {
                adj_I_s[joint_start + l].data[i][j] += adj_M[M_start + row_index(stride, l*6 + i, l*6 + j)];
            }
        }
    } 
}
