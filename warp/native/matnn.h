#pragma once


CUDA_CALLABLE inline int dense_index(int stride, int i, int j)
{
    return i*stride + j;
}

template <bool transpose>
CUDA_CALLABLE inline int dense_index(int rows, int cols, int i, int j)
{
    if (transpose)
        return j*rows + i;
    else
        return i*cols + j;
}


#ifndef __CUDACC__

const int kNumThreadsPerBlock = 1;

template <bool t1, bool t2, bool add>
CUDA_CALLABLE inline void dense_gemm_impl(int m, int n, int p, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    for (int i=0; i < m; i++)
    {
        for (int j=0; j < n; ++j)
        {
            float sum = 0.0f;

            for (int k=0; k < p; ++k)
            {
                sum += A[dense_index<t1>(m, p, i, k)]*B[dense_index<t2>(p, n, k, j)];
            }
            
            if (add)
                C[i*n + j] += sum;
            else
                C[i*n + j] = sum;
        }
    }
}

#else

const int kNumThreadsPerBlock = 256;

template <bool t1, bool t2, bool add>
CUDA_CALLABLE inline void dense_gemm_impl(int m, int n, int p, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    // each thread in the block calculates an output (or more if output dim > block dim)
    for (int e=threadIdx.x; e < m*n; e += blockDim.x)
    {
        const int i=e/n;
        const int j=e%n;

        float sum = 0.0f;

        for (int k=0; k < p; ++k)
        {
            sum += A[dense_index<t1>(m, p, i, k)]*B[dense_index<t2>(p, n, k, j)];
        }
        
        if (add)
            C[i*n + j] += sum;
        else
            C[i*n + j] = sum;
    }
}

#endif


template <bool add=false>
CUDA_CALLABLE inline void dense_gemm(int m, int n, int p, int t1, int t2, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    if (t1 == 0 && t2 == 0)
        dense_gemm_impl<false, false, add>(m, n, p, A, B, C);
    else if (t1 == 1 && t2 == 0)
        dense_gemm_impl<true, false, add>(m, n, p, A, B, C);
    else if (t1 == 0 && t2 == 1)
        dense_gemm_impl<false, true, add>(m, n, p, A, B, C);
    else if (t1 == 1 && t2 == 1)
        dense_gemm_impl<true, true, add>(m, n, p, A, B, C);
}

template <bool add=false>
CUDA_CALLABLE inline void dense_gemm_batched(
    const int* __restrict__ m, const int* __restrict__ n, const int* __restrict__ p, int t1, int t2,
     const int* __restrict__ A_start,  const int* __restrict__ B_start, const int* __restrict__ C_start,
     const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    // on the CPU each thread computes the whole matrix multiply
    // on the GPU each block computes the multiply with one output per-thread
    const int batch = tid()/kNumThreadsPerBlock;

    dense_gemm<add>(m[batch], n[batch], p[batch], t1, t2, A+A_start[batch], B+B_start[batch], C+C_start[batch]);
}




// computes c = b^T*a*b, with a and b being stored in row-major layout
CUDA_CALLABLE inline void dense_quadratic()
{
}

// CUDA_CALLABLE inline void dense_chol(int n, const float* A, float* L)
// {
//     // for each column
//     for (int j=0; j < n; ++j)
//     {
//         for (int i=j; i < n; ++i)
//         {
//             L[dense_index(n, i, j)] = A[dense_index(n, i, j)];
//         }

//         for (int k = 0; k < j; ++k)
//         {
//             const float p = L[dense_index(n, j, k)];
            
//             for (int i=j; i < n; ++i)
//             {
//                 L[dense_index(n, i, j)] -= p*L[dense_index(n, i, k)];
//             }
//         }

//         // scale 
//         const float d = L[dense_index(n, j, j)];
//         const float s = 1.0f/sqrtf(d);
    
//         for (int i=j; i < n; ++i)
//         {
//             L[dense_index(n, i, j)] *=s;
//         }
//     }
// }

void  CUDA_CALLABLE inline dense_chol(int n, const float* __restrict__ A, float regularization, float* __restrict__ L)
{
    for (int j=0; j < n; ++j)
    {
        float s = A[dense_index(n, j, j)] + regularization;

        for (int k=0; k < j; ++k)
        {
            float r = L[dense_index(n, j, k)];
            s -= r*r;
        }

        s = sqrtf(s);
        const float invS = 1.0f/s;

        L[dense_index(n, j, j)] = s;

        for (int i=j+1; i < n; ++i)
        {
            s = A[dense_index(n, i, j)];
            
            for (int k=0; k < j; ++k)
            {
                s -= L[dense_index(n, i, k)]*L[dense_index(n, j, k)];
            }

            L[dense_index(n, i, j)] = s*invS;
        }
    }
}



void CUDA_CALLABLE inline dense_chol_batched(const int* __restrict__ A_start, const int* __restrict__ A_dim, const float* __restrict__ A, float regularization, float* __restrict__ L)
{
    const int batch = tid();
    
    const int n = A_dim[batch];
    const int offset = A_start[batch];
    
    dense_chol(n, A + offset, regularization, L + offset);
}





// Solves (L*L^T)x = b given the Cholesky factor L 
CUDA_CALLABLE inline void dense_subs(int n, const float* __restrict__ L, const float* __restrict__ b, float* __restrict__ x)
{
    // forward substitution
    for (int i=0; i < n; ++i)
    {
        float s = b[i];

        for (int j=0; j < i; ++j)
        {
            s -= L[dense_index(n, i, j)]*x[j];
        }

        x[i] = s/L[dense_index(n, i, i)];
    }

    // backward substitution
    for (int i=n-1; i >= 0; --i)
    {
        float s = x[i];

        for (int j=i+1; j < n; ++j)
        {
            s -= L[dense_index(n, j, i)]*x[j];
        }

        x[i] = s/L[dense_index(n, i, i)];
    }
}

CUDA_CALLABLE inline void dense_solve(int n, const float* __restrict__ A, const float* __restrict__ L, const float* __restrict__ b, float* __restrict__ x)
{
    dense_subs(n, L, b, x);
}

CUDA_CALLABLE inline void dense_solve_batched(
    const int* __restrict__ b_start, const int* A_start, const int* A_dim, 
    const float* __restrict__ A, const float* __restrict__ L, 
    const float* __restrict__ b, float* __restrict__ x)
{
    const int batch = tid();

    dense_solve(A_dim[batch], A + A_start[batch], L + A_start[batch], b + b_start[batch], x + b_start[batch]);
}


CUDA_CALLABLE inline void print_matrix(const char* name, int m, int n, const float* data)
{
    printf("%s = [", name);

    for (int i=0; i < m; ++i)
    {
        for (int j=0; j < n; ++j)
        {
            printf("%f ", data[dense_index(n, i, j)]);
        }

        printf(";\n");
    }

    printf("]\n");
}

// adjoint methods
CUDA_CALLABLE inline void adj_dense_gemm(
    int m, int n, int p, int t1, int t2, const float* A, const float* B, float* C,
    int adj_m, int adj_n, int adj_p, int adj_t1, int adj_t2, float* adj_A, float* adj_B, const float* adj_C)
{

    // print_matrix("A", m, p, A);
    // print_matrix("B", p, n, B);
    // printf("t1: %d t2: %d\n", t1, t2);

    if (t1)
    {
        dense_gemm<true>(p, m, n, 0, 1, B, adj_C, adj_A);
        dense_gemm<true>(p, n, m, int(!t1), 0, A, adj_C, adj_B);
    }
    else
    {
        dense_gemm<true>(m, p, n, 0, int(!t2), adj_C, B, adj_A);
        dense_gemm<true>(p, n, m, int(!t1), 0, A, adj_C, adj_B);
    }
}

CUDA_CALLABLE inline void adj_dense_gemm_batched(
    const int* __restrict__ m, const int* __restrict__ n, const int* __restrict__ p, int t1, int t2,
    const int* __restrict__ A_start,  const int* __restrict__ B_start, const int* __restrict__ C_start,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    // adj
    int* __restrict__ adj_m, int* __restrict__ adj_n, int* __restrict__ adj_p, int adj_t1, int adj_t2,
    int* __restrict__ adj_A_start,  int* __restrict__ adj_B_start, int* __restrict__ adj_C_start,
    float* __restrict__ adj_A, float* __restrict__ adj_B, const float* __restrict__ adj_C)
{
    const int batch = tid()/kNumThreadsPerBlock;

    adj_dense_gemm(m[batch], n[batch], p[batch], t1, t2, A+A_start[batch], B+B_start[batch], C+C_start[batch], 
                   0, 0, 0, 0, 0, adj_A+A_start[batch], adj_B+B_start[batch], adj_C+C_start[batch]);
}


CUDA_CALLABLE inline void adj_dense_chol(
    int n, const float* A, float regularization, float* L,
    int adj_n, const float* adj_A, float adj_regularization, float* adj_L)
{
    // nop, use dense_solve to differentiate through (A^-1)b = x
}

CUDA_CALLABLE inline void adj_dense_chol_batched(
    const int* __restrict__ A_start, const int* __restrict__ A_dim, const float* __restrict__ A, float regularization, float* __restrict__ L,
    const int* __restrict__ adj_A_start, const int* __restrict__ adj_A_dim, const float* __restrict__ adj_A, float adj_regularization, float* __restrict__ adj_L)
{
    // nop, use dense_solve to differentiate through (A^-1)b = x
}


CUDA_CALLABLE inline void adj_dense_subs(
    int n, const float* L, const float* b, float* x,
    int adj_n, const float* adj_L, const float* adj_b, float* adj_x)
{
    // nop, use dense_solve to differentiate through (A^-1)b = x
}


CUDA_CALLABLE inline void adj_dense_solve(
    int n, const float* __restrict__ A, const float* __restrict__ L, const float* __restrict__ b, const float* __restrict__ x,
    int adj_n, float* __restrict__ adj_A, float* __restrict__ adj_L, float* __restrict__ adj_b, const float* __restrict__ adj_x)
{
    // see https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pwp, section 2.3.1
    dense_subs(n, L, adj_x, adj_b);

    // A* = -adj_b*x^T
    for (int i=0; i < n; ++i)
    {
        for (int j=0; j < n; ++j)
        {
            adj_A[dense_index(n, i, j)] += -adj_b[i]*x[j];
        }
    }
}

CUDA_CALLABLE inline void adj_dense_solve_batched(
    const int* __restrict__ b_start, const int* A_start, const int* A_dim, 
    const float* __restrict__ A, const float* __restrict__ L, 
    const float* __restrict__ b, float* __restrict__ x,
    // adj
    int* __restrict__ adj_b_start, int* __restrict__ adj_A_start, int* __restrict__ adj_A_dim, 
    float* __restrict__ adj_A, float* __restrict__ adj_L, 
    float* __restrict__ adj_b, const float* __restrict__ adj_x)
{
    const int batch = tid();

    adj_dense_solve(A_dim[batch], A + A_start[batch], L + A_start[batch], b + b_start[batch], x + b_start[batch],
                    0, adj_A + A_start[batch], adj_L + A_start[batch], adj_b + b_start[batch], adj_x + b_start[batch]);

}
