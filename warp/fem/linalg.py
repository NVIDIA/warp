# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import warp as wp
import warp.types


@wp.func
def generalized_outer(x: Any, y: Any):
    """Generalized outer product allowing for the first argument to be a scalar"""
    return wp.outer(x, y)


@wp.func
def generalized_outer(x: wp.float32, y: wp.vec2):
    return x * y


@wp.func
def generalized_outer(x: wp.float32, y: wp.vec3):
    return x * y


@wp.func
def generalized_outer(x: wp.quatf, y: wp.vec3):
    return generalized_outer(wp.vec4(x[0], x[1], x[2], x[3]), y)


@wp.func
def generalized_inner(x: Any, y: Any):
    """Generalized inner product allowing for the first argument to be a tensor"""
    return wp.dot(x, y)


@wp.func
def generalized_inner(x: float, y: float):
    return x * y


@wp.func
def generalized_inner(x: wp.mat22, y: wp.vec2):
    return x[0] * y[0] + x[1] * y[1]


@wp.func
def generalized_inner(x: wp.mat33, y: wp.vec3):
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]


@wp.func
def basis_coefficient(val: wp.float32, i: int):
    return val


@wp.func
def basis_coefficient(val: Any, i: int):
    return val[i]


@wp.func
def basis_coefficient(val: wp.vec2, i: int, j: int):
    # treat as row vector
    return val[j]


@wp.func
def basis_coefficient(val: wp.vec3, i: int, j: int):
    # treat as row vector
    return val[j]


@wp.func
def basis_coefficient(val: Any, i: int, j: int):
    return val[i, j]


@wp.func
def symmetric_part(x: Any):
    """Symmetric part of a square tensor"""
    return 0.5 * (x + wp.transpose(x))


@wp.func
def spherical_part(x: wp.mat22):
    """Spherical part of a square tensor"""
    return 0.5 * wp.trace(x) * wp.identity(n=2, dtype=float)


@wp.func
def spherical_part(x: wp.mat33):
    """Spherical part of a square tensor"""
    return (wp.trace(x) / 3.0) * wp.identity(n=3, dtype=float)


@wp.func
def skew_part(x: wp.mat22):
    """Skew part of a 2x2 tensor as corresponding rotation angle"""
    return 0.5 * (x[1, 0] - x[0, 1])


@wp.func
def skew_part(x: wp.mat33):
    """Skew part of a 3x3 tensor as the corresponding rotation vector"""
    a = 0.5 * (x[2, 1] - x[1, 2])
    b = 0.5 * (x[0, 2] - x[2, 0])
    c = 0.5 * (x[1, 0] - x[0, 1])
    return wp.vec3(a, b, c)


@wp.func
def householder_qr_decomposition(A: Any):
    """
    QR decomposition of a square matrix using Householder reflections

    Returns Q and R such that Q R = A, Q orthonormal (such that QQ^T = Id), R upper triangular
    """

    x = type(A[0])()
    Q = wp.identity(n=type(x).length, dtype=A.dtype)

    zero = x.dtype(0.0)
    two = x.dtype(2.0)

    for i in range(type(x).length):
        for k in range(type(x).length):
            x[k] = wp.where(k < i, zero, A[k, i])

        alpha = wp.length(x) * wp.sign(x[i])
        x[i] += alpha
        two_over_x_sq = wp.where(alpha == zero, zero, two / wp.length_sq(x))

        A -= wp.outer(two_over_x_sq * x, x * A)
        Q -= wp.outer(Q * x, two_over_x_sq * x)

    return Q, A


@wp.func
def householder_make_hessenberg(A: Any):
    """Transforms a square matrix to Hessenberg form (single lower diagonal) using Householder reflections

    Returns:
        Q and H such that Q H Q^T = A, Q orthonormal, H under Hessenberg form
        If A is symmetric, H will be tridiagonal
    """

    x = type(A[0])()
    Q = wp.identity(n=type(x).length, dtype=A.dtype)

    zero = x.dtype(0.0)
    two = x.dtype(2.0)

    for i in range(1, type(x).length):
        for k in range(type(x).length):
            x[k] = wp.where(k < i, zero, A[k, i - 1])

        alpha = wp.length(x) * wp.sign(x[i])
        x[i] += alpha
        two_over_x_sq = wp.where(alpha == zero, zero, two / wp.length_sq(x))

        # apply on both sides
        A -= wp.outer(two_over_x_sq * x, x * A)
        A -= wp.outer(A * x, two_over_x_sq * x)
        Q -= wp.outer(Q * x, two_over_x_sq * x)

    return Q, A


@wp.func
def solve_triangular(R: Any, b: Any):
    """Solves for R x = b where R is an upper triangular matrix

    Returns x
    """
    zero = b.dtype(0)
    x = type(b)(b.dtype(0))
    for i in range(b.length, 0, -1):
        j = i - 1
        r = b[j] - wp.dot(R[j], x)
        x[j] = wp.where(R[j, j] == zero, zero, r / R[j, j])

    return x


@wp.func
def inverse_qr(A: Any):
    # Computes a square matrix inverse using QR factorization

    Q, R = householder_qr_decomposition(A)

    A_inv = type(A)()
    for i in range(type(A[0]).length):
        A_inv[i] = solve_triangular(R, Q[i])  # ith column of Q^T

    return wp.transpose(A_inv)


@wp.func
def _wilkinson_shift(a: Any, b: Any, c: Any, tol: Any):
    # Wilkinson shift: estimate eigenvalue of 2x2 symmetric matrix [a, c, c, b]
    d = (a - b) * type(tol)(0.5)
    return b + d - wp.sign(d) * wp.sqrt(d * d + c * c)


@wp.func
def _givens_rotation(a: Any, b: Any):
    # Givens rotation [[c -s], [s c]] such that sa+cb =0
    zero = type(a)(0.0)
    one = type(a)(1.0)

    b2 = b * b
    if b2 == zero:
        # id rotation
        return one, zero

    scale = one / wp.sqrt(a * a + b2)
    return a * scale, -b * scale


@wp.func
def tridiagonal_symmetric_eigenvalues_qr(D: Any, L: Any, Q: Any, tol: Any):
    """
    Computes the eigenvalues and eigen vectors of a symmetric tridiagonal matrix using the
    Symmetric tridiagonal QR algorithm with implicit Wilkinson shift

    Args:
        D: Main diagonal of the matrix
        L: Lower diagonal of the matrix, indexed such that L[i] = A[i+1, i]
        Q: Initialization for the eigenvectors, useful if a pre-transformation has been applied, otherwise may be identity
        tol: Tolerance for the diagonalization residual (Linf norm of off-diagonal over diagonal terms)

    Returns a tuple (D: vector of eigenvalues, P: matrix with one eigenvector per row) such that A = P^T D P


    Ref: Arbenz P, Numerical Methods for Solving Large Scale Eigenvalue Problems, Chapter 4 (QR algorithm, Mar 13, 2018)
    """

    two = D.dtype(2.0)

    # so that we can use the type length in expressions
    # this will prevent unrolling by warp, but should be ok for native code
    m = int(0)
    for _ in range(type(D).length):
        m += 1

    start = int(0)
    y = D.dtype(0.0)  # moving buldge
    x = D.dtype(0.0)  # coeff atop buldge

    for _ in range(32 * m):  # failsafe, usually converges faster than that
        # Iterate over all independent (deflated) blocks
        end = int(-1)

        for k in range(m - 1):
            if k >= end:
                # Check if new block is starting
                if k == end or wp.abs(L[k]) <= tol * (wp.abs(D[k]) + wp.abs(D[k + 1])):
                    continue

                # Find end of block
                start = k
                end = start + 1
                while end + 1 < m:
                    if wp.abs(L[end]) <= tol * (wp.abs(D[end + 1]) + wp.abs(D[end])):
                        break
                    end += 1

                # Wilkinson shift (an eigenvalue of the last 2x2 block)
                shift = _wilkinson_shift(D[end - 1], D[end], L[end - 1], tol)

                # start with eliminating lower diag of first column of shifted matrix
                # (i.e. first step of explicit QR factorization)
                # Then all further steps eliminate the buldge (second diag) of the non-shifted matrix
                x = D[start] - shift
                y = L[start]

            c, s = _givens_rotation(x, y)

            # Apply Givens rotation on both sides of tridiagonal matrix

            # middle block
            d = D[k] - D[k + 1]
            z = (two * c * L[k] + d * s) * s
            D[k] -= z
            D[k + 1] += z
            L[k] = d * c * s + (c * c - s * s) * L[k]

            if k > start:
                L[k - 1] = c * x - s * y

            x = L[k]
            y = -s * L[k + 1]  # new buldge
            L[k + 1] *= c

            # apply givens rotation on left of Q
            # note: Q is transposed compared to usual impls, as Warp makes it easier to index rows
            Qk0 = Q[k]
            Qk1 = Q[k + 1]
            Q[k] = c * Qk0 - s * Qk1
            Q[k + 1] = c * Qk1 + s * Qk0

        if end <= 0:
            # We did nothing, so diagonalization must have been achieved
            break

    return D, Q


@wp.func
def symmetric_eigenvalues_qr(A: Any, tol: Any):
    """
    Computes the eigenvalues and eigen vectors of a square symmetric matrix A using the QR algorithm

    Args:
        A: square symmetric matrix
        tol: Tolerance for the diagonalization residual (Linf norm of off-diagonal over diagonal terms)

    Returns a tuple (D: vector of eigenvalues, P: matrix with one eigenvector per row) such that A = P^T D P
    """

    # Put A under Hessenberg form (tridiagonal)
    Q, H = householder_make_hessenberg(A)

    # tridiagonal storage for H
    D = wp.get_diag(H)
    L = type(D)(A.dtype(0.0))
    for i in range(1, type(D).length):
        L[i - 1] = H[i, i - 1]

    Qt = wp.transpose(Q)
    ev, P = tridiagonal_symmetric_eigenvalues_qr(D, L, Qt, tol)
    return ev, P


def array_axpy(x: wp.array, y: wp.array, alpha: float = 1.0, beta: float = 1.0):
    """Performs y = alpha*x + beta*y"""

    dtype = wp.types.type_scalar_type(y.dtype)

    alpha = dtype(alpha)
    beta = dtype(beta)

    if x.shape != y.shape or x.device != y.device:
        raise ValueError("x and y arrays must have the same shape and device")

    # array_axpy requires a custom adjoint; unfortunately we cannot use `wp.func_grad`
    # as generic functions are not supported yet. Instead we use a non-differentiable kernel
    # and record a custom adjoint function on the tape.

    # temporarily disable tape to avoid printing warning that kernel is not differentiable
    (tape, wp.context.runtime.tape) = (wp.context.runtime.tape, None)
    wp.launch(kernel=_array_axpy_kernel, dim=x.shape, device=x.device, inputs=[x, y, alpha, beta])
    wp.context.runtime.tape = tape

    if tape is not None and (x.requires_grad or y.requires_grad):

        def backward_axpy():
            # adj_x += adj_y * alpha
            # adj_y = adj_y * beta
            array_axpy(x=y.grad, y=x.grad, alpha=alpha, beta=1.0)
            if beta != 1.0:
                array_axpy(x=y.grad, y=y.grad, alpha=0.0, beta=beta)

        tape.record_func(backward_axpy, arrays=[x, y])


@wp.kernel(enable_backward=False)
def _array_axpy_kernel(x: wp.array(dtype=Any), y: wp.array(dtype=Any), alpha: Any, beta: Any):
    i = wp.tid()
    y[i] = beta * y[i] + alpha * y.dtype(x[i])
