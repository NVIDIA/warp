# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

###########################################################################
# Example jax_kernel()
#
# Examples of calling a Warp kernel from JAX.
###########################################################################

import math
from functools import partial

import jax
import jax.numpy as jnp

import warp as wp
from warp.jax_experimental.ffi import jax_kernel


@wp.kernel
def add_kernel(a: wp.array(dtype=int), b: wp.array(dtype=int), output: wp.array(dtype=int)):
    tid = wp.tid()
    output[tid] = a[tid] + b[tid]


@wp.kernel
def sincos_kernel(angle: wp.array(dtype=float), sin_out: wp.array(dtype=float), cos_out: wp.array(dtype=float)):
    tid = wp.tid()
    sin_out[tid] = wp.sin(angle[tid])
    cos_out[tid] = wp.cos(angle[tid])


@wp.kernel
def diagonal_kernel(output: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    output[tid] = wp.mat33(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0)


@wp.kernel
def matmul_kernel(
    a: wp.array2d(dtype=float),  # NxK
    b: wp.array2d(dtype=float),  # KxM
    c: wp.array2d(dtype=float),  # NxM
):
    # launch dims should be (N, M)
    i, j = wp.tid()
    N = a.shape[0]
    K = a.shape[1]
    M = b.shape[1]
    if i < N and j < M:
        s = wp.float32(0)
        for k in range(K):
            s += a[i, k] * b[k, j]
        c[i, j] = s


@wp.kernel
def scale_vec_kernel(a: wp.array(dtype=wp.vec2), s: float, output: wp.array(dtype=wp.vec2)):
    tid = wp.tid()
    output[tid] = a[tid] * s


def example1():
    # two inputs and one output
    jax_add = jax_kernel(add_kernel)

    @jax.jit
    def f():
        n = 10
        a = jnp.arange(n, dtype=jnp.int32)
        b = jnp.ones(n, dtype=jnp.int32)
        return jax_add(a, b)

    print(f())


def example2():
    # one input and two outputs
    jax_sincos = jax_kernel(sincos_kernel, num_outputs=2)

    @jax.jit
    def f():
        n = 32
        a = jnp.linspace(0, 2 * math.pi, n)
        return jax_sincos(a)

    s, c = f()
    print(s)
    print()
    print(c)


def example3():
    # multiply vectors by scalar
    jax_scale_vec = jax_kernel(scale_vec_kernel)

    @jax.jit
    def f():
        a = jnp.arange(10, dtype=jnp.float32).reshape((5, 2))  # array of vec2
        s = 2.0
        return jax_scale_vec(a, s)

    b = f()
    print(b)


def example4():
    # multiply vectors by scalar (static arg)
    jax_scale_vec = jax_kernel(scale_vec_kernel)

    # NOTE: scalar arguments must be static compile-time constants
    @partial(jax.jit, static_argnames=["s"])
    def f(a, s):
        return jax_scale_vec(a, s)

    a = jnp.arange(10, dtype=jnp.float32).reshape((5, 2))  # array of vec2
    s = 3.0

    b = f(a, s)
    print(b)


def example5():
    N, M, K = 3, 4, 2

    # specify default launch dims
    jax_matmul = jax_kernel(matmul_kernel, launch_dims=(N, M))

    @jax.jit
    def f():
        a = jnp.full((N, K), 2, dtype=jnp.float32)
        b = jnp.full((K, M), 3, dtype=jnp.float32)

        # use default launch dims
        return jax_matmul(a, b)

    print(f())


def example6():
    # don't specify default launch dims
    jax_matmul = jax_kernel(matmul_kernel)

    @jax.jit
    def f():
        N1, M1, K1 = 3, 4, 2
        a1 = jnp.full((N1, K1), 2, dtype=jnp.float32)
        b1 = jnp.full((K1, M1), 3, dtype=jnp.float32)

        # use custom launch dims
        result1 = jax_matmul(a1, b1, launch_dims=(N1, M1))

        N2, M2, K2 = 4, 3, 2
        a2 = jnp.full((N2, K2), 2, dtype=jnp.float32)
        b2 = jnp.full((K2, M2), 3, dtype=jnp.float32)

        # use custom launch dims
        result2 = jax_matmul(a2, b2, launch_dims=(N2, M2))

        return result1, result2

    r1, r2 = f()
    print(r1)
    print()
    print(r2)


def example7():
    # no inputs and one output
    jax_diagonal = jax_kernel(diagonal_kernel)

    @jax.jit
    def f():
        # launch dimensions determine output size
        return jax_diagonal(launch_dims=4)

    print(f())


def main():
    wp.init()
    wp.load_module(device=wp.get_device())

    examples = [example1, example2, example3, example4, example5, example6, example7]

    for example in examples:
        print("\n===========================================================================")
        print(f"{example.__name__}:")
        example()


if __name__ == "__main__":
    main()
