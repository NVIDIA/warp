# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example jax_kernel()
#
# Examples of calling a Warp kernel from JAX.
###########################################################################

import math

import jax
import jax.numpy as jnp

import warp as wp
from warp.jax_experimental.ffi import jax_kernel


@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), output: wp.array(dtype=float)):
    tid = wp.tid()
    output[tid] = a[tid] + b[tid]


@wp.kernel
def sincos_kernel(angle: wp.array(dtype=float), sin_out: wp.array(dtype=float), cos_out: wp.array(dtype=float)):
    tid = wp.tid()
    sin_out[tid] = wp.sin(angle[tid])
    cos_out[tid] = wp.cos(angle[tid])


@wp.kernel
def matmul_kernel(
    a: wp.array2d(dtype=wp.float32),  # NxK
    b: wp.array2d(dtype=wp.float32),  # KxM
    c: wp.array2d(dtype=wp.float32),  # NxM
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


def example1():
    # two inputs and one output
    jax_add = jax_kernel(add_kernel)

    @jax.jit
    def f():
        n = 16
        a = jnp.arange(0, n, dtype=jnp.float32)
        b = jnp.ones(n, dtype=jnp.float32)
        return jax_add(a, b)

    print(f())


def example2():
    # one input and two outputs
    jax_sincos = jax_kernel(sincos_kernel)

    @jax.jit
    def f():
        n = 32
        a = jnp.linspace(0, 2 * math.pi, n)
        return jax_sincos(a)

    s, c = f()
    print(s)
    print(c)


def example3():
    # static custom launch dims
    N, M, K = 3, 4, 2

    jax_matmul = jax_kernel(matmul_kernel, launch_dims=(N, M))

    @jax.jit
    def f():
        a = jnp.full((N, K), 2, dtype=jnp.float32)
        b = jnp.full((K, M), 3, dtype=jnp.float32)
        return jax_matmul(a, b)

    print(f())


def example4():
    # dynamic custom launch dims
    jax_matmul = jax_kernel(matmul_kernel)

    @jax.jit
    def f():
        N1, M1, K1 = 3, 4, 2
        a1 = jnp.full((N1, K1), 2, dtype=jnp.float32)
        b1 = jnp.full((K1, M1), 3, dtype=jnp.float32)
        result1 = jax_matmul(a1, b1, launch_dims=(N1, M1))

        N2, M2, K2 = 4, 3, 2
        a2 = jnp.full((N2, K2), 2, dtype=jnp.float32)
        b2 = jnp.full((K2, M2), 3, dtype=jnp.float32)
        result2 = jax_matmul(a2, b2, launch_dims=(N2, M2))

        return result1, result2

    r1, r2 = f()
    print(r1)
    print(r2)


def main():
    wp.init()
    wp.load_module(device=wp.get_device())

    examples = [example1, example2, example3, example4]

    for example in examples:
        print("\n===========================================================================")
        print(f"{example.__name__}:")
        example()


if __name__ == "__main__":
    main()
