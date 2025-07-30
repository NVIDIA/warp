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
# Example jax_callable()
#
# Examples of calling annotated Python functions from JAX.
###########################################################################

from functools import partial

import jax
import jax.numpy as jnp

import warp as wp
from warp.jax_experimental.ffi import jax_callable


@wp.kernel
def scale_kernel(a: wp.array(dtype=float), s: float, output: wp.array(dtype=float)):
    tid = wp.tid()
    output[tid] = a[tid] * s


@wp.kernel
def scale_vec_kernel(a: wp.array(dtype=wp.vec2), s: float, output: wp.array(dtype=wp.vec2)):
    tid = wp.tid()
    output[tid] = a[tid] * s


# The Python function to call.
# Note the argument annotations, just like Warp kernels.
def scale_func(
    # inputs
    a: wp.array(dtype=float),
    b: wp.array(dtype=wp.vec2),
    s: float,
    # outputs
    c: wp.array(dtype=float),
    d: wp.array(dtype=wp.vec2),
):
    wp.launch(scale_kernel, dim=a.shape, inputs=[a, s], outputs=[c])
    wp.launch(scale_vec_kernel, dim=b.shape, inputs=[b, s], outputs=[d])


@wp.kernel
def accum_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] += a[tid]


def in_out_func(
    a: wp.array(dtype=float),  # input only
    b: wp.array(dtype=float),  # input and output
    c: wp.array(dtype=float),  # output only
):
    wp.launch(scale_kernel, dim=a.size, inputs=[a, 2.0], outputs=[c])
    wp.launch(accum_kernel, dim=a.size, inputs=[a, b])  # modifies `b`


def example1():
    jax_func = jax_callable(scale_func, num_outputs=2)

    @jax.jit
    def f():
        # inputs
        a = jnp.arange(10, dtype=jnp.float32)
        b = jnp.arange(10, dtype=jnp.float32).reshape((5, 2))  # wp.vec2
        s = 2.0

        # output shapes
        output_dims = {"c": a.shape, "d": b.shape}

        c, d = jax_func(a, b, s, output_dims=output_dims)

        return c, d

    r1, r2 = f()
    print(r1)
    print(r2)


def example2():
    jax_func = jax_callable(scale_func, num_outputs=2)

    # NOTE: scalar arguments must be static compile-time constants
    @partial(jax.jit, static_argnames=["s"])
    def f(a, b, s):
        # output shapes
        output_dims = {"c": a.shape, "d": b.shape}

        c, d = jax_func(a, b, s, output_dims=output_dims)

        return c, d

    # inputs
    a = jnp.arange(10, dtype=jnp.float32)
    b = jnp.arange(10, dtype=jnp.float32).reshape((5, 2))  # wp.vec2
    s = 3.0

    r1, r2 = f(a, b, s)
    print(r1)
    print(r2)


def example3():
    # Using input-output arguments

    jax_func = jax_callable(in_out_func, num_outputs=2, in_out_argnames=["b"])

    f = jax.jit(jax_func)

    a = jnp.ones(10, dtype=jnp.float32)
    b = jnp.arange(10, dtype=jnp.float32)

    b, c = f(a, b)
    print(b)
    print(c)


def main():
    wp.init()
    wp.load_module(device=wp.get_device())

    examples = [example1, example2, example3]

    for example in examples:
        print("\n===========================================================================")
        print(f"{example.__name__}:")
        example()


if __name__ == "__main__":
    main()
