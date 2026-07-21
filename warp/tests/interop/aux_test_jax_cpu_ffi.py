# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Warp JAX FFI wrappers on a CPU-only JAX runtime with two host devices.

Launched as a subprocess by ``test_jax.py`` because ``JAX_PLATFORMS`` and ``XLA_FLAGS``
must be configured before JAX initializes.
"""

import jax
import jax.numpy as jnp
import numpy as np

import warp as wp


@wp.kernel
def triple_kernel(x: wp.array[float], y: wp.array[float]):
    tid = wp.tid()
    y[tid] = 3.0 * x[tid]


def triple_func(x: wp.array[float], y: wp.array[float]):
    wp.launch(triple_kernel, dim=x.shape, inputs=[x], outputs=[y])


def main():
    cpu_devices = jax.devices("cpu")
    assert len(cpu_devices) == 2, cpu_devices

    x = jax.device_put(np.arange(16, dtype=np.float32), cpu_devices[0])
    expected = 3.0 * np.arange(16, dtype=np.float32)
    no_preload_wrappers = (
        ("jax_kernel", wp.jax_kernel(triple_kernel, module_preload_mode=wp.JaxModulePreloadMode.NONE)),
        ("jax_callable", wp.jax_callable(triple_func, module_preload_mode=wp.JaxModulePreloadMode.NONE)),
    )
    for name, wrapper in no_preload_wrappers:
        run = jax.jit(lambda value, wrapper=wrapper: wrapper(value)[0])
        result = run(x)
        jax.block_until_ready(result)
        np.testing.assert_allclose(np.asarray(result), expected, err_msg=f"{name} with module preloading disabled")

    jax_kernel = wp.jax_kernel(triple_kernel)
    jax_callable = wp.jax_callable(triple_func)

    @jax.jit
    def run_kernel(x):
        return jax_kernel(x)[0]

    @jax.jit
    def run_callable(x):
        return jax_callable(x)[0]

    kernel_y = run_kernel(x)
    jax.block_until_ready(kernel_y)
    callable_y = run_callable(x)
    jax.block_until_ready(callable_y)
    np.testing.assert_allclose(np.asarray(kernel_y), expected)
    np.testing.assert_allclose(np.asarray(callable_y), expected)

    sharded_x = jnp.arange(16, dtype=jnp.float32).reshape((2, 8))
    sharded_kernel_y = jax.pmap(lambda row: jax_kernel(row)[0], devices=cpu_devices)(sharded_x)
    jax.block_until_ready(sharded_kernel_y)
    sharded_callable_y = jax.pmap(lambda row: jax_callable(row)[0], devices=cpu_devices)(sharded_x)
    jax.block_until_ready(sharded_callable_y)
    expected = 3.0 * np.arange(16, dtype=np.float32).reshape((2, 8))
    np.testing.assert_allclose(np.asarray(sharded_kernel_y), expected)
    np.testing.assert_allclose(np.asarray(sharded_callable_y), expected)


if __name__ == "__main__":
    main()
