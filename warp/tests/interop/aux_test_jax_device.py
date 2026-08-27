# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe JAX execution on a Warp device in an isolated process.

JAX/XLA compilation can terminate the process through a native fatal error,
which Python exception handling cannot catch. Running the probe in a child
process lets the parent test process inspect its exit status and report the
failure without breaking the parallel test worker pool.
"""

import os
import sys

import warp as wp

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp


def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} DEVICE")

    device = wp.get_device(sys.argv[1])
    with jax.default_device(wp.device_to_jax(device)):
        jax.block_until_ready(jnp.arange(10, dtype=jnp.float32))


if __name__ == "__main__":
    main()
