# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated JAX integration using custom calls.

.. deprecated:: 1.10.0
    This module is deprecated. Use :mod:`warp.jax_experimental.ffi` instead, which
    provides the current implementation of JAX integration and supports JAX 0.4.25
    and later, including JAX 0.8.0+.

This module contains the legacy custom call-based implementation that is only compatible
with JAX 0.4.25 - 0.7.x and is retained for backward compatibility.
"""

from warp._src.jax_experimental.custom_call import jax_kernel as jax_kernel
