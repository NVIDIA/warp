# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Deprecated alias for :mod:`warp.jax.custom_call`.

.. deprecated:: 1.14.0
    Use :mod:`warp.jax.custom_call` instead. This namespace will be removed in Warp 1.16.
"""

from ._deprecation import warn_deprecated_jax_experimental_namespace as _warn_deprecated_jax_experimental_namespace

_warn_deprecated_jax_experimental_namespace("warp.jax_experimental.custom_call", "warp.jax.custom_call")

from warp.jax.custom_call import jax_kernel as jax_kernel
