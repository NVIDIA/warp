# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental APIs for extending Warp kernel compilation with external C++ and CUDA code.

This module must be explicitly imported::

    import warp.experimental.build

See :mod:`warp.experimental` for the stability rules that apply to this module.
"""

from warp._src.external_build import add_builtin as add_builtin
from warp._src.external_build import add_native_type as add_native_type

__all__ = ["add_builtin", "add_native_type"]
