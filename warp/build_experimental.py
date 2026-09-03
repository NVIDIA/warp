# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental utilities for extending Warp kernel compilation.

This module's API is experimental and may change without deprecation during
the external-compilation feature's development.
"""

from warp._src.external_build import add_builtin as add_builtin
from warp._src.external_build import add_native_type as add_native_type

__all__ = ["add_builtin", "add_native_type"]
