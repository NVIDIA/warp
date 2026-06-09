# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Callable target fixture imported as a Python module."""

import warp as wp


@wp.func
def callable_external_module_triple_it(x: float):
    return x * 3.0
