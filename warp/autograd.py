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

"""Utility functions for debugging automatic differentiation.

This module provides functions to evaluate and verify gradients computed by Warp's
automatic differentiation engine. Typical workflows involve computing Jacobian matrices
using both automatic differentiation and finite differences, then comparing them to
verify gradient accuracy.

Usage:
    This module must be explicitly imported::

        import warp.autograd
"""

# isort: skip_file

from warp._src.autograd import gradcheck as gradcheck
from warp._src.autograd import gradcheck_tape as gradcheck_tape
from warp._src.autograd import jacobian as jacobian
from warp._src.autograd import jacobian_fd as jacobian_fd
from warp._src.autograd import jacobian_plot as jacobian_plot


# TODO: Remove after cleaning up the public API.

from warp._src import autograd as _autograd


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_autograd, "warp", name)
