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

"""Iterative linear solvers for sparse linear systems.

This module provides GPU-accelerated iterative methods for solving linear systems,
including conjugate gradient (CG), biconjugate gradient stabilized (BiCGSTAB), conjugate
residual (CR), and generalized minimal residual (GMRES) methods.
"""

# isort: skip_file

from warp._src.optim.linear import LinearOperator as LinearOperator
from warp._src.optim.linear import aslinearoperator as aslinearoperator
from warp._src.optim.linear import bicgstab as bicgstab
from warp._src.optim.linear import cg as cg
from warp._src.optim.linear import cr as cr
from warp._src.optim.linear import gmres as gmres
from warp._src.optim.linear import preconditioner as preconditioner


# TODO: Remove after cleaning up the public API.

from warp._src.optim import linear as _linear


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_linear, "warp.optim", name)
