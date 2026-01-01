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

"""Warp built-in types and functions.

Built-in types and functions are available within Warp kernels and optionally also
from the Warp Python runtime API.

Each built-in function is tagged to indicate where it can be used:

- **Kernel** - Can be called from inside a Warp kernel
- **Python** - Can be called at the Python scope
- **Differentiable** - Propagates gradients when used in reverse mode automatic differentiation

For a listing of the API that is exclusively intended to be used at the Python scope
and run inside the CPython interpreter, see :doc:`/api_reference/warp`.
"""

# Placeholder module listing all built-ins.
# Used to build the language reference documentation.

from warp._src import context as _context
from warp._src.math import *

locals().update(_context.builtin_functions.items())
