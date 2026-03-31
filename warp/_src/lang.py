# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
