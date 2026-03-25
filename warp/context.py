# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: Remove after cleaning up the public API.

from warp._src import context as _context
from warp._src.utils import warn_deprecated_namespace as _warn_deprecated_namespace


def __getattr__(name):
    from warp._src.utils import get_deprecated_api, warn  # noqa: PLC0415

    # Handle special case: Devicelike -> DeviceLike (capital L)
    # This is both a rename AND a relocation to warp.DeviceLike
    if name == "Devicelike":
        warn(
            "The symbol `warp.context.Devicelike` will soon be removed from the public API. Use `warp.DeviceLike` instead.",
            DeprecationWarning,
        )
        return _context.DeviceLike

    return get_deprecated_api(_context, "warp", name)


_warn_deprecated_namespace(__name__)
