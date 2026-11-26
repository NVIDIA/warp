# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated alias of :mod:`warp.experimental.build`, to be removed in Warp 1.24."""

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

from warp._src.logger import log_warning as _log_warning

if _TYPE_CHECKING:
    from warp.experimental.build import add_builtin as add_builtin
    from warp.experimental.build import add_native_type as add_native_type

__all__ = ["add_builtin", "add_native_type"]


def __getattr__(name: str) -> _Any:
    # The forwarded names are deliberately not bound in this module's namespace,
    # so every access (attribute, ``from ... import``, or ``import *``) reaches
    # this function and warns at the caller's location.
    if name in __all__:
        import warp.experimental.build  # noqa: PLC0415

        _log_warning(
            f"wp.build_experimental.{name} is deprecated and will be removed in Warp 1.24. "
            f"Import warp.experimental.build and use warp.experimental.build.{name} instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return getattr(warp.experimental.build, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
