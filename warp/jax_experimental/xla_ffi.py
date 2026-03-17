# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: Remove after cleaning up the public API.

from warp._src.jax_experimental import xla_ffi as _xla_ffi
from warp._src.utils import warn_deprecated_namespace as _warn_deprecated_namespace


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_xla_ffi, "warp.jax_experimental", name)


_warn_deprecated_namespace(__name__)
