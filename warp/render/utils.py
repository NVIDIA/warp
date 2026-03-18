# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: Remove after cleaning up the public API.

from warp._src.render import utils as _utils
from warp._src.utils import warn_deprecated_namespace as _warn_deprecated_namespace


def __getattr__(name):
    # Use simple getattr since namespace warning is already issued by
    # _warn_deprecated_namespace. Individual symbol warnings would be
    # redundant and confusing (suggesting promotion to warp.render)
    return getattr(_utils, name)


_warn_deprecated_namespace(__name__)
