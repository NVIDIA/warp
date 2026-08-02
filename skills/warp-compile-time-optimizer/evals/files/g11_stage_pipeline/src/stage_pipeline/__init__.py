# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Five independent pipeline stages, each in its own Warp module.

The stages have separate lifecycles and separate stable block dimensions, so
they are deliberately not merged: a shared module would compile every kernel
once per block dimension in use.
"""

from __future__ import annotations

__all__ = ["geometry", "reduction", "spectral", "statistics", "transport"]
