# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect and normalize architectures used by Warp builds.

This module maps platform-specific x86-64 and ARM64 names to the
canonical identifiers shared by Warp's build, packaging, and runtime code.
"""

from __future__ import annotations

import platform
from typing import Literal

Architecture = Literal["x86_64", "aarch64"]

_ARCHITECTURE_ALIASES: dict[str, Architecture] = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "arm64": "aarch64",
    "aarch64": "aarch64",
}


def normalize_architecture(machine: str) -> Architecture:
    try:
        return _ARCHITECTURE_ALIASES[machine.strip().lower()]
    except KeyError:
        raise RuntimeError(f"Unrecognized machine architecture {machine!r}") from None


def machine_architecture() -> Architecture:
    return normalize_architecture(platform.machine())
