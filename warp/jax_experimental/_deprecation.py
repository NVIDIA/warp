# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def warn_deprecated_jax_experimental_namespace(old_path: str, new_path: str) -> None:
    from warp._src.logger import log_warning  # noqa: PLC0415

    log_warning(
        f"The `{old_path}` namespace is deprecated and will be removed in Warp 1.16. Use `{new_path}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
