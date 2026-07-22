# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def warn_deprecated_jax_experimental_namespace(old_path: str, replacement: str) -> None:
    from warp._src.logger import log_warning  # noqa: PLC0415

    log_warning(
        f"The `{old_path}` namespace is deprecated and will be removed in Warp 1.18. Use {replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def warn_deprecated_jax_experimental_graph_cache_getter(api_name: str) -> None:
    from warp._src.logger import log_warning  # noqa: PLC0415

    log_warning(
        f"`{api_name}()` is deprecated and will be removed with `warp.jax_experimental`. "
        "Use `jax_func.graph_cache_max` on the callable returned by `warp.jax_callable(...)`; "
        "the default `32` applies to new top-level callables.",
        DeprecationWarning,
        stacklevel=2,
    )


def warn_deprecated_jax_experimental_graph_cache_setter(api_name: str) -> None:
    from warp._src.logger import log_warning  # noqa: PLC0415

    log_warning(
        f"`{api_name}()` is deprecated and will be removed with `warp.jax_experimental`. "
        "Pass `graph_cache_max=<value>` to `warp.jax_callable(...)` or update an existing callable with "
        "`jax_func.graph_cache_max = <value>`.",
        DeprecationWarning,
        stacklevel=2,
    )
