# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-down declarations of Warp module membership.

This module lets public modules (e.g. ``warp.sparse``) declare which Warp module
the kernels/functions/structs defined in the internal ``warp._src`` tree belong
to, instead of each ``_src`` file pointing upward to its public module. The maps
are consulted by ``warp._src.context.get_module()`` when a construct is registered
(registration timing is unchanged—only the target name is inferred differently).

It deliberately has no Warp dependencies so that ``warp/__init__.py`` can populate
the maps at the very top, before importing ``warp._src.context`` (which, through
its import chain, eagerly creates several Warp modules). The declarations must be
in effect before those constructs are created.
"""

from __future__ import annotations

from collections.abc import Iterable

# ``_module_source_map`` is the common case: every construct whose ``__module__``
# is a given source module belongs to one Warp module. ``_module_construct_map``
# is the handpick escape hatch: it pins individual constructs (keyed by their
# ``__qualname__``) to a Warp module regardless of which ``_src`` file defines
# them, so the ``_src`` tree can be reorganized without changing Warp modules.
_module_source_map: dict[str, str] = {}  # source __module__ -> Warp module name
_module_construct_map: dict[tuple[str, str], str] = {}  # (source __module__, qualname) -> Warp module name


def register_module_source(public_module: str, source_module: str) -> None:
    """Declare that all Warp constructs defined in a source module belong to a Warp module.

    Every kernel, function, and struct whose ``__module__`` is ``source_module``
    is registered into the Warp module named ``public_module``. Each source module
    maps to exactly one Warp module.

    This must be called before ``source_module`` is imported, so the mapping is in
    effect when its constructs are created. All declarations therefore live at the
    top of ``warp/__init__.py``, which Python runs to completion before any
    ``warp._src`` submodule can be imported.

    Args:
        public_module: Name of the target Warp module (e.g. ``"warp.sparse"``).
        source_module: Name of the source Python module whose constructs are
            grouped (e.g. ``"warp._src.sparse"``).

    Raises:
        RuntimeError: If ``source_module`` is already mapped to a different Warp module.
    """
    existing = _module_source_map.get(source_module)
    if existing is not None and existing != public_module:
        raise RuntimeError(
            f"Source module '{source_module}' is already registered to Warp module "
            f"'{existing}'; cannot also register it to '{public_module}'."
        )
    _module_source_map[source_module] = public_module


def register_module_constructs(public_module: str, constructs: Iterable[tuple[str, str]]) -> None:
    """Handpick individual Warp constructs into a Warp module.

    Each entry pins a single construct, identified by its source module and
    ``__qualname__``, to ``public_module``. This takes precedence over any
    :func:`register_module_source` declaration for the same construct, allowing the
    internal ``warp._src`` tree to be reorganized while keeping the resulting Warp
    modules stable.

    Like :func:`register_module_source`, this must be called before the constructs
    are created (i.e. before their source modules are imported).

    Args:
        public_module: Name of the target Warp module (e.g. ``"warp.sparse"``).
        constructs: Iterable of ``(source_module, qualname)`` pairs identifying the
            constructs to register, where ``qualname`` is the construct's
            ``__qualname__``.

    Raises:
        RuntimeError: If any construct is already pinned to a different Warp module.
    """
    for source_module, qualname in constructs:
        key = (source_module, qualname)
        existing = _module_construct_map.get(key)
        if existing is not None and existing != public_module:
            raise RuntimeError(
                f"Construct '{qualname}' from '{source_module}' is already registered to Warp "
                f"module '{existing}'; cannot also register it to '{public_module}'."
            )
        _module_construct_map[key] = public_module


def resolve_module_name(py_module: str, qualname: str | None = None) -> str:
    """Resolve the Warp module name a construct should be registered into.

    Resolution order, most specific first:

    1. A handpicked construct declaration (:func:`register_module_constructs`).
    2. A source-module declaration (:func:`register_module_source`).
    3. The source Python module name itself (the default).
    """
    if qualname is not None:
        warp_module = _module_construct_map.get((py_module, qualname))
        if warp_module is not None:
            return warp_module

    warp_module = _module_source_map.get(py_module)
    if warp_module is not None:
        return warp_module

    return py_module
