# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Describe native and Python-implemented Warp functions for the API reference."""

import inspect
import os
import re
from pathlib import Path
from urllib.parse import quote

from sphinx.ext.napoleon.docstring import GoogleDocstring

import warp as wp
from warp._src.context import ApiFunctionRegistration, Function, api_functions, builtin_functions

REPO_ROOT = Path(wp.__file__).parent.parent


def normalize_docstring(doc: str) -> str:
    """Normalize docstrings for consistent RST indentation and formatting."""
    if not doc:
        return ""
    cleaned = inspect.cleandoc(doc)
    if not cleaned:
        return ""
    rst = str(GoogleDocstring(cleaned))
    # Rewrite ``wp.`` aliases in :type:/:rtype: fields so Sphinx cross-references
    # resolve correctly.  Only target field-list lines to avoid mangling code
    # examples that legitimately use ``import warp as wp``.
    if "wp." not in rst:
        return rst
    return re.sub(
        r"^:(rtype|type\s+\w+):.*$", lambda m: re.sub(r"\bwp\.", "warp.", m.group(0)), rst, flags=re.MULTILINE
    )


def _with_defaults(func, args: dict[str, str]) -> dict[str, str]:
    """Append each registered default value to the rendered parameter annotations.

    Uses the same renderer as the type stub so that the documented signature and
    the IDE hint show either the substituted value or ``...`` for an internally
    inferred omission sentinel.

    Args:
        func: The built-in whose ``defaults`` supply the values.
        args: The rendered annotation per ``input_types`` key.

    Returns:
        A new mapping with ``= value`` appended wherever a default is registered.
    """
    result = {}
    for key, annotation in args.items():
        # ``input_types`` keeps the ``*``/``**`` prefix that ``defaults`` omits.
        name = key.lstrip("*")
        if key.startswith("*") or name not in func.defaults:
            result[key] = annotation
            continue

        value = func.defaults[name]
        result[key] = f"{annotation} = {wp._src.context.format_default_value(value)}"

    return result


def declared_overloads(head: Function) -> list[Function]:
    """List declarations without exposing runtime generic-specialization caches."""
    if head.is_builtin():
        return list(head.overloads)
    declarations = [
        f for f in (*head.user_overloads.values(), *head.user_templates.values()) if f.generic_parent is None
    ]
    # The head is already in its family; preserve distinct declarations, even
    # when they share a docstring or have similar rendered annotations.
    return list(dict.fromkeys(declarations)) or [head]


def source_url(obj, source_ref: str) -> str | None:
    """Link to the lines defining a Python object on GitHub, if it has Python source."""
    try:
        filename = inspect.getsourcefile(obj)
        lines, start = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        return None
    if filename is None:
        return None
    relative = Path(os.path.relpath(filename, REPO_ROOT)).as_posix()
    # Modules start at line 0 and are linked as a whole.
    anchor = f"#L{start}-L{start + len(lines) - 1}" if start else ""
    return f"https://github.com/NVIDIA/warp/blob/{quote(source_ref, safe='/')}/{relative}{anchor}"


def describe_function(
    head: Function,
    publication: ApiFunctionRegistration | None = None,
    source_ref: str = "main",
) -> list[dict[str, object]]:
    """Describe every declared overload using the relevant capability contract."""
    all_funcs = declared_overloads(head)
    native = head.is_builtin()
    exported = [f for f in all_funcs if wp._src.context.resolve_exported_function_sig(f) is not None] if native else []
    descriptions = []
    seen = set()
    for func in all_funcs:
        if func.hidden:
            continue
        args = {k: wp._src.context.type_str(v) for k, v in func.input_types.items()}
        args_str = ", ".join(f"{k}: {v}" for k, v in _with_defaults(func, args).items())
        if native:
            try:
                return_type = wp._src.context.type_str(func.value_func(None, None))
            except Exception:
                # The return type of a built-in whose value function cannot be evaluated
                # without concrete arguments is unknown here, not absent.
                return_type = "Any"
            python_callable = any(
                wp._src.codegen.func_match_args(func, list(f.input_types.values()), {}) for f in exported
            )
            differentiable = func.is_differentiable
            source = None
        else:
            # Only the declared annotation: compiling the function would also record
            # its inferred return type, so the output would depend on compilation state.
            annotation = func.adj.arg_types.get("return")
            return_type = None if annotation is None else wp._src.context.type_str(annotation)
            python_callable = publication.python_callable if publication is not None else None
            differentiable = publication.differentiable if publication is not None else None
            source = source_url(func.func, source_ref)
        doc = normalize_docstring(func.doc)
        key = (args_str, return_type, python_callable, differentiable, doc, source)
        if key in seen:
            continue
        seen.add(key)
        descriptions.append(
            {
                "args": args_str,
                "return_type": return_type,
                "python_callable": python_callable,
                "differentiable": differentiable,
                "doc": doc,
                "source_url": source,
            }
        )
    return descriptions


def validate_public_functions(module_name, module, symbols, aliases):
    """Require registrations to agree with the Warp functions a module exports."""
    for symbol in symbols:
        function = getattr(module, symbol)
        if not isinstance(function, Function):
            continue
        public_name = f"{module_name}.{symbol}"
        registration = api_functions.get(public_name)
        if registration is not None:
            if registration.function is not function:
                raise RuntimeError(
                    f"API function '{public_name}' is registered for a different object than its export."
                )
            continue
        # A re-export from a submodule is documented on the submodule's page.
        if symbol in aliases:
            canonical = api_functions.get(f"{aliases[symbol]}.{symbol}")
            if canonical is not None and canonical.function is function:
                continue
        # Only an actual root built-in export can bypass explicit registration.
        if module_name == "warp" and builtin_functions.get(symbol) is function:
            continue
        raise RuntimeError(
            f"Public Warp function '{public_name}' is not registered for documentation. "
            f'Call register_api_function({function.key}, module="{module_name}").'
        )
    for public_name, registration in api_functions.items():
        if registration.module != module_name:
            continue
        symbol = public_name.rsplit(".", 1)[1]
        if getattr(module, symbol, None) is not registration.function:
            raise RuntimeError(f"API function '{public_name}' is registered without a matching public export.")
