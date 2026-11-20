# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overrides for supported public Warp host APIs.

Neutral helpers depend only on the standard library so that array types can
use them during import. Decorators capture the live configuration dictionary
when they are registered, before any wrapped calls execute.
"""

import functools
import inspect
import sys
from contextvars import ContextVar

# Restore suppression and caller location together.
_dispatch_state = ContextVar("warp_function_state", default=(False, None))

# Reuse registered wrappers across startup activation changes.
WARP_FUNCTIONS = {}
_method_bindings = []


def register_api_method(owner, name):
    """Register an explicit class attribute whose wrapper is installed on activation."""
    _method_bindings.append((owner, name))


def set_function_overrides_enabled(enabled):
    """Switch registered bindings in loaded Warp modules at application startup."""
    replacements = {
        id(native if enabled else wrapped): wrapped if enabled else native for wrapped, native in WARP_FUNCTIONS.items()
    }
    for name, module in tuple(sys.modules.items()):
        if module is None or (name != "warp" and not name.startswith("warp.")):
            continue
        namespace = vars(module)
        for attribute, value in tuple(namespace.items()):
            replacement = replacements.get(id(value))
            if replacement is not None:
                namespace[attribute] = replacement
    for owner, name in _method_bindings:
        replacement = replacements.get(id(getattr(owner, name, None)))
        if replacement is not None:
            setattr(owner, name, replacement)


def default_warp_function(cls, func, types, args=(), kwargs=None):
    """Defer to the next override without executing the Warp implementation."""
    return NotImplemented


def get_warp_function_callsite():
    """Return the originating ``(filename, lineno, function)`` or ``None``.

    Store the location rather than frames or code objects so copied contexts
    preserve their origin without retaining the caller's locals.
    """
    return _dispatch_state.get()[1]


def _capture_warp_function_callsite():
    """Capture the application caller without retaining its frame or operands."""
    frame = sys._getframe(2)
    try:
        while frame is not None:
            package = frame.f_globals.get("__package__") or ""
            if package != "warp._src" and not package.startswith("warp._src."):
                return frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name
            frame = frame.f_back
        return None
    finally:
        del frame


def suppress_warp_function(func):
    """Suppress dispatch throughout an internal implementation, such as initialization."""
    import warp.config as config  # noqa: PLC0415

    config_values = config.__dict__

    @functools.wraps(func)
    def suppressed(*args, **kwargs):
        try:
            if not config_values["enable_array_function_overrides"]:
                return func(*args, **kwargs)
            suppressed, origin = _dispatch_state.get()
            if suppressed:
                return func(*args, **kwargs)
            token = _dispatch_state.set((True, origin))
            try:
                return func(*args, **kwargs)
            finally:
                _dispatch_state.reset(token)
        except BaseException:
            # Kept exceptions must not retain operands through forwarding frames.
            del args, kwargs
            raise

    WARP_FUNCTIONS[suppressed] = func
    return suppressed if config_values["enable_array_function_overrides"] else func


def _visit_operand(value, recurse, candidates, seen_type_ids, seen_containers):
    """Collect an operand's hook without allocating a recursive closure per call."""
    cls = type(value)
    hook = getattr(cls, "__warp_function__", None)
    if hook is not None and getattr(hook, "__func__", hook) is not default_warp_function:
        if not callable(hook):
            raise TypeError(f"{cls.__name__}.__warp_function__ must be callable or None")
        if inspect.isfunction(hook) and inspect.getattr_static(cls, "__warp_function__", None) is hook:
            raise TypeError(f"{cls.__name__}.__warp_function__ must be declared with @classmethod")
        identity = id(cls)
        if identity not in seen_type_ids:
            seen_type_ids.add(identity)
            for index, (previous, _) in enumerate(candidates):
                # Compare actual MRO entries by identity, ignoring virtual ABCs.
                if any(previous is ancestor for ancestor in cls.__mro__[1:]):
                    candidates.insert(index, (cls, hook))
                    break
            else:
                candidates.append((cls, hook))
    elif recurse and (cls is list or cls is tuple) and id(value) not in seen_containers:
        seen_containers.add(id(value))
        for item in value:
            _visit_operand(item, True, candidates, seen_type_ids, seen_containers)


class _ArgumentExtractor:
    """Validate calls and collect overrides in declaration order."""

    def __init__(self, signature, sequence_parameters):
        self.signature = signature
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in signature.parameters.values()):
            raise ValueError("Warp API overrides require a fixed signature")
        self.sequence_parameters = frozenset(sequence_parameters)
        unknown = self.sequence_parameters.difference(signature.parameters)
        if unknown:
            raise ValueError(f"Unknown Warp function sequence parameters: {', '.join(sorted(unknown))}")

    def candidates(self, args, kwargs):
        # Bind before looking up hooks, without adding omitted defaults.
        bound = self.signature.bind(*args, **kwargs)
        candidates = []
        seen_type_ids = set()
        seen_containers = set()

        # Signature.bind retains declaration order without adding omitted defaults.
        for name, value in bound.arguments.items():
            _visit_operand(value, name in self.sequence_parameters, candidates, seen_type_ids, seen_containers)
        return candidates


def warp_function(func=None, *, sequence_parameters=(), signature=None):
    """Register a public function for operand-based ``__warp_function__`` dispatch.

    Args:
        func: Function whose default body is executed under dispatch suppression.
        sequence_parameters: Parameters whose exact list and tuple values are
            recursively inspected in addition to direct argument values.
        signature: An explicit signature or callable supplying argument metadata
            for a forwarding function such as ``launch_tiled``.
    """
    if func is None:
        return functools.partial(warp_function, sequence_parameters=sequence_parameters, signature=signature)

    import warp.config as config  # noqa: PLC0415

    config_values = config.__dict__
    dispatch_signature = signature if isinstance(signature, inspect.Signature) else inspect.signature(signature or func)
    extractor = _ArgumentExtractor(dispatch_signature, sequence_parameters)

    @functools.wraps(func)
    def dispatch(*args, **kwargs):
        try:
            if not config_values["enable_array_function_overrides"]:
                return func(*args, **kwargs)
            suppressed, origin = _dispatch_state.get()
            if suppressed:
                if origin is not None:
                    return func(*args, **kwargs)
                token = _dispatch_state.set((True, _capture_warp_function_callsite()))
                try:
                    return func(*args, **kwargs)
                finally:
                    _dispatch_state.reset(token)

            candidates = extractor.candidates(args, kwargs)
            origin = _capture_warp_function_callsite()
            token = _dispatch_state.set((True, origin))
            try:
                if candidates:
                    types = tuple(cls for cls, _ in candidates)
                    for _, hook in candidates:
                        result = hook(dispatch, types, args, kwargs)
                        if result is not NotImplemented:
                            return result
                return func(*args, **kwargs)
            finally:
                _dispatch_state.reset(token)
        except BaseException:
            # Drop our references without mutating hook-observed containers.
            del args, kwargs
            raise

    WARP_FUNCTIONS[dispatch] = func
    return dispatch if config_values["enable_array_function_overrides"] else func
