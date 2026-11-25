# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental bridge for calling Warp SIMT functions from kernels written in cuTile Python.

:func:`export_func` packages a ``@wp.func`` as a ``.tilelib`` that cuTile
kernels call directly, deriving the call contract from the function's
signature. :func:`encode_struct` and :func:`array_pointer_args` encode
struct and array arguments for those calls. See :doc:`the cuTile
interoperability guide </user_guide/interoperability/cutile>` for usage and
the calling convention, and the API reference for each function's contract.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import functools
import hashlib
import keyword
import linecache
import os
import re
import shutil
import subprocess
import types
import uuid
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import warp as wp
import warp._src.build
import warp._src.codegen
import warp._src.context
import warp._src.types
from warp._src import cutile_bytecode
from warp._src.logger import log_warning

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from warp._src.context import DeviceLike


_TILELIB_SCHEMA_VERSION = 4
"""Bump whenever the emitted bytecode contract or tilelib packing layout changes (cache identity)."""

_UNIVERSAL_LENGTHS = (32, 64, 128, 256, 512, 1024)
"""Tile lengths every thread-scope declaration is built for (Tile IR tile dimensions are powers of two).

One LTOIR backs them all: the compiled Warp code is per-invocation and does not depend on the length.
"""

_CANONICAL_BLOCK_DIM = 128
"""Block size baked into thread-scope codegen, so the cache identity does not vary with the ``num_worker_warps`` hint."""

_BVH_QUERY_STACK_SIZE = 32
"""Traversal-stack entries per thread for mesh AABB/sphere and BVH queries; must match ``BVH_QUERY_STACK_SIZE`` in ``bvh.h``."""

_DEFAULT_SLAB_THREADS = 256
"""Smallest worker block a thread-scope traversal slab is sized for (8 warps, the common case)."""


def _struct_contains_array(struct: warp._src.codegen.Struct) -> bool:
    """Return True if ``struct`` has a ``wp.array`` field at any nesting depth."""
    for var in struct.vars.values():
        if warp._src.types.is_array(var.type):
            return True
        if isinstance(var.type, warp._src.codegen.Struct) and _struct_contains_array(var.type):
            return True
    return False


def _scalar_map(ct: types.ModuleType) -> dict[type, tuple[str, object]]:
    """Return the Warp-scalar-type -> (C type string, cuTile dtype) mapping."""
    return {
        wp.float32: ("float", ct.float32),
        wp.float64: ("double", ct.float64),
        wp.int32: ("int", ct.int32),
        wp.int64: ("long long", ct.int64),
        wp.uint32: ("unsigned int", ct.uint32),
        wp.uint64: ("unsigned long long", ct.uint64),
        wp.int16: ("short", ct.int16),
        wp.uint16: ("unsigned short", ct.uint16),
        wp.int8: ("signed char", ct.int8),
        wp.uint8: ("unsigned char", ct.uint8),
    }


def _next_pow2(n: int) -> int:
    """Return the smallest power of two >= n."""
    return 1 << (n - 1).bit_length()


def _prod(shape: Sequence[int]) -> int:
    """Product of an int sequence."""
    r = 1
    for s in shape:
        r *= s
    return r


class _Operand:
    """One foreign-call operand: a tile on the cuTile side, one wrapper parameter.

    ``width`` is the number of tile elements consumed per invocation;
    ``by_value`` selects the rank-0 mem_layout (by-value scalar) ABI, legal
    only for inputs. Every operand is per-invocation: a value shared by all
    invocations is expanded to a full tile by the traced call body, so one
    declaration serves every call pattern.
    """

    def __init__(
        self,
        name: str,
        ctype: str,
        ct_dtype: object,
        width: int = 1,
        by_value: bool = False,
        is_output: bool = False,
        alignment: int = 1,
    ):
        """Record one operand's name, C type, cuTile dtype, width, and ABI flags."""
        if by_value and (is_output or width != 1):
            raise ValueError("By-value operands must be single-element inputs")
        self.name = name
        self.ctype = ctype
        self.ct_dtype = ct_dtype
        self.width = width
        self.by_value = by_value
        self.is_output = is_output
        self.alignment = alignment

    @property
    def cname(self) -> str:
        """C identifier of this operand's wrapper parameter.

        Derived under the reserved ``_wp_`` prefix so any legal Warp parameter
        name (including C++ keywords such as ``auto`` or ``register``) is
        emitted as valid C.
        """
        return f"_wp_p_{self.name}"

    def tile_shape(self, num_threads: int) -> tuple[int]:
        """Rank-1 cuTile tile shape carrying this operand."""
        return (num_threads * self.width,)

    def wrapper_param(self) -> str:
        """C parameter declaration for the extern wrapper."""
        if self.by_value:
            return f"{self.ctype} {self.cname}"
        const = "" if self.is_output else "const "
        return f"{const}{self.ctype}* {self.cname}"


class _ArgSpec:
    """ABI plan for one Warp function parameter or return value.

    Expands the Warp type into one or more foreign operands and provides the
    marshalling code between the flat fragment buffers and the Warp C++ value.
    """

    def __init__(self, ct: types.ModuleType, name: str, wp_type: type, is_output: bool = False):
        """Expand one Warp parameter or return type into its foreign operands."""
        self.name = name
        self.wp_type = wp_type
        self.is_output = is_output

        scalar_map = _scalar_map(ct)
        # resolve through the bases so aliases such as wp.handle (a uint64) are scalars
        scalar_key = next((t for t in getattr(wp_type, "__mro__", ()) if t in scalar_map), None)
        if scalar_key is not None:
            self.kind = "scalar"
            ctype, ct_dtype = scalar_map[scalar_key]
            self.logical = 1
            self.operands = [
                _Operand(
                    name,
                    ctype,
                    ct_dtype,
                    by_value=not is_output,
                    is_output=is_output,
                )
            ]
        elif hasattr(wp_type, "_wp_scalar_type_") and hasattr(wp_type, "_shape_"):
            scalar = wp_type._wp_scalar_type_
            if scalar not in scalar_map:
                raise TypeError(f"Unsupported element type {scalar!r} for argument '{name}'")
            self.kind = "vec" if len(wp_type._shape_) == 1 else "mat"
            ctype, ct_dtype = scalar_map[scalar]
            self.logical = _prod(wp_type._shape_)
            self.operands = [
                _Operand(
                    name,
                    ctype,
                    ct_dtype,
                    width=_next_pow2(self.logical),
                    is_output=is_output,
                )
            ]
        elif isinstance(wp_type, warp._src.codegen.Struct):
            if is_output and _struct_contains_array(wp_type):
                raise TypeError(f"Arrays cannot be returned from a foreign function ('{name}' holds an array field)")
            self.kind = "struct"
            size = ctypes.sizeof(wp_type.ctype)
            align = ctypes.alignment(wp_type.ctype)
            self.logical = size
            self.operands = [
                _Operand(
                    name,
                    "int",
                    ct.int32,
                    width=_next_pow2(max(1, (size + 3) // 4)),
                    is_output=is_output,
                    alignment=max(1, align // 4),
                )
            ]
        elif warp._src.types.is_array(wp_type):
            if is_output:
                raise TypeError("Foreign functions cannot return a wp.array")
            # Accept both wp.array instances and subscript annotations (wp.array[float]).
            concrete = getattr(type(wp_type), "_concrete_cls", None) or type(wp_type)
            if concrete is not warp._src.types.array:
                raise TypeError(f"Unsupported array kind {type(wp_type)!r} for argument '{name}'")
            elem_ctype = warp._src.codegen.Var.type_to_ctype(wp_type.dtype)
            self.kind = "array"
            self.elem_ctype = elem_ctype
            self.logical = 1
            # the address and extents travel as per-invocation by-value tiles
            # like any other scalar; the call body expands the one
            # (address, sizes) tuple a kernel passes into those tiles
            self.operands = [_Operand(f"{name}_ptr", "unsigned long long", ct.uint64, by_value=True)]
            for d in range(wp_type.ndim):
                self.operands.append(_Operand(f"{name}_shape{d}", "int", ct.int32, by_value=True))
        elif wp_type is Any:
            raise TypeError(
                f"Parameter '{name}' is generic (typing.Any); foreign functions need a single concrete signature"
            )
        else:
            raise TypeError(
                f"Unsupported Warp type {getattr(wp_type, '__name__', warp._src.types.type_repr(wp_type))} "
                f"for argument '{name}': supported are "
                "scalar, vector, matrix, quaternion, transform, struct, and contiguous wp.array types"
            )

    def marshal_in(self, var: str) -> str:
        """C statements converting the wrapper parameter(s) into a Warp value."""
        if self.kind == "scalar":
            return f"    const {self.operands[0].ctype} {var} = {self.operands[0].cname};\n"
        cpp_type = warp._src.codegen.Var.type_to_ctype(self.wp_type)
        if self.kind == "array":
            dims = ", ".join(f"{self.operands[1 + d].cname}" for d in range(self.wp_type.ndim))
            return f"    {cpp_type} {var}(reinterpret_cast<{self.elem_ctype}*>({self.operands[0].cname}), {dims}, nullptr);\n"
        if self.kind == "struct":
            # locals use the reserved _wp_ prefix so they can never be
            # shadowed by (or shadow) a user parameter name
            return (
                f"    {cpp_type} {var};\n"
                f"    {{\n"
                f"        char* _wp_pd = (char*)&{var};\n"
                f"        const char* _wp_ps = (const char*){self.operands[0].cname};\n"
                f"        for (int _wp_i = 0; _wp_i < (int)sizeof({cpp_type}); ++_wp_i) _wp_pd[_wp_i] = _wp_ps[_wp_i];\n"
                f"    }}\n"
            )
        code = f"    {cpp_type} {var};\n"
        if self.kind == "vec":
            for i in range(self.logical):
                code += f"    {var}[{i}] = {self.operands[0].cname}[{i}];\n"
        else:
            rows, cols = self.wp_type._shape_
            for i in range(rows):
                for j in range(cols):
                    code += f"    {var}.data[{i}][{j}] = {self.operands[0].cname}[{i * cols + j}];\n"
        return code

    def marshal_out(self, var: str) -> str:
        """C statements storing a Warp value through the output pointer.

        Output fragments are not preloaded from the destination tile, so the
        power-of-two padding lanes beyond the logical elements are zero-filled
        to keep the returned tile fully defined (e.g. for tile-wide reductions).
        """
        if self.kind == "scalar":
            return f"    {self.operands[0].cname}[0] = {var};\n"
        width = self.operands[0].width
        if self.kind == "struct":
            cpp_type = warp._src.codegen.Var.type_to_ctype(self.wp_type)
            # zero the whole tile slice first so every lane beyond
            # sizeof(struct) is defined; bytes inside sizeof follow the
            # returned object's representation (compiler padding included)
            return (
                f"    {{\n"
                f"        char* _wp_pd = (char*){self.operands[0].cname};\n"
                f"        const char* _wp_ps = (const char*)&{var};\n"
                f"        for (int _wp_i = 0; _wp_i < {width * 4}; ++_wp_i) _wp_pd[_wp_i] = 0;\n"
                f"        for (int _wp_i = 0; _wp_i < (int)sizeof({cpp_type}); ++_wp_i) _wp_pd[_wp_i] = _wp_ps[_wp_i];\n"
                f"    }}\n"
            )
        code = ""
        if self.kind == "vec":
            for i in range(self.logical):
                code += f"    {self.operands[0].cname}[{i}] = {var}[{i}];\n"
        else:
            rows, cols = self.wp_type._shape_
            for i in range(rows):
                for j in range(cols):
                    code += f"    {self.operands[0].cname}[{i * cols + j}] = {var}.data[{i}][{j}];\n"
        for i in range(self.logical, width):
            code += f"    {self.operands[0].cname}[{i}] = 0;\n"
        return code


def encode_struct(instance: object) -> np.ndarray:
    """Return a Warp struct instance's bytes as a NumPy ``int32`` array padded to a power-of-two length.

    .. admonition:: Experimental

        The cuTile interop API is experimental and subject to change without a formal
        deprecation cycle. See :func:`export_func` for details.

    The result has the layout a struct operand of :func:`export_func` expects:
    upload it as a ``wp.array`` and load it as an ``int32`` tile (as is for
    one struct shared by every invocation, tiled per invocation otherwise).

    Args:
        instance: An instance of a ``@wp.struct`` class.

    Returns:
        A 1-D NumPy ``int32`` array holding the struct's bytes followed by zero
        padding up to the next power-of-two word count.

    Raises:
        TypeError: If ``instance`` is not a Warp struct instance.

    Example:
        Upload the bytes as a ``wp.array`` and load them as an ``int32`` tile
        of the buffer's full length to share one struct across invocations:

        .. code:: python

            d_apply = wpct.export_func(apply_params)
            blob = wp.array(wpct.encode_struct(params), dtype=wp.int32)  # host
            WORDS = len(blob)


            @ct.kernel(num_worker_warps=TILE // 32)
            def kernel(blob, xs, out):
                p = ct.load(blob, (0,), shape=(WORDS,))  # one struct, shared by every invocation
                x = ct.load(xs, (0,), shape=(TILE,))
                ct.store(out, (0,), d_apply(p, x))
    """
    if not isinstance(instance, warp._src.codegen.StructInstance):
        raise TypeError(f"encode_struct expects a Warp struct instance, got {type(instance).__name__}")
    c = instance.__ctype__()
    raw = ctypes.string_at(ctypes.addressof(c), ctypes.sizeof(c))
    words = _next_pow2(max(1, (len(raw) + 3) // 4))
    buf = np.zeros(words * 4, dtype=np.uint8)
    buf[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return buf.view(np.int32)


def array_pointer_args(arr: wp.array) -> tuple:
    """Return the ``(address, size, ...)`` values that pass a ``wp.array`` to a foreign function.

    .. admonition:: Experimental

        The cuTile interop API is experimental and subject to change without a formal
        deprecation cycle. See :func:`export_func` for details.

    A ``wp.array`` parameter of :func:`export_func` takes a tuple of plain
    values at the call site: the array's data address followed by one
    extent per dimension. Pass the address into the cuTile kernel as a
    ``ct.Constant[int]`` parameter (runtime scalar parameters are 32-bit and
    cannot carry it); the extents may be constants or runtime parameters. Hand
    the tuple to the call.

    Args:
        arr: A contiguous, non-empty CUDA ``wp.array``.

    Returns:
        A tuple ``(address, *shape)`` of Python integers.

    Raises:
        TypeError: If ``arr`` is not a :class:`warp.array`.
        ValueError: If the array is not contiguous, not on a CUDA device, or empty.

    Example:
        Unpack the tuple into the kernel's arguments and pass the address and
        size on to the foreign call as one tuple:

        .. code:: python

            d_gather = wpct.export_func(gather_scale)


            @ct.kernel(num_worker_warps=TILE // 32)
            def kernel(ptr: ct.Constant[int], length: ct.Constant[int], idxs, out):
                i_t = ct.load(idxs, (0,), shape=(TILE,))
                ct.store(out, (0,), d_gather((ptr, length), i_t, 2.5))


            ct.launch(stream, (1, 1, 1), kernel, (*wpct.array_pointer_args(data), idxs, out))
    """
    if not isinstance(arr, wp.array):
        raise TypeError(f"array_pointer_args expects a wp.array, got {type(arr).__name__}")
    if not arr.is_contiguous:
        raise ValueError(f"Foreign array operands must be contiguous, got shape {arr.shape} with strides {arr.strides}")
    if not arr.device.is_cuda:
        raise ValueError(f"Foreign array operands must live on a CUDA device, got '{arr.device}'")
    if arr.ptr is None or arr.size == 0:
        raise ValueError("Foreign array operands must be non-empty")
    return (arr.ptr, *arr.shape)


def _codegen_warp_source(func, num_threads: int) -> str:
    """Generate self-contained CUDA C++ for a ``@wp.func`` without any kernel.

    Uses a synthetic module builder (same pattern as ``@wp.func_grad`` return
    type inference) so the function and its transitive dependencies are built
    and emitted standalone, with the module header baked for ``num_threads``.
    """
    module = warp._src.context.Module("cutile_foreign", None)
    # resolved, not raw, options: this is what wp.launch would compile the
    # function with, including extra_build_options preambles and include
    # directories (e.g. for wp.func_native dependencies) and global flags
    options = func.module.resolve_options(wp.config, block_dim=num_threads)
    options.update(
        {
            # No target architecture: MathDx tile builtins (tile_matmul,
            # tile_fft, tile_cholesky, ...) then take their portable non-MathDx
            # code paths, which need no separately linked device libraries.
            "output_arch": None,
            "enable_backward": False,
            # Deterministic mode injects hidden det_ctx parameters between the
            # regular arguments and multi-return refs; the foreign wrapper
            # cannot supply them and the host kernel is not Warp's anyway.
            "deterministic": wp.config.DeterministicMode.NOT_GUARANTEED,
        }
    )
    builder = warp._src.context.ModuleBuilder(module, options)
    builder.build_function(func)
    # as ModuleBuilder.build() does: custom-gradient callbacks (@wp.func_grad)
    # reached through wp.grad() are queued during the build and built after it
    for deferred in builder.deferred_functions:
        builder.build_function(deferred)
    # The flags that make codegen emit a function's adjoint live on the shared
    # Function objects and are left behind by whatever module built them last.
    # Recompute them for this build exactly as a module build does: no backward
    # kernels, so only the functions this build reaches through wp.grad() get
    # an adjoint. That keeps explicit derivatives correct and the generated
    # source, hence the cache digest, independent of what else the process
    # compiled. Afterwards restore the flags this pass changed.
    adjoints = [f.adj for f in builder.functions if getattr(f, "adj", None) is not None]
    saved = [(adj.used_by_backward_kernel, adj.force_adjoint_codegen) for adj in adjoints]
    try:
        builder._propagate_used_by_backward_kernel()
        return builder.codegen("cuda")
    finally:
        for adj, (used, forced) in zip(adjoints, saved, strict=True):
            adj.used_by_backward_kernel = used
            adj.force_adjoint_codegen = forced


def _cache_dir() -> str:
    """Return (and create) the cuTile foreign-function artifact directory under the Warp kernel cache.

    It holds each function's generated ``.cu``, its LTOIR, and the packed ``.tilelib``.
    """
    if wp.config.kernel_cache_dir is None:
        wp.init()
    path = os.path.join(wp.config.kernel_cache_dir, "cutile")
    os.makedirs(path, exist_ok=True)
    return path


@functools.cache
def _native_headers_fingerprint() -> str:
    """Hash of every ``.h``/``.inl`` under ``warp/native``; part of the LTOIR and tilelib cache identity.

    Cached per process: restart after editing headers in a long-lived session.
    """
    native_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "native")
    h = hashlib.sha256()
    for root, _dirs, files in sorted(os.walk(native_dir)):
        for fname in sorted(files):
            if fname.endswith((".h", ".inl")):
                path = os.path.join(root, fname)
                rel = os.path.relpath(path, native_dir).encode()
                with open(path, "rb") as f:
                    content = f.read()
                # length-framed so (name, content) boundaries stay unambiguous
                h.update(len(rel).to_bytes(8, "little"))
                h.update(rel)
                h.update(len(content).to_bytes(8, "little"))
                h.update(content)
    return h.hexdigest()[:8]


@functools.cache
def _compiler_runtime_fingerprint() -> str:
    """Identity of the embedded NVRTC and the native-library build.

    Both change the generated LTOIR without any Python-visible source change
    (the toolkit NVRTC was built against and the MathDx build setting), so
    both must perturb the cache identity. Cached per process.
    """
    warp._src.context.init()
    runtime = warp._src.context.runtime
    nvrtc_id = f"nvrtc-embedded-{runtime.toolkit_version}"
    mathdx = bool(runtime.core.wp_is_mathdx_enabled())
    return hashlib.sha256(f"{nvrtc_id}|mathdx={mathdx}".encode()).hexdigest()[:8]


_LTOIR_COMPILE_OPTIONS = "c++17|rdc|dlto|no-pch"
"""Fixed part of the LTOIR compile configuration; the per-function part comes from :func:`_ltoir_build_options`."""

_LTOIR_BUILD_OPTION_KEYS = (
    "config",
    "optimization_level",
    "verify_fp",
    "fast_math",
    "fuse_fp",
    "lineinfo",
    "extra_include_dirs",
    "build_dependencies",
)
"""Module options that shape the LTOIR compile, in digest order.

``build_dependencies`` is identity only: the resolved ``extra_build_dependencies``
``(path, content hash)`` pairs, so editing a registered header rebuilds the
foreign code the way it rebuilds the Warp module.
"""


def _ltoir_build_options(func: warp._src.context.Function) -> dict[str, object]:
    """Return the NVRTC build options the function's Warp module would compile with.

    A foreign function should compute what the same ``@wp.func`` computes
    under ``wp.launch``, so the module's ``mode``, ``optimization_level``,
    ``fast_math``, ``fuse_fp``, and ``lineinfo`` options are forwarded to
    NVRTC (``None`` entries inherit from ``warp.config`` like Warp does), as
    is the global ``warp.config.verify_fp`` (a global in Warp proper as
    well). ``fuse_fp=False`` is forwarded but cannot be guaranteed:
    floating-point contraction is decided when tileiras link-time-optimizes
    the kernel, outside Warp's control, so a warning is emitted.
    """
    opts = func.module.resolve_options(wp.config)
    if opts.get("fuse_fp", True) is False:
        log_warning(
            f"Module option fuse_fp=False is not honored by cuTile foreign functions ('{func.key}'): "
            "floating-point contraction is decided when tileiras links the kernel",
            once=True,
        )
    level = opts.get("optimization_level")
    if level is None:
        level = 3  # Warp's CUDA default
    return {
        "config": opts.get("mode") or wp.config.mode,
        "optimization_level": level,
        "verify_fp": opts.get("verify_fp", wp.config.verify_fp),
        "fast_math": opts.get("fast_math", False),
        "fuse_fp": opts.get("fuse_fp", True),
        "lineinfo": opts.get("lineinfo", wp.config.lineinfo),
        "extra_include_dirs": tuple(opts.get("extra_cuda_include_dirs", ())),
        "build_dependencies": tuple(opts.get("extra_build_dependencies", ())),
    }


def _build_options_key(build_options: dict | None) -> str:
    """Canonical digest field for a build-option dict (``None`` means Warp defaults)."""
    if build_options is None:
        build_options = {}
    return "|".join(f"{k}={build_options.get(k)}" for k in _LTOIR_BUILD_OPTION_KEYS)


def _compile_ltoir_warp(source: str, out_path: str, arch: str, build_options: dict | None = None) -> None:
    """Compile CUDA C++ to raw LTOIR with Warp's embedded NVRTC.

    Uses the ``.ltoir`` output mode of ``wp_cuda_compile_program`` (extension
    driven, like ``.ptx``), so no external compiler package is needed. The
    generated source is left at ``<out_path>.cu`` for inspection, like the
    ``.cu`` files Warp keeps beside its kernel modules. ``build_options`` (see :func:`_ltoir_build_options`) selects the same
    mode, optimization level, and floating-point flags Warp would use for the
    function's own module; ``None`` means Warp's defaults.
    """
    if wp.config.kernel_cache_dir is None:
        wp.init()
    build_options = build_options or {}
    # split "sm_121a" into the numeric arch and its feature suffix so
    # arch-suffixed LTOIR targets the same variant the tilelib declares
    arch_digits = arch.split("_", 1)[1]
    arch_num = int(arch_digits.rstrip("af"))
    arch_suffix = arch_digits[len(str(arch_num)) :]
    cu_path = f"{out_path}.cu"
    with open(cu_path, "w") as f:
        f.write(source)
    warp._src.build.build_cuda(
        cu_path,
        arch_num,
        out_path,
        pch_dir=None,
        config=build_options.get("config", "release"),
        optimization_level=build_options.get("optimization_level", 3),
        verify_fp=build_options.get("verify_fp", False),
        fast_math=build_options.get("fast_math", False),
        fuse_fp=build_options.get("fuse_fp", True),
        lineinfo=build_options.get("lineinfo", False),
        arch_suffix=arch_suffix,
        use_precompiled_headers=False,
        extra_include_dirs=list(build_options.get("extra_include_dirs", ())),
    )
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Warp NVRTC failed to produce LTOIR for '{out_path}'")


def _publish_artifact(tmp_path: str, path: str) -> None:
    """Move a staged artifact into place atomically.

    On Windows, ``os.replace`` fails while another process holds the target
    open. The target is named by content, so a concurrent writer's copy is
    equivalent: keep it and drop ours.
    """
    try:
        os.replace(tmp_path, path)
    except PermissionError:
        if not os.path.isfile(path):
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
            raise
        with contextlib.suppress(OSError):
            os.remove(tmp_path)


def _compile_ltoir(source: str, symbol: str, arch: str, build_options: dict | None = None) -> str:
    """Compile CUDA C++ to a raw LTOIR file, cached by symbol, source digest, and build options.

    Raw LTOIR (not a relocatable object) keeps the implementation visible to
    tileiras and nvJitLink so calls can be inlined and ABI mismatches surface
    at link time. The source digest in the cache name keeps a caller-supplied
    ``symbol`` override from reusing stale device code when the function body
    (or an included native header) changes. The generated ``.cu`` is kept
    beside the ``.ltoir``.
    """
    src_digest = _digest_fields(
        _LTOIR_COMPILE_OPTIONS,
        _build_options_key(build_options),
        _native_headers_fingerprint(),
        _compiler_runtime_fingerprint(),
        source,
    )[:16]
    path = os.path.join(_cache_dir(), f"{symbol}.{src_digest}.{arch}.ltoir")
    if os.path.isfile(path):
        return path

    # Staging path must keep the .ltoir suffix (the native compiler selects
    # its output format from the file extension) and be unique per build so
    # concurrent threads in one process cannot clobber each other's staging.
    tmp_path = f"{path}.tmp{os.getpid()}_{uuid.uuid4().hex[:8]}.ltoir"
    _compile_ltoir_warp(source, tmp_path, arch, build_options)
    cu_tmp = f"{tmp_path}.cu"
    if os.path.exists(cu_tmp):
        _publish_artifact(cu_tmp, f"{path[: -len('.ltoir')]}.cu")
    _publish_artifact(tmp_path, path)
    return path


_WRAPPER_SYMBOL_PLACEHOLDER = "__WP_FOREIGN_SYMBOL__"
"""Sentinel symbol used to emit wrapper text before the digest-derived name exists."""

_DECL_NAME_PLACEHOLDER = "__WP_FOREIGN_DECL__"
"""Sentinel declaration name used to emit the bytecode contract before the digest-derived name exists."""


def _digest_fields(*parts) -> str:
    """Hash heterogeneous fields with length framing.

    Length-prefixing each field makes the digest immune to separator
    injection (free-form source text shifting content across field
    boundaries), and ``None`` is encoded distinctly from the string "None".
    """
    h = hashlib.sha256()
    for part in parts:
        data = b"\x00" if part is None else str(part).encode()
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def _tilelib_digest(
    *,
    compiler_runtime: str,
    toolchain_fingerprint: str,
    headers_fingerprint: str,
    source: str,
    wrapper: str,
    interface: bytes,
    impl: bytes,
    arch: str,
    lengths: Collection[int],
    scope: str,
    smem_bytes: int,
    build_options: Mapping | None,
) -> str:
    """Content digest doubling as the tilelib ``Tilelib`` version.

    This is cuTile Python's cache contract: kernels are cached on the tilelib's
    (path, version) alone, so every input that shapes the packed tilelib —
    the generated source, wrapper, and bytecode contract modules (emitted
    against placeholder names), the native headers, the LTOIR compile
    options and compiler runtime, the target architecture, and the tileiras
    identity — must perturb this digest.
    """
    return _digest_fields(
        "tilelib",
        _TILELIB_SCHEMA_VERSION,
        _LTOIR_COMPILE_OPTIONS,
        _build_options_key(dict(build_options) if build_options else None),
        compiler_runtime,
        toolchain_fingerprint,
        headers_fingerprint,
        source,
        wrapper,
        interface.hex(),
        impl.hex(),
        arch,
        list(lengths),
        scope,
        smem_bytes,
    )[:16]


def _check_operand_names(operands: Sequence[_Operand], func_key: str) -> None:
    """Reject operand-name collisions in ``func_key``'s declaration before any toolchain work.

    Return operands are named ``ret{i}`` and array parameters expand into
    ``{name}_ptr``/``{name}_shape{d}`` companions, so a user parameter with one
    of those names would otherwise produce invalid C and declarations far downstream.
    The placeholder tokens are reserved too, since they are substituted
    textually in the emitted wrapper and bytecode. (Wrapper locals cannot
    collide: every operand is emitted under the ``_wp_p_`` prefix.)
    """
    seen = set()
    for op in operands:
        if op.name in seen:
            raise ValueError(
                f"'{func_key}': operand name collision on '{op.name}'; parameter names may not clash with "
                "generated operand names (ret0..retN, <array>_ptr, <array>_shape0..N)"
            )
        if _WRAPPER_SYMBOL_PLACEHOLDER in op.name or _DECL_NAME_PLACEHOLDER in op.name:
            raise ValueError(
                f"'{func_key}': parameter '{op.name}' contains a token reserved by the foreign-function wrapper; rename it"
            )
        seen.add(op.name)


@dataclasses.dataclass(frozen=True)
class _ScratchLayout:
    """Layout of the compiler-provided scratch a foreign call receives, and its block-size bounds.

    The scratch is ``[BVH query stack slab | mesh query stack slab | offset
    slots | tile storage]``: a traversal slab per query family the function
    uses (``None`` offsets mark unused families), then, when the function uses
    Warp tile storage (``smem_bytes > 0``), one offset slot per worker thread
    followed by the storage itself. ``slab_threads`` is the block size the
    slabs were sized for; the wrapper traps when the block exceeds it. A
    default-constructed layout means the call receives no scratch.
    """

    bvh_offset: int | None = None
    mesh_offset: int | None = None
    stack_bytes: int = 0
    smem_bytes: int = 0
    slab_threads: int | None = None

    @classmethod
    def plan(cls, uses_bvh: bool, uses_mesh: bool, smem_bytes: int, num_threads: int, scope: str) -> _ScratchLayout:
        """Lay out the scratch for a declaration.

        Traversal slabs are sized for ``max(256, num_threads)`` worker threads
        at thread scope and for exactly ``num_threads`` at block scope.
        """
        slab_threads = num_threads if scope == "block" else max(_DEFAULT_SLAB_THREADS, num_threads)
        slab_bytes = _BVH_QUERY_STACK_SIZE * 4 * slab_threads
        uses_queries = uses_bvh or uses_mesh
        return cls(
            bvh_offset=0 if uses_bvh else None,
            mesh_offset=(slab_bytes if uses_bvh else 0) if uses_mesh else None,
            stack_bytes=slab_bytes * (int(uses_bvh) + int(uses_mesh)),
            smem_bytes=smem_bytes,
            slab_threads=slab_threads if uses_queries else None,
        )

    @property
    def uses_queries(self) -> bool:
        """Whether any traversal-stack slab is laid out."""
        return self.stack_bytes > 0

    @property
    def used(self) -> bool:
        """Whether the call receives a scratch pointer at all."""
        return self.stack_bytes > 0 or self.smem_bytes > 0

    def total_bytes(self, length: int) -> int:
        """Scratch bytes requested for a declaration serving ``length`` worker threads."""
        return self.stack_bytes + (length * 4 + self.smem_bytes if self.smem_bytes > 0 else 0)

    @staticmethod
    def _slab_expr(offset: int | None) -> str:
        """C expression for a slab at ``offset`` bytes into the scratch (``None`` = unused)."""
        if offset is None:
            return "nullptr"
        return "scratch_smem" if offset == 0 else f"(char*)scratch_smem + {offset}"

    def wrapper_prologue(self) -> str:
        """C statements that check the block size and bind the scratch regions before the function body."""
        body = ""
        block = "(int)(blockDim.x * blockDim.y * blockDim.z)"
        if self.slab_threads is not None:
            # a block larger than the slabs would stride past them: fail loudly
            body += f"    if ({block} > {self.slab_threads}) __trap();\n"
        if self.stack_bytes:
            body += (
                f"    wp::bvh_bind_scratch_stacks({self._slab_expr(self.bvh_offset)}, "
                f"{self._slab_expr(self.mesh_offset)});\n"
            )
        if self.smem_bytes > 0:
            base = f"(char*)scratch_smem + {self.stack_bytes}" if self.stack_bytes else "scratch_smem"
            body += f"    wp::tile_shared_storage_t::bind_external_smem({base});\n"
        return body


def _emit_wrapper(
    symbol: str,
    func: warp._src.context.Function,
    args: Sequence[_ArgSpec],
    rets: Sequence[_ArgSpec],
    operands: Sequence[_Operand],
    layout: _ScratchLayout,
) -> str:
    """Emit the extern "C" destination-passing wrapper for a foreign function.

    ``symbol`` may be ``_WRAPPER_SYMBOL_PLACEHOLDER`` so the wrapper text can
    be folded into the cache digest before the digest-derived symbol name
    exists; substitute the real symbol into the returned text afterwards.
    When ``layout`` lays out any scratch, the compiler-owned scratch arrives
    as one trailing pointer after all fragment parameters and is bound before
    the call.
    """
    wrapper_params = ", ".join(op.wrapper_param() for op in operands)
    body = ""
    if layout.used:
        wrapper_params += ", void* scratch_smem"
        body += layout.wrapper_prologue()
    call_args = []
    for i, a in enumerate(args):
        var = f"_wp_arg{i}"
        body += a.marshal_in(var)
        call_args.append(var)
    body += f"    auto _wp_ret0 = {func.native_func}({', '.join(call_args)});\n"
    body += rets[0].marshal_out("_wp_ret0")
    if layout.smem_bytes > 0:
        # exit barrier: no thread may leave while another is still touching
        # the compiler-owned scratch this call bound to Warp's tile storage
        # (per-thread query slabs need none, and thread scope must not sync)
        body += "    __syncthreads();\n"
    return f'\nextern "C" __device__ void {symbol}({wrapper_params})\n{{\n{body}}}\n'


@dataclasses.dataclass(frozen=True)
class _ForeignPlan:
    """Validated ABI plan for one foreign-function declaration."""

    source: str
    layout: _ScratchLayout
    args: list[_ArgSpec]
    in_ops: list[_Operand]
    out_ops: list[_Operand]
    wrapper_template: str
    lengths: tuple[int, ...]
    fallback_length: int
    return_type: type
    """Warp type of the single return value, captured under the codegen lock."""


def _validate_declaration(func, num_worker_warps: int, scope: str) -> None:
    """Check a declaration's arguments before any codegen, toolchain, or GPU work."""
    if not isinstance(func, warp._src.context.Function):
        raise TypeError(f"export_func expects a @wp.func, got {type(func).__name__}")
    if func.is_builtin():
        raise TypeError(
            f"export_func expects a user-defined @wp.func; '{func.key}' is a Warp builtin. "
            "Wrap the call in a @wp.func with concrete parameter types."
        )
    if len(func.user_overloads) > 1:
        raise NotImplementedError(
            f"'{func.key}' has {len(func.user_overloads)} overloads; foreign functions must have "
            "a single concrete signature (only the first-registered overload could be packaged)"
        )
    if isinstance(num_worker_warps, bool) or not isinstance(num_worker_warps, int):
        raise TypeError(f"num_worker_warps must be an int, got {type(num_worker_warps).__name__}")
    if num_worker_warps < 1 or num_worker_warps > 32 or (num_worker_warps & (num_worker_warps - 1)) != 0:
        raise ValueError(
            f"num_worker_warps must be a power of two between 1 and 32 "
            f"(Tile IR tile dimensions are powers of two), got {num_worker_warps}"
        )
    if scope not in ("thread", "block"):
        raise ValueError(f"Unknown scope '{scope}' (expected 'thread' or 'block')")
    # the key names the generated C++ function, the extern symbol, and the
    # Tile IR declaration, so it and the parameter names must be plain ASCII
    # identifiers (\w would admit Unicode word characters that NVRTC and
    # Tile IR reject)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", func.key):
        raise ValueError(
            f"Function key '{func.key}' is not a valid identifier; pass an identifier name to @wp.func(name=...)"
        )
    if func.key.startswith("_wp_"):
        raise ValueError(f"Function key '{func.key}' uses the reserved prefix '_wp_'; pass another @wp.func(name=...)")
    for placeholder in (_WRAPPER_SYMBOL_PLACEHOLDER, _DECL_NAME_PLACEHOLDER):
        if placeholder in func.key:
            raise ValueError(f"Function name may not contain the reserved token '{placeholder}'")
    for arg in func.adj.args:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", arg.label):
            raise ValueError(f"Parameter name '{arg.label}' of '{func.key}' is not a plain ASCII identifier")
        if arg.label.startswith("_wp_"):
            # the generated call body keeps its own names under this prefix
            raise ValueError(f"Parameter name '{arg.label}' of '{func.key}' uses the reserved prefix '_wp_'")
    if func.defaults:
        raise ValueError(
            f"'{func.key}' has default values for {sorted(func.defaults)}; foreign functions take every "
            "argument explicitly at the call site"
        )


def _plan_foreign(ct: object, func: warp._src.context.Function, num_worker_warps: int, scope: str) -> _ForeignPlan:
    """Validate a foreign declaration and build its ABI plan.

    Everything that needs no toolchain or GPU happens here: argument
    validation, operand expansion, the choice of declared tile lengths,
    standalone codegen, shared-memory/scope legality, the scratch layout,
    and the placeholder-symbol wrapper.
    """
    _validate_declaration(func, num_worker_warps, scope)
    num_threads = num_worker_warps * 32  # the block size the declaration is built for
    try:
        args = [_ArgSpec(ct, arg.label, arg.type) for arg in func.adj.args]
    except TypeError as e:
        raise TypeError(f"'{func.key}': {e}") from None

    # Which tile lengths does this declaration cover? Thread-scope code is
    # per-invocation and length-agnostic, so it is declared for every length
    # and generated once at the canonical block size (the hint does not key
    # it). Block scope pins the set to the single num_threads hint (the block
    # size is baked into the cooperative code). The hint is always among the
    # declared lengths, so it serves as the length of a call whose arguments
    # are all shared values (no tile carries a length).
    universal = scope == "thread"
    lengths = _UNIVERSAL_LENGTHS if universal else (num_threads,)
    # Codegen and the metadata read-back hold Warp's codegen lock as a module
    # build does: a concurrent build of the same function resets its shared
    # ``adj.return_var`` while it runs.
    with warp._src.codegen._codegen_lock:
        source = _codegen_warp_source(func, _CANONICAL_BLOCK_DIM if universal else num_threads)
        # the return type and shared-memory need are only known once the
        # function has been built, and belong to this build's codegen
        ret_vars = tuple(func.adj.return_var or ())
        smem_bytes = func.adj.get_total_required_shared()
    if not ret_vars:
        raise TypeError(f"'{func.key}' has no return value; foreign functions must return exactly one value")
    if len(ret_vars) != 1:
        raise NotImplementedError(f"'{func.key}' returns {len(ret_vars)} values; foreign functions return exactly one")
    try:
        rets = [_ArgSpec(ct, f"ret{i}", v.type, is_output=True) for i, v in enumerate(ret_vars)]
    except TypeError as e:
        raise TypeError(f"'{func.key}': {e}") from None
    in_ops = [op for spec in args for op in spec.operands]
    out_ops = [op for spec in rets for op in spec.operands]
    _check_operand_names(in_ops + out_ops, func.key)

    # Shared memory required by the function (transitive, forward pass only)
    # is served from compiler-owned scratch instead of Warp's own static /
    # dynamic shared memory, which a Tile IR kernel never allocates.
    if smem_bytes > 0 and scope != "block":
        raise ValueError(
            f"'{func.key}' requires {smem_bytes} bytes of block-shared memory "
            '(e.g. it uses Warp tile builtins); call export_func with scope="block"'
        )
    if scope != "block" and _uses_tiled_queries(source):
        raise ValueError(
            f"'{func.key}' uses a block-cooperative tiled query builtin (wp.tile_mesh_query_* / "
            'wp.tile_bvh_query_*), which needs every thread of the block; call export_func with scope="block"'
        )
    layout = _ScratchLayout.plan(
        _uses_bvh_stack_queries(source), _uses_mesh_stack_queries(source), smem_bytes, num_threads, scope
    )
    # The defines selecting scratch-backed traversal stacks and external tile
    # storage are part of the source, hence of the digest: cuTile caches
    # kernels on our version alone, so undigested differences would alias.
    if layout.uses_queries:
        source = "#define BVH_SCRATCH_STACK 1\n" + source
    if layout.smem_bytes > 0:
        source = "#define WP_TILE_EXTERNAL_SHARED_STORAGE\n" + source

    return _ForeignPlan(
        source=source,
        layout=layout,
        args=args,
        in_ops=in_ops,
        out_ops=out_ops,
        wrapper_template=_emit_wrapper(_WRAPPER_SYMBOL_PLACEHOLDER, func, args, rets, in_ops + out_ops, layout),
        lengths=lengths,
        fallback_length=num_threads,
        return_type=ret_vars[0].type,
    )


def _check_scratch_budget(key: str, scratch_bytes: int, device) -> None:
    """Raise if the compiler scratch a declaration requests cannot fit a block on ``device``.

    Traversal slabs grow with the declared ``num_threads`` (128 bytes per
    thread per query family), so a large block using both query families can
    exceed what the GPU allows per block; report that at declaration time with
    the numbers instead of as a launch failure.
    """
    if scratch_bytes <= 0:
        return
    limit = device.max_shared_memory_per_block
    if scratch_bytes > limit:
        raise ValueError(
            f"'{key}' needs {scratch_bytes} bytes of compiler scratch (traversal stacks and tile storage) "
            f"but '{device}' allows at most {limit} bytes of shared memory per block; "
            "declare with a smaller num_worker_warps or split the query families across functions"
        )


def _uses_mesh_stack_queries(source: str) -> bool:
    """Return True if generated source constructs a mesh AABB/sphere iterator query (mesh stack slab).

    The mesh AABB/sphere iterator queries and the BVH queries stripe one
    traversal stack per thread across a shared-memory slab. Warp kernels size
    that slab statically from the compile-time block dim; a foreign function
    does not know the calling kernel's block size, so it requests the slab as
    compiler-provided scratch instead (``BVH_SCRATCH_STACK``), strided by the
    runtime block size. Mesh point and ray queries use thread-local stacks.
    """
    # match the query constructors only: the tiled builtins spell their types
    # mesh_query_*_thread_block_t but keep their own block-sized static buffers
    return re.search(r"\bmesh_query_(?:aabb|sphere)\s*\(", source) is not None


def _uses_bvh_stack_queries(source: str) -> bool:
    """Return True if generated source constructs a BVH query of any kind (BVH stack slab)."""
    return re.search(r"\bbvh_query_(?:aabb|ray|sphere|capsule)\s*\(", source) is not None


def _uses_tiled_queries(source: str) -> bool:
    """Return True if generated source calls a block-cooperative tiled query builtin.

    ``wp.tile_mesh_query_*`` and ``wp.tile_bvh_query_*`` traverse with the
    whole block (barriers, block-wide result buffers in static shared memory
    sized from the compile-time block dim), so they are only sound when the
    foreign function owns the block, i.e. at ``scope="block"`` where the block
    size is pinned to ``num_threads``.
    """
    return re.search(r"\bwp::tile_(?:mesh|bvh)_query_\w*\s*\(", source) is not None


_REQUIRED_TILEIRAS_RELEASE = cutile_bytecode.BYTECODE_VERSION
"""The one ``tileiras`` release (CUDA Toolkit major.minor) that reads the bytecode this bridge emits.

Tile IR documents SIMT interoperability bytecode as a preview that is not
compatible with any other toolkit release.
"""

_REQUIRED_TILEIRAS_TEXT = ".".join(map(str, _REQUIRED_TILEIRAS_RELEASE))
"""The required release as users see it in error messages."""


@functools.cache
def _tileiras_version_banner(tileiras: str) -> str | None:
    """Return the stripped ``--version`` banner of a ``tileiras`` binary, or ``None`` if it cannot be read.

    Runs the binary once per resolved path for the process lifetime; an
    in-place swap at the same path requires a restart to be noticed.
    """
    try:
        return subprocess.run(
            [tileiras, "--version"], check=True, capture_output=True, text=True, errors="replace", timeout=60
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


def _tileiras_release(tileiras: str) -> tuple[int, int] | None:
    """Return the ``(major, minor)`` CUDA release of a ``tileiras`` binary, or ``None`` if unknown.

    Parsed from the ``--version`` banner (``Cuda compilation tools, release
    <major>.<minor>, ...``). ``None`` means the banner could not be read or
    parsed; callers decide whether that is fatal.
    """
    banner = _tileiras_version_banner(tileiras)
    if banner is None:
        return None
    match = re.search(r"release\s+(\d+)\.(\d+)", banner)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _check_tileiras_release(tileiras: str) -> None:
    """Raise ``RuntimeError`` unless ``tileiras`` is the release this bridge requires.

    Another toolkit's ``tileiras`` is found just as readily on ``PATH`` but
    is documented as incompatible with the preview bytecode this bridge
    emits, so the mismatch is reported up front instead of as a packing or
    link error. A binary whose release cannot be read is refused for the
    same reason.
    """
    release = _tileiras_release(tileiras)
    if release is None:
        raise RuntimeError(
            f"Could not determine the CUDA Toolkit release of '{tileiras}' from `tileiras --version`; "
            f"the cuTile bridge requires the {_REQUIRED_TILEIRAS_TEXT} tileiras (install the warp-lang[cutile] pip extra)"
        )
    if release != _REQUIRED_TILEIRAS_RELEASE:
        have = ".".join(map(str, release))
        raise RuntimeError(
            f"'{tileiras}' is the CUDA Toolkit {have} tileiras; the cuTile bridge requires the "
            f"{_REQUIRED_TILEIRAS_TEXT} tileiras (install the warp-lang[cutile] pip extra)"
        )


def _toolchain_fingerprint(tileiras: str) -> str:
    """Identity of the resolved ``tileiras``, for tilelib cache identity.

    Combines the resolved path with the tool's version banner, so switching
    or upgrading the packer (preview bytecode is exact-version-locked)
    repacks tilelibs instead of reusing stale ones. Callers must pass the same
    resolved path they hand to :func:`_pack_tilelib`.
    """
    banner = _tileiras_version_banner(tileiras) or "unknown"
    return hashlib.sha256(f"{tileiras}|{banner}".encode()).hexdigest()[:8]


def _pip_tileiras_dir() -> str | None:
    """Return the ``bin`` directories of the ``nvidia.cu13`` pip packages as a ``PATH``-style list, if any.

    ``pip install "warp-lang[cutile]"`` places ``tileiras`` under the
    ``nvidia.cu13`` namespace package, which cuTile Python itself searches first; the
    bridge mirrors that so a pip-only environment needs no configuration. The
    namespace can span several site directories (a ``uv run --with`` overlay,
    a user site next to a system site), and the wheel holding ``bin/tileiras``
    is not necessarily the first, so every entry that has a ``bin`` is kept.
    """
    try:
        import nvidia.cu13 as cu13  # noqa: PLC0415
    except ImportError:
        return None
    bins = [os.path.join(p, "bin") for p in getattr(cu13, "__path__", [])]
    bins = [b for b in bins if os.path.isdir(b)]
    return os.pathsep.join(bins) if bins else None


def _find_tileiras() -> str:
    """Locate ``tileiras`` for tilelib packing.

    Looks in ``$WARP_TILEIR_TOOLCHAIN`` first (a directory, a ``PATH``-style
    list, or the binary itself), then in the ``nvidia-cuda-tileiras`` pip
    package, then on ``PATH``. A configured toolchain that resolves nothing
    raises instead of falling back, so a typo never silently swaps toolchains.
    """

    name = "tileiras"
    toolchain = os.environ.get("WARP_TILEIR_TOOLCHAIN")
    if toolchain:
        # absolute paths throughout: a bare relative name would make
        # subprocess search PATH instead of running the configured binary
        if os.path.isfile(toolchain) and os.access(toolchain, os.X_OK):
            return os.path.abspath(toolchain)  # the binary itself
        # PATHEXT-aware lookup so Windows toolchains (tileiras.exe) resolve too
        found = shutil.which(name, path=toolchain)
        if found:
            return os.path.abspath(found)
        for entry in toolchain.split(os.pathsep):
            path = os.path.join(entry, name)
            if os.path.exists(path):
                raise RuntimeError(f"'{path}' from WARP_TILEIR_TOOLCHAIN exists but is not an executable file")
        # never silently fall through to another toolchain when the user
        # configured one: a typo would swap toolchains behind their back
        raise RuntimeError(
            f"WARP_TILEIR_TOOLCHAIN='{toolchain}' does not contain a tileiras executable; set it to the "
            f"directory holding the CUDA Toolkit {_REQUIRED_TILEIRAS_TEXT} tileiras (or to the binary itself), or unset it"
        )
    pip_dir = _pip_tileiras_dir()
    if pip_dir:
        found = shutil.which(name, path=pip_dir)
        if found:
            return found
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(
        "'tileiras' not found; install the warp-lang[cutile] pip extra or add the CUDA Toolkit "
        f"{_REQUIRED_TILEIRAS_TEXT} bin directory to PATH"
    )


_SIMPLE_TYPES = {
    "float": cutile_bytecode.SimpleType.F32,
    "double": cutile_bytecode.SimpleType.F64,
    "int": cutile_bytecode.SimpleType.I32,
    "unsigned int": cutile_bytecode.SimpleType.I32,
    "long long": cutile_bytecode.SimpleType.I64,
    "unsigned long long": cutile_bytecode.SimpleType.I64,  # Tile IR tiles have no unsigned dtypes; 64 raw bits
    "short": cutile_bytecode.SimpleType.I16,
    "unsigned short": cutile_bytecode.SimpleType.I16,
    "signed char": cutile_bytecode.SimpleType.I8,
    "unsigned char": cutile_bytecode.SimpleType.I8,
}
"""Wrapper C type of an operand -> Tile IR element type of its tile."""


def _op_fragment_layout(op: _Operand, length: int, scope: str) -> tuple[tuple[tuple, tuple], tuple[tuple, tuple]]:
    """Return the ``(mem_layout, fragmap)`` of one operand as ``(shapes, strides)`` pairs.

    The fragment map's modes are ``(invocations, participating threads,
    elements)``: thread scope runs ``length`` independent invocations of one
    thread each; block scope runs one invocation in which ``length`` threads
    participate. By-value scalars use the rank-0 memory layout. These tuples
    ARE the tile ABI: changing them is an ABI change (bump
    ``_TILELIB_SCHEMA_VERSION``).
    """
    w = op.width
    if scope == "block":
        shapes = (1, length, w)
        strides = (0, w, 1)
        scalar_strides = (0, 1, 0)
    else:
        shapes = (length, 1, w)
        strides = (w, 0, 1)
        scalar_strides = (1, 0, 1)
    if op.by_value:
        return ((), ()), ((shapes[0], shapes[1], 1), scalar_strides)
    return ((w,), (1,)), (shapes, strides)


def _emit_tilelib_modules(
    decl_base: str,
    symbol_base: str,
    in_ops: Sequence[_Operand],
    out_ops: Sequence[_Operand],
    lengths: Collection[int],
    scope: str,
    layout: _ScratchLayout,
    arch: str,
) -> tuple[bytes, bytes]:
    """Emit the tilelib's interface and implementation bytecode modules.

    One extern declaration (``{decl_base}_n{length}``) and one
    ``extern_impl`` (``{symbol_base}_n{length}``) per declared length, all in
    a single module pair. Writable operands trail the inputs and are tied
    one-to-one to the declared results (the form this tileiras build
    resolves): callers pass a destination tile per output and receive the
    value as an SSA result. The implementation is constrained to the LTOIR's
    architecture so a wrong-arch tilelib fails at resolution instead of deep
    inside the device link; block scope declares block synchronization; scratch
    (``[BVH query stack slab | mesh query stack slab | offset slots | tile storage]``)
    comes with the ABI map that only concerns the scratch pointer's address space.
    """
    ops = in_ops + out_ops
    access = ["in"] * len(in_ops) + ["out"] * len(out_ops)
    iface = cutile_bytecode.Module()
    impl = cutile_bytecode.Module()

    def tile_type(op, length):
        """Tile type carrying this operand across the call boundary."""
        return iface.tile(iface.simple(_SIMPLE_TYPES[op.ctype]), [op.tile_shape(length)[0]])

    for length in lengths:
        iface.declare(
            f"{decl_base}_n{length}",
            [tile_type(op, length) for op in ops],
            [tile_type(op, length) for op in out_ops],
            access,
        )
        fragments = []
        for op in ops:
            mem_layout, fragmap = _op_fragment_layout(op, length, scope)
            fragments.append(
                impl.fragment(
                    impl.simple(_SIMPLE_TYPES[op.ctype]),
                    [] if op.by_value else [op.width],
                    mem_layout,
                    fragmap,
                    op.alignment,
                )
            )
        scratch_total = layout.total_bytes(length)
        impl.implement(
            f"{symbol_base}_n{length}",
            f"{decl_base}_n{length}",
            fragments,
            access,
            arch=[arch.replace("_", "")],
            synchronization="cta" if scope == "block" else None,
            scratch=(scratch_total, 16) if scratch_total > 0 else None,
            frontend_abi={"!llvm.ptr<3>": "!llvm.ptr"} if scratch_total > 0 else None,
        )
    return iface.to_bytes(), impl.to_bytes()


def _pack_tilelib(iface: bytes, impl: bytes, ltoir_path: str, out_path: str, tileiras: str) -> None:
    """Pack the interface and implementation bytecode with the LTOIR into ``out_path``.

    ``tileiras`` is the caller's single resolution of the packer — the same
    path folded into the cache digest, so the digest and the packed artifact
    can never describe different toolchain invocations.
    """

    base = out_path[: -len(".tilelib")]
    # Per-build token: staging and intermediate names must be unique so
    # concurrent packers (threads share a PID) never write the same files.
    token = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    paths = {}
    for kind, data in (("interface", iface), ("impl", impl)):
        paths[kind] = f"{base}.{kind}.{token}.tilebc"
        with open(paths[kind], "wb") as f:
            f.write(data)

    # stage and atomically publish: a crash or a concurrent packer must never
    # leave a partial tilelib at the trusted cache path
    tmp_path = f"{out_path}.tmp{token}.tilelib"
    cmd = [
        tileiras,
        "--preview-create-tile-lib",
        f"--preview-interface={paths['interface']}",
        f"--preview-implementation={paths['impl']}",
        f"--preview-implementation={ltoir_path}",
        "-o",
        tmp_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, errors="replace", timeout=600)
    except subprocess.CalledProcessError as e:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise RuntimeError(f"tileiras failed (exit {e.returncode}):\n{e.stdout}{e.stderr}") from e
    except subprocess.TimeoutExpired as e:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise RuntimeError(f"tileiras timed out after {e.timeout} s: {' '.join(cmd)}") from e
    if not os.path.isfile(tmp_path):
        raise RuntimeError(f"tileiras exited successfully but produced no tilelib for '{out_path}'")
    _publish_artifact(tmp_path, out_path)

    # intermediates are kept on failure for debugging, removed on success
    for path in paths.values():
        with contextlib.suppress(OSError):
            os.remove(path)


def _type_name(spec: _ArgSpec) -> str:
    """Short Warp type name of a parameter for messages (``float32``, ``vec3f``, ``array``, a struct's key)."""
    t = spec.wp_type
    if warp._src.types.is_array(t):
        return "array"
    if isinstance(t, warp._src.codegen.Struct):
        return t.key
    if t is warp._src.types.handle:
        return "handle"  # type_repr() has no case for the id alias and would print its module path
    return warp._src.types.type_repr(t)


def _describe_param(spec: _ArgSpec) -> str:
    """One ``name: type form`` entry of a declaration's call signature for its repr and docstring."""
    tname = _type_name(spec)
    op = spec.operands[0]
    if spec.kind == "array":
        return f"{spec.name}: wp.{tname} (address, sizes)"
    if op.width == 1:
        return f"{spec.name}: wp.{tname} tile (TILE,) or shared value"
    return f"{spec.name}: wp.{tname} tile (TILE * {op.width},) or shared ({op.width},)"


def _int_bounds(dtype) -> tuple[int, int]:
    """Return the inclusive ``(low, high)`` range of a cuTile integer dtype, from its name (``int32``, ``uint64``, ...)."""
    name = str(getattr(dtype, "name", dtype))
    m = re.fullmatch(r"(u?)int(\d+)", name)
    if m is None:
        raise ValueError(f"not an integer dtype: {name}")
    bits = int(m.group(2))
    return (0, 2**bits - 1) if m.group(1) else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)


def _signed_twin(ct, dtype) -> tuple:
    """Return the signed cuTile dtype of the same width as an unsigned ``dtype``, or an empty tuple."""
    twins = {ct.uint8: ct.int8, ct.uint16: ct.int16, ct.uint32: ct.int32, ct.uint64: ct.int64}
    return (twins[dtype],) if dtype in twins else ()


def _python_def_name(key: str) -> str:
    """Return ``key`` as a usable Python ``def`` name: a Warp key such as ``class`` gets a trailing underscore."""
    return f"{key}_" if keyword.iskeyword(key) else key


def _make_invoke(
    arg_specs: list[_ArgSpec],
    base_name: str,
    tilelib: object,
    variants: dict,
    fallback_length: int,
    out_dtype: object,
    ct: object,
    tp: object,
    func_name: str | None = None,
) -> Callable:
    """Generate the traced call body for a declaration set (1:1 with the Warp signature).

    cuTile Python traces source through a restricted AST subset (no lists,
    ``raise``, ``isinstance``, ...), so argument classification cannot be
    written generically inside the traced call. Instead, the exact unrolled
    statements are generated per declaration set. Each argument is classified
    at trace time (``ct.static_eval``, full Python) as *shared* or
    *per-invocation* from its type and shape alone: a plain Python number, a
    rank-0 tile, or a ``(width,)`` tile is one value shared by every
    invocation; a rank-1 tile of ``(TILE * width,)`` is per-invocation; an
    array always takes one ``(address, sizes)`` tuple of plain values. The
    two cannot alias: a shared operand is at most ``(width,)`` and a
    per-invocation one at least ``(32 * width,)``. ``TILE`` is read from the
    first per-invocation operand (a width-1 operand when one exists, since a
    scalar shape cannot alias another declared length times a padding
    width), or is ``fallback_length`` when every argument is shared. Shared
    values are then expanded in the kernel to full ``(TILE * width,)`` tiles
    (``ct.full`` / ``ct.broadcast_to``), so one all-per-invocation declaration
    serves every call pattern.

    The generated function carries the Warp parameter names, so
    argument-count errors name them, and it opens with ``ct.static_assert``
    checks that report a wrong rank, element type, length, or an unusable
    value, naming the function and parameter, before cuTile's own operand
    type check could only report an opaque mismatch.

    ``variants`` maps each declared tile length to its
    ``(decl_name, out_shape, out_types)`` (``out_types`` in the preview call's
    ``(dtype, shape)`` form); when several lengths are declared, the body
    branches on the trace-time ``TILE`` constant (folded by the tracer) to
    pick the matching declaration. Values the call needs (tile dtypes, the
    tilelib) are placed in the generated function's globals under the
    ``_wp_`` prefix, which the declaration reserves, and the source is seeded
    into ``linecache`` so the tracer's ``inspect.findsource`` can read it
    without a file on disk.
    """

    label = func_name or base_name  # the Warp key, as diagnostics name it
    func_name = _python_def_name(label)
    ns = {
        "_wp_ct": ct,
        "_wp_fcall": tp.foreign_call,
        "_wp_libs": (tilelib,),
        "_wp_odt": out_dtype,
        "_wp_isinstance": isinstance,
        "_wp_len": len,
        "_wp_tuple": tuple,
        "_wp_int": int,
        "_wp_float": float,
        "_wp_bool": bool,
        "_wp_isfile": os.path.isfile,
        "_wp_path": tilelib.path,
    }
    for length, (decl_name, out_shape, out_types) in variants.items():
        ns[f"_wp_fname_n{length}"] = decl_name
        ns[f"_wp_oshape_n{length}"] = out_shape
        ns[f"_wp_otypes_n{length}"] = out_types
    lengths = tuple(variants)
    prefix = f"{label}(): "

    def check(cond: str, message: str) -> str:
        """Emit one trace-time assertion; ``message`` is the body of an f-string."""
        return f'    _wp_ct.static_assert({cond}, f"{prefix}{message}")'

    def plain_int_check(value: str, what: str, spec: _ArgSpec, op: _Operand) -> str:
        """Reject a plain value that is not an in-range integer for an integer operand (``ct.full`` would truncate or wrap it)."""
        lo, hi = _int_bounds(op.ct_dtype)
        return check(
            f"_wp_isinstance({value}, _wp_ct.Tile) or "
            f"(_wp_isinstance({value}, _wp_int) and not _wp_isinstance({value}, _wp_bool) and {lo} <= {value} <= {hi})",
            f"{what} (wp.{_type_name(spec)}) expects an integer in [{lo}, {hi}], got {{{value}}}",
        )

    def plain_float_check(value: str, what: str, spec: _ArgSpec) -> str:
        """Reject a plain value that is not a real number for a floating-point operand."""
        return check(
            f"_wp_isinstance({value}, _wp_ct.Tile) or "
            f"(_wp_isinstance({value}, (_wp_int, _wp_float)) and not _wp_isinstance({value}, _wp_bool))",
            f"{what} (wp.{_type_name(spec)}) expects a number, got {{{value}}}",
        )

    def plain_value_check(value: str, what: str, spec: _ArgSpec, op: _Operand) -> str:
        """Reject a plain value of the wrong kind for ``op``: integers must be in range, floats must be numbers."""
        if op.ctype in ("float", "double"):
            return plain_float_check(value, what, spec)
        return plain_int_check(value, what, spec, op)

    checks, expansions, operands = [], [], []
    candidates = []  # (value expr, param name, type name, operand) that may carry TILE
    for spec in arg_specs:
        name = spec.name
        tname = _type_name(spec)
        if spec.kind == "array":
            # one tuple of plain values from array_pointer_args(), each
            # element expanded to a per-invocation tile once TILE is known
            n = len(spec.operands)
            checks.append(
                check(
                    f"_wp_isinstance({name}, _wp_tuple) and _wp_len({name}) == {n}",
                    f"parameter '{name}' (wp.array) expects a tuple of {n} plain values (address, then one size "
                    f"per dimension) from array_pointer_args(), got {{{name}}}",
                )
            )
            # unpacked right after its check, before any later check reads the parts
            unpacked = ", ".join(f"_wp_av_{name}_{j}" for j in range(n))
            checks.append(f"    {unpacked} = {name}")
            for j, op in enumerate(spec.operands):
                value = f"_wp_av_{name}_{j}"
                what = f"element {j} of parameter '{name}'"
                checks.append(
                    check(
                        f"not _wp_isinstance({value}, _wp_ct.Tile) or _wp_len({value}.shape) == 0",
                        f"{what} expects a plain Python number (or a rank-0 tile), got {{{value}}}",
                    )
                )
                checks.append(plain_value_check(value, what, spec, op))
                # element keys carry their own prefix so an array "x" cannot collide with a scalar "x_0"
                ns[f"_wp_adt_{name}_{j}"] = op.ct_dtype
                ns[f"_wp_adts_{name}_{j}"] = (op.ct_dtype, *_signed_twin(ct, op.ct_dtype))
                # a rank-0 tile (runtime kernel parameter) must already have the operand's dtype
                checks.append(
                    check(
                        f"not _wp_isinstance({value}, _wp_ct.Tile) or {value}.dtype in _wp_adts_{name}_{j}",
                        f"{what} expects {' or '.join(str(getattr(d, 'name', d)) for d in (op.ct_dtype, *_signed_twin(ct, op.ct_dtype)))} "
                        f"elements, got {{{value}.dtype}}",
                    )
                )
                expansions.append(f"    _wp_at_{name}_{j} = _wp_ct.full((_wp_n,), {value}, dtype=_wp_adt_{name}_{j})")
                operands.append(f"_wp_at_{name}_{j}")
            continue

        op = spec.operands[0]
        w = op.width
        # Tile IR has no unsigned dtypes: an unsigned operand accepts the
        # signed tile of the same width, whose bits pass through unchanged
        accepted = (op.ct_dtype, *_signed_twin(ct, op.ct_dtype))
        ns[f"_wp_dt_{name}"] = accepted
        ns[f"_wp_dts_{name}"] = op.ct_dtype
        dt_text = " or ".join(str(getattr(d, "name", d)) for d in accepted)
        shared = f"_wp_s_{name}"
        if w == 1:
            # scalar: a plain number, rank-0 tile, or (1,) tile is shared;
            # a rank-1 tile is per-invocation
            checks.append(
                check(
                    f"not _wp_isinstance({name}, _wp_ct.Tile) or _wp_len({name}.shape) <= 1",
                    f"parameter '{name}' (wp.{tname}) expects a (TILE,) tile, or one shared value "
                    f"(a plain number or a rank-0 tile), got {{{name}}}",
                )
            )
            checks.append(
                f"    {shared} = _wp_ct.static_eval(not _wp_isinstance({name}, _wp_ct.Tile) "
                f"or _wp_len({name}.shape) == 0 or {name}.shape == (1,))"
            )
            checks.append(f"    _wp_p_{name} = _wp_ct.static_eval(not _wp_isinstance({name}, _wp_ct.Tile))")
            checks.append(plain_value_check(name, f"parameter '{name}'", spec, op))
            checks.append(
                check(
                    f"not _wp_isinstance({name}, _wp_ct.Tile) or {name}.dtype in _wp_dt_{name}",
                    f"parameter '{name}' expects {dt_text} elements, got {{{name}.dtype}}",
                )
            )
            # a fresh name per operand: the tracer rejects re-binding a
            # parameter to a tile of another shape
            expansions.append(f"    if {shared}:")
            expansions.append(f"        if _wp_p_{name}:")
            expansions.append(f"            _wp_a_{name} = _wp_ct.full((_wp_n,), {name}, dtype=_wp_dts_{name})")
            expansions.append("        else:")
            expansions.append(f"            _wp_a_{name} = _wp_ct.broadcast_to(_wp_ct.reshape({name}, (1,)), (_wp_n,))")
            expansions.append("    else:")
            expansions.append(f"        _wp_a_{name} = {name}")
        else:
            # vector / matrix / struct: a (width,) tile is shared, a
            # (TILE * width,) tile is per-invocation
            checks.append(
                check(
                    f"_wp_isinstance({name}, _wp_ct.Tile) and _wp_len({name}.shape) == 1",
                    f"parameter '{name}' (wp.{tname}) expects a rank-1 tile of shape (TILE * {w},), "
                    f"or one shared ({w},) tile, got {{{name}}}",
                )
            )
            checks.append(f"    {shared} = _wp_ct.static_eval({name}.shape == ({w},))")
            checks.append(
                check(
                    f"{name}.dtype in _wp_dt_{name}",
                    f"parameter '{name}' expects {dt_text} elements, got {{{name}.dtype}}",
                )
            )
            expansions.append(f"    if {shared}:")
            expansions.append(
                f"        _wp_a_{name} = _wp_ct.reshape(_wp_ct.broadcast_to(_wp_ct.reshape({name}, (1, {w})), "
                f"(_wp_n, {w})), (_wp_n * {w},))"
            )
            expansions.append("    else:")
            expansions.append(f"        _wp_a_{name} = {name}")
        operands.append(f"_wp_a_{name}")
        candidates.append((name, tname, op))

    # TILE is read from the first per-invocation operand, preferring width-1
    # operands: scalar shapes cannot alias another declared length times a
    # padding width, so an underpadded wide operand is caught by the
    # consistency check instead of silently regrouping under a smaller
    # length. With every argument shared, TILE is the declaration's hint.
    ordered = sorted(candidates, key=lambda item: item[2].width > 1)
    count_expr = str(fallback_length)
    key_expr = "'num_worker_warps'"
    for name, _, op in reversed(ordered):
        count_expr = f"({name}.shape[0] // {op.width}) if not _wp_s_{name} else {count_expr}"
        key_expr = f"'{name}' if not _wp_s_{name} else {key_expr}"
    checks.append(f"    _wp_n = _wp_ct.static_eval({count_expr})")
    checks.append(
        check(
            f"_wp_n in {lengths}",
            f"tile length {{_wp_n}} (from parameter '{{{key_expr}}}') is not a declared length; "
            f"this declaration serves {lengths}",
        )
    )
    for name, tname, op in ordered:
        checks.append(
            check(
                f"_wp_s_{name} or {name}.shape[0] == _wp_n * {op.width}",
                f"parameter '{name}' (wp.{tname}, width {op.width}) expects shape ({{_wp_n * {op.width}}},) "
                f"to match parameter '{{{key_expr}}}' (tile length {{_wp_n}}), got {{{name}.shape}}",
            )
        )

    ops = ", ".join(operands)

    def call_lines(length, indent):
        """Emit the destination tile + foreign_call pair for one length."""
        return [
            f"{indent}_wp_dest = _wp_ct.zeros(_wp_oshape_n{length}, dtype=_wp_odt)",
            f"{indent}return _wp_fcall(tilelibs=_wp_libs, symbol_name=_wp_fname_n{length}, constant_params=(), "
            f"inputs=({ops}{', ' if ops else ''}_wp_dest,), output_types=_wp_otypes_n{length})[0]",
        ]

    # a handle outlives its artifact if the kernel cache is cleared; say so
    # instead of letting tileiras fail on a missing file
    stale = check(
        "_wp_isfile(_wp_path)", "its .tilelib is missing (cleared from the kernel cache?); call export_func() again"
    )
    body = [stale, *checks, *expansions]
    if len(variants) > 1:
        for length in variants:
            if length == fallback_length:
                continue
            body.append(f"    if _wp_n == {length}:")
            body.extend(call_lines(length, "        "))
    body.extend(call_lines(fallback_length, "    "))
    params = ", ".join(spec.name for spec in arg_specs)
    src = (
        f"def {func_name}({params}):\n"
        f'    """Traced foreign call for {base_name} (generated)."""\n' + "\n".join(body) + "\n"
    )
    # Keyed by content, not a random token: cuTile Python embeds the filename in the
    # traced kernel's bytecode (as a location), so a per-process name would
    # defeat its cross-process cache for every kernel that calls a foreign
    # function. Two declarations of one function (different lengths or
    # scopes) generate different source and so keep separate traceback entries.
    fname = f"<wp_foreign_invoke_{base_name}_{hashlib.sha256(src.encode()).hexdigest()[:8]}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
    exec(compile(src, fname, "exec"), ns)
    return ns[func_name]


@dataclasses.dataclass(frozen=True)
class TilelibForeignFunction:
    """A Warp function packaged as a ``.tilelib``, callable inside cuTile kernels.

    Instances are returned by :func:`export_func`; users do not
    construct them. Calling one inside a ``@ct.kernel`` body records a
    ``cuda.tile_preview.foreign_call`` to the declaration matching the
    call-site tile length (``lengths`` lists the lengths the declaration set
    was built for) and yields the result as one tile. Calls are traced by
    cuTile as device code, so the returned object is not portable across
    processes; declare the function again in each process (the cached
    ``.tilelib`` makes that cheap).
    """

    name: str
    """Base declaration name inside the ``.tilelib`` (per-length names append ``_n<length>``)."""
    symbol: str
    """Base linkable symbol of the compiled wrapper (per-length symbols append ``_n<length>``)."""
    path: str
    """Path of the packed ``.tilelib`` file in the Warp kernel cache."""
    version: str
    """Content digest; cuTile Python caches kernels on (declaration name, version)."""
    lengths: tuple[int, ...]
    """Tile lengths this declaration set was built for."""
    tilelib: object = dataclasses.field(repr=False, compare=False)
    """``cuda.tile_preview.Tilelib`` (path + content-digest version) handing the library to cuTile."""
    invoke: object = dataclasses.field(repr=False, compare=False)
    """Generated call body — traced by cuTile; dispatches to the declared tile lengths."""
    signature: str = dataclasses.field(compare=False)
    """Call signature in Warp terms, e.g. ``blend(x: wp.float32 tile (TILE,) or shared value, ...) -> wp.float32 tile``."""

    def __call__(self, *inputs, **kwargs):
        """Invoke the foreign function inside a ``@ct.kernel`` body (positional or keyword, by Warp parameter name).

        cuTile Python traces this method as device code, so it must stay a plain
        forwarding call.
        """
        return self.invoke(*inputs, **kwargs)

    def __repr__(self) -> str:
        """Summarize the declaration: its call signature and the tile lengths it serves."""
        return f"<export_func {self.signature}; lengths={self.lengths}>"


def export_func(
    func: wp.Function,
    num_worker_warps: int = 4,
    scope: Literal["thread", "block"] = "thread",
    device: DeviceLike = None,
):
    """Package a Warp ``@wp.func`` as a cuTile foreign function (a ``.tilelib``) callable inside ``@ct.kernel`` bodies.

    .. admonition:: Experimental

        The cuTile interop API is experimental and subject to change without a formal
        deprecation cycle; it builds on cuTile preview interfaces that may change
        between CUDA Toolkit and cuTile Python releases.

    Compilation and packaging happen when this function is called, on the
    current machine, and the result is cached in the Warp kernel cache.
    Requires the CUDA Toolkit 13.5 ``tileiras``, found in the
    ``nvidia-cuda-tileiras`` wheel that ``warp-lang[cutile]`` installs, else
    on ``PATH``.

    Args:
        func: A concrete (non-generic) ``@wp.func`` with scalar, vector, matrix,
            quaternion, transform, struct, or contiguous ``wp.array`` parameters
            (``float16``, ``bfloat16``, and ``bool`` excepted) and exactly one
            return value.
        num_worker_warps: Worker-warp count of the calling kernel, as in
            ``@ct.kernel(num_worker_warps=...)``; a power of two between 1 and
            32. It can normally be left at its default: a thread-scope
            declaration is built for every power-of-two tile length from 32
            to 1024 and call sites select the length through their tiles. It
            matters in three cases. For ``scope="block"`` the declaration is
            built for the single tile length ``num_worker_warps * 32``, every
            call must use it, and it must equal the calling kernel's
            ``num_worker_warps``, or ``tileiras`` rejects the kernel when it
            compiles it. For a call whose arguments are all shared values (no
            tile carries a length) it is the number of invocations the call
            runs. For functions using mesh AABB/sphere or BVH iterator queries
            it sets the block the traversal-stack scratch covers:
            ``max(8, num_worker_warps)`` warps at thread scope, exactly
            ``num_worker_warps`` at block scope. A calling kernel whose
            compiled block is larger stops with a device trap, a sticky CUDA
            error, so declare at least the kernel's ``num_worker_warps``.
        scope: Execution scope: ``"thread"`` (one independent invocation per
            tile element; the function must not synchronize across threads) or
            ``"block"`` (one invocation with all worker threads participating
            together, enabling block-cooperative Warp builtins that use
            barriers; declared to Tile IR as ``cta`` synchronization).
        device: CUDA device whose architecture the declaration is built for.
            Defaults to the current Warp device.

    Returns:
        ``TilelibForeignFunction``: A callable for use inside ``@ct.kernel``
        bodies with one argument per Warp parameter, positional or by name.
        Each argument is either per-invocation (a ``(TILE,)`` or
        ``(TILE * width,)`` tile) or shared by every invocation (a plain
        Python number or rank-0 tile for a scalar, a ``(width,)`` tile for a
        vector, matrix, or struct, a tuple of plain values for an array); the
        call tells the two apart by shape and expands shared values to the
        tile length of the per-invocation ones. It returns
        one tile (see :doc:`the cuTile interoperability guide
        </user_guide/interoperability/cutile>` for the full calling
        convention). Its ``repr`` shows the call signature in Warp terms and
        the tile lengths it serves.

    Raises:
        ImportError: If the cuTile packages are missing or older than 1.8.1.
        TypeError: If ``func`` is not a user-defined ``@wp.func``, ``num_worker_warps``
            is not an int, a parameter or return type cannot cross the
            boundary, or the function returns nothing.
        ValueError: If ``num_worker_warps`` or ``scope`` is invalid, the function or a parameter
            has a name the generated code cannot use (not a plain ASCII
            identifier, the reserved ``_wp_`` prefix, or a generated operand
            name such as ``ret0``), the function has default parameter values,
            it uses block-cooperative builtins or tiled queries at thread scope,
            its scratch exceeds the device's per-block shared memory, or
            ``device`` is not a CUDA device.
        NotImplementedError: If the function has several overloads or return
            values.
        RuntimeError: If ``tileiras`` cannot be found or is not the CUDA
            Toolkit 13.5 release, or NVRTC or ``tileiras`` fails to build the
            function.

    Example:
        Declare once on the host, then call the result inside a ``@ct.kernel``
        with one argument per Warp parameter:

        .. code:: python

            @wp.func
            def scale_add(x: float, y: float) -> float:
                return 2.0 * x + y


            wp_scale_add = wpct.export_func(scale_add)


            @ct.kernel(num_worker_warps=TILE // 32)
            def kernel(xs, ys, out, out_shared):
                x = ct.load(xs, (0,), shape=(TILE,))
                y = ct.load(ys, (0,), shape=(TILE,))
                ct.store(out, (0,), wp_scale_add(x, y))  # one call per tile element
                ct.store(out_shared, (0,), wp_scale_add(x, 1.0))  # y shared by every invocation
    """
    try:
        import cuda.tile as ct  # noqa: PLC0415
        import cuda.tile_preview as tp  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            'cuTile Python is required for warp.experimental.cutile; install it with: pip install "warp-lang[cutile]"'
        ) from e
    if not (hasattr(tp, "Tilelib") and hasattr(tp, "foreign_call")):
        raise ImportError(
            "the installed cuda-tile-preview lacks the Tilelib/foreign_call API; cuda-tile 1.8.1 or later is "
            'required: pip install -U "warp-lang[cutile]"'
        )

    plan = _plan_foreign(ct, func, num_worker_warps, scope)
    # the declaration is built for one device's architecture; checked after
    # argument validation so declaration errors surface without a GPU
    device = wp.get_device(device)
    if not device.is_cuda:
        raise ValueError(f"Foreign functions must target a CUDA device; got '{device}'")
    in_ops, out_ops, lengths, fallback_length = plan.in_ops, plan.out_ops, plan.lengths, plan.fallback_length
    # the function must compute what its own module would under wp.launch
    build_options = _ltoir_build_options(func)
    _check_scratch_budget(func.key, plan.layout.total_bytes(num_worker_warps * 32), device)

    # Resolve the packer once and use the same resolution for both the digest
    # and the packing step, so the digest always describes the tool that
    # produced the artifact.
    tileiras = _find_tileiras()
    _check_tileiras_release(tileiras)
    arch = f"sm_{device.arch}"

    def emit(decl_base, symbol_base):
        """Emit the interface/implementation module pair for the declared lengths."""
        return _emit_tilelib_modules(decl_base, symbol_base, in_ops, out_ops, lengths, scope, plan.layout, arch)

    iface_template, impl_template = emit(_DECL_NAME_PLACEHOLDER, _WRAPPER_SYMBOL_PLACEHOLDER)

    digest = _tilelib_digest(
        compiler_runtime=_compiler_runtime_fingerprint(),
        toolchain_fingerprint=_toolchain_fingerprint(tileiras),
        headers_fingerprint=_native_headers_fingerprint(),
        source=plan.source,
        wrapper=plan.wrapper_template,
        interface=iface_template,
        impl=impl_template,
        arch=arch,
        lengths=lengths,
        scope=scope,
        smem_bytes=plan.layout.smem_bytes,
        build_options=build_options,
    )
    # the digest makes the declared name collision-free (func.key alone is
    # qualname-only and can collide across Python modules; _plan_foreign
    # guarantees it is a plain identifier)
    symbol = f"wp_foreign_{func.key}_{digest}"
    decl_name = f"{func.key}_{digest}"

    # one NVRTC compile: the shared per-invocation code plus one thin
    # extern "C" wrapper (= one linkable symbol) per declared length
    wrappers = "".join(
        plan.wrapper_template.replace(_WRAPPER_SYMBOL_PLACEHOLDER, f"{symbol}_n{length}") for length in lengths
    )
    # the symbol carries the full content digest, so an existing tilelib under
    # it was packed from identical inputs and needs neither NVRTC nor tileiras
    tilelib_path = os.path.join(_cache_dir(), f"{symbol}.{arch}.tilelib")
    if not os.path.isfile(tilelib_path):
        ltoir_path = _compile_ltoir(plan.source + wrappers, symbol, arch, build_options=build_options)
        iface, impl = emit(decl_name, symbol)
        _pack_tilelib(iface, impl, ltoir_path, tilelib_path, tileiras)

    tilelib = tp.Tilelib(path=tilelib_path, version=digest)
    variants = {
        length: (
            f"{decl_name}_n{length}",
            out_ops[0].tile_shape(length),
            tuple((op.ct_dtype, op.tile_shape(length)) for op in out_ops),
        )
        for length in lengths
    }
    out_dtype = out_ops[0].ct_dtype
    ret_spec = _ArgSpec(ct, "ret0", plan.return_type, is_output=True)
    signature = f"{func.key}({', '.join(_describe_param(a) for a in plan.args)}) -> wp.{_type_name(ret_spec)} tile"
    invoke = _make_invoke(plan.args, decl_name, tilelib, variants, fallback_length, out_dtype, ct, tp, func.key)
    invoke.__doc__ = f"{signature} (generated cuTile call body)."
    return TilelibForeignFunction(
        name=decl_name,
        symbol=symbol,
        path=tilelib_path,
        version=digest,
        lengths=lengths,
        tilelib=tilelib,
        invoke=invoke,
        signature=signature,
    )
