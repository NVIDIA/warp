# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Tile IR bytecode writer for the cuTile foreign-function bridge.

A ``.tilelib`` is packed from two Tile IR bytecode modules: an *interface*
holding the extern declaration a kernel calls, and an *implementation*
holding the ``extern_impl`` entry that binds that declaration to a linkable
symbol with per-operand fragment maps. The CUDA Toolkit ships no tool that
assembles these modules from text, so this module writes the bytecode
directly. It covers exactly the subset the bridge emits: rank-0/rank-1
tiles of simple element types, register fragments, access modes, and the
optional implementation attributes (architecture, synchronization, scratch,
frontend ABI).

The encoding follows the format written by cuTile Python's own bytecode writer
(``cuda.tile._bytecode``), restricted to bytecode version 13.5 and to
zero-function modules.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

BYTECODE_VERSION = (13, 5)
"""Tile IR bytecode version written: the first with fragment-map externs."""

_MAGIC = b"\x7fTileIR\x00"


# --- primitive encodings --------------------------------------------------------


def _varint(x: int, buf: bytearray) -> None:
    """Append ``x`` as an unsigned LEB128 varint."""
    if x < 0:
        raise ValueError(f"varint must be non-negative, got {x}")
    while x >= 0x80:
        buf.append((x & 0x7F) | 0x80)
        x >>= 7
    buf.append(x)


def _signed_varint(x: int, buf: bytearray) -> None:
    """Append ``x`` as a zigzag-style signed varint (sign in the low bit)."""
    x <<= 1
    if x < 0:
        x = ~x
    _varint(x, buf)


def _int_list(values: Sequence[int], byte_width: int, buf: bytearray) -> None:
    """Append a counted list of fixed-width little-endian signed integers."""
    _varint(len(values), buf)
    for v in values:
        buf.extend(int(v).to_bytes(byte_width, "little", signed=True))


class _Table:
    """Interning table: bytes -> dense id in first-use order (``start`` is the first id)."""

    def __init__(self, start: int = 0) -> None:
        """Start an empty table whose first entry gets index ``start``."""
        self._items: dict[bytes, int] = {}
        self._start = start

    def __getitem__(self, key: bytes) -> int:
        """Return the id of ``key``, assigning the next id on first use."""
        if key not in self._items:
            self._items[key] = len(self._items) + self._start
        return self._items[key]

    def __len__(self) -> int:
        """Number of interned entries."""
        return len(self._items)

    def encode(self, index_size: int, buf: bytearray) -> None:
        """Append the table: count, aligned offset index, then the concatenated items."""
        items = sorted(self._items.items(), key=lambda kv: kv[1])
        _varint(len(items), buf)
        _pad(index_size, buf)
        offset = 0
        for data, _ in items:
            buf.extend(offset.to_bytes(index_size, "little"))
            offset += len(data)
        for data, _ in items:
            buf.extend(data)


def _pad(alignment: int, buf: bytearray) -> None:
    """Pad ``buf`` to ``alignment`` with the Tile IR filler byte."""
    buf.extend(b"\xcb" * (-len(buf) % alignment))


# --- tags -------------------------------------------------------------------------


class SimpleType(enum.Enum):
    """Element type tags (bytecode type table)."""

    I1 = b"\x00"
    I8 = b"\x01"
    I16 = b"\x02"
    I32 = b"\x03"
    I64 = b"\x04"
    F16 = b"\x05"
    BF16 = b"\x06"
    F32 = b"\x07"
    F64 = b"\x09"


class _Composite(enum.Enum):
    """Composite type tags."""

    Tile = b"\x0d"
    Function = b"\x10"
    Fragment = b"\xd2\x01"  # preview


class _Attr(enum.Enum):
    """Attribute tags used by the extern entries."""

    String = b"\x05"
    Array = b"\x06"
    Dictionary = b"\x0a"
    Access = b"\xdf\x01"
    ExprInt = b"\xd1\x01"
    LayoutTuple = b"\xd8\x01"
    Layout = b"\xd9\x01"


class _Section(enum.IntEnum):
    """Section ids; bit 7 marks an aligned section."""

    EndOfBytecode = 0x00
    String = 0x01
    Func = 0x02
    Debug = 0x03
    Constant = 0x04
    Type = 0x05


_ACCESS = {"in": 0, "out": 1, "inout": 2}
_SYNCHRONIZATION = {"thread": 0, "warp": 1, "warpgroup": 2, "cta": 3}
_MEMORY_SPACE = {"rmem": 0, "smem": 1, "tmem": 2}
_ACCESS_ATTR_NAME = b"cuda_tile.preview$access"


# --- module writer ----------------------------------------------------------------


class Module:
    """One zero-function Tile IR bytecode module carrying extern entries.

    Build types with :meth:`tile`, :meth:`fragment`, and :meth:`function`,
    add entries with :meth:`declare` or :meth:`implement`, then call
    :meth:`to_bytes`. Entries must reference symbols by name; the string and
    type tables are assembled on output.
    """

    def __init__(self) -> None:
        """Start an empty module with fresh type, string, and function tables."""
        self._strings = _Table()
        self._types = _Table()
        self._entries: list[bytes] = []
        # the two predefined type ids every table starts with
        self.simple(SimpleType.I1)
        self.simple(SimpleType.I32)

    # types

    def simple(self, t: SimpleType) -> int:
        """Return the type id of a simple element type."""
        return self._types[t.value]

    def tile(self, element: int, shape: Sequence[int]) -> int:
        """Return the type id of a tile of ``element`` with a concrete ``shape`` (rank 0 allowed)."""
        buf = bytearray(_Composite.Tile.value)
        _varint(element, buf)
        _int_list(shape, 8, buf)
        return self._types[bytes(buf)]

    def function(self, params: Sequence[int], results: Sequence[int]) -> int:
        """Return the type id of a function type."""
        buf = bytearray(_Composite.Function.value)
        _varint(len(params), buf)
        for t in params:
            _varint(t, buf)
        _varint(len(results), buf)
        for t in results:
            _varint(t, buf)
        return self._types[bytes(buf)]

    def fragment(
        self,
        element: int,
        shape: Sequence[int],
        mem_layout: tuple[Sequence[int], Sequence[int]],
        fragmap: tuple[Sequence[int], Sequence[int]],
        alignment: int = 1,
        memory_space: str = "rmem",
    ) -> int:
        """Return the type id of a register fragment: per-invocation ``shape`` elements with the given layouts.

        ``mem_layout`` and ``fragmap`` are ``(shapes, strides)`` pairs of
        plain integers; ``alignment`` is in elements.
        """
        buf = bytearray(_Composite.Fragment.value)
        buf.extend(self._layout_tuple(shape))  # symbolic shape (all concrete)
        _int_list(shape, 8, buf)  # concrete shape
        _varint(element, buf)
        _varint(_MEMORY_SPACE[memory_space], buf)
        buf.extend(self._layout(*mem_layout))
        buf.extend(self._layout(*fragmap))
        buf.extend(self._expr_int(alignment))
        return self._types[bytes(buf)]

    # entries

    def declare(self, name: str, params: Sequence[int], results: Sequence[int], access: Sequence[str]) -> None:
        """Add an extern declaration: a callable with typed operands, per-operand access modes, and results."""
        if len(access) != len(params):
            raise ValueError("One access mode per declared operand is required")
        buf = bytearray()
        _varint(self._strings[name.encode()], buf)
        _varint(self.function(params, results), buf)
        buf.append(0x01)  # FormNormal; a zero body length marks the declaration as extern
        _varint(0, buf)  # no debug location
        self._access_array(access, buf)
        _varint(0, buf)  # no body
        buf.extend(_Attr.Array.value)
        _varint(0, buf)  # no constant parameters
        self._entries.append(bytes(buf))

    def implement(
        self,
        symbol: str,
        implements: str,
        fragments: Sequence[int],
        access: Sequence[str],
        *,
        arch: Sequence[str] | None = None,
        synchronization: str | None = None,
        scratch: tuple[int, int] | None = None,
        frontend_abi: Mapping[str, str] | None = None,
    ) -> None:
        """Add an ``extern_impl`` entry binding ``implements`` to the linkable ``symbol``.

        ``fragments`` are the operands' fragment type ids in declaration
        order; ``scratch`` is ``(size_bytes, alignment_bytes)`` of
        compiler-provided shared memory; ``frontend_abi`` maps Tile IR LLVM
        types to the implementation's (e.g. the scratch pointer's address
        space). ``synchronization`` of ``None`` keeps the default (thread).
        """
        if len(access) != len(fragments):
            raise ValueError("One access mode per implementation operand is required")
        fn = self.function(fragments, [])
        body = bytearray()
        flags = 1 << 2  # access
        if scratch is not None:
            flags |= 1 << 4
        if arch is not None:
            flags |= 1 << 6
        if frontend_abi is not None:
            flags |= 1 << 7
        if synchronization is not None:
            flags |= 1 << 8
        _varint(flags, body)
        _varint(self._strings[symbol.encode()], body)
        _varint(self._strings[implements.encode()], body)
        _varint(fn, body)
        self._access_array(access, body, tagged=False)
        if scratch is not None:
            body.extend(self._expr_int(scratch[0]))
            body.extend(self._expr_int(scratch[1]))
        if arch is not None:
            _varint(len(arch), body)
            for a in arch:
                body.extend(_Attr.String.value)
                _varint(self._strings[a.encode()], body)
        if frontend_abi is not None:
            _varint(len(frontend_abi), body)
            for tile_ty, frontend_ty in frontend_abi.items():
                _varint(self._strings[tile_ty.encode()], body)
                body.extend(_Attr.String.value)
                _varint(self._strings[frontend_ty.encode()], body)
        if synchronization is not None:
            _varint(_SYNCHRONIZATION[synchronization], body)
        _varint(1, body)  # one region ...
        _varint(0, body)  # ... that is empty

        buf = bytearray()
        _varint(self._strings[symbol.encode()], buf)
        _varint(fn, buf)
        buf.append(0x05)  # private | FormExternImpl
        _varint(0, buf)  # no debug location
        buf.extend(body)
        _varint(0, buf)  # no body
        self._entries.append(bytes(buf))

    # output

    def to_bytes(self) -> bytes:
        """Serialize the module: header, function, constant, debug, type, and string sections."""
        out = bytearray(_MAGIC)
        out.append(BYTECODE_VERSION[0])
        out.append(BYTECODE_VERSION[1])
        out.extend((1).to_bytes(2, "little"))  # preview content flag (replaces the version tag)

        func = bytearray()
        _varint(len(self._entries), func)
        for e in self._entries:
            func.extend(e)
        self._section(_Section.Func, 8, func, out)

        self._section(_Section.Constant, 8, self._encode_empty_table(8), out)

        debug = bytearray()
        _varint(0, debug)  # functions with debug info
        _pad(4, debug)
        _varint(0, debug)  # operations with debug info
        _pad(8, debug)
        # the decoder rejects an empty attribute table, so register one null entry
        attrs = _Table(start=1)
        attrs[b"\x00"]
        attrs.encode(4, debug)
        self._section(_Section.Debug, 8, debug, out)

        types = bytearray()
        self._types.encode(4, types)
        self._section(_Section.Type, 4, types, out)

        strings = bytearray()
        self._strings.encode(4, strings)
        self._section(_Section.String, 4, strings, out)

        out.append(_Section.EndOfBytecode)
        return bytes(out)

    # helpers

    @staticmethod
    def _encode_empty_table(index_size: int) -> bytearray:
        """Encode a table with no items."""
        buf = bytearray()
        _Table().encode(index_size, buf)
        return buf

    @staticmethod
    def _section(section: _Section, alignment: int, payload: bytearray, out: bytearray) -> None:
        """Append one section: id (aligned bit set), length, alignment, padding, payload."""
        out.append(int(section) | 0x80)
        _varint(len(payload), out)
        _varint(alignment, out)
        _pad(alignment, out)
        out.extend(payload)

    @staticmethod
    def _expr_int(value: int) -> bytes:
        """Encode a symbolic integer expression holding a constant."""
        buf = bytearray(_Attr.ExprInt.value)
        _signed_varint(value, buf)
        return bytes(buf)

    def _layout_tuple(self, values: Sequence[int]) -> bytes:
        """Encode a tuple of constant symbolic expressions."""
        buf = bytearray(_Attr.LayoutTuple.value)
        _varint(len(values), buf)
        for v in values:
            buf.extend(self._expr_int(v))
        return bytes(buf)

    def _layout(self, shapes: Sequence[int], strides: Sequence[int]) -> bytes:
        """Encode a ``(shapes):(strides)`` layout attribute."""
        buf = bytearray(_Attr.Layout.value)
        for tup in (shapes, strides):
            _varint(len(tup), buf)
            for v in tup:
                buf.extend(self._expr_int(v))
        return bytes(buf)

    def _access_array(self, access: Sequence[str], buf: bytearray, *, tagged: bool = True) -> None:
        """Encode per-operand access modes as an array of ``{cuda_tile.preview$access = <mode>}`` dictionaries."""
        if tagged:
            buf.extend(_Attr.Array.value)
        _varint(len(access), buf)
        for mode in access:
            buf.extend(_Attr.Dictionary.value)
            _varint(1, buf)
            _varint(self._strings[_ACCESS_ATTR_NAME], buf)
            buf.extend(_Attr.Access.value)
            _varint(_ACCESS[mode], buf)
