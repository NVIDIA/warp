# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Track the on-disk and static TLS footprint of Warp's native libraries."""

import struct
import sys
from pathlib import Path

from asv_runner.benchmarks.mark import skip_benchmark_if

import warp as wp


def _get_elf_tls_allocation_size(path):
    """Return the in-memory size of an ELF image's TLS segment."""
    data = Path(path).read_bytes()

    if data[:4] != b"\x7fELF":
        raise ValueError(f"Not an ELF image: {path}")

    # EI_CLASS selects 32-bit or 64-bit header layouts, while EI_DATA selects
    # the byte order used by every multi-byte field in the image.
    elf_class = data[4]
    byte_order = data[5]
    if byte_order == 1:
        byte_order_prefix = "<"
    elif byte_order == 2:
        byte_order_prefix = ">"
    else:
        raise ValueError(f"Unsupported ELF byte order: {byte_order}")

    def unpack_from(format_string, offset):
        return struct.unpack_from(byte_order_prefix + format_string, data, offset)[0]

    if elf_class == 2:
        program_header_offset = unpack_from("Q", 0x20)
        program_header_entry_size = unpack_from("H", 0x36)
        program_header_count = unpack_from("H", 0x38)
        memory_size_offset = 40
        memory_size_format = "Q"
    elif elf_class == 1:
        program_header_offset = unpack_from("I", 0x1C)
        program_header_entry_size = unpack_from("H", 0x2A)
        program_header_count = unpack_from("H", 0x2C)
        memory_size_offset = 20
        memory_size_format = "I"
    else:
        raise ValueError(f"Unsupported ELF class: {elf_class}")

    for program_header_index in range(program_header_count):
        entry_offset = program_header_offset + program_header_index * program_header_entry_size
        if unpack_from("I", entry_offset) == 7:  # PT_TLS
            # p_memsz includes both the initialized template and zero-filled
            # TLS data, unlike p_filesz, which covers only the template.
            return unpack_from(memory_size_format, entry_offset + memory_size_offset)

    return 0


class NativeLibraryFootprint:
    """Track Linux native-library footprint metrics in bytes."""

    def setup(self):
        bin_directory = Path(wp.__file__).resolve().parent / "bin"
        self.warp_library_path = bin_directory / "warp.so"
        self.warp_clang_library_path = bin_directory / "warp-clang.so"

    @skip_benchmark_if(not sys.platform.startswith("linux"))
    def track_warp_library_size(self):
        return self.warp_library_path.stat().st_size

    track_warp_library_size.unit = "bytes"

    @skip_benchmark_if(not sys.platform.startswith("linux"))
    def track_warp_clang_library_size(self):
        return self.warp_clang_library_path.stat().st_size

    track_warp_clang_library_size.unit = "bytes"

    @skip_benchmark_if(not sys.platform.startswith("linux"))
    def track_warp_tls_allocation_size(self):
        return _get_elf_tls_allocation_size(self.warp_library_path)

    track_warp_tls_allocation_size.unit = "bytes"
