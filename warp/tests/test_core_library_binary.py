# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test structural properties of Warp's platform-specific core libraries.

The tests in this module inspect shared-library binary metadata and enforce
constraints that prevent platform-specific resource regressions.
"""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

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


def _get_pe_tls_allocation_size(path):
    """Return the initialized and zero-filled TLS size of a PE image."""
    data = Path(path).read_bytes()

    def unpack_from(format_string, offset):
        return struct.unpack_from(format_string, data, offset)[0]

    pe_offset = unpack_from("<I", 0x3C)  # DOS header e_lfanew field
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise ValueError(f"Not a PE image: {path}")

    coff_offset = pe_offset + 4
    section_count = unpack_from("<H", coff_offset + 2)
    optional_header_size = unpack_from("<H", coff_offset + 16)
    optional_header_offset = coff_offset + 20
    optional_header_magic = unpack_from("<H", optional_header_offset)

    # PE32+ and PE32 place their data-directory tables at different offsets.
    if optional_header_magic == 0x20B:
        data_directories_offset = optional_header_offset + 112
        address_format = "<Q"
    elif optional_header_magic == 0x10B:
        data_directories_offset = optional_header_offset + 96
        address_format = "<I"
    else:
        raise ValueError(f"Unsupported PE optional-header magic: 0x{optional_header_magic:x}")

    # IMAGE_DIRECTORY_ENTRY_TLS is the data-directory entry at index 9.
    tls_directory_rva = unpack_from("<I", data_directories_offset + 9 * 8)
    if tls_directory_rva == 0:
        return 0

    sections_offset = optional_header_offset + optional_header_size
    for section_index in range(section_count):
        section_offset = sections_offset + section_index * 40
        virtual_size = unpack_from("<I", section_offset + 8)
        virtual_address = unpack_from("<I", section_offset + 12)
        raw_size = unpack_from("<I", section_offset + 16)
        raw_offset = unpack_from("<I", section_offset + 20)
        mapped_size = max(virtual_size, raw_size)

        if virtual_address <= tls_directory_rva < virtual_address + mapped_size:
            # Data directories use RVAs; translate through the containing
            # section to locate the TLS directory in the file image.
            tls_directory_offset = raw_offset + tls_directory_rva - virtual_address
            address_size = struct.calcsize(address_format)
            start_address = unpack_from(address_format, tls_directory_offset)
            end_address = unpack_from(address_format, tls_directory_offset + address_size)
            # SizeOfZeroFill follows four pointer-sized fields in both PE32 TLS
            # directory variants.
            zero_fill_size = unpack_from("<I", tls_directory_offset + 4 * address_size)
            return end_address - start_address + zero_fill_size

    raise ValueError(f"TLS directory RVA 0x{tls_directory_rva:x} is not mapped by a PE section")


class TestCoreLibraryBinary(unittest.TestCase):
    def test_elf_tls_allocation_size_includes_zero_fill(self):
        """Verify ELF TLS allocation includes zero-filled memory."""
        program_header_offset = 0x40
        file_size = 32
        memory_size = 96

        # Build a minimal ELF64 image with one PT_TLS program header. Its file
        # size covers initialized data, while its larger memory size adds zero-fill.
        image = bytearray(0x100)
        image[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
        struct.pack_into("<Q", image, 0x20, program_header_offset)
        struct.pack_into("<H", image, 0x36, 56)
        struct.pack_into("<H", image, 0x38, 1)
        struct.pack_into("<I", image, program_header_offset, 7)
        struct.pack_into("<Q", image, program_header_offset + 32, file_size)
        struct.pack_into("<Q", image, program_header_offset + 40, memory_size)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tls.so"
            path.write_bytes(image)
            self.assertEqual(_get_elf_tls_allocation_size(path), memory_size)

    def test_pe_tls_allocation_size_includes_zero_fill(self):
        """Verify PE TLS allocation includes zero-filled memory."""
        pe_offset = 0x80
        optional_header_offset = pe_offset + 24
        optional_header_size = 0xF0
        sections_offset = optional_header_offset + optional_header_size
        tls_directory_rva = 0x1000
        tls_directory_offset = 0x200
        raw_template_size = 32
        zero_fill_size = 96

        # Build a minimal PE32+ image with one section and point its TLS data
        # directory at an RVA mapped by that section.
        image = bytearray(0x400)
        struct.pack_into("<I", image, 0x3C, pe_offset)
        image[pe_offset : pe_offset + 4] = b"PE\0\0"
        struct.pack_into("<H", image, pe_offset + 6, 1)
        struct.pack_into("<H", image, pe_offset + 20, optional_header_size)
        struct.pack_into("<H", image, optional_header_offset, 0x20B)
        struct.pack_into("<I", image, optional_header_offset + 112 + 9 * 8, tls_directory_rva)
        struct.pack_into("<I", image, sections_offset + 8, 0x200)
        struct.pack_into("<I", image, sections_offset + 12, tls_directory_rva)
        struct.pack_into("<I", image, sections_offset + 16, 0x200)
        struct.pack_into("<I", image, sections_offset + 20, tls_directory_offset)

        # The TLS directory's address range covers initialized template data;
        # SizeOfZeroFill extends the allocation beyond that raw range.
        struct.pack_into("<Q", image, tls_directory_offset, 0x180000000)
        struct.pack_into("<Q", image, tls_directory_offset + 8, 0x180000000 + raw_template_size)
        struct.pack_into("<I", image, tls_directory_offset + 32, zero_fill_size)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tls.dll"
            path.write_bytes(image)
            self.assertEqual(_get_pe_tls_allocation_size(path), raw_template_size + zero_fill_size)

    @unittest.skipUnless(sys.platform.startswith("linux"), "ELF TLS segments only apply on Linux")
    def test_linux_core_library_tls_allocation_size(self):
        """Ensure the core library does not contain a substantial static TLS allocation."""
        shared_library_path = Path(wp.__file__).resolve().parent / "bin" / "warp.so"
        tls_allocation_size = _get_elf_tls_allocation_size(shared_library_path)

        self.assertLessEqual(
            tls_allocation_size,
            64 * 1024,
            f"warp.so has a {tls_allocation_size}-byte ELF TLS segment",
        )

    @unittest.skipUnless(sys.platform == "win32", "PE TLS templates only apply on Windows")
    def test_windows_core_library_tls_allocation_size(self):
        """Ensure the core library does not reserve substantial memory for every host thread."""
        dll_path = Path(wp.__file__).resolve().parent / "bin" / "warp.dll"
        tls_allocation_size = _get_pe_tls_allocation_size(dll_path)

        self.assertLessEqual(
            tls_allocation_size,
            64 * 1024,
            f"warp.dll has {tls_allocation_size} bytes of TLS data allocated for every Windows thread",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
