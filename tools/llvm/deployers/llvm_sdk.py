# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conan deployer producing Warp's normalized, Conan-free LLVM SDK tree.

The deployed tree is the product consumers see; nothing Conan-specific
survives in it. Shape:

    llvm-sdk/
    ├── include/{llvm,clang,...}   # all installed headers
    ├── lib/                       # static libraries only, flat
    └── licenses/{llvm,clang}/LICENSE.TXT
"""

import os
import shutil

# Static-archive extensions across all supported hosts: .a for ELF/Mach-O
# (Linux, macOS), .lib for MSVC (Windows). Anything in lib/ not ending in one
# of these is skipped, keeping the deployed tree static-only. Extend this only
# if a new host uses a different static-archive suffix, or if the SDK is ever
# meant to ship shared libraries (.so/.dylib/.dll), which today it does not.
_STATIC_LIB_SUFFIXES = (".a", ".lib")


def deploy(graph, output_folder, **kwargs):
    """Assemble the normalized llvm-sdk tree from the clang-warp package.

    Conan's deployer entry point (invoked via --deployer). Finds the clang-warp
    node in the dependency graph and copies its headers, licenses, and static
    libraries into output_folder/llvm-sdk/, leaving nothing Conan-specific
    behind. See the module docstring for the resulting layout.

    Args:
        graph: Resolved Conan dependency graph, scanned for the clang-warp node.
        output_folder: Destination root; the SDK tree is written to llvm-sdk/ within it.
        **kwargs: Extra arguments Conan passes to deployers; unused here.
    """
    for node in graph.nodes:
        if node.conanfile.name != "clang-warp":
            continue
        package_folder = node.conanfile.package_folder
        sdk_root = os.path.join(output_folder, "llvm-sdk")
        if os.path.exists(sdk_root):
            shutil.rmtree(sdk_root)

        shutil.copytree(os.path.join(package_folder, "include"), os.path.join(sdk_root, "include"))
        shutil.copytree(os.path.join(package_folder, "licenses"), os.path.join(sdk_root, "licenses"))

        lib_src = os.path.join(package_folder, "lib")
        lib_dst = os.path.join(sdk_root, "lib")
        os.makedirs(lib_dst)
        skipped = []
        for name in sorted(os.listdir(lib_src)):
            src = os.path.join(lib_src, name)
            if os.path.isfile(src) and name.endswith(_STATIC_LIB_SUFFIXES):
                shutil.copy2(src, os.path.join(lib_dst, name))
            else:
                skipped.append(name)
        if skipped:
            node.conanfile.output.info(f"llvm_sdk deployer skipped: {skipped}")
        node.conanfile.output.success(f"LLVM SDK deployed to {sdk_root}")
