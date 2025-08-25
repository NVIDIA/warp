# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is an 'offline' build of the core warp runtime libraries
# designed to be executed as part of CI / developer workflows, not
# as part of the user runtime (since it requires CUDA toolkit, etc)

from __future__ import annotations

import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys

import warp.build_dll as build_dll
from warp.build import clear_kernel_cache, clear_lto_cache
from warp.context import export_builtins


def find_cuda_sdk() -> str | None:
    # check environment variables
    for env in ["WARP_CUDA_PATH", "CUDA_HOME", "CUDA_PATH"]:
        cuda_sdk = os.environ.get(env)
        if cuda_sdk is not None:
            print(f"Using CUDA Toolkit path '{cuda_sdk}' provided through the '{env}' environment variable")
            return cuda_sdk

    # use which/where to locate the nvcc compiler program
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        cuda_sdk = os.path.dirname(os.path.dirname(nvcc))  # strip the executable name and bin folder
        print(f"Using CUDA Toolkit path '{cuda_sdk}' found through 'which nvcc'")
        return cuda_sdk

    # check default paths
    if platform.system() == "Windows":
        cuda_paths = glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*.*")
        if cuda_paths:
            # Choose the newest installed toolkit
            def version_key(p: str) -> tuple[int, int]:
                base = os.path.basename(p)  # e.g., "v12.5"
                ver = base[1:].split(".")  # drop leading 'v'
                return (int(ver[0]), int(ver[1]) if len(ver) > 1 else 0)

            cuda_sdk = max(cuda_paths, key=version_key)
            print(f"Using CUDA Toolkit path '{cuda_sdk}' found at default path")
            return cuda_sdk
    else:
        usr_local_cuda = "/usr/local/cuda"
        if os.path.exists(usr_local_cuda):
            cuda_sdk = usr_local_cuda
            print(f"Using CUDA Toolkit path '{cuda_sdk}' found at default path")
            return cuda_sdk

    return None


def find_libmathdx(cuda_toolkit_major_version: int, base_path: str) -> str | None:
    libmathdx_path = os.environ.get("LIBMATHDX_HOME")

    if libmathdx_path:
        print(f"Using libmathdx path '{libmathdx_path}' provided through the 'LIBMATHDX_HOME' environment variable")
        return libmathdx_path

    # Fetch libmathdx from https://developer.nvidia.com/cublasdx-downloads using Packman
    if platform.system() == "Windows":
        packman = os.path.join(base_path, "tools", "packman", "packman.cmd")
    elif platform.system() == "Linux":
        packman = os.path.join(base_path, "tools", "packman", "packman")
    else:
        raise RuntimeError(f"Unsupported platform for libmathdx: {platform.system()}")

    try:
        output = subprocess.check_output(
            [
                packman,
                "pull",
                "--verbose",
                "--platform",
                f"{platform.system()}-{build_dll.machine_architecture()}".lower(),
                "--include-tag",
                f"cu{cuda_toolkit_major_version}",
                os.path.join(base_path, "deps", "libmathdx-deps.packman.xml"),
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Only print on verbose; caller controls this flag via build_dll.verbose_cmd
        if build_dll.verbose_cmd:
            print(output, end="")
    except subprocess.CalledProcessError as e:
        print(e.output)

        # Check if the libmathdx target directory exists and is not a symbolic link
        libmathdx_target_dir = os.path.join(base_path, "_build", "target-deps", "libmathdx")
        if os.path.exists(libmathdx_target_dir) and not os.path.islink(libmathdx_target_dir):
            print(f"\nError: {libmathdx_target_dir} exists and is not a symbolic link.")
            print("Please try deleting this folder and running the script again.")
        raise

    # Success
    return os.path.join(base_path, "_build", "target-deps", "libmathdx")


def lib_name(name: str) -> str:
    """Return platform-specific shared library name."""
    if platform.system() == "Windows":
        return f"{name}.dll"
    elif platform.system() == "Darwin":
        return f"lib{name}.dylib"
    else:
        return f"{name}.so"


def generate_exports_header_file(base_path: str) -> None:
    """Generates warp/native/exports.h, which lets built-in functions be callable from outside kernels."""
    export_path = os.path.join(base_path, "warp", "native", "exports.h")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    try:
        with open(export_path, "w") as f:
            copyright_notice = """/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

"""
            f.write(copyright_notice)
            export_builtins(f)

        print(f"Finished writing {export_path}")
    except FileNotFoundError:
        print(f"Error: The file '{export_path}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied. Unable to write to '{export_path}'.")
    except OSError as e:
        print(f"Error: An OS-related error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Warp build script")
    parser.add_argument("--msvc_path", type=str, help="Path to MSVC compiler (optional if already on PATH)")
    parser.add_argument("--sdk_path", type=str, help="Path to WinSDK (optional if already on PATH)")
    parser.add_argument("--cuda_path", type=str, help="Path to CUDA SDK")
    parser.add_argument("--libmathdx_path", type=str, help="Path to libmathdx (optional if LIBMATHDX_HOME is defined)")
    parser.add_argument(
        "--mode",
        type=str,
        default="release",
        help="Build configuration, default 'release'",
        choices=["release", "debug"],
    )

    parser.add_argument(
        "--clang_build_toolchain",
        action="store_true",
        help="(Linux only) Use Clang compiler for building both CPU and GPU code during library compilation (default: use host compiler and NVCC)",
    )
    parser.set_defaults(clang_build_toolchain=False)

    # Note argparse.BooleanOptionalAction can be used here when Python 3.9+ becomes the minimum supported version
    parser.add_argument("--verbose", action="store_true", help="Verbose building output, default enabled")
    parser.add_argument("--no_verbose", dest="verbose", action="store_false")
    parser.set_defaults(verbose=True)

    parser.add_argument(
        "--verify_fp",
        action="store_true",
        help="Verify kernel inputs and outputs are finite after each launch, default disabled",
    )
    parser.add_argument("--no_verify_fp", dest="verify_fp", action="store_false")
    parser.set_defaults(verify_fp=False)

    parser.add_argument("--fast_math", action="store_true", help="Enable fast math on library, default disabled")
    parser.add_argument("--no_fast_math", dest="fast_math", action="store_false")
    parser.set_defaults(fast_math=False)

    parser.add_argument("--quick", action="store_true", help="Only generate PTX code")
    parser.set_defaults(quick=False)

    parser.add_argument("-mp", "--multi_process", type=int, default=1, help="Number of concurrent build tasks.")

    group_clang_llvm = parser.add_argument_group("Clang/LLVM Options")
    group_clang_llvm.add_argument("--llvm_path", type=str, help="Path to an existing LLVM installation")
    group_clang_llvm.add_argument(
        "--build_llvm", action="store_true", help="Build Clang/LLVM compiler from source, default disabled"
    )
    group_clang_llvm.add_argument("--no_build_llvm", dest="build_llvm", action="store_false")
    group_clang_llvm.set_defaults(build_llvm=False)
    group_clang_llvm.add_argument(
        "--llvm_source_path", type=str, help="Path to the LLVM project source code (optional, repo cloned if not set)"
    )
    group_clang_llvm.add_argument(
        "--debug_llvm", action="store_true", help="Enable LLVM compiler code debugging, default disabled"
    )
    group_clang_llvm.add_argument("--no_debug_llvm", dest="debug_llvm", action="store_false")
    group_clang_llvm.set_defaults(debug_llvm=False)
    group_clang_llvm.add_argument(
        "--standalone", action="store_true", help="Use standalone LLVM-based JIT compiler, default enabled"
    )
    group_clang_llvm.add_argument("--no_standalone", dest="standalone", action="store_false")
    group_clang_llvm.set_defaults(standalone=True)

    parser.add_argument("--libmathdx", action="store_true", help="Build Warp with MathDx support, default enabled")
    parser.add_argument("--no_libmathdx", dest="libmathdx", action="store_false")
    parser.set_defaults(libmathdx=True)

    parser.add_argument(
        "--compile_time_trace",
        action="store_true",
        help="Output a 'build_warp_time_trace.json' trace file for the NVCC compilation process, default disabled",
    )

    args = parser.parse_args(argv)

    # resolve base paths
    base_path = os.path.dirname(os.path.realpath(__file__))
    build_path = os.path.join(base_path, "warp")

    if args.verbose:
        print(args)

    # propagate verbosity to build subsystem
    build_dll.verbose_cmd = args.verbose

    # setup CUDA Toolkit path
    if platform.system() == "Darwin":
        args.cuda_path = None
    else:
        if not args.cuda_path:
            args.cuda_path = find_cuda_sdk()

        # libmathdx needs to be used with a build of Warp that supports CUDA
        if args.libmathdx:
            if not args.libmathdx_path and args.cuda_path:
                major, _ = build_dll.get_cuda_toolkit_version(args.cuda_path)
                args.libmathdx_path = find_libmathdx(major, base_path)
        else:
            args.libmathdx_path = None

    # setup MSVC and WinSDK paths
    if platform.system() == "Windows":
        if args.msvc_path or args.sdk_path:
            # user provided MSVC and Windows SDK
            assert args.msvc_path and args.sdk_path, "--msvc_path and --sdk_path must be used together."
            args.host_compiler = build_dll.set_msvc_env(msvc_path=args.msvc_path, sdk_path=args.sdk_path)
        else:
            # attempt to find MSVC in environment (will set vcvars)
            args.host_compiler = build_dll.find_host_compiler()
            if not args.host_compiler:
                print("Warp build error: Could not find MSVC compiler")
                return 1

    try:
        # Generate warp/native/export.h
        generate_exports_header_file(base_path)

        # build warp.dll
        cpp_sources = [
            "native/warp.cpp",
            "native/crt.cpp",
            "native/error.cpp",
            "native/cuda_util.cpp",
            "native/mesh.cpp",
            "native/hashgrid.cpp",
            "native/reduce.cpp",
            "native/runlength_encode.cpp",
            "native/sort.cpp",
            "native/sparse.cpp",
            "native/volume.cpp",
            "native/mathdx.cpp",
            "native/coloring.cpp",
        ]
        warp_cpp_paths = [os.path.join(build_path, cpp) for cpp in cpp_sources]

        if args.cuda_path is None:
            print("Warning: CUDA toolchain not found, building without CUDA support")
            warp_cu_paths = None
        else:
            cuda_sources = [
                "native/bvh.cu",
                "native/mesh.cu",
                "native/sort.cu",
                "native/hashgrid.cu",
                "native/reduce.cu",
                "native/runlength_encode.cu",
                "native/scan.cu",
                "native/sparse.cu",
                "native/volume.cu",
                "native/volume_builder.cu",
                "native/warp.cu",
            ]
            warp_cu_paths = [os.path.join(build_path, cu) for cu in cuda_sources]

        if args.libmathdx and args.libmathdx_path is None:
            print("Warning: libmathdx not found, building without MathDx support")

        warp_dll_path = os.path.join(build_path, f"bin/{lib_name('warp')}")
        build_dll.build_dll(args, dll_path=warp_dll_path, cpp_paths=warp_cpp_paths, cu_paths=warp_cu_paths)

        # build warp-clang.dll
        if args.standalone:
            import build_llvm

            if args.build_llvm:
                build_llvm.build_llvm_clang_from_source(args)

            build_llvm.build_warp_clang(args, lib_name("warp-clang"))

    except Exception as e:
        print(f"Warp build error: {e}")
        return 1

    try:
        is_gitlab_ci = os.getenv("GITLAB_CI") is not None
        if not (is_gitlab_ci and platform.system() == "Windows"):
            # Clear kernel cache (also initializes Warp)
            clear_kernel_cache()
            clear_lto_cache()
        else:
            print("Skipping kernel cache clearing in GitLab CI on Windows")
    except Exception as e:
        print(f"Unable to clear kernel cache: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
