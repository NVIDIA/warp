# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
# ]
# ///

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
import datetime
import glob
import os
import platform
import shutil
import subprocess
import sys

import build_llvm
import warp._src.build_dll as build_dll
import warp.config as config
from warp._src.context import export_builtins


def handle_ci_nightly_build(base_path: str) -> str | None:
    """Update version for nightly builds in scheduled CI pipeline.

    Returns:
        Updated version string if nightly build was triggered, None otherwise.
    """
    ci_pipeline_source = os.environ.get("CI_PIPELINE_SOURCE")
    if ci_pipeline_source != "schedule":
        return None

    print("Detected scheduled CI pipeline - updating version for nightly build")

    # Import CI publishing tools
    sys.path.insert(0, os.path.join(base_path, "tools", "ci", "publishing"))
    from set_nightly_version import (  # noqa: PLC0415
        increment_minor,
        write_new_version_to_config,
        write_new_version_to_version_file,
    )
    from update_git_hash import get_git_hash, update_git_hash_in_config  # noqa: PLC0415

    # Paths
    version_file = os.path.join(base_path, "VERSION.md")
    config_file = os.path.join(base_path, "warp", "config.py")

    # Read base version
    with open(version_file) as f:
        base_version = f.readline().strip()

    # Generate nightly version
    if "dev" in base_version:
        dev_index = base_version.find("dev")
        base_version_incremented = base_version[:dev_index].rstrip(".")
    else:
        base_version_incremented = increment_minor(base_version)

    dateint = datetime.date.today().strftime("%Y%m%d")
    dev_version_string = f"{base_version_incremented}.dev{dateint}"

    # Update files
    write_new_version_to_version_file(version_file, dev_version_string, dry_run=False)
    write_new_version_to_config(config_file, dev_version_string, dry_run=False)

    # Update git hash
    git_hash = get_git_hash()
    if git_hash:
        update_git_hash_in_config(config_file, git_hash, dry_run=False)

    return dev_version_string


def generate_version_header(base_path: str, version: str) -> None:
    """Generate version.h with WP_VERSION_STRING macro."""
    version_header_path = os.path.join(base_path, "warp", "native", "version.h")
    current_year = datetime.date.today().year

    copyright_notice = f"""/*
 * SPDX-FileCopyrightText: Copyright (c) {current_year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    with open(version_header_path, "w") as f:
        f.write(copyright_notice)
        f.write("#ifndef WP_VERSION_H\n")
        f.write("#define WP_VERSION_H\n\n")
        f.write(f'#define WP_VERSION_STRING "{version}"\n\n')
        f.write("#endif  // WP_VERSION_H\n")

    print(f"Generated {version_header_path} with version {version}")


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


def validate_libmathdx_path(libmathdx_path: str) -> bool:
    """Validate that libmathdx path exists and has required directory structure.

    Args:
        libmathdx_path: Path to libmathdx installation to validate.

    Returns:
        True if valid, False otherwise (with error message printed).
    """
    if not os.path.isdir(libmathdx_path):
        print(f"Error: libmathdx path does not exist or is not a directory: {libmathdx_path}")
        return False

    # Check for required subdirectories
    libmathdx_lib_subdir = "lib/x64" if platform.system() == "Windows" else "lib"
    required_dirs = {
        "include": os.path.join(libmathdx_path, "include"),
        libmathdx_lib_subdir: os.path.join(libmathdx_path, libmathdx_lib_subdir),
    }

    for name, path in required_dirs.items():
        if not os.path.isdir(path):
            print(f"Error: libmathdx installation is missing '{name}' directory: {path}")
            return False

    return True


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
    parser = argparse.ArgumentParser(
        description="Build Warp native libraries with optional CUDA, LLVM, and MathDx support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General options
    parser.add_argument(
        "--mode",
        type=str,
        choices=["release", "debug"],
        default="release",
        help="Build configuration mode",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="Number of concurrent build tasks",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable verbose build output",
    )
    parser.add_argument(
        "--compile-time-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate compilation profiling trace file 'build_warp_time_trace.json' (does not affect output binary)",
    )

    # Toolchain paths
    group_toolchain = parser.add_argument_group("Toolchain Paths")
    group_toolchain.add_argument(
        "--msvc-path",
        type=str,
        help="Path to MSVC compiler (Windows only, optional if on PATH)",
    )
    group_toolchain.add_argument(
        "--sdk-path",
        type=str,
        help="Path to Windows SDK (Windows only, optional if on PATH)",
    )
    group_toolchain.add_argument(
        "--cuda-path",
        type=str,
        help="Path to CUDA Toolkit installation (auto-detected via WARP_CUDA_PATH, CUDA_HOME, CUDA_PATH, or nvcc)",
    )
    group_toolchain.add_argument(
        "--libmathdx-path",
        type=str,
        help="Path to NVIDIA libmathdx installation (optional if LIBMATHDX_HOME is set)",
    )

    # Build options
    group_build = parser.add_argument_group("Build Options")
    group_build.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build with CUDA support (auto-detects CUDA Toolkit). Use --no-cuda for a CPU-only build",
    )
    group_build.add_argument(
        "--clang-build-toolchain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Clang for both CPU and GPU compilation (Linux only, experimental)",
    )
    group_build.add_argument(
        "--use-libmathdx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build with NVIDIA libmathdx (includes cuBLASDx/cuFFTDx/cuSOLVERDx) for tile operations: matrix multiplication, FFT, and linear solvers",
    )
    group_build.add_argument(
        "--verify-fp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Verify floating-point values are finite after each kernel launch",
    )
    group_build.add_argument(
        "--fast-math",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable fast math optimizations (may reduce numerical accuracy)",
    )
    group_build.add_argument(
        "--quick",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fast build mode: compile for minimal GPU architectures (PTX-only for sm_75), disable CUDA forward compatibility",
    )

    # Clang/LLVM options
    group_clang_llvm = parser.add_argument_group(
        "Clang/LLVM Options",
        "Options for building LLVM compiler support (used for CPU kernels, optionally for GPU via runtime config)",
    )
    group_clang_llvm.add_argument(
        "--standalone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build warp-clang library for CPU kernel compilation (disabling makes only CUDA devices available)",
    )
    group_clang_llvm.add_argument(
        "--llvm-path",
        type=str,
        help="Path to existing LLVM installation (used for warp-clang library and adds bin to PATH for --clang-build-toolchain)",
    )
    group_clang_llvm.add_argument(
        "--build-llvm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build Clang/LLVM from source (takes ~60 minutes)",
    )
    group_clang_llvm.add_argument(
        "--llvm-source-path",
        type=str,
        help="Path to LLVM source code for building (only used with --build-llvm; defaults to external/llvm-project submodule)",
    )
    group_clang_llvm.add_argument(
        "--debug-llvm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build LLVM with debug symbols and assertions enabled",
    )

    args = parser.parse_args(argv)

    # Validate mutually exclusive LLVM options
    if args.llvm_path and args.build_llvm:
        print("Error: --llvm-path and --build-llvm are mutually exclusive.")
        print("  Use --llvm-path to use an existing LLVM installation")
        print("  Use --build-llvm to build LLVM from source")
        return 1

    # Validate --no-cuda conflicts
    if not args.cuda:
        if args.cuda_path:
            print("Error: --no-cuda and --cuda-path are mutually exclusive.")
            return 1
        if args.clang_build_toolchain:
            print("Error: --clang-build-toolchain requires CUDA (incompatible with --no-cuda).")
            return 1

    # Warn if building on Intel Mac (cross-compiling for ARM64)
    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        print("=" * 80)
        print("WARNING: Building Warp on Intel-based macOS")
        print("=" * 80)
        print("You are building Warp for ARM64 (Apple Silicon) on an Intel Mac.")
        print("The resulting binaries will NOT run on this machine.")
        print()
        print("Intel-based macOS is no longer supported for running Warp.")
        print("Use Warp 1.9.x or earlier if you need to run Warp on Intel Mac.")
        print("=" * 80)
        print()

    # resolve base paths
    base_path = os.path.dirname(os.path.realpath(__file__))
    build_path = os.path.join(base_path, "warp")

    if args.verbose:
        print(args)

    # propagate verbosity to build subsystem
    build_dll.verbose_cmd = args.verbose

    # check LLVM build dependencies early if --build-llvm is set
    if args.build_llvm:
        try:
            build_llvm.check_build_dependencies(verbose=args.verbose)
        except RuntimeError as e:
            print(f"Warp build error: {e}")
            return 1

    # setup CUDA Toolkit path
    if platform.system() == "Darwin" or not args.cuda:
        if not args.cuda:
            print("CUDA support disabled (--no-cuda)")
        args.cuda_path = None
        args.libmathdx_path = None
    else:
        if not args.cuda_path:
            args.cuda_path = find_cuda_sdk()

        # libmathdx needs to be used with a build of Warp that supports CUDA
        if args.use_libmathdx:
            if not args.libmathdx_path and args.cuda_path:
                major, _ = build_dll.get_cuda_toolkit_version(args.cuda_path)
                args.libmathdx_path = find_libmathdx(major, base_path)
        else:
            args.libmathdx_path = None

    # Validate libmathdx path (from any source: CLI, environment, or Packman)
    if args.libmathdx_path:
        if not validate_libmathdx_path(args.libmathdx_path):
            return 1

    # setup MSVC and WinSDK paths
    if platform.system() == "Windows":
        if args.msvc_path or args.sdk_path:
            # user provided MSVC and Windows SDK
            if not (args.msvc_path and args.sdk_path):
                print("Error: --msvc-path and --sdk-path must be used together")
                return 1
            args.host_compiler = build_dll.set_msvc_env(msvc_path=args.msvc_path, sdk_path=args.sdk_path)
        else:
            # attempt to find MSVC in environment (will set vcvars)
            args.host_compiler = build_dll.find_host_compiler()
            if not args.host_compiler:
                print("Warp build error: Could not find MSVC compiler")
                return 1
    else:
        args.host_compiler = build_dll.find_host_compiler()
        if not args.host_compiler:
            print("Warp build error: Could not find C++ compiler")
            return 1

    try:
        # Handle CI nightly builds (returns updated version string if triggered, else None)
        nightly_version = handle_ci_nightly_build(base_path)

        if nightly_version is not None:
            build_version = nightly_version
        else:
            build_version = config.version

        if args.verbose:
            print(f"Building Warp version {build_version}")

        # Generate warp/native/export.h
        generate_exports_header_file(base_path)

        # Generate warp/native/version.h
        generate_version_header(base_path, build_version)

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
            "native/texture.cpp",
            "native/mathdx.cpp",
            "native/coloring.cpp",
        ]
        warp_cpp_paths = [os.path.join(build_path, cpp) for cpp in cpp_sources]

        if args.cuda_path is None:
            if args.cuda:
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

            # libmathdx is only needed when building with CUDA
            if args.use_libmathdx and args.libmathdx_path is None:
                print("Error: libmathdx not found. MathDx support is enabled but libmathdx could not be located.")
                print("  Either:")
                print("    - Install libmathdx and set LIBMATHDX_HOME environment variable")
                print("    - Use --libmathdx-path to specify the installation path")
                print("    - Use --no-use-libmathdx to build without MathDx support")
                return 1

        warp_dll_path = os.path.join(build_path, f"bin/{lib_name('warp')}")
        build_dll.build_dll(args, dll_path=warp_dll_path, cpp_paths=warp_cpp_paths, cu_paths=warp_cu_paths)

        # build warp-clang.dll
        if args.standalone:
            if args.build_llvm:
                build_llvm.build_llvm_clang_from_source(args)

            build_llvm.build_warp_clang(args, lib_name("warp-clang"))

    except Exception as e:
        print(f"Warp build error: {e}")
        return 1

    try:
        is_gitlab_ci_windows = os.getenv("GITLAB_CI") is not None and platform.system() == "Windows"
        is_intel_mac = platform.system() == "Darwin" and platform.machine() == "x86_64"

        if is_gitlab_ci_windows or is_intel_mac:
            if is_gitlab_ci_windows:
                print("Skipping kernel cache clearing in GitLab CI on Windows")
            if is_intel_mac:
                print("Skipping kernel cache clearing on Intel Mac (binaries built for ARM64)")
        else:
            # Clear kernel cache in subprocess (ensures fresh import of updated config.py)
            print("Clearing kernel cache...")
            sys.stdout.flush()
            sys.stderr.flush()
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from warp._src.build import clear_kernel_cache, clear_lto_cache; clear_kernel_cache(); clear_lto_cache()",
                ],
                cwd=base_path,
                check=False,
            )
            if result.returncode != 0:
                print(f"Warning: Failed to clear kernel cache (exit code {result.returncode})")

            # Flush build output before printing diagnostics so log ordering is correct
            sys.stdout.flush()
            sys.stderr.flush()

            # Print build diagnostics (subprocess ensures fresh import of rebuilt libraries)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import warp; warp.print_diagnostics()",
                ],
                cwd=base_path,
                check=False,
            )
    except Exception as e:
        print(f"Unable to clear kernel cache: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
