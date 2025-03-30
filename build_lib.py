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

import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys

from warp.build_dll import build_dll, find_host_compiler, machine_architecture, set_msvc_env, verbose_cmd
from warp.context import export_builtins

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

parser.add_argument("--build_llvm", action="store_true", help="Build Clang/LLVM compiler from source, default disabled")
parser.add_argument("--no_build_llvm", dest="build_llvm", action="store_false")
parser.set_defaults(build_llvm=False)

parser.add_argument(
    "--llvm_source_path", type=str, help="Path to the LLVM project source code (optional, repo cloned if not set)"
)

parser.add_argument("--debug_llvm", action="store_true", help="Enable LLVM compiler code debugging, default disabled")
parser.add_argument("--no_debug_llvm", dest="debug_llvm", action="store_false")
parser.set_defaults(debug_llvm=False)

parser.add_argument("--standalone", action="store_true", help="Use standalone LLVM-based JIT compiler, default enabled")
parser.add_argument("--no_standalone", dest="standalone", action="store_false")
parser.set_defaults(standalone=True)

parser.add_argument("--libmathdx", action="store_true", help="Build Warp with MathDx support, default enabled")
parser.add_argument("--no_libmathdx", dest="libmathdx", action="store_false")
parser.set_defaults(libmathdx=True)

args = parser.parse_args()

# set build output path off this file
base_path = os.path.dirname(os.path.realpath(__file__))
build_path = os.path.join(base_path, "warp")

print(args)

verbose_cmd = args.verbose


def find_cuda_sdk():
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
        cuda_paths = glob.glob("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*.*")
        if len(cuda_paths) >= 1:
            cuda_sdk = cuda_paths[0]
            print(f"Using CUDA Toolkit path '{cuda_sdk}' found at default path")
            return cuda_sdk

    else:
        usr_local_cuda = "/usr/local/cuda"
        if os.path.exists(usr_local_cuda):
            cuda_sdk = usr_local_cuda
            print(f"Using CUDA Toolkit path '{cuda_sdk}' found at default path")
            return cuda_sdk

    return None


def find_libmathdx():
    libmathdx_path = os.environ.get("LIBMATHDX_HOME")

    if libmathdx_path:
        print(f"Using libmathdx path '{libmathdx_path}' provided through the 'LIBMATHDX_HOME' environment variable")
        return libmathdx_path
    else:
        # Fetch libmathdx from https://developer.nvidia.com/cublasdx-downloads using Packman
        if platform.system() == "Windows":
            packman = "tools\\packman\\packman.cmd"
        elif platform.system() == "Linux":
            packman = "./tools/packman/packman"
        else:
            raise RuntimeError(f"Unsupported platform for libmathdx: {platform.system()}")

        try:
            subprocess.check_output(
                [
                    packman,
                    "pull",
                    "--verbose",
                    "--platform",
                    f"{platform.system()}-{machine_architecture()}".lower(),
                    os.path.join(base_path, "deps", "libmathdx-deps.packman.xml"),
                ],
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(e.output)

            # Check if the libmathdx target directory exists and is not a symbolic link
            libmathdx_target_dir = os.path.join(base_path, "_build", "target-deps", "libmathdx")
            if os.path.exists(libmathdx_target_dir) and not os.path.islink(libmathdx_target_dir):
                print(f"\nError: {libmathdx_target_dir} exists and is not a symbolic link.")
                print("Please try deleting this folder and running the script again.")
            raise e

        # Success
        return os.path.join(base_path, "_build", "target-deps", "libmathdx", "libmathdx")


# setup CUDA Toolkit path
if platform.system() == "Darwin":
    args.cuda_path = None
else:
    if not args.cuda_path:
        args.cuda_path = find_cuda_sdk()

    # libmathdx needs to be used with a build of Warp that supports CUDA
    if args.libmathdx:
        if not args.libmathdx_path and args.cuda_path:
            args.libmathdx_path = find_libmathdx()
    else:
        args.libmathdx_path = None


# setup MSVC and WinSDK paths
if platform.system() == "Windows":
    if args.msvc_path or args.sdk_path:
        # user provided MSVC and Windows SDK
        assert args.msvc_path and args.sdk_path, "--msvc_path and --sdk_path must be used together."

        args.host_compiler = set_msvc_env(msvc_path=args.msvc_path, sdk_path=args.sdk_path)
    else:
        # attempt to find MSVC in environment (will set vcvars)
        args.host_compiler = find_host_compiler()

        if not args.host_compiler:
            print("Warp build error: Could not find MSVC compiler")
            sys.exit(1)


# return platform specific shared library name
def lib_name(name):
    if platform.system() == "Windows":
        return f"{name}.dll"
    elif platform.system() == "Darwin":
        return f"lib{name}.dylib"
    else:
        return f"{name}.so"


def generate_exports_header_file():
    """Generates warp/native/exports.h, which lets built-in functions be callable from outside kernels"""

    # set build output path off this file
    export_path = os.path.join(base_path, "warp", "native", "exports.h")

    try:
        with open(export_path, "w") as f:
            # Add copyright notice using a triple-quoted string
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


try:
    # Generate warp/native/export.h
    generate_exports_header_file()

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
        "native/marching.cpp",
        "native/mathdx.cpp",
        "native/coloring.cpp",
    ]
    warp_cpp_paths = [os.path.join(build_path, cpp) for cpp in cpp_sources]

    if args.cuda_path is None:
        print("Warning: CUDA toolchain not found, building without CUDA support")
        warp_cu_path = None
    else:
        warp_cu_path = os.path.join(build_path, "native/warp.cu")

    if args.libmathdx and args.libmathdx_path is None:
        print("Warning: libmathdx not found, building without MathDx support")

    warp_dll_path = os.path.join(build_path, f"bin/{lib_name('warp')}")

    build_dll(args, dll_path=warp_dll_path, cpp_paths=warp_cpp_paths, cu_path=warp_cu_path)

    # build warp-clang.dll
    if args.standalone:
        import build_llvm

        if args.build_llvm:
            build_llvm.build_from_source(args)

        build_llvm.build_warp_clang(args, lib_name("warp-clang"))

except Exception as e:
    # output build error
    print(f"Warp build error: {e}")

    # report error
    sys.exit(1)
