# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import concurrent.futures
import functools
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import time

from warp._src.utils import ScopedTimer

_wp_module_name_ = "warp.build_dll"

verbose_cmd = True  # print command lines before executing them

MIN_CTK_VERSION = (12, 0)


def machine_architecture() -> str:
    """Return a canonical machine architecture string.
    - "x86_64" for x86-64, aka. AMD64, aka. x64
    - "aarch64" for AArch64, aka. ARM64
    """
    machine = platform.machine()
    if machine == "x86_64" or machine == "AMD64":
        return "x86_64"
    if machine == "aarch64" or machine == "arm64":
        return "aarch64"
    raise RuntimeError(f"Unrecognized machine architecture {machine}")


def run_cmd(cmd):
    if verbose_cmd:
        print(cmd)

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        # Print output even on success to show warnings
        if output:
            decoded_output = output.decode()
            if decoded_output.strip():  # Only print if not just whitespace
                # In parallel builds, associate output with its command for clarity
                # Use single print to avoid interleaving with other processes
                print(f"Output from: {cmd}\n{decoded_output}")
        return output
    except subprocess.CalledProcessError as e:
        # Single print to avoid interleaving in parallel builds
        print(f"Command failed with exit code {e.returncode}: {cmd}\nCommand output was:\n{e.output.decode()}")
        raise e


# cut-down version of vcvars64.bat that allows using
# custom toolchain locations, returns the compiler program path
def set_msvc_env(msvc_path, sdk_path):
    if "INCLUDE" not in os.environ:
        os.environ["INCLUDE"] = ""

    if "LIB" not in os.environ:
        os.environ["LIB"] = ""

    msvc_path = os.path.abspath(msvc_path)
    sdk_path = os.path.abspath(sdk_path)

    os.environ["INCLUDE"] += os.pathsep + os.path.join(msvc_path, "include")
    os.environ["INCLUDE"] += os.pathsep + os.path.join(sdk_path, "include/winrt")
    os.environ["INCLUDE"] += os.pathsep + os.path.join(sdk_path, "include/um")
    os.environ["INCLUDE"] += os.pathsep + os.path.join(sdk_path, "include/ucrt")
    os.environ["INCLUDE"] += os.pathsep + os.path.join(sdk_path, "include/shared")

    os.environ["LIB"] += os.pathsep + os.path.join(msvc_path, "lib/x64")
    os.environ["LIB"] += os.pathsep + os.path.join(sdk_path, "lib/ucrt/x64")
    os.environ["LIB"] += os.pathsep + os.path.join(sdk_path, "lib/um/x64")

    os.environ["PATH"] += os.pathsep + os.path.join(msvc_path, "bin/HostX64/x64")
    os.environ["PATH"] += os.pathsep + os.path.join(sdk_path, "bin/x64")

    return os.path.join(msvc_path, "bin", "HostX64", "x64", "cl.exe")


def find_host_compiler() -> str:
    """Find the host C++ compiler.

    On Windows, checks for pre-configured Visual Studio environment before
    attempting auto-configuration. On Unix/Linux, respects $CXX environment
    variable if set.

    Returns:
        Path to compiler executable, or empty string if not found (Windows only).
        Note: Empty string return allows build_lib.py to handle error gracefully.
    """
    if os.name == "nt":
        # Check if Visual Studio environment already configured (conda, Docker, vcvars64, etc.)
        # VCINSTALLDIR and VCToolsVersion are set by vcvars64.bat
        if os.environ.get("VCINSTALLDIR") or os.environ.get("VCToolsVersion"):
            if verbose_cmd:
                print("Visual Studio environment already configured, skipping vcvars64.bat")

            cl_path = shutil.which("cl.exe")
            if cl_path:
                if verbose_cmd:
                    print(f"Using cl.exe from pre-configured environment: {cl_path}")
                return cl_path
            else:
                # Fall through to auto-configuration if cl.exe not actually available
                if verbose_cmd:
                    print("Warning: VS environment variables set but cl.exe not found, attempting auto-configuration")

        vswhere_path = r"%ProgramFiles(x86)%/Microsoft Visual Studio/Installer/vswhere.exe"
        vswhere_path = os.path.expandvars(vswhere_path)
        if not os.path.exists(vswhere_path):
            return ""  # Signal to caller that VS not found

        vs_path = run_cmd(f'"{vswhere_path}" -latest -property installationPath').decode().rstrip()
        vsvars_path = os.path.join(vs_path, "VC\\Auxiliary\\Build\\vcvars64.bat")

        if not os.path.exists(vsvars_path):
            return ""  # Signal to caller that VS environment script not found

        output = run_cmd(f'"{vsvars_path}" && set').decode()

        for line in output.splitlines():
            pair = line.split("=", 1)
            if len(pair) >= 2:
                os.environ[pair[0]] = pair[1]

        cl_path = shutil.which("cl.exe")
        cl_version = os.environ["VCToolsVersion"].split(".")

        # ensure at least VS2019 version, see list of MSVC versions here https://en.wikipedia.org/wiki/Microsoft_Visual_C%2B%2B
        cl_required_major = 14
        cl_required_minor = 29

        if int(cl_version[0]) < cl_required_major or (
            (int(cl_version[0]) == cl_required_major) and (int(cl_version[1]) < cl_required_minor)
        ):
            print(
                f"Warp: MSVC found but compiler version too old, found {cl_version[0]}.{cl_version[1]}, but must be {cl_required_major}.{cl_required_minor} or higher, kernel host compilation will be disabled."
            )
            return ""  # Signal to caller that version too old

        return cl_path

    else:
        # Respect $CXX environment variable (conda, cross-compilation, custom compilers, etc.)
        cxx = os.environ.get("CXX", "").strip()
        if cxx:
            if verbose_cmd:
                print(f"Using C++ compiler from $CXX: {cxx}")
            return cxx

        gxx_path = shutil.which("g++")
        if gxx_path:
            if verbose_cmd:
                print(f"Using g++ found in PATH: {gxx_path}")
            return gxx_path
        else:
            if verbose_cmd:
                print("Warning: Could not locate g++, falling back to 'g++'")
            return "g++"


def get_cuda_toolkit_version(cuda_home) -> tuple[int, int]:
    try:
        # Get nvcc command using intelligent discovery
        nvcc_cmd = find_nvcc_executable(cuda_home)

        # Remove quotes for subprocess call (subprocess handles paths directly)
        nvcc_executable = nvcc_cmd.strip('"')
        if not nvcc_executable:
            raise ValueError("nvcc command is empty")

        nvcc_version_output = subprocess.check_output([nvcc_executable, "--version"]).decode("utf-8")

        # search for release substring (e.g., "release 11.5")
        m = re.search(r"release (\d+)\.(\d+)", nvcc_version_output)
        if m is not None:
            major, minor = map(int, m.groups())
            return (major, minor)
        else:
            raise Exception("Failed to parse NVCC output")

    except Exception as e:
        print(f"Warning: Failed to determine CUDA Toolkit version: {e}")
        return MIN_CTK_VERSION


@functools.lru_cache(maxsize=1)  # Avoid duplicate verbose output when called multiple times per build
def find_nvcc_executable(cuda_home) -> str:
    """Find nvcc executable, maintaining consistency with cuda_home.

    Detection order prioritizes consistency between compiler and headers/libs:
    1. If cuda_home is set → use its nvcc (ensures consistency with headers/libs)
    2. Otherwise check PATH → use nvcc from PATH (conda, modules, containers)
    3. Fall back to assuming nvcc in PATH

    This ensures the nvcc compiler version matches the CUDA headers/libraries,
    preventing compilation failures from version mismatches.

    Args:
        cuda_home: Path to CUDA installation (may be None). Can come from
            CUDA_HOME env var, --cuda_path argument, or auto-detection.

    Returns:
        String command to invoke nvcc (either "nvcc" or quoted full path)
    """
    # Determine correct executable name for platform
    nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"

    # First priority: If cuda_home is provided, use its nvcc
    # This handles CUDA_HOME env var, --cuda_path arg, and ensures consistency
    if cuda_home:
        nvcc_path = os.path.join(cuda_home, "bin", nvcc_name)
        if os.path.exists(nvcc_path):
            if verbose_cmd:
                print(f"Using nvcc from cuda_home: {nvcc_path}")
            return f'"{nvcc_path}"'

    # Second priority: nvcc in PATH (conda, modules, containers)
    # Note: shutil.which() automatically handles .exe extension on Windows
    nvcc_in_path = shutil.which("nvcc")
    if verbose_cmd:
        if nvcc_in_path:
            print(f"Using nvcc from PATH: {nvcc_in_path}")
        else:
            print("Warning: nvcc not found in PATH or cuda_home, compilation will likely fail")

    return nvcc_name


def quote(path):
    return '"' + path + '"'


def add_llvm_bin_to_path(args):
    """Add the LLVM bin directory to the PATH environment variable if it's set.

    Args:
        args: The argument namespace containing llvm_path.

    Returns:
        ``True`` if the PATH was updated, ``False`` otherwise.
    """
    if not hasattr(args, "llvm_path") or not args.llvm_path:
        return False

    # Construct the bin directory path
    llvm_bin_path = os.path.join(args.llvm_path, "bin")

    # Check if the directory exists
    if not os.path.isdir(llvm_bin_path):
        print(f"Warning: LLVM bin directory not found at {llvm_bin_path}")
        return False

    # Add to PATH environment variable
    os.environ["PATH"] = llvm_bin_path + os.pathsep + os.environ.get("PATH", "")

    print(f"Added {llvm_bin_path} to PATH")
    return True


def _get_architectures_cu12(
    ctk_version: tuple[int, int], arch: str, target_platform: str, quick_build: bool = False
) -> tuple[list[str], list[str]]:
    """Get architecture flags for CUDA 12.x."""
    gencode_opts = []
    clang_arch_flags = []

    if quick_build:
        gencode_opts = ["-gencode=arch=compute_52,code=compute_52", "-gencode=arch=compute_75,code=compute_75"]
        clang_arch_flags = ["--cuda-gpu-arch=sm_52", "--cuda-gpu-arch=sm_75"]
    else:
        if arch == "aarch64" and target_platform == "linux" and ctk_version == (12, 9):
            # Skip certain architectures for aarch64 with CUDA 12.9 due to CCCL bug
            print(
                "[INFO] Skipping sm_52, sm_60, sm_61, and sm_70 targets for ARM due to a CUDA Toolkit bug. "
                "See https://nvidia.github.io/warp/installation.html#cuda-12-9-limitation-on-linux-arm-platforms "
                "for details."
            )
        else:
            gencode_opts.extend(
                [
                    "-gencode=arch=compute_52,code=sm_52",  # Maxwell
                    "-gencode=arch=compute_60,code=sm_60",  # Pascal
                    "-gencode=arch=compute_61,code=sm_61",
                    "-gencode=arch=compute_70,code=sm_70",  # Volta
                ]
            )
            clang_arch_flags.extend(
                [
                    "--cuda-gpu-arch=sm_52",
                    "--cuda-gpu-arch=sm_60",
                    "--cuda-gpu-arch=sm_61",
                    "--cuda-gpu-arch=sm_70",
                ]
            )

        # Desktop architectures
        gencode_opts.extend(
            [
                "-gencode=arch=compute_75,code=sm_75",  # Turing
                "-gencode=arch=compute_75,code=compute_75",  # Turing (PTX)
                "-gencode=arch=compute_80,code=sm_80",  # Ampere
                "-gencode=arch=compute_86,code=sm_86",
                "-gencode=arch=compute_89,code=sm_89",  # Ada
                "-gencode=arch=compute_90,code=sm_90",  # Hopper
            ]
        )
        clang_arch_flags.extend(
            [
                "--cuda-gpu-arch=sm_75",  # Turing
                "--cuda-gpu-arch=sm_80",  # Ampere
                "--cuda-gpu-arch=sm_86",
                "--cuda-gpu-arch=sm_89",  # Ada
                "--cuda-gpu-arch=sm_90",  # Hopper
            ]
        )

        if ctk_version >= (12, 8):
            gencode_opts.extend(["-gencode=arch=compute_100,code=sm_100", "-gencode=arch=compute_120,code=sm_120"])
            clang_arch_flags.extend(["--cuda-gpu-arch=sm_100", "--cuda-gpu-arch=sm_120"])

        # Mobile architectures for aarch64 Linux
        if arch == "aarch64" and target_platform == "linux":
            gencode_opts.extend(
                [
                    "-gencode=arch=compute_87,code=sm_87",  # Orin
                    "-gencode=arch=compute_53,code=sm_53",  # X1
                    "-gencode=arch=compute_62,code=sm_62",  # X2
                    "-gencode=arch=compute_72,code=sm_72",  # Xavier
                ]
            )
            clang_arch_flags.extend(
                [
                    "--cuda-gpu-arch=sm_87",
                    "--cuda-gpu-arch=sm_53",
                    "--cuda-gpu-arch=sm_62",
                    "--cuda-gpu-arch=sm_72",
                ]
            )

            # Thor support in CUDA 12.8+
            if ctk_version >= (12, 8):
                gencode_opts.append("-gencode=arch=compute_101,code=sm_101")  # Thor (CUDA 12 numbering)
                clang_arch_flags.append("--cuda-gpu-arch=sm_101")

            if ctk_version >= (12, 9):
                gencode_opts.append("-gencode=arch=compute_121,code=sm_121")
                clang_arch_flags.append("--cuda-gpu-arch=sm_121")

        # PTX for future hardware (use highest available compute capability)
        if ctk_version >= (12, 9):
            gencode_opts.extend(["-gencode=arch=compute_121,code=compute_121"])
        elif ctk_version >= (12, 8):
            gencode_opts.extend(["-gencode=arch=compute_120,code=compute_120"])
        else:
            gencode_opts.append("-gencode=arch=compute_90,code=compute_90")

    return gencode_opts, clang_arch_flags


def _get_architectures_cu13(
    ctk_version: tuple[int, int], arch: str, target_platform: str, quick_build: bool = False
) -> tuple[list[str], list[str]]:
    """Get architecture flags for CUDA 13.x."""
    gencode_opts = []
    clang_arch_flags = []

    if quick_build:
        gencode_opts = ["-gencode=arch=compute_75,code=compute_75"]
        clang_arch_flags = ["--cuda-gpu-arch=sm_75"]
    else:
        # Desktop architectures
        gencode_opts.extend(
            [
                "-gencode=arch=compute_75,code=sm_75",  # Turing
                "-gencode=arch=compute_75,code=compute_75",  # Turing (PTX)
                "-gencode=arch=compute_80,code=sm_80",  # Ampere
                "-gencode=arch=compute_86,code=sm_86",
                "-gencode=arch=compute_89,code=sm_89",  # Ada
                "-gencode=arch=compute_90,code=sm_90",  # Hopper
                "-gencode=arch=compute_100,code=sm_100",  # Blackwell
                "-gencode=arch=compute_120,code=sm_120",  # Blackwell
            ]
        )
        clang_arch_flags.extend(
            [
                "--cuda-gpu-arch=sm_75",  # Turing
                "--cuda-gpu-arch=sm_80",  # Ampere
                "--cuda-gpu-arch=sm_86",
                "--cuda-gpu-arch=sm_89",  # Ada
                "--cuda-gpu-arch=sm_90",  # Hopper
                "--cuda-gpu-arch=sm_100",  # Blackwell
                "--cuda-gpu-arch=sm_120",  # Blackwell
            ]
        )

        # Mobile architectures for aarch64 Linux
        if arch == "aarch64" and target_platform == "linux":
            gencode_opts.extend(
                [
                    "-gencode=arch=compute_87,code=sm_87",  # Orin
                    "-gencode=arch=compute_110,code=sm_110",  # Thor
                    "-gencode=arch=compute_121,code=sm_121",  # Spark
                ]
            )
            clang_arch_flags.extend(
                [
                    "--cuda-gpu-arch=sm_87",
                    "--cuda-gpu-arch=sm_110",
                    "--cuda-gpu-arch=sm_121",
                ]
            )

        # PTX for future hardware (use highest available compute capability)
        gencode_opts.extend(["-gencode=arch=compute_121,code=compute_121"])

    return gencode_opts, clang_arch_flags


def build_dll_for_arch(args, dll_path, cpp_paths, cu_paths, arch, libs: list[str] | None = None, mode=None):
    mode = args.mode if (mode is None) else mode
    cuda_home = args.cuda_path
    cuda_cmd = None

    # Add LLVM bin directory to PATH
    add_llvm_bin_to_path(args)

    if args.quick or cu_paths is None:
        cuda_compat_enabled = "WP_ENABLE_CUDA_COMPATIBILITY=0"
    else:
        cuda_compat_enabled = "WP_ENABLE_CUDA_COMPATIBILITY=1"

    if libs is None:
        libs = []

    warp_home_path = pathlib.Path(__file__).parent.parent
    warp_home = warp_home_path.resolve()

    if args.verbose:
        print(f"Building {dll_path}")

    native_dir = os.path.join(warp_home, "native")

    if cu_paths:
        # check CUDA Toolkit version
        ctk_version = get_cuda_toolkit_version(cuda_home)
        if ctk_version < MIN_CTK_VERSION:
            raise Exception(
                f"CUDA Toolkit version {MIN_CTK_VERSION[0]}.{MIN_CTK_VERSION[1]}+ is required (found {ctk_version[0]}.{ctk_version[1]} in {cuda_home})"
            )

        # Get architecture flags based on CUDA version
        if ctk_version >= (13, 0):
            gencode_opts, clang_arch_flags = _get_architectures_cu13(ctk_version, arch, sys.platform, args.quick)
        else:
            gencode_opts, clang_arch_flags = _get_architectures_cu12(ctk_version, arch, sys.platform, args.quick)

        nvcc_opts = [
            *gencode_opts,
            "-t0",  # multithreaded compilation
            "--extended-lambda",
            "-diag-suppress=221",  # suppress "floating-point value does not fit" warning from INFINITY macro in CUDA headers
        ]

        # Clang options
        clang_opts = [
            *clang_arch_flags,
            "-std=c++17",
            "-xcuda",
            f'--cuda-path="{cuda_home}"',
        ]

        # CUDA 13+ moved CUB into CCCL directory structure
        if ctk_version >= (13, 0):
            clang_opts.append(f'-I"{cuda_home}/include/cccl"')
            # CCCL has #pragma unroll directives that Clang can't always satisfy
            # Suppress optimizer warnings to avoid -Werror failures
            clang_opts.append("-Wno-pass-failed")

        if args.compile_time_trace:
            if ctk_version >= (12, 8):
                nvcc_opts.append("--fdevice-time-trace=_build/build_lib_@filename@_compile-time-trace")
            else:
                print("Warp warning: CUDA version is less than 12.8, compile_time_trace is not supported")

        if args.fast_math:
            nvcc_opts.append("--use_fast_math")

        # Get nvcc executable (checks PATH first, then CUDA_HOME)
        nvcc_cmd = find_nvcc_executable(cuda_home)

    # is the library being built with CUDA enabled?
    cuda_enabled = "WP_ENABLE_CUDA=1" if (cu_paths is not None) else "WP_ENABLE_CUDA=0"

    if args.libmathdx_path:
        libmathdx_includes = f' -I"{args.libmathdx_path}/include"'
        mathdx_enabled = "WP_ENABLE_MATHDX=1"
    else:
        libmathdx_includes = ""
        mathdx_enabled = "WP_ENABLE_MATHDX=0"

    if os.name == "nt":
        if args.host_compiler:
            host_linker = os.path.join(os.path.dirname(args.host_compiler), "link.exe")
        else:
            raise RuntimeError("Warp build error: No host compiler was found")

        cpp_includes = f' /I"{warp_home_path.parent}/external/llvm-project/out/install/{mode}-{arch}/include"'
        cpp_includes += f' /I"{warp_home_path.parent}/_build/host-deps/llvm-project/release-{arch}/include"'
        cuda_includes = f' /I"{cuda_home}/include"' if cu_paths else ""
        includes = cpp_includes + cuda_includes

        # nvrtc_static.lib is built with /MT and _ITERATOR_DEBUG_LEVEL=0 so if we link it in we must match these options
        if cu_paths or mode != "debug":
            runtime = "/MT"
            iter_dbg = "_ITERATOR_DEBUG_LEVEL=0"
            debug = "NDEBUG"
        else:
            runtime = "/MTd"
            iter_dbg = "_ITERATOR_DEBUG_LEVEL=2"
            debug = "_DEBUG"

        cpp_flags = f'/nologo /std:c++17 /GR- /EHsc {runtime} /D "{debug}" /D "{cuda_enabled}" /D "{mathdx_enabled}" /D "{cuda_compat_enabled}" /D "{iter_dbg}" /I"{native_dir}" {includes} '

        if args.mode == "debug":
            cpp_flags += "/FS /Zi /Od /D WP_ENABLE_DEBUG=1"
            linkopts = ["/DLL", "/DEBUG"]
        elif args.mode == "release":
            cpp_flags += "/Ox /D WP_ENABLE_DEBUG=0"
            linkopts = ["/DLL"]
        else:
            raise RuntimeError(f"Unrecognized build configuration (debug, release), got: {args.mode}")

        if args.verify_fp:
            cpp_flags += ' /D "WP_VERIFY_FP"'

        if args.fast_math:
            cpp_flags += " /fp:fast"

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures, wall_clock = [], time.perf_counter_ns()

            cpp_cmds = []
            for cpp_path in cpp_paths:
                cpp_out = cpp_path + ".obj"
                linkopts.append(quote(cpp_out))
                # Add warning suppressions for clang.cpp to avoid LLVM header warnings
                extra_flags = ""
                if "clang/clang.cpp" in cpp_path.replace("\\", "/"):
                    extra_flags = " /wd4624"  # suppress C4624: destructor was implicitly defined as deleted
                cpp_cmd = f'"{args.host_compiler}" {cpp_flags}{extra_flags} -c "{cpp_path}" /Fo"{cpp_out}"'
                cpp_cmds.append(cpp_cmd)

            if args.jobs <= 1:
                with ScopedTimer("build", active=args.verbose):
                    for cpp_cmd in cpp_cmds:
                        run_cmd(cpp_cmd)
            else:
                futures = [executor.submit(run_cmd, cmd=cpp_cmd) for cpp_cmd in cpp_cmds]

            cuda_cmds = []
            if cu_paths:
                for cu_path in cu_paths:
                    cu_out = cu_path + ".o"

                    _nvcc_opts = [
                        opt.replace("@filename@", os.path.basename(cu_path).replace(".", "_")) for opt in nvcc_opts
                    ]

                    if mode == "debug":
                        cuda_cmd = f'{nvcc_cmd} --std=c++17 --compiler-options=/MT,/Zi,/Od -g -G -O0 -DNDEBUG -D_ITERATOR_DEBUG_LEVEL=0 -I"{native_dir}" -line-info {" ".join(_nvcc_opts)} -DWP_ENABLE_CUDA=1 -D{mathdx_enabled} {libmathdx_includes} -o "{cu_out}" -c "{cu_path}"'
                    elif mode == "release":
                        cuda_cmd = f'{nvcc_cmd} --std=c++17 -O3 {" ".join(_nvcc_opts)} -I"{native_dir}" -DNDEBUG -DWP_ENABLE_CUDA=1 -D{mathdx_enabled} {libmathdx_includes} -o "{cu_out}" -c "{cu_path}"'

                    cuda_cmds.append(cuda_cmd)

                    linkopts.append(quote(cu_out))

                linkopts.append(
                    f'cudart_static.lib nvrtc_static.lib nvrtc-builtins_static.lib nvptxcompiler_static.lib ws2_32.lib user32.lib /LIBPATH:"{cuda_home}/lib/x64"'
                )

                if args.libmathdx_path:
                    linkopts.append(f'nvJitLink_static.lib /LIBPATH:"{args.libmathdx_path}/lib/x64" mathdx_static.lib')

            if args.jobs <= 1:
                with ScopedTimer("build_cuda", active=args.verbose):
                    for cuda_cmd in cuda_cmds:
                        run_cmd(cuda_cmd)
            else:
                futures.extend([executor.submit(run_cmd, cmd=cuda_cmd) for cuda_cmd in cuda_cmds])

            if futures:
                done, pending = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
                for d in done:
                    if e := d.exception():
                        for f in pending:
                            f.cancel()
                        raise e
                elapsed = (time.perf_counter_ns() - wall_clock) / 1000000.0
                print(f"build took {elapsed:.2f} ms ({args.jobs:d} workers)")

        with ScopedTimer("link", active=args.verbose):
            link_cmd = f'"{host_linker}" {" ".join(linkopts + libs)} /out:"{dll_path}"'
            run_cmd(link_cmd)

    else:
        # Unix compilation
        cuda_compiler = "clang++" if getattr(args, "clang_build_toolchain", False) else "nvcc"
        cpp_compiler = "clang++" if getattr(args, "clang_build_toolchain", False) else args.host_compiler

        cpp_includes = f' -I"{warp_home_path.parent}/external/llvm-project/out/install/{mode}-{arch}/include"'
        cpp_includes += f' -I"{warp_home_path.parent}/_build/host-deps/llvm-project/release-{arch}/include"'
        cuda_includes = f' -I"{cuda_home}/include"' if cu_paths else ""
        includes = cpp_includes + cuda_includes

        if sys.platform == "darwin":
            version = f"--target={arch}-apple-macos11"
        else:
            compiler_name = os.path.basename(cpp_compiler)
            # Check for GCC compilers (g++, g++-11, x86_64-linux-gnu-g++, etc.)
            # Exclude clang to avoid false match on "clang++" which ends with "g++"
            if "clang" not in compiler_name and (compiler_name.endswith("g++") or "g++-" in compiler_name):
                version = "-fabi-version=13"  # GCC 8.2+
            else:
                version = ""

        cpp_flags = f'-Werror -Wuninitialized {version} --std=c++17 -fno-rtti -D{cuda_enabled} -D{mathdx_enabled} -D{cuda_compat_enabled} -fPIC -fvisibility=hidden -D_GLIBCXX_USE_CXX11_ABI=0 -I"{native_dir}" {includes} '

        if mode == "debug":
            cpp_flags += "-O0 -g -D_DEBUG -DWP_ENABLE_DEBUG=1 -fkeep-inline-functions"

        if mode == "release":
            cpp_flags += "-O3 -DNDEBUG -DWP_ENABLE_DEBUG=0"

        if args.verify_fp:
            cpp_flags += " -DWP_VERIFY_FP"

        if args.fast_math:
            cpp_flags += " -ffast-math"

        ld_inputs = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures, wall_clock = [], time.perf_counter_ns()

            cpp_cmds = []
            for cpp_path in cpp_paths:
                cpp_out = cpp_path + ".o"
                ld_inputs.append(quote(cpp_out))
                cpp_cmd = f'{cpp_compiler} {cpp_flags} -c "{cpp_path}" -o "{cpp_out}"'
                cpp_cmds.append(cpp_cmd)

            if args.jobs <= 1:
                with ScopedTimer("build", active=args.verbose):
                    for cpp_cmd in cpp_cmds:
                        run_cmd(cpp_cmd)
            else:
                futures = [executor.submit(run_cmd, cmd=cpp_cmd) for cpp_cmd in cpp_cmds]

            cuda_cmds = []
            if cu_paths:
                for cu_path in cu_paths:
                    cu_out = cu_path + ".o"

                    _nvcc_opts = [
                        opt.replace("@filename@", os.path.basename(cu_path).replace(".", "_")) for opt in nvcc_opts
                    ]

                    if cuda_compiler == "nvcc":
                        if mode == "debug":
                            cuda_cmd = f'{nvcc_cmd} --std=c++17 -g -G -O0 --compiler-options -fPIC,-fvisibility=hidden -D_DEBUG -D_ITERATOR_DEBUG_LEVEL=0 -line-info {" ".join(_nvcc_opts)} -DWP_ENABLE_CUDA=1 -I"{native_dir}" -D{mathdx_enabled} {libmathdx_includes} -o "{cu_out}" -c "{cu_path}"'
                        elif mode == "release":
                            cuda_cmd = f'{nvcc_cmd} --std=c++17 -O3 --compiler-options -fPIC,-fvisibility=hidden {" ".join(_nvcc_opts)} -DNDEBUG -DWP_ENABLE_CUDA=1 -I"{native_dir}" -D{mathdx_enabled} {libmathdx_includes} -o "{cu_out}" -c "{cu_path}"'
                    else:
                        # Use Clang compiler
                        if mode == "debug":
                            cuda_cmd = f'clang++ -Werror -Wuninitialized -Wno-unknown-cuda-version {" ".join(clang_opts)} -g -O0 -fPIC -fvisibility=hidden -D_DEBUG -D_ITERATOR_DEBUG_LEVEL=0 -DWP_ENABLE_CUDA=1 -I"{native_dir}" -D{mathdx_enabled} {libmathdx_includes} -o "{cu_out}" -c "{cu_path}"'
                        elif mode == "release":
                            cuda_cmd = f'clang++ -Werror -Wuninitialized -Wno-unknown-cuda-version {" ".join(clang_opts)} -O3 -fPIC -fvisibility=hidden -DNDEBUG -DWP_ENABLE_CUDA=1 -I"{native_dir}" -D{mathdx_enabled} {libmathdx_includes} -o "{cu_out}" -c "{cu_path}"'

                    cuda_cmds.append(cuda_cmd)

                    ld_inputs.append(quote(cu_out))

                ld_inputs.append(
                    f'-L"{cuda_home}/lib64" -lcudart_static -lnvrtc_static -lnvrtc-builtins_static -lnvptxcompiler_static -lpthread -ldl -lrt'
                )

                if args.libmathdx_path:
                    ld_inputs.append(f"-lnvJitLink_static -L{args.libmathdx_path}/lib -lmathdx_static")

            if args.jobs <= 1:
                with ScopedTimer("build_cuda", active=args.verbose):
                    for cuda_cmd in cuda_cmds:
                        run_cmd(cuda_cmd)
            else:
                futures.extend([executor.submit(run_cmd, cmd=cuda_cmd) for cuda_cmd in cuda_cmds])

            if futures:
                done, pending = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
                for d in done:
                    if e := d.exception():
                        for f in pending:
                            f.cancel()
                        raise e
                elapsed = (time.perf_counter_ns() - wall_clock) / 1000000.0
                print(f"build took {elapsed:.2f} ms ({args.jobs:d} workers)")

        if sys.platform == "darwin":
            opt_no_undefined = "-Wl,-undefined,error"
            opt_exclude_libs = ""
        else:
            opt_no_undefined = "-Wl,--no-undefined"
            opt_exclude_libs = "-Wl,--exclude-libs,ALL"

        with ScopedTimer("link", active=args.verbose):
            origin = "@loader_path" if (sys.platform == "darwin") else "$ORIGIN"
            link_cmd = f"{cpp_compiler} {version} -shared -Wl,-rpath,'{origin}' {opt_no_undefined} {opt_exclude_libs} -o '{dll_path}' {' '.join(ld_inputs + libs)}"
            run_cmd(link_cmd)

            # Strip symbols to reduce the binary size
            if mode == "release":
                if sys.platform == "darwin":
                    run_cmd(f"strip -x {dll_path}")  # Strip all local symbols
                else:  # Linux
                    # Strip all symbols except for those needed to support debugging JIT-compiled code
                    run_cmd(
                        f"strip --strip-all --keep-symbol=__jit_debug_register_code --keep-symbol=__jit_debug_descriptor {dll_path}"
                    )


def build_dll(args, dll_path, cpp_paths, cu_paths, libs=None):
    if sys.platform == "darwin":
        # build for ARM64 only (may be cross-compiled from Intel Mac)
        build_dll_for_arch(args, dll_path, cpp_paths, cu_paths, "aarch64", libs)
    else:
        build_dll_for_arch(args, dll_path, cpp_paths, cu_paths, machine_architecture(), libs)
