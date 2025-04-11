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

import os
import platform
import subprocess
import sys
from typing import List, Optional

from warp.utils import ScopedTimer

verbose_cmd = True  # print command lines before executing them


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
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print("Command failed with exit code:", e.returncode)
        print("Command output was:")
        print(e.output.decode())
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


def find_host_compiler():
    if os.name == "nt":
        # try and find an installed host compiler (msvc)
        # runs vcvars and copies back the build environment

        vswhere_path = r"%ProgramFiles(x86)%/Microsoft Visual Studio/Installer/vswhere.exe"
        vswhere_path = os.path.expandvars(vswhere_path)
        if not os.path.exists(vswhere_path):
            return ""

        vs_path = run_cmd(f'"{vswhere_path}" -latest -property installationPath').decode().rstrip()
        vsvars_path = os.path.join(vs_path, "VC\\Auxiliary\\Build\\vcvars64.bat")

        output = run_cmd(f'"{vsvars_path}" && set').decode()

        for line in output.splitlines():
            pair = line.split("=", 1)
            if len(pair) >= 2:
                os.environ[pair[0]] = pair[1]

        cl_path = run_cmd("where cl.exe").decode("utf-8").rstrip()
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
            return ""

        return cl_path

    else:
        # try and find g++
        return run_cmd("which g++").decode()


def get_cuda_toolkit_version(cuda_home):
    try:
        # the toolkit version can be obtained by running "nvcc --version"
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        nvcc_version_output = subprocess.check_output([nvcc_path, "--version"]).decode("utf-8")
        # search for release substring (e.g., "release 11.5")
        import re

        m = re.search(r"(?<=release )\d+\.\d+", nvcc_version_output)
        if m is not None:
            return tuple(int(x) for x in m.group(0).split("."))
        else:
            raise Exception("Failed to parse NVCC output")

    except Exception as e:
        print(f"Failed to determine CUDA Toolkit version: {e}")


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


def build_dll_for_arch(args, dll_path, cpp_paths, cu_path, arch, libs: Optional[List[str]] = None, mode=None):
    mode = args.mode if (mode is None) else mode
    cuda_home = args.cuda_path
    cuda_cmd = None

    # Add LLVM bin directory to PATH
    add_llvm_bin_to_path(args)

    if args.quick or cu_path is None:
        cuda_compat_enabled = "WP_ENABLE_CUDA_COMPATIBILITY=0"
    else:
        cuda_compat_enabled = "WP_ENABLE_CUDA_COMPATIBILITY=1"

    if libs is None:
        libs = []

    import pathlib

    warp_home_path = pathlib.Path(__file__).parent
    warp_home = warp_home_path.resolve()

    if args.verbose:
        print(f"Building {dll_path}")

    native_dir = os.path.join(warp_home, "native")

    if cu_path:
        # check CUDA Toolkit version
        min_ctk_version = (11, 5)
        ctk_version = get_cuda_toolkit_version(cuda_home) or min_ctk_version
        if ctk_version < min_ctk_version:
            raise Exception(
                f"CUDA Toolkit version {min_ctk_version[0]}.{min_ctk_version[1]}+ is required (found {ctk_version[0]}.{ctk_version[1]} in {cuda_home})"
            )

        if ctk_version[0] < 12 and args.libmathdx_path:
            print("MathDx support requires at least CUDA 12, skipping")
            args.libmathdx_path = None

        # NVCC gencode options
        gencode_opts = []

        # Clang architecture flags
        clang_arch_flags = []

        if args.quick:
            # minimum supported architectures (PTX)
            gencode_opts += ["-gencode=arch=compute_52,code=compute_52", "-gencode=arch=compute_75,code=compute_75"]
            clang_arch_flags += ["--cuda-gpu-arch=sm_52", "--cuda-gpu-arch=sm_75"]
        else:
            # generate code for all supported architectures
            gencode_opts += [
                # SASS for supported desktop/datacenter architectures
                "-gencode=arch=compute_52,code=sm_52",  # Maxwell
                "-gencode=arch=compute_60,code=sm_60",  # Pascal
                "-gencode=arch=compute_61,code=sm_61",
                "-gencode=arch=compute_70,code=sm_70",  # Volta
                "-gencode=arch=compute_75,code=sm_75",  # Turing
                "-gencode=arch=compute_80,code=sm_80",  # Ampere
                "-gencode=arch=compute_86,code=sm_86",
            ]

            # TODO: Get this working with sm_52, sm_60, sm_61
            clang_arch_flags += [
                # SASS for supported desktop/datacenter architectures
                "--cuda-gpu-arch=sm_52",
                "--cuda-gpu-arch=sm_60",
                "--cuda-gpu-arch=sm_61",
                "--cuda-gpu-arch=sm_70",  # Volta
                "--cuda-gpu-arch=sm_75",  # Turing
                "--cuda-gpu-arch=sm_80",  # Ampere
                "--cuda-gpu-arch=sm_86",
            ]

            if arch == "aarch64" and sys.platform == "linux":
                gencode_opts += [
                    # SASS for supported mobile architectures (e.g. Tegra/Jetson)
                    "-gencode=arch=compute_53,code=sm_53",  # X1
                    "-gencode=arch=compute_62,code=sm_62",  # X2
                    "-gencode=arch=compute_72,code=sm_72",  # Xavier
                    "-gencode=arch=compute_87,code=sm_87",  # Orin
                ]

                clang_arch_flags += [
                    # SASS for supported mobile architectures
                    "--cuda-gpu-arch=sm_53",  # X1
                    "--cuda-gpu-arch=sm_62",  # X2
                    "--cuda-gpu-arch=sm_72",  # Xavier
                    "--cuda-gpu-arch=sm_87",  # Orin
                ]

            if ctk_version >= (12, 8):
                # Support for Blackwell is available with CUDA Toolkit 12.8+
                gencode_opts += [
                    "-gencode=arch=compute_89,code=sm_89",  # Ada
                    "-gencode=arch=compute_90,code=sm_90",  # Hopper
                    "-gencode=arch=compute_100,code=sm_100",  # Blackwell
                    "-gencode=arch=compute_120,code=sm_120",  # Blackwell
                    "-gencode=arch=compute_120,code=compute_120",  # PTX for future hardware
                ]

                clang_arch_flags += [
                    "--cuda-gpu-arch=sm_89",  # Ada
                    "--cuda-gpu-arch=sm_90",  # Hopper
                    "--cuda-gpu-arch=sm_100",  # Blackwell
                    "--cuda-gpu-arch=sm_120",  # Blackwell
                ]
            elif ctk_version >= (11, 8):
                # Support for Ada and Hopper is available with CUDA Toolkit 11.8+
                gencode_opts += [
                    "-gencode=arch=compute_89,code=sm_89",  # Ada
                    "-gencode=arch=compute_90,code=sm_90",  # Hopper
                    "-gencode=arch=compute_90,code=compute_90",  # PTX for future hardware
                ]

                clang_arch_flags += [
                    "--cuda-gpu-arch=sm_89",  # Ada
                    "--cuda-gpu-arch=sm_90",  # Hopper
                ]
            else:
                gencode_opts += [
                    "-gencode=arch=compute_86,code=compute_86",  # PTX for future hardware
                ]

                clang_arch_flags += [
                    "--cuda-gpu-arch=sm_86",  # PTX for future hardware
                ]

        nvcc_opts = [
            *gencode_opts,
            "-t0",  # multithreaded compilation
            "--extended-lambda",
        ]

        # Clang options
        clang_opts = [
            *clang_arch_flags,
            "-std=c++17",
            "-xcuda",
            f'--cuda-path="{cuda_home}"',
        ]

        if args.compile_time_trace:
            if ctk_version >= (12, 8):
                nvcc_opts.append("--fdevice-time-trace=build_lib_compile-time-trace")
            else:
                print("Warp warning: CUDA version is less than 12.8, compile_time_trace is not supported")

        if args.fast_math:
            nvcc_opts.append("--use_fast_math")

    # is the library being built with CUDA enabled?
    cuda_enabled = "WP_ENABLE_CUDA=1" if (cu_path is not None) else "WP_ENABLE_CUDA=0"

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
        cuda_includes = f' /I"{cuda_home}/include"' if cu_path else ""
        includes = cpp_includes + cuda_includes

        # nvrtc_static.lib is built with /MT and _ITERATOR_DEBUG_LEVEL=0 so if we link it in we must match these options
        if cu_path or mode != "debug":
            runtime = "/MT"
            iter_dbg = "_ITERATOR_DEBUG_LEVEL=0"
            debug = "NDEBUG"
        else:
            runtime = "/MTd"
            iter_dbg = "_ITERATOR_DEBUG_LEVEL=2"
            debug = "_DEBUG"

        cpp_flags = f'/nologo /std:c++17 /GR- {runtime} /D "{debug}" /D "{cuda_enabled}" /D "{mathdx_enabled}" /D "{cuda_compat_enabled}" /D "{iter_dbg}" /I"{native_dir}" {includes} '

        if args.mode == "debug":
            cpp_flags += "/Zi /Od /D WP_ENABLE_DEBUG=1"
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

        with ScopedTimer("build", active=args.verbose):
            for cpp_path in cpp_paths:
                cpp_out = cpp_path + ".obj"
                linkopts.append(quote(cpp_out))

                cpp_cmd = f'"{args.host_compiler}" {cpp_flags} -c "{cpp_path}" /Fo"{cpp_out}"'
                run_cmd(cpp_cmd)

        if cu_path:
            cu_out = cu_path + ".o"

            if mode == "debug":
                cuda_cmd = f'"{cuda_home}/bin/nvcc" --std=c++17 --compiler-options=/MT,/Zi,/Od -g -G -O0 -DNDEBUG -D_ITERATOR_DEBUG_LEVEL=0 -I"{native_dir}" -line-info {" ".join(nvcc_opts)} -DWP_ENABLE_CUDA=1 -D{mathdx_enabled} {libmathdx_includes} -o "{cu_out}" -c "{cu_path}"'

            elif mode == "release":
                cuda_cmd = f'"{cuda_home}/bin/nvcc" --std=c++17 -O3 {" ".join(nvcc_opts)} -I"{native_dir}" -DNDEBUG -DWP_ENABLE_CUDA=1 -D{mathdx_enabled} {libmathdx_includes} -o "{cu_out}" -c "{cu_path}"'

            with ScopedTimer("build_cuda", active=args.verbose):
                run_cmd(cuda_cmd)
                linkopts.append(quote(cu_out))
                linkopts.append(
                    f'cudart_static.lib nvrtc_static.lib nvrtc-builtins_static.lib nvptxcompiler_static.lib ws2_32.lib user32.lib /LIBPATH:"{cuda_home}/lib/x64"'
                )

                if args.libmathdx_path:
                    linkopts.append(f'nvJitLink_static.lib /LIBPATH:"{args.libmathdx_path}/lib" mathdx_static.lib')

        with ScopedTimer("link", active=args.verbose):
            link_cmd = f'"{host_linker}" {" ".join(linkopts + libs)} /out:"{dll_path}"'
            run_cmd(link_cmd)

    else:
        # Unix compilation
        cuda_compiler = "clang++" if getattr(args, "clang_build_toolchain", False) else "nvcc"
        cpp_compiler = "clang++" if getattr(args, "clang_build_toolchain", False) else "g++"

        cpp_includes = f' -I"{warp_home_path.parent}/external/llvm-project/out/install/{mode}-{arch}/include"'
        cpp_includes += f' -I"{warp_home_path.parent}/_build/host-deps/llvm-project/release-{arch}/include"'
        cuda_includes = f' -I"{cuda_home}/include"' if cu_path else ""
        includes = cpp_includes + cuda_includes

        if sys.platform == "darwin":
            version = f"--target={arch}-apple-macos11"
        else:
            if cpp_compiler == "g++":
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

        with ScopedTimer("build", active=args.verbose):
            for cpp_path in cpp_paths:
                cpp_out = cpp_path + ".o"
                ld_inputs.append(quote(cpp_out))

                build_cmd = f'{cpp_compiler} {cpp_flags} -c "{cpp_path}" -o "{cpp_out}"'
                run_cmd(build_cmd)

        if cu_path:
            cu_out = cu_path + ".o"

            if cuda_compiler == "nvcc":
                if mode == "debug":
                    cuda_cmd = f'"{cuda_home}/bin/nvcc" --std=c++17 -g -G -O0 --compiler-options -fPIC,-fvisibility=hidden -D_DEBUG -D_ITERATOR_DEBUG_LEVEL=0 -line-info {" ".join(nvcc_opts)} -DWP_ENABLE_CUDA=1 -I"{native_dir}" -D{mathdx_enabled} {libmathdx_includes} -o "{cu_out}" -c "{cu_path}"'
                elif mode == "release":
                    cuda_cmd = f'"{cuda_home}/bin/nvcc" --std=c++17 -O3 --compiler-options -fPIC,-fvisibility=hidden {" ".join(nvcc_opts)} -DNDEBUG -DWP_ENABLE_CUDA=1 -I"{native_dir}" -D{mathdx_enabled} {libmathdx_includes} -o "{cu_out}" -c "{cu_path}"'
            else:
                # Use Clang compiler
                if mode == "debug":
                    cuda_cmd = f'clang++ -Werror -Wuninitialized -Wno-unknown-cuda-version {" ".join(clang_opts)} -g -O0 -fPIC -fvisibility=hidden -D_DEBUG -D_ITERATOR_DEBUG_LEVEL=0 -DWP_ENABLE_CUDA=1 -I"{native_dir}" -D{mathdx_enabled} {libmathdx_includes} -o "{cu_out}" -c "{cu_path}"'
                elif mode == "release":
                    cuda_cmd = f'clang++ -Werror -Wuninitialized -Wno-unknown-cuda-version {" ".join(clang_opts)} -O3 -fPIC -fvisibility=hidden -DNDEBUG -DWP_ENABLE_CUDA=1 -I"{native_dir}" -D{mathdx_enabled} {libmathdx_includes} -o "{cu_out}" -c "{cu_path}"'

            with ScopedTimer("build_cuda", active=args.verbose):
                run_cmd(cuda_cmd)

                ld_inputs.append(quote(cu_out))
                ld_inputs.append(
                    f'-L"{cuda_home}/lib64" -lcudart_static -lnvrtc_static -lnvrtc-builtins_static -lnvptxcompiler_static -lpthread -ldl -lrt'
                )

                if args.libmathdx_path:
                    ld_inputs.append(f"-lnvJitLink_static -L{args.libmathdx_path}/lib -lmathdx_static")

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


def build_dll(args, dll_path, cpp_paths, cu_path, libs=None):
    if sys.platform == "darwin":
        # create a universal binary by combining x86-64 and AArch64 builds
        build_dll_for_arch(args, dll_path + "-x86_64", cpp_paths, cu_path, "x86_64", libs)
        build_dll_for_arch(args, dll_path + "-aarch64", cpp_paths, cu_path, "aarch64", libs)

        run_cmd(f"lipo -create -output {dll_path} {dll_path}-x86_64 {dll_path}-aarch64")
        os.remove(f"{dll_path}-x86_64")
        os.remove(f"{dll_path}-aarch64")

    else:
        build_dll_for_arch(args, dll_path, cpp_paths, cu_path, machine_architecture(), libs)
