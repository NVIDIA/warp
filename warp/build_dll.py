# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import os
import subprocess

import warp.config
from warp.utils import ScopedTimer


def run_cmd(cmd, capture=False):
    if warp.config.verbose:
        print(cmd)

    try:
        return subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print(e.stdout.decode())
        if e.stderr:
            print(e.stderr.decode())
        raise (e)


# cut-down version of vcvars64.bat that allows using
# custom toolchain locations
def set_msvc_compiler(msvc_path, sdk_path):
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

    warp.config.host_compiler = os.path.join(msvc_path, "bin", "HostX64", "x64", "cl.exe")


def find_host_compiler():
    if os.name == "nt":
        try:
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

            if (
                (int(cl_version[0]) < cl_required_major)
                or (int(cl_version[0]) == cl_required_major)
                and int(cl_version[1]) < cl_required_minor
            ):
                print(
                    f"Warp: MSVC found but compiler version too old, found {cl_version[0]}.{cl_version[1]}, but must be {cl_required_major}.{cl_required_minor} or higher, kernel host compilation will be disabled."
                )
                return ""

            return cl_path

        except Exception as e:
            # couldn't find host compiler
            return ""
    else:
        # try and find g++
        try:
            return run_cmd("which g++").decode()
        except:
            return ""


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


def build_dll_for_arch(dll_path, cpp_paths, cu_path, libs, mode, arch, verify_fp=False, fast_math=False, quick=False):
    cuda_home = warp.config.cuda_path
    cuda_cmd = None

    if quick:
        cutlass_includes = ""
        cutlass_enabled = "WP_ENABLE_CUTLASS=0"
    else:
        cutlass_home = "warp/native/cutlass"
        cutlass_includes = f'-I"{cutlass_home}/include" -I"{cutlass_home}/tools/util/include"'
        cutlass_enabled = "WP_ENABLE_CUTLASS=1"

    if quick or cu_path is None:
        cuda_compat_enabled = "WP_ENABLE_CUDA_COMPATIBILITY=0"
    else:
        cuda_compat_enabled = "WP_ENABLE_CUDA_COMPATIBILITY=1"

    import pathlib

    warp_home_path = pathlib.Path(__file__).parent
    warp_home = warp_home_path.resolve()
    nanovdb_home = warp_home_path.parent / "_build/host-deps/nanovdb/include"

    # output stale, rebuild
    if warp.config.verbose:
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

        gencode_opts = []

        if quick:
            # minimum supported architectures (PTX)
            gencode_opts += ["-gencode=arch=compute_52,code=compute_52", "-gencode=arch=compute_75,code=compute_75"]
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
                # SASS for supported mobile architectures (e.g. Tegra/Jetson)
                # "-gencode=arch=compute_53,code=sm_53",
                # "-gencode=arch=compute_62,code=sm_62",
                # "-gencode=arch=compute_72,code=sm_72",
                # "-gencode=arch=compute_87,code=sm_87",
            ]

            # support for Ada and Hopper is available with CUDA Toolkit 11.8+
            if ctk_version >= (11, 8):
                gencode_opts += [
                    "-gencode=arch=compute_89,code=sm_89",  # Ada
                    "-gencode=arch=compute_90,code=sm_90",  # Hopper
                    # PTX for future hardware
                    "-gencode=arch=compute_90,code=compute_90",
                ]
            else:
                gencode_opts += [
                    # PTX for future hardware
                    "-gencode=arch=compute_86,code=compute_86",
                ]

        nvcc_opts = gencode_opts + [
            "-t0",  # multithreaded compilation
            "--extended-lambda",
        ]

        if fast_math:
            nvcc_opts.append("--use_fast_math")

    # is the library being built with CUDA enabled?
    cuda_enabled = "WP_ENABLE_CUDA=1" if (cu_path is not None) else "WP_ENABLE_CUDA=0"

    if os.name == "nt":
        if warp.config.host_compiler:
            host_linker = os.path.join(os.path.dirname(warp.config.host_compiler), "link.exe")
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

        if warp.config.mode == "debug":
            cpp_flags = f'/nologo {runtime} /Zi /Od /D "{debug}" /D WP_ENABLE_DEBUG=1 /D "{cuda_enabled}" /D "{cutlass_enabled}" /D "{cuda_compat_enabled}" /D "{iter_dbg}" /I"{native_dir}" /I"{nanovdb_home}" {includes}'
            linkopts = ["/DLL", "/DEBUG"]
        elif warp.config.mode == "release":
            cpp_flags = f'/nologo {runtime} /Ox /D "{debug}" /D WP_ENABLE_DEBUG=0 /D "{cuda_enabled}" /D "{cutlass_enabled}" /D "{cuda_compat_enabled}" /D "{iter_dbg}" /I"{native_dir}" /I"{nanovdb_home}" {includes}'
            linkopts = ["/DLL"]
        else:
            raise RuntimeError(f"Unrecognized build configuration (debug, release), got: {mode}")

        if verify_fp:
            cpp_flags += ' /D "WP_VERIFY_FP"'

        if fast_math:
            cpp_flags += " /fp:fast"

        with ScopedTimer("build", active=warp.config.verbose):
            for cpp_path in cpp_paths:
                cpp_out = cpp_path + ".obj"
                linkopts.append(quote(cpp_out))

                cpp_cmd = f'"{warp.config.host_compiler}" {cpp_flags} -c "{cpp_path}" /Fo"{cpp_out}"'
                run_cmd(cpp_cmd)

        if cu_path:
            cu_out = cu_path + ".o"

            if mode == "debug":
                cuda_cmd = f'"{cuda_home}/bin/nvcc" --compiler-options=/MT,/Zi,/Od -g -G -O0 -DNDEBUG -D_ITERATOR_DEBUG_LEVEL=0 -I"{native_dir}" -I"{nanovdb_home}" -line-info {" ".join(nvcc_opts)} -DWP_ENABLE_CUDA=1 -D{cutlass_enabled} {cutlass_includes} -o "{cu_out}" -c "{cu_path}"'

            elif mode == "release":
                cuda_cmd = f'"{cuda_home}/bin/nvcc" -O3 {" ".join(nvcc_opts)} -I"{native_dir}" -I"{nanovdb_home}" -DNDEBUG -DWP_ENABLE_CUDA=1 -D{cutlass_enabled} {cutlass_includes} -o "{cu_out}" -c "{cu_path}"'

            with ScopedTimer("build_cuda", active=warp.config.verbose):
                run_cmd(cuda_cmd)
                linkopts.append(quote(cu_out))
                linkopts.append(
                    f'cudart_static.lib nvrtc_static.lib nvrtc-builtins_static.lib nvptxcompiler_static.lib ws2_32.lib user32.lib /LIBPATH:"{cuda_home}/lib/x64"'
                )

        with ScopedTimer("link", active=warp.config.verbose):
            link_cmd = f'"{host_linker}" {" ".join(linkopts + libs)} /out:"{dll_path}"'
            run_cmd(link_cmd)

    else:
        cpp_includes = f' -I"{warp_home_path.parent}/external/llvm-project/out/install/{mode}-{arch}/include"'
        cpp_includes += f' -I"{warp_home_path.parent}/_build/host-deps/llvm-project/release-{arch}/include"'
        cuda_includes = f' -I"{cuda_home}/include"' if cu_path else ""
        includes = cpp_includes + cuda_includes

        if sys.platform == "darwin":
            target = f"--target={arch}-apple-macos11"
        else:
            target = ""

        if mode == "debug":
            cpp_flags = f'{target} -O0 -g -fno-rtti -D_DEBUG -DWP_ENABLE_DEBUG=1 -D{cuda_enabled} -D{cutlass_enabled} -D{cuda_compat_enabled} -fPIC -fvisibility=hidden --std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0 -fkeep-inline-functions -I"{native_dir}" {includes}'

        if mode == "release":
            cpp_flags = f'{target} -O3 -DNDEBUG -DWP_ENABLE_DEBUG=0 -D{cuda_enabled} -D{cutlass_enabled} -D{cuda_compat_enabled} -fPIC -fvisibility=hidden --std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0 -I"{native_dir}" {includes}'

        if verify_fp:
            cpp_flags += " -DWP_VERIFY_FP"

        if fast_math:
            cpp_flags += " -ffast-math"

        ld_inputs = []

        with ScopedTimer("build", active=warp.config.verbose):
            for cpp_path in cpp_paths:
                cpp_out = cpp_path + ".o"
                ld_inputs.append(quote(cpp_out))

                build_cmd = f'g++ {cpp_flags} -c "{cpp_path}" -o "{cpp_out}"'
                run_cmd(build_cmd)

        if cu_path:
            cu_out = cu_path + ".o"

            if mode == "debug":
                cuda_cmd = f'"{cuda_home}/bin/nvcc" -g -G -O0 --compiler-options -fPIC,-fvisibility=hidden -D_DEBUG -D_ITERATOR_DEBUG_LEVEL=0 -line-info {" ".join(nvcc_opts)} -DWP_ENABLE_CUDA=1 -I"{native_dir}" -D{cutlass_enabled} {cutlass_includes} -o "{cu_out}" -c "{cu_path}"'

            elif mode == "release":
                cuda_cmd = f'"{cuda_home}/bin/nvcc" -O3 --compiler-options -fPIC,-fvisibility=hidden {" ".join(nvcc_opts)} -DNDEBUG -DWP_ENABLE_CUDA=1 -I"{native_dir}" -D{cutlass_enabled} {cutlass_includes} -o "{cu_out}" -c "{cu_path}"'

            with ScopedTimer("build_cuda", active=warp.config.verbose):
                run_cmd(cuda_cmd)

                ld_inputs.append(quote(cu_out))
                ld_inputs.append(
                    f'-L"{cuda_home}/lib64" -lcudart_static -lnvrtc_static -lnvrtc-builtins_static -lnvptxcompiler_static -lpthread -ldl -lrt'
                )

        if sys.platform == "darwin":
            opt_no_undefined = "-Wl,-undefined,error"
            opt_exclude_libs = ""
        else:
            opt_no_undefined = "-Wl,--no-undefined"
            opt_exclude_libs = "-Wl,--exclude-libs,ALL"

        with ScopedTimer("link", active=warp.config.verbose):
            origin = "@loader_path" if (sys.platform == "darwin") else "$ORIGIN"
            link_cmd = f"g++ {target} -shared -Wl,-rpath,'{origin}' {opt_no_undefined} {opt_exclude_libs} -o '{dll_path}' {' '.join(ld_inputs + libs)}"
            run_cmd(link_cmd)

            # Strip symbols to reduce the binary size
            if sys.platform == "darwin":
                run_cmd(f"strip -x {dll_path}")  # Strip all local symbols
            else:  # Linux
                # Strip all symbols except for those needed to support debugging JIT-compiled code
                run_cmd(
                    f"strip --strip-all --keep-symbol=__jit_debug_register_code --keep-symbol=__jit_debug_descriptor {dll_path}"
                )


def build_dll(dll_path, cpp_paths, cu_path, libs=[], mode="release", verify_fp=False, fast_math=False, quick=False):
    if sys.platform == "darwin":
        # create a universal binary by combining x86-64 and AArch64 builds
        build_dll_for_arch(dll_path + "-x86_64", cpp_paths, cu_path, libs, mode, "x86_64", verify_fp, fast_math, quick)
        build_dll_for_arch(dll_path + "-arm64", cpp_paths, cu_path, libs, mode, "arm64", verify_fp, fast_math, quick)

        run_cmd(f"lipo -create -output {dll_path} {dll_path}-x86_64 {dll_path}-arm64")
        os.remove(f"{dll_path}-x86_64")
        os.remove(f"{dll_path}-arm64")

    else:
        build_dll_for_arch(dll_path, cpp_paths, cu_path, libs, mode, "x86_64", verify_fp, fast_math, quick)
