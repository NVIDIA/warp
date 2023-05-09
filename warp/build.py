# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import subprocess
import ctypes
import _ctypes

import warp.config
import warp.utils
from warp.utils import ScopedTimer
from warp.thirdparty import appdirs

# return from funtions without type -> C++ compile error
# array[i,j] += x -> augassign not handling target of subscript


def run_cmd(cmd, capture=False):
    if warp.config.verbose:
        print(cmd)

    try:
        return subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output.decode())
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


# builds cuda source to PTX or CUBIN using NVRTC (output type determined by output_path extension)
def build_cuda(cu_path, arch, output_path, config="release", verify_fp=False, fast_math=False):
    src_file = open(cu_path)
    src = src_file.read().encode("utf-8")
    src_file.close()

    inc_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "native").encode("utf-8")
    output_path = output_path.encode("utf-8")

    err = warp.context.runtime.core.cuda_compile_program(
        src, arch, inc_path, config == "debug", warp.config.verbose, verify_fp, fast_math, output_path
    )
    if err:
        raise Exception("CUDA build failed")


# load PTX or CUBIN as a CUDA runtime module (input type determined by input_path extension)
def load_cuda(input_path, device):
    if not device.is_cuda:
        raise ("Not a CUDA device")

    return warp.context.runtime.core.cuda_load_module(device.context, input_path.encode("utf-8"))


def quote(path):
    return '"' + path + '"'


def build_dll(
    dll_path, cpp_paths, cu_path, libs=[], mode="release", verify_fp=False, fast_math=False, quick=False
):
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

    # ensure that dll is not loaded in the process
    force_unload_dll(dll_path)

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
        # try loading warp-clang.dll, except when we're building warp-clang.dll or warp.dll
        clang = None
        if os.path.basename(dll_path) != "warp-clang.dll" and os.path.basename(dll_path) != "warp.dll":
            try:
                clang = warp.build.load_dll(f"{warp_home_path}/bin/warp-clang.dll")
            except RuntimeError as e:
                clang = None

        if warp.config.host_compiler:
            host_linker = os.path.join(os.path.dirname(warp.config.host_compiler), "link.exe")
        elif not clang:
            raise RuntimeError("Warp build error: No host or bundled compiler was found")

        cpp_includes = f' /I"{warp_home_path.parent}/external/llvm-project/out/install/{mode}/include"'
        cpp_includes += f' /I"{warp_home_path.parent}/_build/host-deps/llvm-project/include"'
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

        if "/NODEFAULTLIB" in libs:
            runtime = "/sdl- /GS-"  # don't specify a runtime, and disable security checks with depend on it

        if mode == "debug":
            cpp_flags = f'/nologo {runtime} /Zi /Od /D "{debug}" /D "WP_CPU" /D "{cuda_enabled}" /D "{cutlass_enabled}" /D "{cuda_compat_enabled}" /D "{iter_dbg}" /I"{native_dir}" /I"{nanovdb_home}" {includes}'
            linkopts = ["/DLL", "/DEBUG"]
        elif mode == "release":
            cpp_flags = f'/nologo {runtime} /Ox /D "{debug}" /D "WP_CPU" /D "{cuda_enabled}" /D "{cutlass_enabled}" /D "{cuda_compat_enabled}" /D "{iter_dbg}" /I"{native_dir}" /I"{nanovdb_home}" {includes}'
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

                if clang:
                    with open(cpp_path, "rb") as cpp:
                        clang.compile_cpp(cpp.read(), native_dir.encode("utf-8"), cpp_out.encode("utf-8"))

                else:
                    cpp_cmd = f'"{warp.config.host_compiler}" {cpp_flags} -c "{cpp_path}" /Fo"{cpp_out}"'
                    run_cmd(cpp_cmd)

        if cu_path:
            cu_out = cu_path + ".o"

            if mode == "debug":
                cuda_cmd = f'"{cuda_home}/bin/nvcc" --compiler-options=/MT,/Zi,/Od -g -G -O0 -DNDEBUG -D_ITERATOR_DEBUG_LEVEL=0 -I"{native_dir}" -I"{nanovdb_home}" -line-info {" ".join(nvcc_opts)} -DWP_CUDA -DWP_ENABLE_CUDA=1 -D{cutlass_enabled} {cutlass_includes} -o "{cu_out}" -c "{cu_path}"'

            elif mode == "release":
                cuda_cmd = f'"{cuda_home}/bin/nvcc" -O3 {" ".join(nvcc_opts)} -I"{native_dir}" -I"{nanovdb_home}" -DNDEBUG -DWP_CUDA -DWP_ENABLE_CUDA=1 -D{cutlass_enabled} {cutlass_includes} -o "{cu_out}" -c "{cu_path}"'

            with ScopedTimer("build_cuda", active=warp.config.verbose):
                run_cmd(cuda_cmd)
                linkopts.append(quote(cu_out))
                linkopts.append(
                    f'cudart_static.lib nvrtc_static.lib nvrtc-builtins_static.lib nvptxcompiler_static.lib ws2_32.lib user32.lib /LIBPATH:"{cuda_home}/lib/x64"'
                )

        with ScopedTimer("link", active=warp.config.verbose):
            # Link into a DLL, unless we have LLVM to load the object code directly
            if not clang:
                link_cmd = f'"{host_linker}" {" ".join(linkopts + libs)} /out:"{dll_path}"'
                run_cmd(link_cmd)

    else:
        clang = None
        try:
            if sys.platform == "darwin":
                # try loading libwarp-clang.dylib, except when we're building libwarp-clang.dylib or libwarp.dylib
                if os.path.basename(dll_path) != "libwarp-clang.dylib" and os.path.basename(dll_path) != "libwarp.dylib":
                    clang = warp.build.load_dll(f"{warp_home_path}/bin/libwarp-clang.dylib")
            else:  # Linux
                # try loading warp-clang.so, except when we're building warp-clang.so or warp.so
                if os.path.basename(dll_path) != "warp-clang.so" and os.path.basename(dll_path) != "warp.so":
                    clang = warp.build.load_dll(f"{warp_home_path}/bin/warp-clang.so")
        except RuntimeError as e:
            clang = None

        cpp_includes = f' -I"{warp_home_path.parent}/external/llvm-project/out/install/{mode}/include"'
        cpp_includes += f' -I"{warp_home_path.parent}/_build/host-deps/llvm-project/include"'
        cuda_includes = f' -I"{cuda_home}/include"' if cu_path else ""
        includes = cpp_includes + cuda_includes

        if mode == "debug":
            cpp_flags = f'-O0 -g -D_DEBUG -DWP_CPU -D{cuda_enabled} -D{cutlass_enabled} -D{cuda_compat_enabled} -fPIC -fvisibility=hidden --std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0 -fkeep-inline-functions -I"{native_dir}" {includes}'

        if mode == "release":
            cpp_flags = f'-O3 -DNDEBUG -DWP_CPU -D{cuda_enabled} -D{cutlass_enabled} -D{cuda_compat_enabled} -fPIC -fvisibility=hidden --std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0 -I"{native_dir}" {includes}'

        if verify_fp:
            cpp_flags += " -DWP_VERIFY_FP"

        if fast_math:
            cpp_flags += " -ffast-math"

        ld_inputs = []

        with ScopedTimer("build", active=warp.config.verbose):
            for cpp_path in cpp_paths:
                cpp_out = cpp_path + ".o"
                ld_inputs.append(quote(cpp_out))

                if clang:
                    with open(cpp_path, "rb") as cpp:
                        clang.compile_cpp(cpp.read(), native_dir.encode("utf-8"), cpp_out.encode("utf-8"))

                else:
                    build_cmd = f'g++ {cpp_flags} -c "{cpp_path}" -o "{cpp_out}"'
                    run_cmd(build_cmd)

        if cu_path:
            cu_out = cu_path + ".o"

            if mode == "debug":
                cuda_cmd = f'"{cuda_home}/bin/nvcc" -g -G -O0 --compiler-options -fPIC,-fvisibility=hidden -D_DEBUG -D_ITERATOR_DEBUG_LEVEL=0 -line-info {" ".join(nvcc_opts)} -DWP_CUDA -DWP_ENABLE_CUDA=1 -I"{native_dir}" -D{cutlass_enabled} {cutlass_includes} -o "{cu_out}" -c "{cu_path}"'

            elif mode == "release":
                cuda_cmd = f'"{cuda_home}/bin/nvcc" -O3 --compiler-options -fPIC,-fvisibility=hidden {" ".join(nvcc_opts)} -DNDEBUG -DWP_CUDA -DWP_ENABLE_CUDA=1 -I"{native_dir}" -D{cutlass_enabled} {cutlass_includes} -o "{cu_out}" -c "{cu_path}"'

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
            # Link into a DLL, unless we have LLVM to load the object code directly
            if not clang:
                origin = "@loader_path" if (sys.platform == "darwin") else "$ORIGIN"
                link_cmd = f"g++ -shared -Wl,-rpath,'{origin}' {opt_no_undefined} {opt_exclude_libs} -o '{dll_path}' {' '.join(ld_inputs + libs)}"
                run_cmd(link_cmd)


def load_dll(dll_path):
    try:
        if sys.version_info[0] > 3 or sys.version_info[0] == 3 and sys.version_info[1] >= 8:
            dll = ctypes.CDLL(dll_path, winmode=0)
        else:
            dll = ctypes.CDLL(dll_path)
    except OSError:
        raise RuntimeError(f"Failed to load the shared library '{dll_path}'")
    return dll


def unload_dll(dll):
    handle = dll._handle
    del dll

    # force garbage collection to eliminate any Python references to the dll
    import gc

    gc.collect()

    # platform dependent unload, removes *all* references to the dll
    # note this should only be performed if you know there are no dangling
    # refs to the dll inside the Python program
    if os.name == "nt":
        max_attempts = 100
        for i in range(max_attempts):
            result = ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(handle))
            if result == 0:
                return
    else:
        _ctypes.dlclose(handle)


def force_unload_dll(dll_path):
    try:
        # force load/unload of the dll from the process
        dll = load_dll(dll_path)
        unload_dll(dll)

    except Exception as e:
        return


kernel_bin_dir = None
kernel_gen_dir = None


def init_kernel_cache(path=None):
    """Initialize kernel cache directory.

    This function is used during Warp initialization, but it can also be called directly to change the cache location.
    If the path is not explicitly specified, a default location will be chosen based on OS-specific conventions.

    To change the default cache location, set warp.config.kernel_cache_dir before calling warp.init().
    """

    warp_root_dir = os.path.dirname(os.path.realpath(__file__))
    warp_bin_dir = os.path.join(warp_root_dir, "bin")

    if path is not None:
        cache_root_dir = os.path.realpath(path)
    else:
        cache_root_dir = appdirs.user_cache_dir(
            appname="warp", appauthor="NVIDIA Corporation", version=warp.config.version
        )

    cache_bin_dir = os.path.join(cache_root_dir, "bin")
    cache_gen_dir = os.path.join(cache_root_dir, "gen")

    if not os.path.isdir(cache_root_dir):
        # print("Creating cache directory '%s'" % cache_root_dir)
        os.makedirs(cache_root_dir, exist_ok=True)

    if not os.path.isdir(cache_gen_dir):
        # print("Creating codegen directory '%s'" % cache_gen_dir)
        os.makedirs(cache_gen_dir, exist_ok=True)

    if not os.path.isdir(cache_bin_dir):
        # print("Creating binary directory '%s'" % cache_bin_dir)
        os.makedirs(cache_bin_dir, exist_ok=True)

    warp.config.kernel_cache_dir = cache_root_dir

    global kernel_bin_dir, kernel_gen_dir
    kernel_bin_dir = cache_bin_dir
    kernel_gen_dir = cache_gen_dir


def clear_kernel_cache():
    """Clear the kernel cache."""

    import glob

    paths = []

    if kernel_bin_dir is not None and os.path.isdir(kernel_bin_dir):
        pattern = os.path.join(kernel_bin_dir, "wp_*")
        paths += glob.glob(pattern)

    if kernel_gen_dir is not None and os.path.isdir(kernel_gen_dir):
        pattern = os.path.join(kernel_gen_dir, "wp_*")
        paths += glob.glob(pattern)

    for p in paths:
        if os.path.isfile(p):
            os.remove(p)
