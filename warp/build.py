# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import ctypes
import _ctypes

import warp.config
import warp.utils
from warp.utils import ScopedTimer
from warp.thirdparty import appdirs


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


def build_cpu(dll_path, cpp_path, mode="release", verify_fp=False, fast_math=False):
    # output stale, rebuild
    if warp.config.verbose:
        print(f"Building {dll_path}")

    import pathlib

    warp_home_path = pathlib.Path(__file__).parent
    warp_home = warp_home_path.resolve()
    native_dir = os.path.join(warp_home, "native")

    try:
        if os.name == "nt":
            clang = load_dll(f"{warp_home_path}/bin/warp-clang.dll")
        elif sys.platform == "darwin":
            clang = load_dll(f"{warp_home_path}/bin/libwarp-clang.dylib")
        else:  # Linux
            clang = load_dll(f"{warp_home_path}/bin/warp-clang.so")
    except RuntimeError as e:
        clang = None

    cpp_out = cpp_path + ".o"

    with ScopedTimer("build", active=warp.config.verbose):
        with open(cpp_path, "rb") as cpp:
            clang.compile_cpp(cpp.read(), native_dir.encode("utf-8"), cpp_out.encode("utf-8"), mode == "debug")


def load_dll(dll_path):
    if sys.platform == "win32":
        if dll_path[-4:] != ".dll":
            return None
    elif sys.platform == "darwin":
        if dll_path[-6:] != ".dylib":
            return None
    else:
        if dll_path[-3:] != ".so":
            return None

    try:
        if sys.version_info[0] > 3 or sys.version_info[0] == 3 and sys.version_info[1] >= 8:
            dll = ctypes.CDLL(dll_path, winmode=0)
        else:
            dll = ctypes.CDLL(dll_path)
    except OSError:
        raise RuntimeError(f"Failed to load the shared library '{dll_path}'")
    return dll


def unload_dll(dll):
    if dll is None:
        return

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
