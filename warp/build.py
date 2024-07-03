# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

import warp.config
from warp.thirdparty import appdirs


# builds cuda source to PTX or CUBIN using NVRTC (output type determined by output_path extension)
def build_cuda(cu_path, arch, output_path, config="release", verify_fp=False, fast_math=False):
    with open(cu_path, "rb") as src_file:
        src = src_file.read()
        cu_path = cu_path.encode("utf-8")
        inc_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "native").encode("utf-8")
        output_path = output_path.encode("utf-8")

        if warp.config.llvm_cuda:
            warp.context.runtime.llvm.compile_cuda(src, cu_path, inc_path, output_path, False)

        else:
            err = warp.context.runtime.core.cuda_compile_program(
                src, arch, inc_path, config == "debug", warp.config.verbose, verify_fp, fast_math, output_path
            )
            if err != 0:
                raise Exception(f"CUDA kernel build failed with error code {err}")


# load PTX or CUBIN as a CUDA runtime module (input type determined by input_path extension)
def load_cuda(input_path, device):
    if not device.is_cuda:
        raise RuntimeError("Not a CUDA device")

    return warp.context.runtime.core.cuda_load_module(device.context, input_path.encode("utf-8"))


def build_cpu(obj_path, cpp_path, mode="release", verify_fp=False, fast_math=False):
    with open(cpp_path, "rb") as cpp:
        src = cpp.read()
        cpp_path = cpp_path.encode("utf-8")
        inc_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "native").encode("utf-8")
        obj_path = obj_path.encode("utf-8")

        err = warp.context.runtime.llvm.compile_cpp(src, cpp_path, inc_path, obj_path, mode == "debug", verify_fp)
        if err != 0:
            raise Exception(f"CPU kernel build failed with error code {err}")


def init_kernel_cache(path=None):
    """Initialize kernel cache directory.

    This function is used during Warp initialization, but it can also be called directly to change the cache location.
    If the path is not explicitly specified, a default location will be chosen based on OS-specific conventions.

    To change the default cache location, set warp.config.kernel_cache_dir before calling warp.init().
    """

    if path is not None:
        cache_root_dir = os.path.realpath(path)
    elif "WARP_CACHE_PATH" in os.environ:
        cache_root_dir = os.path.realpath(os.environ.get("WARP_CACHE_PATH"))
    else:
        cache_root_dir = appdirs.user_cache_dir(appname="warp", appauthor="NVIDIA", version=warp.config.version)

    warp.config.kernel_cache_dir = cache_root_dir

    os.makedirs(warp.config.kernel_cache_dir, exist_ok=True)


def clear_kernel_cache() -> None:
    """Clear the kernel cache directory of previously generated source code and compiler artifacts.

    Only directories beginning with ``wp_`` will be deleted.
    This function only clears the cache for the current Warp version.
    """

    warp.context.init()

    import shutil

    is_intialized = warp.context.runtime is not None
    assert is_intialized, "The kernel cache directory is not configured; wp.init() has not been called yet or failed."

    for item in os.listdir(warp.config.kernel_cache_dir):
        item_path = os.path.join(warp.config.kernel_cache_dir, item)
        if os.path.isdir(item_path) and item.startswith("wp_"):
            # Remove the directory and its contents
            shutil.rmtree(item_path, ignore_errors=True)
