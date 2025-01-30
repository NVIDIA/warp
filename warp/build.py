# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ctypes
import os

import warp.config
from warp.thirdparty import appdirs

# From nvJitLink.h
nvJitLink_input_type = {"cubin": 1, "ptx": 2, "ltoir": 3, "fatbin": 4, "object": 5, "library": 6}


# builds cuda source to PTX or CUBIN using NVRTC (output type determined by output_path extension)
def build_cuda(
    cu_path,
    arch,
    output_path,
    config="release",
    verify_fp=False,
    fast_math=False,
    fuse_fp=True,
    lineinfo=False,
    ltoirs=None,
    fatbins=None,
) -> None:
    with open(cu_path, "rb") as src_file:
        src = src_file.read()
        cu_path_bytes = cu_path.encode("utf-8")
        program_name_bytes = os.path.basename(cu_path).encode("utf-8")
        inc_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "native").encode("utf-8")
        output_path = output_path.encode("utf-8")

        if warp.config.llvm_cuda:
            warp.context.runtime.llvm.compile_cuda(src, cu_path_bytes, inc_path, output_path, False)

        else:
            if ltoirs is None:
                ltoirs = []
            if fatbins is None:
                fatbins = []

            link_data = list(ltoirs) + list(fatbins)
            num_link = len(link_data)
            arr_link = (ctypes.c_char_p * num_link)(*link_data)
            arr_link_sizes = (ctypes.c_size_t * num_link)(*[len(l) for l in link_data])
            link_input_types = [nvJitLink_input_type["ltoir"]] * len(ltoirs) + [nvJitLink_input_type["fatbin"]] * len(
                fatbins
            )
            arr_link_input_types = (ctypes.c_int * num_link)(*link_input_types)
            err = warp.context.runtime.core.cuda_compile_program(
                src,
                program_name_bytes,
                arch,
                inc_path,
                0,
                None,
                config == "debug",
                warp.config.verbose,
                verify_fp,
                fast_math,
                fuse_fp,
                lineinfo,
                output_path,
                num_link,
                arr_link,
                arr_link_sizes,
                arr_link_input_types,
            )
            if err != 0:
                raise Exception(f"CUDA kernel build failed with error code {err}")


# load PTX or CUBIN as a CUDA runtime module (input type determined by input_path extension)
def load_cuda(input_path, device):
    if not device.is_cuda:
        raise RuntimeError("Not a CUDA device")

    return warp.context.runtime.core.cuda_load_module(device.context, input_path.encode("utf-8"))


def build_cpu(obj_path, cpp_path, mode="release", verify_fp=False, fast_math=False, fuse_fp=True):
    with open(cpp_path, "rb") as cpp:
        src = cpp.read()
        cpp_path = cpp_path.encode("utf-8")
        inc_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "native").encode("utf-8")
        obj_path = obj_path.encode("utf-8")

        err = warp.context.runtime.llvm.compile_cpp(
            src, cpp_path, inc_path, obj_path, mode == "debug", verify_fp, fuse_fp
        )
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
