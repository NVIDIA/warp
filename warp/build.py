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

import ctypes
import errno
import hashlib
import json
import os
import time
from pathlib import Path

import warp.config
from warp.thirdparty import appdirs
from warp.types import *

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
    LTO artifacts are not affected.
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


def clear_lto_cache() -> None:
    """Clear the LTO cache directory of previously generated LTO code.

    The LTO cache is stored within a subdirectory of the kernel cache directory.
    This function only clears the cache for the current Warp version.
    """

    warp.context.init()

    import shutil

    is_intialized = warp.context.runtime is not None
    assert is_intialized, "The kernel cache directory is not configured; wp.init() has not been called yet or failed."

    lto_path = os.path.join(warp.config.kernel_cache_dir, "lto")
    if os.path.isdir(lto_path):
        # Remove the lto directory and its contents
        shutil.rmtree(lto_path, ignore_errors=True)


def safe_rename(src, dst, attempts=5, delay=0.1):
    for i in range(attempts):
        try:
            os.rename(src, dst)
            return
        except FileExistsError:
            return
        except OSError as e:
            if e.errno == errno.ENOTEMPTY:
                # if directory exists we assume another process
                # got there first, in which case we will copy
                # our output to the directory manually in second step
                return
            else:
                # otherwise assume directory creation failed e.g.: access denied
                # on Windows we see occasional failures to rename directories due to
                # some process holding a lock on a file to be moved to workaround
                # this we make multiple attempts to rename with some delay
                if i < attempts - 1:
                    time.sleep(delay)
                else:
                    print(
                        f"Could not update Warp cache with compiled binaries, trying to rename {src} to {dst}, error {e}"
                    )
                    raise e


def hash_symbol(symbol):
    ch = hashlib.sha256()
    ch.update(symbol.encode("utf-8"))
    return ch.hexdigest()


def get_lto_cache_dir():
    lto_dir = os.path.join(warp.config.kernel_cache_dir, "lto")
    return lto_dir


def get_cached_lto(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            lto_code_data = f.read()
        return lto_code_data
    else:
        return None


def get_cached_lto_meta(path, symbol):
    if os.path.exists(path):
        with open(path, "r") as f:
            keys = json.load(f)
        value = keys[symbol]
        return value
    else:
        return None


def build_lto_dot(M, N, K, adtype, bdtype, cdtype, alayout, blayout, clayout, arch, num_threads, builder):
    # TODO: MathDx doesn't yet have heuristics for Blackwell
    arch = min(arch, 90)

    # Maps Python/Warp types to C++ types and enums
    def cublasdx_type_map(dtype):
        if dtype == float16:
            return ("wp::float16", 3, 0)
        if dtype == float32:
            return ("wp::float32", 5, 0)
        if dtype == float64:
            return ("wp::float64", 6, 0)
        if dtype == vec2h:
            return ("wp::vec2h", 3, 1)
        if dtype == vec2f:
            return ("wp::vec2f", 5, 1)
        if dtype == vec2d:
            return ("wp::vec2d", 6, 1)
        raise TypeError("Unsupported input type in tile_matmul")

    def cublasdx_arrangement_map(layout):
        if layout == "colmajor":
            return 0  # CUBLASDX_ARRANGEMENT_COL_MAJOR
        if layout == "rowmajor":
            return 1  # CUBLASDX_ARRANGEMENT_ROW_MAJOR
        raise ValueError("Unsupported layout in tile_matmul")

    (a_dtype, a_prec, a_type) = cublasdx_type_map(adtype)
    (b_dtype, b_prec, b_type) = cublasdx_type_map(bdtype)
    (c_dtype, c_prec, c_type) = cublasdx_type_map(cdtype)
    a_arrangement = cublasdx_arrangement_map(alayout)
    b_arrangement = cublasdx_arrangement_map(blayout)
    c_arrangement = cublasdx_arrangement_map(clayout)

    if a_type != b_type or a_type != c_type:
        raise TypeError("time_matmul(A, B, C) requires all inputs to be real or complex")

    element_type = a_type

    lto_symbol = f"dot_{M}_{N}_{K}_{arch}_{num_threads}_{a_arrangement}_{b_arrangement}_{c_arrangement}_{a_prec}_{b_prec}_{c_prec}_{element_type}"

    # early out if LTO for this symbol is already cached in current module
    if lto_symbol in builder.ltoirs:
        return lto_symbol, builder.ltoirs[lto_symbol]

    # hash symbol and determine output path
    h = hash_symbol(lto_symbol)

    lto_dir = get_lto_cache_dir()
    lto_name = f"{h[:7]}.lto"
    lto_path = os.path.join(lto_dir, lto_name)

    # early out if LTO for this symbol is already built but not cached in current module
    lto_code_data = get_cached_lto(lto_path)

    if lto_code_data is not None:
        builder.ltoirs[lto_symbol] = lto_code_data
        builder.ltoirs_decl[lto_symbol] = (
            f"void {lto_symbol}({c_dtype}, {a_dtype}*, {b_dtype}*, {c_dtype}, {c_dtype}*);"
        )

        return lto_symbol, lto_code_data

    # create a temporary (process unique) dir for build outputs before moving to the binary dir
    build_dir = f"{lto_dir}_p{os.getpid()}"

    # dir may exist from previous attempts / runs / archs
    Path(build_dir).mkdir(parents=True, exist_ok=True)

    # temporary path to compile to in build_dir
    temp_lto_path = os.path.join(build_dir, lto_name)

    # compile LTO
    result = warp.context.runtime.core.cuda_compile_dot(
        temp_lto_path.encode("utf-8"),
        lto_symbol.encode("utf-8"),
        0,
        None,
        None,
        arch,
        M,
        N,
        K,
        a_prec,
        b_prec,
        c_prec,
        element_type,
        a_arrangement,
        b_arrangement,
        c_arrangement,
        num_threads,
    )

    if not result:
        if Path(temp_lto_path).exists():
            Path(temp_lto_path).unlink()
        raise RuntimeError("Failed to compile tile_matmul")
    else:
        with open(temp_lto_path, "rb") as f:
            lto_code_data = f.read()

    builder.ltoirs[lto_symbol] = lto_code_data
    builder.ltoirs_decl[lto_symbol] = f"void {lto_symbol}({c_dtype}, {a_dtype}*, {b_dtype}*, {c_dtype}, {c_dtype}*);"

    # try to move process outputs to cache
    safe_rename(build_dir, lto_dir)

    if os.path.exists(lto_dir):
        if not os.path.exists(lto_path):
            # copy output file to the destination lto dir
            try:
                os.rename(temp_lto_path, lto_path)
            except (OSError, FileExistsError):
                # another process likely updated the lto dir first
                pass

    if build_dir:
        import shutil

        # clean up build_dir used for this process
        shutil.rmtree(build_dir, ignore_errors=True)

    return lto_symbol, lto_code_data


def build_lto_solver(M, N, solver, solver_enum, fill_mode, arch, precision_enum, num_threads, parameter_list, builder):
    # TODO: MathDx doesn't yet have heuristics for Blackwell
    arch = min(arch, 90)

    lto_symbol = f"{solver}_{M}_{N}_{arch}_{precision_enum}"
    ltoir_decl = f"void {lto_symbol}{parameter_list};"

    # early out if LTO for this symbol is already cached in current module
    if lto_symbol in builder.ltoirs:
        return lto_symbol, builder.ltoirs[lto_symbol]

    # hash symbol and determine output path
    h = hash_symbol(lto_symbol)

    lto_dir = get_lto_cache_dir()
    lto_name = f"{h[:7]}.lto"
    lto_path = os.path.join(lto_dir, lto_name)

    # we also cache a universal fatbin binary for this symbol
    universal_fatbin_name = f"{h[:7]}_fatbin.lto"
    universal_fatbin_path = os.path.join(lto_dir, universal_fatbin_name)

    lto_code_data = get_cached_lto(lto_path)
    universal_fatbin_code_data = get_cached_lto(universal_fatbin_path)

    # early out if LTO for this symbol is already built but not cached in current module
    if lto_code_data is not None and universal_fatbin_code_data is not None:
        builder.ltoirs[lto_symbol] = lto_code_data
        builder.ltoirs_decl[lto_symbol] = ltoir_decl
        builder.fatbins[lto_symbol] = universal_fatbin_code_data

        return lto_symbol, lto_code_data

    # create a temporary (process unique) dir for build outputs before moving to the binary dir
    build_dir = f"{lto_dir}_p{os.getpid()}"

    # dir may exist from previous attempts / runs / archs
    Path(build_dir).mkdir(parents=True, exist_ok=True)

    # temporary paths to compile to in build_dir
    temp_lto_path = os.path.join(build_dir, lto_name)
    temp_universal_fatbin_path = os.path.join(build_dir, universal_fatbin_name)

    # compile LTO
    result = warp.context.runtime.core.cuda_compile_solver(
        temp_universal_fatbin_path.encode("utf-8"),
        temp_lto_path.encode("utf-8"),
        lto_symbol.encode("utf-8"),
        0,
        None,
        None,
        arch,
        M,
        N,
        solver_enum,
        precision_enum,
        fill_mode,
        num_threads,
    )

    if not result:
        for path in [temp_universal_fatbin_path, temp_lto_path]:
            if Path(path).exists():
                Path(path).unlink()
        raise RuntimeError("Failed to compile tile_cholesky")

    else:
        with open(temp_lto_path, "rb") as f:
            lto_code_data = f.read()
        with open(temp_universal_fatbin_path, "rb") as f:
            universal_fatbin_code_data = f.read()

    builder.ltoirs[lto_symbol] = lto_code_data
    builder.ltoirs_decl[lto_symbol] = ltoir_decl
    builder.fatbins[lto_symbol] = universal_fatbin_code_data

    # try to move process outputs to lto cache
    safe_rename(build_dir, lto_dir)

    if os.path.exists(lto_dir):
        for p in [(lto_path, temp_lto_path), (universal_fatbin_path, temp_universal_fatbin_path)]:
            path, temp_path = p
            if not os.path.exists(path):
                # copy output file to the destination lto dir
                try:
                    os.rename(temp_path, path)
                except (OSError, FileExistsError):
                    # another process likely updated the lto dir first
                    pass

    if build_dir:
        import shutil

        # clean up build_dir used for this process
        shutil.rmtree(build_dir, ignore_errors=True)

    return lto_symbol, lto_code_data


def build_lto_fft(arch, size, ept, direction, dir, precision, builder):
    # TODO: MathDx doesn't yet have heuristics for Blackwell
    arch = min(arch, 90)

    lto_symbol = f"fft_{size}_{ept}_{arch}_{direction}_{precision}"

    # early out if LTO for this symbol is already cached in current module
    if lto_symbol in builder.ltoirs:
        return lto_symbol, builder.ltoirs[lto_symbol], builder.shared_memory_bytes[lto_symbol]

    # hash symbol and determine output path
    h = hash_symbol(lto_symbol)

    lto_dir = get_lto_cache_dir()
    lto_name = f"{h[:7]}.lto"
    lto_path = os.path.join(lto_dir, lto_name)

    # we also cache shared memory requirements for this kernel in a .meta file
    meta_name = f"{h[:7]}.meta"
    meta_path = os.path.join(lto_dir, meta_name)

    # early out if LTO for this symbol is already built but not cached in current module
    lto_code_data = get_cached_lto(lto_path)
    shared_memory_bytes = get_cached_lto_meta(meta_path, lto_symbol)

    if lto_code_data is not None and shared_memory_bytes is not None:
        builder.ltoirs[lto_symbol] = lto_code_data
        builder.shared_memory_bytes[lto_symbol] = shared_memory_bytes

        return lto_symbol, lto_code_data, shared_memory_bytes

    # create a temporary (process unique) dir for build outputs before moving to the binary dir
    build_dir = f"{lto_dir}_p{os.getpid()}"

    # dir may exist from previous attempts / runs / archs
    Path(build_dir).mkdir(parents=True, exist_ok=True)

    # temporary paths to compile to in build_dir
    temp_lto_path = os.path.join(build_dir, lto_name)
    temp_meta_path = os.path.join(build_dir, meta_name)

    # compile LTO
    shared_memory_size = ctypes.c_int(0)

    result = warp.context.runtime.core.cuda_compile_fft(
        temp_lto_path.encode("utf-8"),
        lto_symbol.encode("utf-8"),
        0,
        None,
        None,
        arch,
        size,
        ept,
        dir,
        precision,
        ctypes.byref(shared_memory_size),
    )

    shared_memory_bytes = Tile.round_up(shared_memory_size.value)

    if not result:
        if Path(temp_lto_path).exists():
            Path(temp_lto_path).unlink()
        raise RuntimeError("Failed to compile tile_fft")

    else:
        with open(temp_lto_path, "rb") as f:
            lto_code_data = f.read()

        # output meta file with shared memory requirements for this lto_symbol
        meta = {}
        meta[lto_symbol] = shared_memory_bytes

        with open(temp_meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    builder.ltoirs[lto_symbol] = lto_code_data
    builder.shared_memory_bytes[lto_symbol] = shared_memory_bytes

    # try to move process outputs to cache
    safe_rename(build_dir, lto_dir)

    if os.path.exists(lto_dir):
        for p in [(lto_path, temp_lto_path), (meta_path, temp_meta_path)]:
            path, temp_path = p
            if not os.path.exists(path):
                # copy output file to the destination lto dir
                try:
                    os.rename(temp_path, path)
                except (OSError, FileExistsError):
                    # another process likely updated the lto dir first
                    pass

    if build_dir:
        import shutil

        # clean up build_dir used for this process
        shutil.rmtree(build_dir, ignore_errors=True)

    return lto_symbol, lto_code_data, shared_memory_bytes
