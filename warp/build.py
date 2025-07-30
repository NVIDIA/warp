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
    compile_time_trace=False,
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
            err = warp.context.runtime.core.wp_cuda_compile_program(
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
                compile_time_trace,
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

    return warp.context.runtime.core.wp_cuda_load_module(device.context, input_path.encode("utf-8"))


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

        if os.name == "nt" and os.path.isabs(cache_root_dir) and not cache_root_dir.startswith("\\\\?\\"):
            # Add Windows long-path prefix, accounting for UNC shares.
            if cache_root_dir.startswith("\\\\"):
                # UNC path  \\server\share\…  →  \\?\UNC\server\share\…
                cache_root_dir = "\\\\?\\UNC\\" + cache_root_dir.lstrip("\\")
            else:
                # Drive-letter path  C:\…  →  \\?\C:\…
                cache_root_dir = "\\\\?\\" + cache_root_dir

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
        with open(path) as f:
            keys = json.load(f)
        value = keys[symbol]
        return value
    else:
        return None


def _build_lto_base(lto_symbol, compile_func, builder, extra_files=None):
    """Generic LTO build function that handles caching, file operations and process management.

    Args:
        lto_symbol: Unique identifier for the LTO operation
        compile_func: Function to compile the specific LTO
            (receives a dictionary of build paths)
        builder: Builder object to store results
        extra_files: Dictionary of additional file types to handle (e.g.,
            {".meta": None, ".fatbin": None}). Values are the functions to get
            the cached file data.

    Returns:
        Tuple containing lto_code_data followed by any extra data from extra_files
    """
    if extra_files is None:
        extra_files = {}

    # Hash symbol and set up paths
    h = hash_symbol(lto_symbol)
    lto_dir = get_lto_cache_dir()
    lto_name = f"{h[:7]}.lto"
    lto_path = os.path.join(lto_dir, lto_name)

    # Set up paths for extra files
    file_paths = {".lto": lto_path}
    temp_file_paths = {}

    for ext, _ in extra_files.items():
        name = f"{h[:7]}{ext}"
        file_paths[ext] = os.path.join(lto_dir, name)

    # Check if already built but not cached
    lto_code_data = get_cached_lto(lto_path)
    if lto_code_data is not None:
        # Get the cached data for the extra files and early return
        all_files_cached = True
        for ext, getter in extra_files.items():
            if getter and os.path.exists(file_paths[ext]):
                cached_data = getter(file_paths[ext])
                if cached_data is None:
                    all_files_cached = False
                    break
                extra_files[ext] = cached_data
            elif getter:  # If there's a getter but file doesn't exist
                all_files_cached = False
                break

        if all_files_cached:
            if not extra_files:
                return (lto_code_data,)
            else:
                return (lto_code_data, *[extra_files[ext] for ext in extra_files.keys()])

    # Create process-dependent temporary build directory
    build_dir = f"{lto_dir}_p{os.getpid()}"
    Path(build_dir).mkdir(parents=True, exist_ok=True)

    # Set up temporary paths for the build outputs
    for ext, path in file_paths.items():
        temp_file_paths[ext] = os.path.join(build_dir, os.path.basename(path))

    # Compile LTO with the specialized function
    result, outputs = compile_func(temp_file_paths)

    if not result:
        # Clean up and fail
        for path in temp_file_paths.values():
            if Path(path).exists():
                Path(path).unlink()
        raise RuntimeError(f"Failed to compile {lto_symbol}")

    # Move outputs to cache
    safe_rename(build_dir, lto_dir)

    # If build_dir couldn't be moved by a rename, move the outputs one-by-one to lto_dir
    if os.path.exists(lto_dir):
        for ext, path in file_paths.items():
            if not os.path.exists(path):
                try:
                    # copy output file to the destination lto dir
                    os.rename(temp_file_paths[ext], path)
                except (OSError, FileExistsError):
                    # another process likely updated the lto dir first
                    pass

    # Clean up the temporary build directory
    if build_dir:
        import shutil

        shutil.rmtree(build_dir, ignore_errors=True)

    if not extra_files:
        return (outputs[".lto"],)
    else:
        return (outputs[".lto"], *[outputs[ext] for ext in extra_files.keys()])


def build_lto_dot(M, N, K, adtype, bdtype, cdtype, alayout, blayout, clayout, arch, num_threads, builder):
    arch = 120 if arch > 121 else arch

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
        raise TypeError("tile_matmul(A, B, C) requires all inputs to be real or complex")

    element_type = a_type

    lto_symbol = f"dot_{M}_{N}_{K}_{arch}_{num_threads}_{a_arrangement}_{b_arrangement}_{c_arrangement}_{a_prec}_{b_prec}_{c_prec}_{element_type}"

    def compile_lto_dot(temp_paths):
        result = warp.context.runtime.core.wp_cuda_compile_dot(
            temp_paths[".lto"].encode("utf-8"),
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

        if result:
            with open(temp_paths[".lto"], "rb") as f:
                lto_code_data = f.read()
            return True, {".lto": lto_code_data}
        return False, {}

    # Early out if already cached in module
    if lto_symbol in builder.ltoirs:
        lto_code_data = builder.ltoirs[lto_symbol]
    else:
        (lto_code_data,) = _build_lto_base(lto_symbol, compile_lto_dot, builder, {})

        # Update builder
        builder.ltoirs[lto_symbol] = lto_code_data
        builder.ltoirs_decl[lto_symbol] = (
            f"void {lto_symbol}({c_dtype}*, {a_dtype}*, {b_dtype}*, {c_dtype}*, {c_dtype}*);"
        )

    return lto_symbol, lto_code_data


def build_lto_solver(
    M,
    N,
    NRHS,
    solver,
    solver_enum,
    side_enum,
    diag_enum,
    alayout,
    blayout,
    fill_mode,
    arch,
    precision_enum,
    num_threads,
    parameter_list,
    builder,
):
    arch = 120 if arch > 121 else arch

    def cusolverdx_arrangement_map(layout):
        if layout == "colmajor":
            return 0  # CUSOLVERDX_ARRANGEMENT_COL_MAJOR
        if layout == "rowmajor":
            return 1  # CUSOLVERDX_ARRANGEMENT_ROW_MAJOR
        raise ValueError("Unsupported layout in tile_matmul")

    a_arrangement = cusolverdx_arrangement_map(alayout)
    b_arrangement = cusolverdx_arrangement_map(blayout)

    lto_symbol = f"{solver}_{M}_{N}_{NRHS}_{arch}_{num_threads}_{a_arrangement}_{b_arrangement}_{precision_enum}_{side_enum if side_enum >= 0 else 'x'}_{diag_enum if diag_enum >= 0 else 'x'}_{fill_mode}"

    def compile_lto_solver(temp_paths):
        # compile LTO
        result = warp.context.runtime.core.wp_cuda_compile_solver(
            temp_paths["_fatbin.lto"].encode("utf-8"),
            temp_paths[".lto"].encode("utf-8"),
            lto_symbol.encode("utf-8"),
            0,
            None,
            None,
            arch,
            M,
            N,
            NRHS,
            solver_enum,
            side_enum,
            diag_enum,
            precision_enum,
            a_arrangement,
            b_arrangement,
            fill_mode,
            num_threads,
        )

        if result:
            with open(temp_paths[".lto"], "rb") as f:
                lto_code_data = f.read()
            with open(temp_paths["_fatbin.lto"], "rb") as f:
                universal_fatbin_code_data = f.read()
            return True, {".lto": lto_code_data, "_fatbin.lto": universal_fatbin_code_data}
        return False, {}

    # Early out if already cached in module
    if lto_symbol in builder.ltoirs:
        lto_code_data = builder.ltoirs[lto_symbol]
    else:
        lto_code_data, universal_fatbin_code_data = _build_lto_base(
            lto_symbol, compile_lto_solver, builder, {"_fatbin.lto": get_cached_lto}
        )

        # Update builder
        builder.ltoirs[lto_symbol] = lto_code_data
        builder.ltoirs_decl[lto_symbol] = f"void {lto_symbol}{parameter_list};"
        builder.fatbins[lto_symbol] = universal_fatbin_code_data

    return lto_symbol, lto_code_data


def build_lto_fft(arch, size, ept, direction, dir, precision, builder):
    arch = 120 if arch > 121 else arch

    lto_symbol = f"fft_{size}_{ept}_{arch}_{direction}_{precision}"

    def compile_lto_fft(temp_paths):
        shared_memory_size = ctypes.c_int(0)

        result = warp.context.runtime.core.wp_cuda_compile_fft(
            temp_paths[".lto"].encode("utf-8"),
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

        if result:
            with open(temp_paths[".lto"], "rb") as f:
                lto_code_data = f.read()

            shared_memory_bytes = tile.round_up(shared_memory_size.value)

            # output meta file with shared memory requirements for this lto_symbol
            meta = {}
            meta[lto_symbol] = shared_memory_bytes

            with open(temp_paths[".meta"], "w") as meta_file:
                json.dump(meta, meta_file)

            return True, {".lto": lto_code_data, ".meta": shared_memory_bytes}

        return False, {}

    # Early out if already cached in module
    if lto_symbol in builder.ltoirs and lto_symbol in builder.shared_memory_bytes:
        lto_code_data = builder.ltoirs[lto_symbol]
        shared_memory_bytes = builder.shared_memory_bytes[lto_symbol]
    else:
        lto_code_data, shared_memory_bytes = _build_lto_base(
            lto_symbol, compile_lto_fft, builder, {".meta": lambda path: get_cached_lto_meta(path, lto_symbol)}
        )

        # Update builder
        builder.ltoirs[lto_symbol] = lto_code_data
        builder.shared_memory_bytes[lto_symbol] = shared_memory_bytes

    return lto_symbol, lto_code_data, shared_memory_bytes
