# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ctypes
import os
import platform
import re
import sys
import warnings
from importlib.metadata import PackageNotFoundError, files

CUDA_HOME = None
MATHDX_HOME = None
CUTLASS_HOME = None


PLATFORM_LINUX = sys.platform.startswith("linux")
PLATFORM_WIN = sys.platform.startswith("win32")


def _conda_get_target_name():
    if PLATFORM_LINUX:
        plat = platform.processor()
        if plat == "aarch64":
            return "sbsa-linux"
        else:
            return f"{plat}-linux"
    elif PLATFORM_WIN:
        return "x64"
    else:
        raise AssertionError


def _check_cuda_home():
    # We need some CUDA headers for compiling mathDx headers.
    # We assume users properly managing their local envs (ex: no mix-n-match).
    global CUDA_HOME

    # Try wheel
    try:
        # We need CUDA 12+ for device API support
        cudart = files("nvidia-cuda-runtime-cu12")
        cccl = files("nvidia-cuda-cccl-cu12")
        # use cuda_fp16.h (which we need) as a proxy
        cudart = [f for f in cudart if "cuda_fp16.h" in str(f)][0]
        cudart = os.path.join(os.path.dirname(cudart.locate()), "..")
        # use cuda/std/type_traits as a proxy
        cccl = min([f for f in cccl if re.match(".*cuda\\/std\\/type_traits.*", str(f))], key=lambda x: len(str(x)))
        cccl = os.path.join(os.path.dirname(cccl.locate()), "../../..")
    except PackageNotFoundError:
        pass
    except ValueError:
        # cccl wheel is buggy (headers missing), skip using wheels
        pass
    else:
        CUDA_HOME = (cudart, cccl)
        return

    # Try conda
    if "CONDA_PREFIX" in os.environ:
        if PLATFORM_LINUX:
            conda_include = os.path.join(
                os.environ["CONDA_PREFIX"], "targets", f"{_conda_get_target_name()}", "include"
            )
        elif PLATFORM_WIN:
            conda_include = os.path.join(os.environ["CONDA_PREFIX"], "Library", "include")
        else:
            assert AssertionError
        if os.path.isfile(os.path.join(conda_include, "cuda_fp16.h")) and os.path.isfile(
            os.path.join(conda_include, "cuda/std/type_traits")
        ):
            CUDA_HOME = (os.path.join(conda_include, ".."),)
            return

    # Try local
    CUDA_PATH = os.environ.get("CUDA_PATH", None)
    CUDA_HOME = os.environ.get("CUDA_HOME", None)
    if CUDA_PATH is None and CUDA_HOME is None:
        raise RuntimeError(
            "cudart headers not found. Depending on how you install nvmath-python and other CUDA packages,\n"
            "you may need to perform one of the steps below:\n"
            "  - conda install -c conda-forge cuda-cudart-dev cuda-cccl cuda-version=12\n"
            "  - export CUDA_HOME=/path/to/CUDA/Toolkit"
        )
    elif CUDA_PATH is not None and CUDA_HOME is None:
        CUDA_HOME = CUDA_PATH
    elif CUDA_PATH is not None and CUDA_HOME is not None:
        if CUDA_HOME != CUDA_PATH:
            warnings.warn(
                "Both CUDA_HOME and CUDA_PATH are set but not consistent. " "Ignoring CUDA_PATH...", stacklevel=2
            )
    CUDA_HOME = (CUDA_HOME,)


def _check_mathdx_home():
    # Find mathDx headers
    global MATHDX_HOME

    # Try wheel
    try:
        MATHDX_HOME = files("nvidia-mathdx")
    except PackageNotFoundError:
        pass
    else:
        # use cufftdx.hpp as a proxy
        MATHDX_HOME = [f for f in MATHDX_HOME if "cufftdx.hpp" in str(f)][0]
        MATHDX_HOME = os.path.join(os.path.dirname(MATHDX_HOME.locate()), "..")
        return

    # Try conda
    if "CONDA_PREFIX" in os.environ:
        if PLATFORM_LINUX:
            conda_include = os.path.join(os.environ["CONDA_PREFIX"], "include")
        elif PLATFORM_WIN:
            conda_include = os.path.join(os.environ["CONDA_PREFIX"], "Library", "include")
        if os.path.isfile(os.path.join(conda_include, "cufftdx.hpp")):
            MATHDX_HOME = os.path.join(conda_include, "..")
            return

    # Try local
    if "MATHDX_HOME" not in os.environ:
        raise RuntimeError(
            "mathDx headers not found. Depending on how you install nvmath-python and other CUDA packages, "
            "you may need to perform one of the steps below:\n"
            "   - pip install nvidia-mathdx\n"
            "   - conda install -c conda-forge mathdx\n"
            "   - export MATHDX_HOME=/path/to/mathdx"
        )
    else:
        MATHDX_HOME = os.environ["MATHDX_HOME"]


def get_mathdx_include_dirs():
    _check_mathdx_home()

    global MATHDX_HOME
    return (MATHDX_HOME + "/include").encode("utf-8")


def get_cuda_include_dirs():
    _check_cuda_home()

    global CUDA_HOME
    include_dirs = [(f"{h}" + "/include").encode("utf-8") for h in CUDA_HOME]
    arr_include_dirs = (ctypes.c_char_p * len(include_dirs))()
    arr_include_dirs[:] = include_dirs
    return arr_include_dirs
