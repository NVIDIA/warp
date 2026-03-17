# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile Warp kernel to CUBIN for C++ integration.

This script demonstrates the simplest case: compiling a single kernel in the
__main__ module to a single CUBIN file. The generated CUBIN can then be loaded
and launched from C++ code.

Usage:
    python compile_kernel.py
    # or with uv:
    uv run compile_kernel.py
"""

import glob
import os

import warp as wp


@wp.kernel
def saxpy(alpha: float, x: wp.array[float], y: wp.array[float]):
    """SAXPY: Single-Precision A·X Plus Y

    Computes: ``y = alpha * x + y``.
    """
    tid = wp.tid()
    y[tid] = alpha * x[tid] + y[tid]


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(output_dir, exist_ok=True)

    # Compile the kernel to CUBIN
    print("Compiling Warp kernel to CUBIN...")
    device = wp.get_device()
    print(f"Target device: {device.name}")
    print(f"Target architecture: sm_{device.get_cuda_compile_arch()}")

    wp.compile_aot_module("__main__", module_dir=output_dir, use_ptx=False, strip_hash=True)

    # Get kernel name programmatically (after compilation, hash is computed)
    forward_kernel_name = f"{saxpy.get_mangled_name()}_cuda_kernel_forward"
    print(f"Kernel name: {forward_kernel_name}")

    # Verify CUBIN file was generated
    cubin_files = glob.glob(os.path.join(output_dir, "wp___main__.sm*.cubin"))
    if not cubin_files:
        raise RuntimeError(f"No CUBIN files generated in {output_dir}")

    print(f"\nGenerated CUBIN file: {os.path.basename(cubin_files[0])}")
    print("\nCompilation complete!")
