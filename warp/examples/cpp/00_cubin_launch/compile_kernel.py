# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
def saxpy(alpha: wp.float32, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32)):
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
