# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Used to test reloading module references through functions bound to kernel-locals."""

import warp as wp
import warp.tests.aux_test_reference as ref
import warp.tests.aux_test_reference_reference as refref


@wp.kernel
def kern(expect: float):
    # Bind the referenced module's function to a kernel-local before calling it.
    f = ref.magic
    wp.expect_eq(f(), expect)


@wp.kernel
def kern_tuple(expect: float):
    # Bind functions from two modules to locals via tuple unpacking. Only this kernel
    # references aux_test_reference_reference directly, so the dependency on it exercises
    # the tuple branch of Module._find_references.
    f, g = ref.magic, refref.more_magic
    wp.expect_eq(f() + g(), expect)


def run(expect, device):
    wp.launch(kern, dim=1, inputs=[expect], device=device)
    wp.synchronize_device(device)
