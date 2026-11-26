# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise array-family specialization against cold and warm caches in a fresh process."""

import json
import sys
from typing import Any
from unittest import mock

import numpy as np

import warp as wp
from warp._src import build, types
from warp.tests.test_fabricarray import _create_fabric_array_interface


@wp.kernel(enable_backward=False)
def read_generic(source: Any, out: wp.array[float]):
    out[wp.tid()] = source[wp.tid()] * 2.0


@wp.kernel(enable_backward=False)
def read_regular(source: wp.array[float], out: wp.array[float]):
    out[wp.tid()] = source[wp.tid()] * 2.0


def main(device_name, cache_state):
    wp.init()
    device = wp.get_device(device_name)
    families = (wp.array, wp.indexedarray, wp.fabricarray, wp.indexedfabricarray)
    expected = np.arange(1, 5, dtype=np.float32)
    storage = wp.array(expected, dtype=float, device=device)
    indices = wp.array([3, 1, 0, 2], dtype=wp.int32, device=device)
    fabric = wp.fabricarray(_create_fabric_array_interface(storage, "a", [2, 2]), "a")
    sources = (
        storage,
        wp.indexedarray(storage, indices),
        fabric,
        wp.indexedfabricarray(fabric, indices),
    )
    values = (expected, expected[[3, 1, 0, 2]], expected, expected[[3, 1, 0, 2]])
    out = wp.empty(4, dtype=float, device=device)

    # Register every family before the first module load so adding unrelated
    # specializations does not rebuild the module during this comparison.
    for family in families:
        wp.overload(read_generic, [family[float], wp.array[float]])
    signatures = []
    module_hash = read_generic.module.hash_module()
    with (
        mock.patch.object(build, "build_cpu", wraps=build.build_cpu) as build_cpu,
        mock.patch.object(build, "build_cuda", wraps=build.build_cuda) as build_cuda,
    ):
        for family, source, expected_values in zip(families, sources, values, strict=True):
            subclass = type(f"Child{family.__name__}", (family,), {})
            sibling = type(f"Sibling{family.__name__}", (family,), {})
            overload = None
            for cls in (family, subclass, sibling):
                source.__class__ = cls
                arg_types = read_generic.infer_argument_types([source, out])
                signature = types.get_signature(arg_types)
                candidate = read_generic.add_overload(arg_types)
                if overload is None:
                    overload = candidate
                    signatures.append(signature)
                elif candidate is not overload or signature != signatures[-1]:
                    raise AssertionError(f"{family.__name__} subclasses created a different specialization")
                wp.launch(read_generic, 4, inputs=[source], outputs=[out], device=device)
                np.testing.assert_allclose(out.numpy(), expected_values * 2.0)
                if read_generic.module.hash_module() != module_hash:
                    raise AssertionError("An application subclass changed the compiled module hash")

        # Rejections must still happen after the specialized code is loaded,
        # including when that code came from another process's cache.
        wp.launch(read_regular, 4, inputs=[storage], outputs=[out], device=device)
        rejections = 0
        for source in sources[1:]:
            try:
                wp.launch(read_regular, 4, inputs=[source], outputs=[out], device=device)
            except RuntimeError as error:
                if "expects an array of type" not in str(error):
                    raise
                rejections += 1
            else:
                raise AssertionError("A wrong-family input reached a cached regular-array kernel")

        native_builds = build_cpu.call_count + build_cuda.call_count
        if cache_state == "warm" and native_builds:
            raise AssertionError("The warm-cache process compiled native code again")
        if cache_state == "cold" and not native_builds:
            raise AssertionError("The cold-cache process did not compile its kernel module")

    print(
        "ARRAY_SUBCLASS_CACHE_RESULT="
        + json.dumps(
            {
                "signatures": signatures,
                "module_hash": module_hash.hex(),
                "native_builds": native_builds,
                "wrong_family_rejections": rejections,
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
