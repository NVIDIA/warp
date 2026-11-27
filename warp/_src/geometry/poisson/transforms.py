# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference cofactor inverses in the input precision for world queries."""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp

wp.set_module_options({"fuse_fp": False})


@lru_cache(None)
def precision_kernels(dtype):
    mat33 = wp.types.matrix(shape=(3, 3), dtype=dtype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=dtype)

    @wp.func
    def _det3(m: mat33):
        return (
            m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
            - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
            + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
        )

    @wp.func
    def _minor(m: mat44, row: int, col: int):
        out = mat33(dtype(0.0))
        rr = int(0)
        for r in range(4):
            if r != row:
                cc = int(0)
                for c in range(4):
                    if c != col:
                        out[rr, cc] = m[r, c]
                        cc += 1
                rr += 1
        return _det3(out)

    @wp.func
    def reference_inverse(m: mat44):
        det = dtype(0.0)
        for c in range(4):
            if c % 2 == 0:
                det += m[0, c] * _minor(m, 0, c)
            else:
                det -= m[0, c] * _minor(m, 0, c)
        out = mat44(dtype(0.0))
        for r in range(4):
            for c in range(4):
                value = _minor(m, r, c)
                if (r + c) % 2 != 0:
                    value = -value
                out[c, r] = value / det
        return out

    @wp.kernel
    def evaluation_transform_kernel(normalization: wp.array[dtype], transforms: wp.array[mat44]):
        s = normalization[0]
        m = mat44(
            dtype(0.0),
            s,
            dtype(0.0),
            normalization[1],
            dtype(0.0),
            dtype(0.0),
            s,
            normalization[2],
            s,
            dtype(0.0),
            dtype(0.0),
            normalization[3],
            dtype(0.0),
            dtype(0.0),
            dtype(0.0),
            dtype(1.0),
        )
        inverse = reference_inverse(m)
        transforms[0] = inverse
        transforms[1] = reference_inverse(inverse)

    return SimpleNamespace(
        _det3=_det3,
        _minor=_minor,
        reference_inverse=reference_inverse,
        evaluation_transform=evaluation_transform_kernel,
    )
