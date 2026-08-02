# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Triton-backed inference service.

Triton JIT-compiles the kernel on the first request after a deploy, which is why
the first call is slow. The compiled artifacts land in a cache directory under
/tmp, so a pod restart throws them away and the next first call pays again.
"""

import os

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    out_ptr,
    in_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    row_start = in_ptr + row * in_row_stride
    values = tl.load(row_start + cols, mask=mask, other=-float("inf"))

    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    result = numerator / tl.sum(numerator, axis=0)

    out_start = out_ptr + row * out_row_stride
    tl.store(out_start + cols, result, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax backed by the Triton kernel above."""
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](
        out,
        x,
        x.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
    )
    return out


def handle_request(batch: torch.Tensor) -> torch.Tensor:
    """Serve one request. The first call pays the Triton JIT compile."""
    return softmax(batch)


def main() -> None:
    batch = torch.randn(8, 512, device="cuda")
    print(handle_request(batch).sum().item())


if __name__ == "__main__":
    main()
