# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from contact_cuda import compact_contacts as _compact_contacts


def compact_contacts(
    candidate_pairs: torch.Tensor,
    separations: torch.Tensor,
    cutoff: float,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Compact valid contacts while preserving stable candidate order."""
    if candidate_pairs.device.type != "cuda" or separations.device.type != "cuda":
        raise ValueError("contact inputs must be CUDA-resident")
    if candidate_pairs.dtype != torch.int32 or separations.dtype != torch.float32:
        raise TypeError("contacts require int32 pairs and float32 separations")

    pairs, count, overflow = _compact_contacts(
        candidate_pairs,
        separations,
        cutoff,
        capacity,
    )
    return pairs, count, bool(overflow)
