# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F


def feature_stage(
    images: torch.Tensor,
    convolution_weight: torch.Tensor,
    projection_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply the dense feature-extraction stage."""
    features = F.conv2d(images, convolution_weight, padding=1)
    features = F.layer_norm(features, features.shape[-3:])
    pooled = F.adaptive_avg_pool2d(F.gelu(features), output_size=1).flatten(1)
    return pooled @ projection_weight
